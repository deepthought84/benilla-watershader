//! The stylised water's **wave field** — the one source of every wave a body makes on the water,
//! so that a swimmer's wake is water actually moving rather than a picture of a wake.
//!
//! **Where this pushes, it is the alternative to `water_fx`'s wake, not an addition to it.** That
//! module is the reference's own `CWater0Ripple`: a pool of *records*, each a splash stencil
//! (`splash.blp` / `wake.blp`) stretched over a static patch of geometry by a texture matrix as the
//! record ages. It is byte-verified against the 1.12 client and it stays exactly as it is on the
//! reference lane. What it cannot do is what a painted thing never can: two wakes crossing do not
//! interfere, a wake does not stay where it was made when the body turns, and nothing that is not
//! a unit can disturb the water at all. So on the stylised lane ([`super::WaterStyle::Stylised`])
//! this pushes for every body in or under the water, from ankle depth on
//! ([`benilla_world::water_fx::params::wave_push`] is the decals' gate: there they leave the wake to this),
//! and there the decal wake is silenced; the rings and the step-in splash stay decals.
//!
//! ## What it solves and how
//!
//! Linear deep-water gravity waves, `ω² = g·k`, under a moving surface pressure: in Fourier space
//! each wavenumber `k` of the height `ĥ` obeys
//!
//! ```text
//! ĥ'' = −ω²·(ĥ + D̂)        ω = √(g·|k|)
//! ```
//!
//! where `D̂` is the pressure patch's footprint in yards of static depression (a body floating on
//! water presses it down by the water it displaces: `D̂ = Σ bodies a·Gaussian(σ)`, each centred on
//! its body). That equation is solved **exactly** per step for any `Δt` — a rotation of
//! `(ĥ + D̂, v̂/ω)` by `ω·Δt` — so there is no stability limit, no substep cadence and no numerical
//! dispersion: the field's dispersion is the water's. A body dragged through the lattice leaves a
//! Kelvin wedge behind it because that is what `ω² = g·k` does to a moving source: every component
//! whose phase speed `√(g/k)·secθ` matches the body's stays with it, and their stationary phase
//! piles up along the 19.47° cusp lines, with the transverse crests between them and the divergent
//! ones outside; none of that is a case anywhere in this file. When the body turns or stops, the
//! waves it has made keep their own speeds and stay where they were made — the source simply moves
//! or stops moving. An ordinary wave equation (the scheme this replaces) carries every wavelength at
//! one speed and so draws rings or a fixed `arcsin(c/v)` wedge, never the V the water draws.
//!
//! The step is spectral: the height and its rate are packed into one complex field
//! (`h + i·v`, two reals for the price of one FFT), transformed with `rustfft`, separated by
//! Hermitian symmetry, rotated per cell, repacked and transformed back. The pressure footprint is
//! never rasterised: a Gaussian's transform is a Gaussian times a phase, both separable, so each
//! body costs two 1-D tables and an outer product. A body under the surface still presses on it,
//! through the water above: its footprint is filtered by `e^{−|k|·depth}` (the evanescent decay of
//! a source at that depth, exact for a point source), so a diver makes a soft, long hump and no
//! ripple, which fades to nothing as he goes down.
//!
//! Damping is the old law in time units ([`DAMP_TAU`], the e-fold the 90 Hz scheme's per-step
//! retention gave) plus a viscous `ν·k²` ([`VISCOSITY`]) that takes the two-cell ripples out
//! first and leaves the long Kelvin arms alone. Strokes are impulses on the rate field behind the
//! body, which the same equation turns into Cauchy–Poisson rings. Foam is a field of its own,
//! generated where the simulated surface heaves faster than calm water can (the face of a moving
//! body's trough and bow, a stroke's kick) and where a stroke kicks the water, ageing in place: a
//! body at rest sits in its trough and makes none.
//!
//! ## The window
//!
//! The lattice is not the world — it is a [`SIM_YARDS`]-square window centred on the viewer,
//! carried along as they move. It moves in **whole cells only**: a fractional scroll would need the
//! field resampled every frame, and resampling a wave field is a low-pass filter, so the ripples
//! would visibly dissolve whenever the player walked. A whole-cell shift is a memmove, and it is
//! exact. The transform is periodic: a wave leaving the window would come in on the far side, which
//! the world has no reason for. [`EDGE_ABSORB`] cells of progressive damping around the rim swallow
//! it instead, and the shader fades the whole contribution out over the same band, so a wave
//! leaving the simulated area leaves quietly.

use std::sync::Arc;

use bevy::asset::RenderAssetUsages;
use bevy::ecs::entity::EntityHashMap;
use bevy::image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::{
    Extent3d, TexelCopyBufferLayout, TextureDimension, TextureFormat, TextureUsages,
};
use bevy::render::renderer::RenderQueue;
use bevy::render::texture::GpuImage;
use bevy::render::{Render, RenderApp, RenderSystems};
use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

use super::reflect::WaterReflectData;
use super::{WaterChunkInfo, WaterIndex, WaterStyle};
use crate::world_unit::{ViewerUnit, WorldUnit};
use benilla_assets::coords::bevy_to_wow;
use benilla_assets::preview as lever;

/// Lattice resolution, per side. 192 = 2⁶·3, a size the mixed-radix transform takes in its stride.
const SIM_CELLS: usize = 192;

/// How much world the window covers, per side, in yards. With [`SIM_CELLS`] this puts a cell at a
/// quarter of a yard: fine enough that the shortest wave the lattice can carry (two cells, half a
/// yard) is a ripple rather than a swell, and coarse enough that the window still reaches
/// twenty-four yards out — past the distance at which a wake is worth resolving.
const SIM_YARDS: f32 = 48.0;

/// Cell pitch in yards.
const SIM_CELL: f32 = SIM_YARDS / SIM_CELLS as f32;

/// Gravity in yards per second squared (9.81 m/s²): the dispersion's one constant.
const GRAVITY: f32 = 10.7283;

/// The step's quantum, seconds: a frame's time is rounded to it (the remainder carried to the
/// next), so a steady frame rate steps by one `dt` frame after frame and the per-cell rotation
/// tables are reused rather than recomputed. Half of it is the most the field's clock is ever off
/// the world's.
const SIM_DT_QUANTUM: f32 = 1.0 / 1200.0;

/// The longest step the field takes at once, seconds. The integrator is exact for any step; the
/// cap is for the sources, which move between steps, so a stall does not drag a body a yard in one
/// step and ring the window. Past it the simulation simply loses time, which looks like the water
/// briefly slowing and is the right failure.
const SIM_DT_MAX: f32 = 0.1;

/// The field's e-fold time, seconds: the old scheme's per-step retention of 0.9955 at 90 steps a
/// second. Ripples have to die out eventually or the window fills with the memory of everywhere the
/// player has been; the reference footage shows a trail still legible fifteen yards behind the
/// swimmer, which at swimming pace is several seconds old, and this keeps a third of it there.
const DAMP_TAU: f32 = 2.46;

/// Viscous damping `ν·k²`, yd²/s: 0.6/s on the two-cell ripple (k = 12.6 rad/yd), nothing on the
/// transverse Kelvin wave at swim speed (k = 0.48, 0.001/s). The lattice's finest waves are the
/// ones a body's edge excites and a byte cannot draw; they go first.
const VISCOSITY: f32 = 0.004;

/// Width of the absorbing rim, in cells, and its retention per 1/90 s at the outermost live cell —
/// see the module doc. Linear from the floor up to no extra loss at the band's inner edge, gradual
/// on purpose: an abrupt change in damping is itself an impedance step, and a wave reflects off one
/// of those too.
const EDGE_ABSORB: usize = 14;
const EDGE_ABSORB_FLOOR: f32 = 0.72;
const EDGE_ABSORB_RATE: f32 = 90.0;

/// A body's footprint σ over its height, and the footprint's bounds in yards: a human (2.03 yd)
/// presses on a σ = 0.45 yd Gaussian, a yard across at half depth.
const BODY_SIGMA_PER_HEIGHT: f32 = 0.22;
const BODY_SIGMA_RANGE: [f32; 2] = [0.2, 2.5];

/// The static depression a body floating at the surface makes, over its σ: 0.11 yd for a human,
/// about the water a body displaces over a square yard. A body at rest sits in it and the water is
/// still; under way the trough runs with the body and its faces heave ([`FOAM_BREAK_SPEED`]).
const BODY_STATIC_PER_SIGMA: f32 = 0.25;

/// The dynamic part, over σ at swim speed, growing as the speed squared (the bow's stagnation head
/// does): what a body under way presses beyond what it floats on.
const BODY_DYNAMIC_PER_SIGMA: f32 = 0.30;

/// The speed the dynamic part is set against, yd/s: a swimmer's.
const SWIM_SPEED: f32 = 4.72;

/// How much of a body is under the surface when it swims on it: a human's feet float 1.52 yd down
/// of its 2.03 (`preview_rows`' phantom). Water deeper than that over the feet is water over the
/// body, and the body is submerged by the difference.
const SWIM_FLOAT_FRAC: f32 = 0.75;

/// How far ahead of the body's position its pressure sits when under way, in yards: the bow. A
/// real bow wave starts at the bow, and the model's position is its middle.
const PUSH_BOW_OFFSET: f32 = 0.55;

/// How deep the water under a body's feet has to be before it presses on the surface at all, in
/// yards: ankle deep, since shallower the body is walking on wet sand (the decals' gate too).
const PUSH_MIN_DEPTH: f32 = 0.35;

/// How far a body must move in one frame to count as moving at all, in yards. Small, but not zero:
/// a streamed position that jitters by a thousandth of a yard is a body standing still.
const MOVED_EPSILON: f32 = 1.0e-3;

/// The speed estimate's smoothing, seconds: a streamed unit's position arrives in bursts.
const SPEED_TAU: f32 = 0.25;

/// A stroke every this many yards swum at the surface (two a second at swim speed, one arm then
/// the other), from this speed on; the kick's σ over the body's, how far behind and beside the body
/// it lands (in the body's σ), the rate it adds to the water (yd/s) and the foam it leaves.
const STROKE_YD: f32 = 2.4;
const STROKE_SPEED_MIN: f32 = 1.0;
const STROKE_SIGMA: f32 = 0.5;
const STROKE_BACK: f32 = 0.8;
const STROKE_SIDE: f32 = 0.7;
const STROKE_RATE: f32 = 0.6;
const STROKE_FOAM: f32 = 0.25;

/// Foam: its life in seconds (an e-fold), the surface's vertical speed past which the water breaks
/// and aerates, yd/s, and how fast it whitens there (per second per yd/s over the break). A
/// swimmer's trough and bow heave at up to 0.9 yd/s within a yard of the body, the Kelvin arms at
/// under 0.3 a few yards back, and a body at rest not at all: the churn at the head and the kicked
/// trail are the moving body's, and a wader standing still has no ring of foam. (A threshold on
/// the slope whitened the static trough of every body at rest for ever.) The rate keeps a
/// swimmer's churn near 0.4 of full cover, so the lace reads as a raft breaking up, not a
/// sheet; the strokes' kicks top it up in patches.
const FOAM_TAU: f32 = 2.0;
const FOAM_BREAK_SPEED: f32 = 0.3;
const FOAM_BREAK_RATE: f32 = 2.0;

/// `$WOW_PREVIEW_WAKE_FOAM_GAIN=<0-8>` (1): the breaking foam's rate over [`FOAM_BREAK_RATE`] and
/// the strokes' foam over [`STROKE_FOAM`], read once, to set the look.
fn foam_gain() -> f32 {
    static GAIN: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *GAIN.get_or_init(|| lever::number("WAKE_FOAM_GAIN", 1.0, 0.0, 8.0))
}

/// Below this peak height and rate (yd, yd/s) the field is under a byte of the texture: with no
/// body pressing and no stroke owed, it is cleared once and the step is skipped until one comes.
const QUIET: f32 = 2.0e-4;

/// The encode's scales: the slope that fills a channel's half range before the soft limit (the
/// Kelvin arms of a swimmer run 0.005-0.03, the trough's flank 0.15; the shader has the gain), and
/// the height that fills the blue channel's, yards.
const ENC_SLOPE: f32 = 8.0;
const ENC_HEIGHT: f32 = 4.0;

/// One body's press on the surface this step.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Source {
    /// Its centre in world XZ yards.
    at: Vec2,
    /// The footprint's σ, yards.
    sigma: f32,
    /// The depression's depth, yards (negative height).
    amount: f32,
    /// How far under the surface the body is, yards: 0 at the surface.
    submerged: f32,
}

/// The spectral lattice: the transform and the per-cell tables the step reads.
struct Spectral {
    fwd: Arc<dyn Fft<f32>>,
    inv: Arc<dyn Fft<f32>>,
    scratch: Vec<Complex<f32>>,
    /// The signed wavenumber along one axis per index, rad/yd (0 at the Nyquist index: no
    /// footprint lands there).
    k1: Vec<f32>,
    /// Per cell: |k| and ω = √(g·|k|).
    kmag: Vec<f32>,
    omega: Vec<f32>,
    /// The last step's rotation, per cell: (cos ω·dt, sin(ω·dt)/ω, ω·sin ω·dt) and the decay,
    /// kept while the step stays the same (a capture's fixed step, a steady frame rate:
    /// [`SIM_DT_QUANTUM`]). At ω = 0 (the mean, and the Nyquist axes, which carry nothing) the
    /// middle term is `dt`, the drift of a free mode.
    rot_dt: f32,
    rot_cs: Vec<[f32; 3]>,
    rot_decay: Vec<f32>,
}

impl Spectral {
    fn new() -> Self {
        let n = SIM_CELLS;
        let mut planner = FftPlanner::new();
        let fwd = planner.plan_fft_forward(n);
        let inv = planner.plan_fft_inverse(n);
        let scratch = vec![
            Complex::new(0.0, 0.0);
            fwd.get_inplace_scratch_len()
                .max(inv.get_inplace_scratch_len())
        ];
        let dk = std::f32::consts::TAU / SIM_YARDS;
        let k1: Vec<f32> = (0..n)
            .map(|j| {
                if j == n / 2 {
                    0.0
                } else if j < n / 2 {
                    j as f32 * dk
                } else {
                    (j as f32 - n as f32) * dk
                }
            })
            .collect();
        let mut kmag = vec![0.0; n * n];
        let mut omega = vec![0.0; n * n];
        for y in 0..n {
            for x in 0..n {
                let k = (k1[x] * k1[x] + k1[y] * k1[y]).sqrt();
                kmag[y * n + x] = k;
                omega[y * n + x] = (GRAVITY * k).sqrt();
            }
        }
        Self {
            fwd,
            inv,
            scratch,
            k1,
            kmag,
            omega,
            rot_dt: 0.0,
            rot_cs: vec![[1.0, 0.0, 0.0]; n * n],
            rot_decay: vec![1.0; n * n],
        }
    }

    /// The rotation tables for a step of `dt`, recomputed only when the step changes.
    fn rotation(&mut self, dt: f32) {
        if dt == self.rot_dt {
            return;
        }
        self.rot_dt = dt;
        let decay0 = (-dt / DAMP_TAU).exp();
        for i in 0..SIM_CELLS * SIM_CELLS {
            let w = self.omega[i];
            let (s, c) = (w * dt).sin_cos();
            self.rot_cs[i] = [c, if w > 0.0 { s / w } else { dt }, w * s];
            let k = self.kmag[i];
            self.rot_decay[i] = decay0 * (-dt * VISCOSITY * k * k).exp();
        }
    }

    /// The 2-D transform in place: every row, a transpose, every row again, and back.
    fn fft2(&mut self, z: &mut [Complex<f32>], inverse: bool) {
        let plan = if inverse { &self.inv } else { &self.fwd };
        rows_fft(plan, z, &mut self.scratch);
        transpose(z);
        rows_fft(plan, z, &mut self.scratch);
        transpose(z);
    }
}

/// Every row of the lattice through `plan`. (Bands of rows on the compute pool measured slower in
/// the client, 2.0 ms a frame against 1.4: the pool is busy with the frame's other systems.)
fn rows_fft(plan: &Arc<dyn Fft<f32>>, z: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) {
    plan.process_with_scratch(z, scratch);
}

/// A square matrix's transpose in place.
fn transpose(z: &mut [Complex<f32>]) {
    let n = SIM_CELLS;
    for y in 0..n {
        for x in (y + 1)..n {
            z.swap(y * n + x, x * n + y);
        }
    }
}

/// The live height field and the image the shader reads it through.
#[derive(Resource)]
pub(crate) struct RippleSim {
    /// The texture the liquid materials bind: R/G the surface slope, B the height, A the foam.
    /// Same slope convention as the tiling ripple map beside it (`super::ripple`), so the shader
    /// reads the two the same way.
    pub(crate) image: Handle<Image>,
    /// The height (yards) and its rate (yd/s), and the foam on it (0-1).
    h: Vec<f32>,
    v: Vec<f32>,
    foam: Vec<f32>,
    /// The packed field `h + i·v` and the step's output, kept allocated across frames.
    z: Vec<Complex<f32>>,
    next: Vec<Complex<f32>>,
    /// The press of one body, separable: its per-axis tables.
    gx: Vec<Complex<f32>>,
    gy: Vec<Complex<f32>>,
    spectral: Spectral,
    /// The encoded RGBA the render world uploads — staged here so the step and the encode stay in
    /// the main world and the render side only ever copies bytes.
    pixels: Vec<u8>,
    /// The window's lower corner, as a whole number of cells in **Bevy** XZ. Whole cells is the
    /// point — see the module doc.
    origin: IVec2,
    /// Has the window ever been placed? Until it has, there is nothing to scroll *from*.
    placed: bool,
    /// Frame time not yet stepped, under half a [`SIM_DT_QUANTUM`].
    accum: f32,
    /// The field is flat and cleared ([`QUIET`]): nothing to step until a body presses.
    quiet: bool,
}

impl RippleSim {
    fn new(image: Handle<Image>) -> Self {
        let n = SIM_CELLS * SIM_CELLS;
        Self {
            image,
            h: vec![0.0; n],
            v: vec![0.0; n],
            foam: vec![0.0; n],
            z: vec![Complex::new(0.0, 0.0); n],
            next: vec![Complex::new(0.0, 0.0); n],
            gx: vec![Complex::new(0.0, 0.0); SIM_CELLS],
            gy: vec![Complex::new(0.0, 0.0); SIM_CELLS],
            spectral: Spectral::new(),
            pixels: vec![0u8; n * 4],
            origin: IVec2::ZERO,
            placed: false,
            accum: 0.0,
            quiet: false,
        }
    }

    /// This frame's step from the frame's time: the quantum's multiple nearest the time owed.
    fn take_dt(&mut self, frame: f32) -> f32 {
        let owed = self.accum + frame.clamp(0.0, SIM_DT_MAX);
        let dt = (owed / SIM_DT_QUANTUM).round() * SIM_DT_QUANTUM;
        self.accum = owed - dt;
        dt
    }

    /// The window's lower corner in world yards.
    fn corner(&self) -> Vec2 {
        self.origin.as_vec2() * SIM_CELL
    }

    /// A world XZ point in cells from the corner, cell centres at half integers.
    fn cells(&self, at: Vec2) -> Vec2 {
        (at - self.corner()) / SIM_CELL
    }

    /// Carry the window to a new centre, shifting the field by whole cells and clearing whatever
    /// scrolls in. A jump further than the window is a teleport: nothing of the old field is
    /// relevant, so it all goes.
    fn recentre(&mut self, centre: Vec2) {
        let want = IVec2::new(
            (centre.x / SIM_CELL).round() as i32 - (SIM_CELLS / 2) as i32,
            (centre.y / SIM_CELL).round() as i32 - (SIM_CELLS / 2) as i32,
        );
        if !self.placed {
            self.origin = want;
            self.placed = true;
            return;
        }
        let d = want - self.origin;
        if d == IVec2::ZERO {
            return;
        }
        self.origin = want;
        if d.x.unsigned_abs() as usize >= SIM_CELLS || d.y.unsigned_abs() as usize >= SIM_CELLS {
            self.h.fill(0.0);
            self.v.fill(0.0);
            self.foam.fill(0.0);
            return;
        }
        // The cell that was at (x + d) is now at x. Copying in the direction of travel would
        // overwrite sources before reading them, so each axis walks the way that cannot.
        for buf in [&mut self.h, &mut self.v, &mut self.foam] {
            let rows: Box<dyn Iterator<Item = usize>> = if d.y > 0 {
                Box::new(0..SIM_CELLS)
            } else {
                Box::new((0..SIM_CELLS).rev())
            };
            for y in rows {
                let sy = y as i32 + d.y;
                let cols: Box<dyn Iterator<Item = usize>> = if d.x > 0 {
                    Box::new(0..SIM_CELLS)
                } else {
                    Box::new((0..SIM_CELLS).rev())
                };
                for x in cols {
                    let sx = x as i32 + d.x;
                    let v = if (0..SIM_CELLS as i32).contains(&sx)
                        && (0..SIM_CELLS as i32).contains(&sy)
                    {
                        buf[sy as usize * SIM_CELLS + sx as usize]
                    } else {
                        0.0
                    };
                    buf[y * SIM_CELLS + x] = v;
                }
            }
        }
    }

    /// Add a stroke at `at` (world XZ) of `sigma` yards to `which` (0 the rate, 1 the foam): the
    /// kick and the foam it leaves, the only things added in real space. The kick is a Ricker
    /// (`(1 − r²/2σ²)·e^{−r²/2σ²}`, `amount` its peak rate), which displaces water without adding
    /// any: a plain Gaussian's net volume went into the longest waves, a slow hump that swamped the
    /// ring. The foam is the Gaussian, capped at 1.
    fn splat(&mut self, which: usize, at: Vec2, sigma: f32, amount: f32) {
        let c = self.cells(at);
        let s = sigma / SIM_CELL;
        let r = 3.5 * s;
        let lo = ((c - r).floor().max(Vec2::ZERO)).as_ivec2();
        let hi = ((c + r).ceil().min(Vec2::splat((SIM_CELLS - 1) as f32))).as_ivec2();
        if lo.x > hi.x || lo.y > hi.y {
            return;
        }
        let buf = if which == 0 {
            &mut self.v
        } else {
            &mut self.foam
        };
        for y in lo.y..=hi.y {
            for x in lo.x..=hi.x {
                let d2 = (Vec2::new(x as f32 + 0.5, y as f32 + 0.5) - c).length_squared();
                let q = 0.5 * d2 / (s * s);
                let g = (-q).exp();
                let i = y as usize * SIM_CELLS + x as usize;
                buf[i] = if which == 0 {
                    buf[i] + amount * (1.0 - q) * g
                } else {
                    (buf[i] + amount * g).min(1.0)
                };
            }
        }
    }

    /// One body's footprint tables: per axis, the Gaussian's transform times the phase that puts
    /// it at the body, `√(2π)·s·e^{−k²σ²/2}·e^{−i·k·x}` with `s` the σ in cells (the transform of
    /// a Gaussian sampled on the lattice, as the inverse divides by N² later).
    fn body_tables(&mut self, src: &Source) {
        let c = self.cells(src.at);
        // The lattice's cell `j` is at (j + ½) cells; the phase is against the body's offset from
        // cell 0's centre.
        let q = (c - 0.5) * SIM_CELL;
        let s = src.sigma / SIM_CELL;
        let norm = (std::f32::consts::TAU).sqrt() * s;
        let g2 = 0.5 * src.sigma * src.sigma;
        for j in 0..SIM_CELLS {
            let k = self.spectral.k1[j];
            let env = if j == SIM_CELLS / 2 {
                0.0
            } else {
                norm * (-g2 * k * k).exp()
            };
            let (sx, cx) = (-k * q.x).sin_cos();
            let (sy, cy) = (-k * q.y).sin_cos();
            self.gx[j] = Complex::new(cx * env, sx * env);
            self.gy[j] = Complex::new(cy * env, sy * env);
        }
    }

    /// Whether the field is under [`QUIET`] everywhere, foam included; if so it is zeroed, so the
    /// step can be skipped with the texture showing flat water.
    fn settle(&mut self) -> bool {
        let peak = |b: &[f32]| b.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        if peak(&self.h) < QUIET && peak(&self.v) < QUIET && peak(&self.foam) < 1.0 / 512.0 {
            self.h.fill(0.0);
            self.v.fill(0.0);
            self.foam.fill(0.0);
            return true;
        }
        false
    }

    /// One step of `dt` seconds under the bodies' press: the exact rotation of every wavenumber's
    /// `(ĥ + D̂, v̂/ω)` by `ω·dt`, damped; then the rim.
    fn step(&mut self, dt: f32, sources: &[Source]) {
        let n = SIM_CELLS;
        let nn = n * n;
        for ((z, &h), &v) in self.z.iter_mut().zip(&self.h).zip(&self.v) {
            *z = Complex::new(h, v);
        }
        let mut z = std::mem::take(&mut self.z);
        self.spectral.fft2(&mut z, false);
        // The forcing: every body's footprint, summed per cell.
        let mut forcing = std::mem::take(&mut self.next);
        forcing.fill(Complex::new(0.0, 0.0));
        for src in sources {
            if src.amount == 0.0 {
                continue;
            }
            self.body_tables(src);
            let sub = src.submerged.max(0.0);
            for y in 0..n {
                let gy = self.gy[y] * src.amount;
                for x in 0..n {
                    let i = y * n + x;
                    let mut g = gy * self.gx[x];
                    if sub > 0.0 {
                        // A source under the surface reaches it through the water: `e^{−|k|·d}`.
                        g *= (-self.spectral.kmag[i] * sub).exp();
                    }
                    forcing[i] += g;
                }
            }
        }
        self.spectral.rotation(dt);
        for y in 0..n {
            let my = (n - y) % n;
            for x in 0..n {
                let i = y * n + x;
                let m = my * n + (n - x) % n;
                if i == 0 {
                    // The mean: the window's water has no mean to move, so a trough is matched by
                    // an immeasurable lift over the whole window.
                    forcing[0] = Complex::new(0.0, 0.0);
                    continue;
                }
                // `z = h + i·v`, both real: ĥ = (ẑ(k) + ẑ*(−k))/2, v̂ = (ẑ(k) − ẑ*(−k))/(2i).
                let zj = z[i];
                let zm = z[m].conj();
                let hh = (zj + zm) * 0.5;
                let vv = (zj - zm) * Complex::new(0.0, -0.5);
                let d = forcing[i];
                let eta = hh + d;
                let [c, sw, ws] = self.spectral.rot_cs[i];
                let decay = self.spectral.rot_decay[i];
                let eta2 = (eta * c + vv * sw) * decay;
                let vv2 = (vv * c - eta * ws) * decay;
                let hh2 = eta2 - d;
                forcing[i] = hh2 + vv2 * Complex::new(0.0, 1.0);
            }
        }
        // `forcing` now holds the stepped spectrum; back to the lattice.
        self.spectral.fft2(&mut forcing, true);
        let inv_nn = 1.0 / nn as f32;
        for ((f, h), v) in forcing.iter().zip(&mut self.h).zip(&mut self.v) {
            *h = f.re * inv_nn;
            *v = f.im * inv_nn;
        }
        self.z = z;
        self.next = forcing;
        self.absorb_rim(dt);
    }

    /// The rim: the outer ring held at zero and [`EDGE_ABSORB`] cells inside it damped
    /// progressively, so an outward wave loses its amplitude on the way out instead of coming in
    /// again on the far side.
    fn absorb_rim(&mut self, dt: f32) {
        let n = SIM_CELLS;
        let steps = dt * EDGE_ABSORB_RATE;
        let factor: Vec<f32> = (0..EDGE_ABSORB)
            .map(|e| {
                (EDGE_ABSORB_FLOOR + (1.0 - EDGE_ABSORB_FLOOR) * (e as f32 / EDGE_ABSORB as f32))
                    .powf(steps)
            })
            .collect();
        for y in 0..n {
            let ey = y.min(n - 1 - y);
            for x in 0..n {
                let edge = ey.min(x.min(n - 1 - x));
                if edge >= EDGE_ABSORB {
                    continue;
                }
                let i = y * n + x;
                let f = if edge == 0 { 0.0 } else { factor[edge] };
                self.h[i] *= f;
                self.v[i] *= f;
                self.foam[i] *= f;
            }
        }
    }

    /// The foam ages and the heaving faces whiten, then the field is encoded into the bound
    /// texture's bytes: the central-difference slope in R/G and the height in B, each signed
    /// around the byte's middle and soft-limited, the foam in A. Same slope layout as the tiling
    /// ripple map next door, which is not a coincidence — the shader reads the two the same way.
    fn encode(&mut self, dt: f32, breaking: bool) {
        let n = SIM_CELLS;
        let keep = (-dt / FOAM_TAU).exp();
        let rate = FOAM_BREAK_RATE * foam_gain();
        let soft = |x: f32| x / (1.0 + x * x).sqrt();
        for y in 0..n {
            let (ym, yp) = (y.saturating_sub(1), (y + 1).min(n - 1));
            for x in 0..n {
                let (xm, xp) = (x.saturating_sub(1), (x + 1).min(n - 1));
                let dx = (self.h[y * n + xp] - self.h[y * n + xm]) * (0.5 / SIM_CELL);
                let dy = (self.h[yp * n + x] - self.h[ym * n + x]) * (0.5 / SIM_CELL);
                let i = y * n + x;
                let mut f = self.foam[i] * keep;
                if breaking {
                    let heave = self.v[i].abs();
                    f = (f + dt * rate * (heave - FOAM_BREAK_SPEED).max(0.0)).min(1.0);
                }
                self.foam[i] = f;
                let p = i * 4;
                self.pixels[p] = ((soft(dx * ENC_SLOPE) * 0.5 + 0.5) * 255.0) as u8;
                self.pixels[p + 1] = ((soft(dy * ENC_SLOPE) * 0.5 + 0.5) * 255.0) as u8;
                self.pixels[p + 2] = ((soft(self.h[i] * ENC_HEIGHT) * 0.5 + 0.5) * 255.0) as u8;
                self.pixels[p + 3] = (f.clamp(0.0, 1.0) * 255.0) as u8;
            }
        }
    }
}

/// One frame's encoded field on its way to the GPU.
///
/// **This exists because the obvious way does not work.** Writing the new pixels into the `Image`
/// asset each frame — `Assets::get_mut`, then mutate — makes Bevy tear the old GPU texture down and
/// build a new one, while the liquid materials' bind groups go on pointing at the old one. Nothing
/// errors: the water simply shows the texture's startup contents for ever, which in this case is a
/// flat field, which looks exactly like a simulation that is running and producing nothing. A whole
/// live capture went into finding that out (and a wrong reading of a swimmer's shadow as a wake on
/// the way).
///
/// So the asset is created once and never touched, and the per-frame contents are copied into the
/// texture the render world already made — the same shape as the reflection's parameter buffer next
/// door, and for the same underlying reason. The `Arc` is what makes the per-frame extract a
/// pointer copy rather than 147 KB.
#[derive(Resource, Clone, Default, ExtractResource)]
pub(crate) struct RippleFrame {
    id: AssetId<Image>,
    pixels: Option<Arc<Vec<u8>>>,
}

/// Copy this frame's field into the texture the materials are already bound to.
fn upload_wake(
    queue: Res<RenderQueue>,
    images: Res<RenderAssets<GpuImage>>,
    frame: Option<Res<RippleFrame>>,
) {
    let Some(frame) = frame else { return };
    let Some(pixels) = frame.pixels.as_ref() else {
        return;
    };
    // Absent for the first frame or two, while the image is still being prepared.
    let Some(gpu) = images.get(frame.id) else {
        return;
    };
    queue.write_texture(
        gpu.texture.as_image_copy(),
        pixels,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * SIM_CELLS as u32),
            rows_per_image: Some(SIM_CELLS as u32),
        },
        Extent3d {
            width: SIM_CELLS as u32,
            height: SIM_CELLS as u32,
            depth_or_array_layers: 1,
        },
    );
}

/// The lattice's own texture: no mips (it is rewritten every frame and never minified far), and
/// **clamped**, so a sample taken outside the window returns the rim — which the absorbing band has
/// already brought to zero.
fn sim_image() -> Image {
    let mut image = Image::new_uninit(
        Extent3d {
            width: SIM_CELLS as u32,
            height: SIM_CELLS as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        TextureFormat::Rgba8Unorm,
        // Render world only. The asset is created once and **never touched again** — see
        // [`upload_wake`] for why the per-frame contents do not go through it.
        RenderAssetUsages::RENDER_WORLD,
    );
    // The render side copies into this texture every frame rather than replacing it.
    image.texture_descriptor.usage |= TextureUsages::COPY_DST;
    // Flat water: zero slope and zero height, signed around the byte's middle, and no foam.
    image.data = Some(
        [128u8, 128, 128, 0]
            .iter()
            .copied()
            .cycle()
            .take(SIM_CELLS * SIM_CELLS * 4)
            .collect(),
    );
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::ClampToEdge,
        address_mode_v: ImageAddressMode::ClampToEdge,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        ..default()
    });
    image
}

fn setup_ripple_sim(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let image = images.add(sim_image());
    commands.insert_resource(RippleSim::new(image));
}

/// What the driver remembers of a body between frames: where it was, how fast it has been going,
/// which way, and the stroke owed.
pub(crate) struct Track {
    last: Vec3,
    speed: f32,
    heading: Vec2,
    credit: f32,
    side: f32,
}

/// A body's press on the surface from its feet (Bevy), its height (yards, scaled) and the water's
/// depth over its feet: `None` out of the water or on wet sand. The footprint is the body's
/// ([`BODY_SIGMA_PER_HEIGHT`], narrowing with a wader's legs); the depression what it floats on,
/// fading as it goes under, plus what its speed presses; the submersion is the water over a body
/// that swims at the surface.
fn body_source(feet: Vec2, height: f32, depth: f32, speed: f32, heading: Vec2) -> Option<Source> {
    if depth < PUSH_MIN_DEPTH || !depth.is_finite() {
        return None;
    }
    let float = (SWIM_FLOAT_FRAC * height).max(0.1);
    let wet = (depth / float).min(1.0);
    let sigma =
        (BODY_SIGMA_PER_HEIGHT * height * wet).clamp(BODY_SIGMA_RANGE[0], BODY_SIGMA_RANGE[1]);
    let submerged = (depth - float).max(0.0);
    let under = (-submerged / sigma).exp();
    let u = (speed / SWIM_SPEED).min(2.0);
    let deep = (-submerged / (4.0 * sigma)).exp();
    let amount = sigma * (BODY_STATIC_PER_SIGMA * under + BODY_DYNAMIC_PER_SIGMA * u * u * deep);
    if amount < 1e-4 {
        return None;
    }
    let bow = PUSH_BOW_OFFSET * (speed / SWIM_SPEED).clamp(0.0, 1.0) * under;
    Some(Source {
        at: feet + heading * bow,
        sigma,
        amount,
        submerged,
    })
}

/// Carry the window, take this frame's bodies, integrate, and publish.
///
/// Inert on the reference lane, where `water_fx`'s decals are the wake and this would be a second
/// one: the strength lane goes to zero and the field is left alone.
#[allow(clippy::too_many_arguments)]
pub(crate) fn drive_ripple_sim(
    style: Res<WaterStyle>,
    time: Res<Time>,
    mut sim: ResMut<RippleSim>,
    mut frame: ResMut<RippleFrame>,
    mut data: ResMut<WaterReflectData>,
    viewer: Res<crate::view::Viewer>,
    cam: Query<&GlobalTransform, With<crate::view::WorldCamera>>,
    units: Query<(Entity, &Transform, &WorldUnit), Without<ViewerUnit>>,
    water: Query<&WaterChunkInfo>,
    index: Res<WaterIndex>,
    // A unit's transform is streamed without a velocity, and a wake is made of velocity.
    mut tracks: Local<EntityHashMap<Track>>,
    // The phantom swimmer's body (captures only), a source like the avatar.
    phantom: Res<super::preview_rows::PhantomBody>,
    // `$WOW_WAKE_COST=1`: the step's CPU time, logged every 600 stepped frames.
    mut cost: Local<Option<(u32, f64)>>,
    mut cost_read: Local<bool>,
) {
    if !*cost_read {
        *cost_read = true;
        if std::env::var_os("WOW_WAKE_COST").is_some_and(|v| v == "1") {
            *cost = Some((0, 0.0));
        }
    }
    // x/y = the window's lower corner, z = 1/extent, w = strength.
    let mut params = [0.0f32; 4];
    if !style.is_stylised() {
        data.0[16..20].copy_from_slice(&params);
        return;
    }
    // The window follows the avatar when there is one and the camera otherwise — a free-flying
    // eye (the world viewer, a capture run) still wants water that moves under it.
    let Some(centre) = viewer
        .at
        .or_else(|| cam.iter().next().map(|t| t.translation()))
    else {
        data.0[16..20].copy_from_slice(&params);
        return;
    };
    sim.recentre(Vec2::new(centre.x, centre.z));

    // The surface height under a point, through the same index the foam decals use — a hash lookup
    // per unit, not a walk over every loaded surface.
    let depth_at = |p: Vec3| -> Option<f32> {
        let w = bevy_to_wow(p);
        index
            .over(w[0], w[1])
            .iter()
            .find_map(|&e| water.get(e).ok()?.surface_z_at(w[0], w[1]))
            .map(|surface| surface - w[2])
    };

    let dt = sim.take_dt(time.delta_secs());
    let strokes = lever::on(lever::RINGS);
    let foaming = lever::on(lever::WAKE_FOAM);

    // Every body in range: the avatar, the phantom, the wading units.
    let mut bodies: Vec<(Entity, Vec3, f32)> = Vec::new();
    if let Some(body) = viewer.at {
        bodies.push((Entity::PLACEHOLDER, body, viewer.height));
    }
    if let Some((body, height)) = phantom.0 {
        bodies.push((super::preview_rows::PHANTOM, body, height));
    }
    for (entity, transform, unit) in &units {
        if unit.wades {
            bodies.push((entity, transform.translation, unit.height * unit.scale));
        }
    }
    let mut sources: Vec<Source> = Vec::new();
    let mut kicks: Vec<(Vec2, f32)> = Vec::new();
    for &(entity, p, height) in &bodies {
        let track = tracks.entry(entity).or_insert(Track {
            last: p,
            speed: 0.0,
            heading: Vec2::Y,
            credit: 0.0,
            side: 1.0,
        });
        let step = Vec2::new(p.x - track.last.x, p.z - track.last.z);
        track.last = p;
        let moved = step.length();
        if dt > 0.0 {
            let v = if moved > MOVED_EPSILON {
                (moved / dt).min(20.0)
            } else {
                0.0
            };
            track.speed += (v - track.speed) * (1.0 - (-dt / SPEED_TAU).exp());
            if moved > MOVED_EPSILON {
                track.heading = step / moved;
            }
        }
        let Some(depth) = depth_at(p) else { continue };
        let feet = Vec2::new(p.x, p.z);
        let Some(src) = body_source(feet, height, depth, track.speed, track.heading) else {
            continue;
        };
        // Strokes: one every `STROKE_YD` swum at the surface, on alternating sides behind the
        // body; at most one a frame however long the frame was.
        if strokes && src.submerged < src.sigma && track.speed > STROKE_SPEED_MIN {
            track.credit += moved / STROKE_YD;
            if track.credit >= 1.0 {
                track.credit = (track.credit - 1.0).min(1.0);
                track.side = -track.side;
                let side = Vec2::new(track.heading.y, -track.heading.x);
                let at = feet - track.heading * (STROKE_BACK * src.sigma)
                    + side * (STROKE_SIDE * src.sigma * track.side);
                kicks.push((at, src.sigma * STROKE_SIGMA));
            }
        }
        sources.push(src);
    }
    // A body that has gone (out of range, dead, despawned) must not keep state here, or the map
    // grows for as long as the session runs.
    if tracks.len() > bodies.len() {
        tracks.retain(|e, _| bodies.iter().any(|(b, _, _)| b == e));
    }

    // Nothing pressing and the water already flat: the texture holds flat water, nothing to do.
    let idle = sources.is_empty() && kicks.is_empty();
    if !idle {
        sim.quiet = false;
    }
    if dt > 0.0 && !sim.quiet {
        let started = cost.is_some().then(std::time::Instant::now);
        for &(at, sigma) in &kicks {
            sim.splat(0, at, sigma, STROKE_RATE);
            if foaming {
                sim.splat(1, at, sigma, (STROKE_FOAM * foam_gain()).min(1.0));
            }
        }
        sim.step(dt, &sources);
        if idle && sim.settle() {
            sim.quiet = true;
        }
        sim.encode(dt, foaming);
        if let (Some(t0), Some((frames, total))) = (started, cost.as_mut()) {
            *frames += 1;
            *total += t0.elapsed().as_secs_f64();
            if *frames == 600 {
                info!(
                    "wave field: {:.0} us a stepped frame over {frames} frames, {} bodies pressing",
                    *total * 1e6 / *frames as f64,
                    sources.len()
                );
                (*frames, *total) = (0, 0.0);
            }
        }
        let RippleSim { pixels, image, .. } = &mut *sim;
        *frame = RippleFrame {
            id: image.id(),
            pixels: Some(Arc::new(std::mem::take(pixels))),
        };
        // `Arc` is what the extract clones each frame, so the staging buffer is handed over rather
        // than copied; take a fresh one for the next step.
        *pixels = vec![0u8; SIM_CELLS * SIM_CELLS * 4];
    }

    let corner = sim.corner();
    params = [corner.x, corner.y, 1.0 / SIM_YARDS, 1.0];
    data.0[16..20].copy_from_slice(&params);
}

pub(super) fn register(app: &mut App) {
    app.init_resource::<RippleFrame>()
        .add_plugins(ExtractResourcePlugin::<RippleFrame>::default())
        .add_systems(
            Startup,
            setup_ripple_sim
                .after(benilla_assets::AssetSet::Open)
                // The materials bind this image, and a material built before it exists would have
                // to be rebuilt to get it — the same ordering the reflection target needs.
                .before(super::WaterPassSet),
        )
        // After every mover has run for the frame, ahead of the other writer of the lane buffer.
        .add_systems(
            PostUpdate,
            drive_ripple_sim.before(super::reflect::drive_reflection),
        );
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render_app.add_systems(Render, upload_wake.in_set(RenderSystems::PrepareResources));
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f32 = 1.0 / 60.0;

    fn sim() -> RippleSim {
        let mut s = RippleSim::new(Handle::default());
        s.recentre(Vec2::splat(SIM_YARDS * 0.5));
        s
    }

    fn at(s: &RippleSim, p: Vec2) -> f32 {
        let c = s.cells(p).as_ivec2();
        s.h[c.y as usize * SIM_CELLS + c.x as usize]
    }

    /// A human swimming at `speed` along +x through the window's middle for `seconds`, from the
    /// left; returns the body's final position.
    fn swim(s: &mut RippleSim, speed: f32, seconds: f32, submerged: f32) -> Vec2 {
        let mut x = 6.0;
        let steps = (seconds / DT) as usize;
        for i in 0..steps {
            x += speed * DT;
            // Eased in over the first second, as a body gathering way is, so the start does not
            // ring the window.
            let ease = (i as f32 * DT).min(1.0);
            let src = Source {
                at: Vec2::new(x, SIM_YARDS * 0.5),
                sigma: 0.45,
                amount: 0.45 * (BODY_STATIC_PER_SIGMA + BODY_DYNAMIC_PER_SIGMA) * ease,
                submerged,
            };
            s.step(DT, &[src]);
        }
        Vec2::new(x, SIM_YARDS * 0.5)
    }

    /// Drive it hard and check it stays finite and bounded: the whole thing is worthless if it
    /// diverges.
    #[test]
    fn the_field_stays_bounded_under_a_hammering() {
        let mut s = sim();
        for i in 0..600 {
            let t = i as f32 * DT;
            let src = Source {
                at: Vec2::new(10.0 + 6.0 * t, 24.0),
                sigma: 0.45,
                amount: 0.3,
                submerged: 0.0,
            };
            s.splat(0, Vec2::new(10.0 + 6.0 * t, 24.5), 0.2, 1.0);
            s.step(DT, &[src]);
        }
        assert!(
            s.h.iter()
                .chain(s.v.iter())
                .all(|v| v.is_finite() && v.abs() < 10.0),
            "the wave field left its stable region"
        );
    }

    /// A body at rest floats in the trough it displaces: after a second the surface under it is
    /// down by about its static amount and the far water is flat.
    #[test]
    fn a_floating_body_rests_in_its_own_trough() {
        let mut s = sim();
        let c = Vec2::splat(SIM_YARDS * 0.5);
        let src = Source {
            at: c,
            sigma: 0.45,
            amount: 0.11,
            submerged: 0.0,
        };
        for _ in 0..180 {
            s.step(DT, &[src]);
        }
        let under = at(&s, c);
        assert!(
            (-0.13..-0.08).contains(&under),
            "the trough under a floating body is {under}, not about −0.11"
        );
        assert!(
            at(&s, c + Vec2::new(8.0, 0.0)).abs() < 0.005,
            "a body at rest moves water eight yards away"
        );
    }

    /// Dispersion is the one property that separates this from the wave equation it replaces: a
    /// wave's frequency is `√(g·k)`, so a 12-yd wave runs twice as fast as a 3-yd one. Each is laid
    /// on the lattice as a wide packet (clear of the rim: the longest modes are not local, and a
    /// rim's worth of change reaches the centre within a step) and stepped a quarter second; the
    /// centre's fall against its start, the known damping divided out, is `cos ω·t`.
    #[test]
    fn waves_disperse_as_deep_water_does() {
        let omega_of = |cycles: f32| -> f32 {
            let mut s = sim();
            let k = std::f32::consts::TAU * cycles / SIM_YARDS;
            let mid = SIM_YARDS * 0.5;
            for y in 0..SIM_CELLS {
                for x in 0..SIM_CELLS {
                    let (px, py) = ((x as f32 + 0.5) * SIM_CELL, (y as f32 + 0.5) * SIM_CELL);
                    let env = (-((px - mid).powi(2) + (py - mid).powi(2)) / (2.0 * 64.0)).exp();
                    s.h[y * SIM_CELLS + x] = (k * px).cos() * env;
                }
            }
            let c = Vec2::splat(SIM_YARDS * 0.5);
            let h0 = at(&s, c);
            let steps = 15;
            for _ in 0..steps {
                s.step(DT, &[]);
            }
            let t = steps as f32 * DT;
            let decay = (-t / DAMP_TAU - t * VISCOSITY * k * k).exp();
            (at(&s, c) / (h0 * decay)).clamp(-1.0, 1.0).acos() / t
        };
        for cycles in [4.0, 16.0] {
            let k = std::f32::consts::TAU * cycles / SIM_YARDS;
            let want = (GRAVITY * k).sqrt();
            let got = omega_of(cycles);
            assert!(
                (got - want).abs() < 0.03 * want,
                "a {}-yd wave runs at ω = {got}, not √(g·k) = {want}",
                SIM_YARDS / cycles
            );
        }
        let ratio = omega_of(16.0) / omega_of(4.0);
        assert!(
            (ratio - 2.0).abs() < 0.1,
            "ω(3 yd) / ω(12 yd) = {ratio}, not 2"
        );
    }

    /// A moving body's waves lie in Kelvin's wedge: nearly all the energy behind the body sits
    /// within a half angle near 19.47° of its track, and the water ahead of it is flat.
    #[test]
    fn a_moving_body_leaves_a_kelvin_wedge() {
        let mut s = sim();
        let body = swim(&mut s, SWIM_SPEED, 5.0, 0.0);
        let n = SIM_CELLS;
        let (mut inside, mut outside, mut ahead) = (0.0f32, 0.0f32, 0.0f32);
        for y in EDGE_ABSORB..n - EDGE_ABSORB {
            for x in EDGE_ABSORB..n - EDGE_ABSORB {
                let p = Vec2::new(x as f32 + 0.5, y as f32 + 0.5) * SIM_CELL + s.corner();
                let d = p - body;
                let e = s.h[y * n + x] * s.h[y * n + x];
                if d.x > 1.5 {
                    ahead += e;
                } else if d.x < -3.0 {
                    let tan = d.y.abs() / -d.x;
                    if tan < 0.42 {
                        inside += e;
                    } else {
                        outside += e;
                    }
                }
            }
        }
        assert!(inside > 0.0, "no wake behind the body");
        assert!(
            outside < 0.08 * inside,
            "energy outside the wedge ({outside}) against inside ({inside}): no Kelvin V"
        );
        assert!(
            ahead < 0.05 * inside,
            "energy ahead of the body ({ahead}) against the wake ({inside})"
        );
        // And an ordinary wedge test from the other side: the arms do reach out toward the cusp
        // (past 12° some energy must lie), so the V is a V and not a line.
        let mut arms = 0.0f32;
        for y in EDGE_ABSORB..n - EDGE_ABSORB {
            for x in EDGE_ABSORB..n - EDGE_ABSORB {
                let p = Vec2::new(x as f32 + 0.5, y as f32 + 0.5) * SIM_CELL + s.corner();
                let d = p - body;
                if d.x < -3.0 && d.y.abs() / -d.x > 0.21 && d.y.abs() / -d.x < 0.42 {
                    arms += s.h[y * n + x] * s.h[y * n + x];
                }
            }
        }
        assert!(
            arms > 0.1 * inside,
            "no arms toward the cusp: a line, not a V"
        );
    }

    /// The wedge's half angle, measured on the solver's own field: 5-9 yd behind a body that has
    /// swum five seconds at swim speed, the outer lobe of the height's energy by angle off the
    /// track (the arms; inside it, the body's groove and the transverse waves) peaks near
    /// Kelvin's 19.47°. Printed for the perf note
    /// (`-- --nocapture`).
    #[test]
    fn the_arms_ridge_lies_at_kelvins_angle() {
        let mut s = sim();
        let body = swim(&mut s, SWIM_SPEED, 5.0, 0.0);
        let n = SIM_CELLS;
        let mut bins = [0.0f32; 40];
        for y in EDGE_ABSORB..n - EDGE_ABSORB {
            for x in EDGE_ABSORB..n - EDGE_ABSORB {
                let p = Vec2::new(x as f32 + 0.5, y as f32 + 0.5) * SIM_CELL + s.corner();
                let d = p - body;
                let r = d.length();
                if d.x > -1.0 || !(5.0..9.0).contains(&r) {
                    continue;
                }
                let a = (d.y.abs() / -d.x).atan().to_degrees();
                if let Some(b) = bins.get_mut(a as usize) {
                    *b += s.h[y * n + x] * s.h[y * n + x];
                }
            }
        }
        // Inside 15° lies the body's own groove and the transverse waves; the arms are the
        // outermost ridge, the last local peak of the energy that still holds a fifth of its
        // greatest, over three-degree means (a lattice of quarter-yard cells beads a degree's
        // ring): the cusp line.
        let mean: Vec<f32> = (0..40)
            .map(|a: usize| {
                bins[a.saturating_sub(1)..(a + 2).min(40)]
                    .iter()
                    .sum::<f32>()
                    / 3.0
            })
            .collect();
        let peak = mean.iter().fold(0.0f32, |m, e| m.max(*e));
        let ridge = (15..39)
            .rfind(|&a| mean[a] >= mean[a - 1] && mean[a] >= mean[a + 1] && mean[a] > 0.25 * peak)
            .unwrap_or(0);
        let total: f32 = bins.iter().sum();
        let mut acc = 0.0;
        let a90 = bins
            .iter()
            .position(|e| {
                acc += e;
                acc >= 0.9 * total
            })
            .unwrap_or(40);
        println!(
            "wedge 5-9 yd behind: the arms' ridge at {ridge}-{} deg, 90% of the energy within {a90} deg",
            ridge + 1
        );
        assert!(
            (17..=21).contains(&ridge),
            "the arms' ridge lies at {ridge} deg, not near 19.47"
        );
    }

    /// The waves detach: once the body stops, the far part of its wake goes on travelling and the
    /// crest that was six yards behind is somewhere else a second later, while the body's own
    /// trough stays put.
    #[test]
    fn the_wake_detaches_from_a_body_that_stops() {
        let mut s = sim();
        let body = swim(&mut s, SWIM_SPEED, 4.0, 0.0);
        let probe = body - Vec2::new(6.0, 0.0);
        let before: Vec<f32> = (0..24)
            .map(|i| at(&s, probe - Vec2::new(i as f32 * SIM_CELL * 2.0, 0.0)))
            .collect();
        let rest = Source {
            at: body,
            sigma: 0.45,
            amount: 0.45 * BODY_STATIC_PER_SIGMA,
            submerged: 0.0,
        };
        for _ in 0..60 {
            s.step(DT, &[rest]);
        }
        let after: Vec<f32> = (0..24)
            .map(|i| at(&s, probe - Vec2::new(i as f32 * SIM_CELL * 2.0, 0.0)))
            .collect();
        let change: f32 = before
            .iter()
            .zip(&after)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / before.iter().map(|a| a.abs()).sum::<f32>().max(1e-6);
        assert!(
            change > 0.5,
            "the wake behind a stopped body did not move on (relative change {change})"
        );
        assert!(
            at(&s, body) < -0.05,
            "the body's trough did not stay with the body"
        );
    }

    /// A body under the surface presses softly: its surface mark is a fraction of the floating
    /// body's, and smooth (no cell-scale ripple).
    #[test]
    fn a_submerged_body_makes_a_soft_hump() {
        let steep = |submerged: f32| -> (f32, f32) {
            let mut s = sim();
            swim(&mut s, SWIM_SPEED, 3.0, submerged);
            let peak = s.h.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let n = SIM_CELLS;
            let mut slope = 0.0f32;
            for y in 1..n - 1 {
                for x in 1..n - 1 {
                    let dx = (s.h[y * n + x + 1] - s.h[y * n + x - 1]) * (0.5 / SIM_CELL);
                    slope = slope.max(dx.abs());
                }
            }
            (peak, slope)
        };
        let (p0, s0) = steep(0.0);
        let (p2, s2) = steep(2.0);
        assert!(
            p2 < 0.5 * p0,
            "a body two yards down marks the surface as much as one on it"
        );
        assert!(
            s2 < 0.15 * s0,
            "a body two yards down ripples the surface ({s2} against {s0})"
        );
    }

    /// Scrolling must carry the field with the world, not smear it: a feature sitting at a world
    /// position has to still be at that world position after the window moves under it.
    #[test]
    fn scrolling_the_window_keeps_features_at_their_world_position() {
        let mut s = sim();
        let p = Vec2::new(SIM_YARDS * 0.5 + 4.0, SIM_YARDS * 0.5);
        s.splat(0, p, 0.5, 1.0);
        s.step(DT, &[]);
        let before = at(&s, p);
        assert!(before.abs() > 1e-4, "the kick did not land");
        s.recentre(Vec2::new(SIM_YARDS * 0.5 + 5.0, SIM_YARDS * 0.5 + 3.0));
        assert!(
            (at(&s, p) - before).abs() < 1e-6,
            "the field did not travel with the window"
        );
    }

    /// A teleport out of the window leaves nothing behind — otherwise the wake of wherever you were
    /// standing arrives with you.
    #[test]
    fn a_jump_past_the_window_clears_the_field() {
        let mut s = sim();
        s.splat(0, Vec2::splat(SIM_YARDS * 0.5), 1.0, 1.0);
        s.step(DT, &[]);
        assert!(s.h.iter().any(|v| v.abs() > 1e-4));
        s.recentre(Vec2::splat(SIM_YARDS * 0.5 + 4000.0));
        assert!(
            s.h.iter().all(|v| *v == 0.0),
            "a teleport carried the old water with it"
        );
    }

    /// Foam is laid where the water broke and ages there: a stroke's foam stays at the stroke
    /// after the body has gone on, and fades.
    #[test]
    fn foam_ages_where_it_was_made() {
        let mut s = sim();
        let p = Vec2::splat(SIM_YARDS * 0.5);
        s.splat(1, p, 0.25, STROKE_FOAM);
        let c = s.cells(p).as_ivec2();
        let i = c.y as usize * SIM_CELLS + c.x as usize;
        let fresh = s.foam[i];
        assert!(fresh > 0.6 * STROKE_FOAM, "{fresh}");
        for _ in 0..60 {
            s.step(DT, &[]);
            s.encode(DT, true);
        }
        let aged = s.foam[i];
        assert!(
            aged < fresh && aged > 0.3 * fresh,
            "foam after a second: {aged} of {fresh}"
        );
        assert_eq!(s.pixels[i * 4 + 3], (aged * 255.0) as u8);
        // A body at rest sits in its trough and the water there is still: no ring of foam. A body
        // under way heaves the water at its flanks, and the foam it whips up stays behind it.
        let still = Source {
            at: p,
            sigma: 0.45,
            amount: 0.11,
            submerged: 0.0,
        };
        let mut t = sim();
        for _ in 0..240 {
            t.step(DT, &[still]);
            t.encode(DT, true);
        }
        let ring = t.foam.iter().fold(0.0f32, |m, f| m.max(*f));
        assert!(
            ring < 0.05,
            "a body at rest whitens the water round it ({ring})"
        );
        let mut u = sim();
        let body = swim(&mut u, SWIM_SPEED, 4.0, 0.0);
        // `swim` does not encode; lay the foam over its last second again with the encode.
        for i in 0..60 {
            let src = Source {
                at: body + Vec2::new(SWIM_SPEED * DT * i as f32, 0.0),
                sigma: 0.45,
                amount: 0.45 * (BODY_STATIC_PER_SIGMA + BODY_DYNAMIC_PER_SIGMA),
                submerged: 0.0,
            };
            u.step(DT, &[src]);
            u.encode(DT, true);
        }
        let end = body + Vec2::new(SWIM_SPEED, 0.0);
        let foam_at = |q: Vec2| {
            let c = u.cells(q).as_ivec2();
            u.foam[c.y as usize * SIM_CELLS + c.x as usize]
        };
        let near = (-8..=8)
            .map(|j| foam_at(end + Vec2::new(-1.0, j as f32 * 0.1)))
            .fold(0.0f32, f32::max);
        let behind = (-8..=8)
            .map(|j| foam_at(end + Vec2::new(-3.0, j as f32 * 0.1)))
            .fold(0.0f32, f32::max);
        assert!(
            near > 0.05,
            "a swimmer under way leaves no foam at its flanks ({near})"
        );
        assert!(
            behind > 0.01,
            "the foam does not stay in the water behind ({behind})"
        );
    }

    /// The idle skip: a field with anything in it is not quiet; flat water under a byte is, and is
    /// cleared.
    #[test]
    fn a_flat_field_settles_and_a_live_one_does_not() {
        let mut s = sim();
        assert!(s.settle(), "an empty field is not quiet");
        s.splat(0, Vec2::splat(SIM_YARDS * 0.5), 0.5, 0.6);
        s.step(DT, &[]);
        assert!(!s.settle(), "a kicked field passed for quiet");
        for v in s.h.iter_mut().chain(s.v.iter_mut()) {
            *v *= 1e-4;
        }
        assert!(s.settle());
        assert!(s.h.iter().chain(s.v.iter()).all(|v| *v == 0.0));
    }

    /// The step's cost, for the perf note: a frame of two swimming bodies with strokes, stepped
    /// and encoded, in microseconds. Ignored (a timing, not a check):
    /// `cargo test --release -p benilla-water --lib step_cost -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn step_cost() {
        let mut s = sim();
        let frames = 600;
        let mut src = [
            Source {
                at: Vec2::new(10.0, 20.0),
                sigma: 0.45,
                amount: 0.2,
                submerged: 0.0,
            },
            Source {
                at: Vec2::new(10.0, 28.0),
                sigma: 0.45,
                amount: 0.2,
                submerged: 2.0,
            },
        ];
        let t = std::time::Instant::now();
        for i in 0..frames {
            for b in src.iter_mut() {
                b.at.x += SWIM_SPEED * DT;
                if b.at.x > 40.0 {
                    b.at.x = 8.0;
                }
            }
            if i % 30 == 0 {
                s.splat(0, src[0].at - Vec2::new(0.4, 0.3), 0.22, STROKE_RATE);
                s.splat(1, src[0].at - Vec2::new(0.4, 0.3), 0.22, STROKE_FOAM);
            }
            s.step(DT, &src);
            s.encode(DT, true);
        }
        let per_frame = t.elapsed().as_secs_f64() * 1e6 / frames as f64;
        let t = std::time::Instant::now();
        for _ in 0..frames {
            s.step(DT, &[]);
        }
        let step_alone = t.elapsed().as_secs_f64() * 1e6 / frames as f64;
        println!(
            "wave field: {per_frame:.0} us a frame (two bodies, one under, strokes, encode); the step alone {step_alone:.0} us"
        );
    }

    /// The press is the body's: nothing on wet sand, the whole static trough on a swimmer at the
    /// surface, a fading one as he goes under, and a bow ahead only under way.
    #[test]
    fn a_body_presses_by_its_depth_and_speed() {
        let feet = Vec2::ZERO;
        assert!(body_source(feet, 2.03, 0.2, 0.0, Vec2::Y).is_none());
        let still = body_source(feet, 2.03, 1.5, 0.0, Vec2::Y).unwrap();
        assert!((still.amount - still.sigma * BODY_STATIC_PER_SIGMA).abs() < 1e-5);
        assert!((still.sigma - 0.44).abs() < 0.02);
        assert_eq!(still.at, feet);
        assert_eq!(still.submerged, 0.0);
        let swimming = body_source(feet, 2.03, 1.5, SWIM_SPEED, Vec2::Y).unwrap();
        assert!(swimming.amount > still.amount);
        assert!((swimming.at.y - PUSH_BOW_OFFSET).abs() < 1e-5);
        let diving = body_source(feet, 2.03, 3.5, SWIM_SPEED, Vec2::Y).unwrap();
        assert!((diving.submerged - 2.0).abs() < 0.03);
        assert!(diving.amount < 0.5 * swimming.amount);
        assert!(diving.at.y < 0.1, "a diver's bow is at the body");
    }
}
