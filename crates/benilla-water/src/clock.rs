//! The stylised water's **clock**: one unwrapped time in f64, and every phase `liquid.wgsl`'s
//! stylised lane animates by, reduced by its own period here and published in the reflection
//! block's [`CLOCK_LANES`].
//!
//! Bevy's `globals.time` wraps at 3600 s (`bevy_render` `globals.rs`), so a phase read off it jumps
//! once an hour unless its period divides the hour. The clock sums the virtual clock's deltas, so a
//! capture holds it while the scene builds and steps it exactly while the shot ages. The reference
//! lane keeps `globals.time`: its 24 fps flip (30 frames in 1.25 s, `0x68aac0`) and its 10 s magma
//! scroll both divide the hour. The stylised magma grade keeps it too, with the rest of magma and
//! slime.

use bevy::prelude::*;

use benilla_world::liquid::CLOCK_LANES;

/// The clock block's length in floats.
const CLOCK_FLOATS: usize = CLOCK_LANES.end - CLOCK_LANES.start;

/// The five ambient ripple octaves' drifts in map tiles a second, longest first — the order
/// `stylised_water` sums them in.
const RIPPLE_DRIFT: [[f64; 2]; 5] = [
    [0.012, 0.007],
    [0.016, 0.021],
    [-0.021, 0.010],
    [0.014, -0.026],
    [0.041, 0.033],
];

/// The shore bubbles' three reads of the map (`foam_noise`): the slow warp, then the two octaves.
const FOAM_DRIFT: [[f64; 2]; 3] = [[0.004, -0.003], [0.010, 0.014], [-0.021, 0.008]];

/// The flow map's restart period in seconds: `liquid.wgsl`'s `FLOW_PERIOD_S`, which also sets how
/// far one phase carries the ripple along the current.
const FLOW_PERIOD_S: f64 = 1.6;

/// One breath of the shore swell (`SHORE_RUNUP_YARDS` up the shore and back), in seconds.
const SHORE_SWELL_S: f64 = 5.5;

/// Seconds of water time since startup, never wrapped.
#[derive(Resource, Default, Clone, Copy, Debug)]
pub(crate) struct WaterClock {
    secs: f64,
}

impl WaterClock {
    /// The clock block at this time — see [`clock_lanes`].
    pub(crate) fn lanes(&self) -> [f32; CLOCK_FLOATS] {
        clock_lanes(self.secs)
    }
}

/// The block at `secs`: `0..16` the eight map offsets as `(u, v)` pairs, ripples then foam, each
/// `drift · t mod 1` (the map tiles, so a whole tile is no move); `16` the flow restart phase and
/// `17` the swell's, as fractions of their periods; `18..20` spare.
fn clock_lanes(secs: f64) -> [f32; CLOCK_FLOATS] {
    let mut lanes = [0.0; CLOCK_FLOATS];
    for (i, drift) in RIPPLE_DRIFT.iter().chain(&FOAM_DRIFT).enumerate() {
        lanes[2 * i] = cycle(drift[0] * secs);
        lanes[2 * i + 1] = cycle(drift[1] * secs);
    }
    lanes[16] = cycle(secs / FLOW_PERIOD_S);
    lanes[17] = cycle(secs / SHORE_SWELL_S);
    lanes
}

/// `x mod 1`, taken in f64 and then narrowed; a remainder that rounds up to 1 in f32 is 0, the
/// same phase.
fn cycle(x: f64) -> f32 {
    let f = x.rem_euclid(1.0) as f32;
    if f < 1.0 {
        f
    } else {
        0.0
    }
}

/// Step the clock by the virtual clock's delta, which is zero while it is paused.
fn advance_water_clock(time: Res<Time<Virtual>>, mut clock: ResMut<WaterClock>) {
    clock.secs += time.delta_secs_f64();
}

pub(super) fn register(app: &mut App) {
    // `PreUpdate`, after `First` has advanced the virtual clock: every system of the frame, and the
    // reflection block's writer in `PostUpdate`, reads the frame's own time.
    app.init_resource::<WaterClock>()
        .add_systems(PreUpdate, advance_water_clock);
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::time::{TimePlugin, TimeUpdateStrategy};
    use std::time::Duration;

    /// Each lane's rate in cycles a second, in lane order.
    fn rates() -> Vec<f64> {
        let mut rates: Vec<f64> = RIPPLE_DRIFT
            .iter()
            .chain(&FOAM_DRIFT)
            .flat_map(|d| [d[0], d[1]])
            .collect();
        rates.push(1.0 / FLOW_PERIOD_S);
        rates.push(1.0 / SHORE_SWELL_S);
        rates
    }

    /// `b - a` as the shortest signed step round the cycle.
    fn step(a: f32, b: f32) -> f64 {
        let d = f64::from(b) - f64::from(a);
        d - d.round()
    }

    /// Across Bevy's 3600 s wrap every published phase moves by exactly its rate times the time
    /// step, where the wrapped clock the shader read before jumps; and again a week in, where the
    /// f64 reduction still holds.
    #[test]
    fn every_phase_steps_evenly_across_the_hour() {
        let rates = rates();
        assert_eq!(rates.len(), 18, "one rate per published lane");
        for (before, after) in [(3599.9, 3600.1), (604_799.9, 604_800.1)] {
            let (a, b) = (clock_lanes(before), clock_lanes(after));
            for (lane, rate) in rates.iter().enumerate() {
                let want = rate * (after - before);
                let want = want - want.round();
                let got = step(a[lane], b[lane]);
                assert!(
                    (got - want).abs() < 1e-6,
                    "lane {lane} at {before}..{after} s stepped {got}, not {want}"
                );
            }
            assert_eq!(a[18..], [0.0; 2], "the spare lanes stay zero");
        }
        // The old clock, `globals.time` wrapped at 3600 s, jumps on every lane whose period does
        // not divide the hour: all but the 0.010 drifts and the 1.6 s flow restart.
        let mut popped = 0;
        for (lane, rate) in rates.iter().enumerate() {
            let turns = rate * 3600.0;
            if (turns - turns.round()).abs() < 1e-9 {
                continue;
            }
            let wrapped = |t: f64| (rate * t.rem_euclid(3600.0)).rem_euclid(1.0) as f32;
            let want = rate * 0.2;
            let jump = step(wrapped(3599.9), wrapped(3600.1)) - (want - want.round());
            assert!(jump.abs() > 1e-3, "lane {lane} would not have popped");
            popped += 1;
        }
        assert_eq!(popped, 15);
    }

    /// Every lane is a phase in `[0, 1)`, even where the f64 remainder rounds up in f32.
    #[test]
    fn phases_stay_in_the_unit_cycle() {
        for secs in [0.0, 1e-9, 1.6 - 1e-12, 3600.0, 1e7 + 0.3] {
            for (lane, v) in clock_lanes(secs).iter().enumerate() {
                assert!((0.0..1.0).contains(v), "lane {lane} at {secs} s is {v}");
            }
        }
    }

    /// A capture's clock: held at zero while the virtual clock is paused, then stepped by exactly
    /// the fixed frame, so two runs reach the shutter on the same bits.
    #[test]
    fn the_capture_clock_is_frozen_then_stepped_exactly() {
        let run = || {
            let mut app = App::new();
            app.add_plugins(TimePlugin)
                .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_nanos(
                    16_666_667,
                )))
                .init_resource::<WaterClock>()
                .add_systems(PreUpdate, advance_water_clock);
            app.world_mut().resource_mut::<Time<Virtual>>().pause();
            for _ in 0..40 {
                app.update();
            }
            let held = app.world().resource::<WaterClock>().secs;
            app.world_mut().resource_mut::<Time<Virtual>>().unpause();
            for _ in 0..300 {
                app.update();
            }
            (held, *app.world().resource::<WaterClock>())
        };
        let (held, clock) = run();
        assert_eq!(held, 0.0, "the clock moved while the capture held it");
        assert!(
            (clock.secs - 300.0 * 0.016_666_667).abs() < 1e-9,
            "300 aging frames made {} s",
            clock.secs
        );
        let (_, again) = run();
        assert_eq!(clock.secs.to_bits(), again.secs.to_bits());
        assert_eq!(clock.lanes(), again.lanes());
    }

    /// The flow period is the shader's too: it sets how far one phase carries the ripple there.
    #[test]
    fn the_flow_period_is_the_shaders() {
        let wgsl = include_str!("../../benilla-assets/src/shaders/liquid.wgsl");
        assert!(
            wgsl.contains(&format!("const FLOW_PERIOD_S: f32 = {FLOW_PERIOD_S};")),
            "liquid.wgsl's FLOW_PERIOD_S is not {FLOW_PERIOD_S}"
        );
    }
}
