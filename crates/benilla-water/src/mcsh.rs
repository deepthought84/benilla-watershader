//! **The lighting sun's terrain shadow around the camera** (the ADT chunks' MCSH bits), for the
//! body's light: a 2048² `R8Unorm` torus over 32×32 MCNK chunks (1066.7 yd), one MCSH texel to a
//! texel, bound to every liquid material at 122 and read by `liquid.wgsl`'s `lighting_sun_mcsh`.
//!
//! The terrain's own shadow arrays are per tile and render-world only, so the water reads the
//! resident CPU copy instead: each `AdtTile` keeps its chunks' 64×64 MCSH bytes (255 shadowed), and
//! a chunk's block lands at its torus slot, `(global chunk mod 32)·64`. The window follows the
//! camera in whole chunks with a two-chunk hysteresis; the chunks it takes in are queued and copied
//! at most [`SHADOW_WRITES_PER_FRAME`] a frame, nearest first, into the texture the render world
//! already made (never through the `Image` asset, which would rebuild the texture under the
//! materials' bind groups; see `ripple_sim`'s `RippleFrame`). A chunk with no MCSH, or no tile, is
//! lit.
//!
//! The window is valid while every chunk it reads holds its own block: the slot of a chunk a
//! recentre takes in still holds the shadow of the chunk 32 chunks away until its copy lands (a lit
//! slot needs none for a chunk with no tile), so those inside the read extent go first, and the
//! window reads lit while any waits (a teleport's first frames).
//!
//! The image exists only while the look is Improved Water: on the reference lane the materials
//! bind the one-texel lit placeholder and no system here runs but the toggle.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::{
    Extent3d, Origin3d, TexelCopyBufferLayout, TexelCopyTextureInfo, TextureAspect,
    TextureDimension, TextureFormat, TextureUsages,
};
use bevy::render::renderer::RenderQueue;
use bevy::render::texture::GpuImage;
use bevy::render::{Render, RenderApp, RenderSystems};

use benilla_assets::materials::LiquidMaterial;
use benilla_assets::AdtTile;
use benilla_formats::{CHUNK_SIZE, SHADOW_MAP_SIZE, TILE_SIZE};

use super::{WaterPassImages, WaterStyle};
use crate::view::WorldCamera;

/// MCSH texels along a chunk's edge.
const CHUNK_TEXELS: u32 = SHADOW_MAP_SIZE;
/// Bytes in one chunk's MCSH block.
const CHUNK_BYTES: usize = (CHUNK_TEXELS * CHUNK_TEXELS) as usize;
/// One MCSH texel's edge in yards.
const TEXEL_YD: f32 = CHUNK_SIZE / CHUNK_TEXELS as f32;
/// Chunks along the window's edge, and the torus's period in chunks.
const WINDOW_CHUNKS: i32 = 32;
/// The torus's edge in texels: 32 chunks of 64.
const MCSH_TEXELS: u32 = WINDOW_CHUNKS as u32 * CHUNK_TEXELS;
/// The map's north-west corner from its centre in yards, 32 tiles: where global chunk 0 starts.
const MAP_HALF_YD: f32 = 32.0 * TILE_SIZE;
/// Chunks the map is read on either side of the window's centre: 15, one inside the window's edge.
const READ_CHUNKS: i32 = WINDOW_CHUNKS / 2 - 1;
/// How far from the window's centre the map is read, yards: [`READ_CHUNKS`] less half a texel, so
/// the bilinear read stays on the read chunks' texels.
const HALF_EXTENT_YD: f32 = READ_CHUNKS as f32 * CHUNK_SIZE - 0.5 * TEXEL_YD;
/// The window recentres once the camera is this many chunks from its centre.
const RECENTRE_CHUNKS: i32 = 2;
/// Most chunk blocks copied a frame: 64 × 4 KiB.
const SHADOW_WRITES_PER_FRAME: usize = 64;

/// The global chunk (column east, row south from the map's north-west corner) whose north-west
/// corner is the raw WoW point `nw`, a chunk's first vertex.
fn chunk_of_nw(nw: [f32; 3]) -> IVec2 {
    IVec2::new(
        ((MAP_HALF_YD - nw[1]) / CHUNK_SIZE).round() as i32,
        ((MAP_HALF_YD - nw[0]) / CHUNK_SIZE).round() as i32,
    )
}

/// The global chunk under Bevy `x`, `z` yards.
fn chunk_at(xz: Vec2) -> IVec2 {
    ((xz + Vec2::splat(MAP_HALF_YD)) / CHUNK_SIZE)
        .floor()
        .as_ivec2()
}

/// The torus slot chunk `k`'s block lands in, as an index into its 32×32 slots.
fn slot(k: IVec2) -> usize {
    let s = k.rem_euclid(IVec2::splat(WINDOW_CHUNKS));
    (s.y * WINDOW_CHUNKS + s.x) as usize
}

/// The texel of the torus where chunk `k`'s block starts.
fn torus_origin(k: IVec2) -> UVec2 {
    k.rem_euclid(IVec2::splat(WINDOW_CHUNKS)).as_uvec2() * CHUNK_TEXELS
}

/// Whether chunk `k` lies within `half` chunks before and `half - 1` after chunk `centre` on both
/// axes.
fn within(centre: IVec2, k: IVec2, half: i32) -> bool {
    let d = k - centre;
    d.x >= -half && d.x < half && d.y >= -half && d.y < half
}

/// Whether chunk `k` is in the window centred on chunk `centre`: 16 chunks before it to 15 after.
fn in_window(centre: IVec2, k: IVec2) -> bool {
    within(centre, k, WINDOW_CHUNKS / 2)
}

/// Whether the shader reads chunk `k` of the window centred on chunk `centre`.
fn in_read(centre: IVec2, k: IVec2) -> bool {
    within(centre, k, READ_CHUNKS)
}

/// The window the body's shadow map covers: its centre in Bevy x and z, its half extent in yards,
/// and whether the map holds it. The default holds none, and the body reads every surface as lit.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct WaterShadowWindow {
    centre: Vec2,
    half_extent: f32,
    valid: bool,
}

impl WaterShadowWindow {
    /// The window around chunk `centre`: its north-west corner is the window's middle.
    fn around(centre: IVec2, valid: bool) -> Self {
        Self {
            centre: centre.as_vec2() * CHUNK_SIZE - Vec2::splat(MAP_HALF_YD),
            half_extent: HALF_EXTENT_YD,
            valid,
        }
    }

    /// The body's `shadow` row: centre x, centre z, half extent, valid (1) or not (0).
    pub(crate) fn lanes(&self) -> [f32; 4] {
        [
            self.centre.x,
            self.centre.y,
            self.half_extent,
            f32::from(u8::from(self.valid)),
        ]
    }
}

/// The shadow torus and what it holds: the resident chunks by global chunk, the window's centre
/// and the chunks queued for a copy.
#[derive(Resource)]
pub(crate) struct WaterShadow {
    /// The image the materials bind: the torus while the look is Improved Water, else the
    /// placeholder.
    image: Handle<Image>,
    /// One lit texel.
    placeholder: Handle<Image>,
    /// The torus is built.
    allocated: bool,
    /// Frames since the torus was added, 0 on its own frame: its texture may reach the render
    /// world a frame after the asset, so nothing is copied until it is 1.
    resident_frames: u32,
    /// The chunk the window is centred on; `None` until the camera has been seen.
    centre: Option<IVec2>,
    /// Every resident chunk: its tile and its index in the tile's chunks.
    chunks: HashMap<IVec2, (AssetId<AdtTile>, u16)>,
    /// The window's chunks whose block has still to be copied.
    pending: HashSet<IVec2>,
    /// The pending chunks a recentre took in, whose slot still holds another chunk's shadow.
    stale: HashSet<IVec2>,
    /// The torus slots ([`slot`]) whose block has any shadow; the rest are lit.
    shaded: Vec<bool>,
}

impl WaterShadow {
    fn new(placeholder: Handle<Image>) -> Self {
        Self {
            image: placeholder.clone(),
            placeholder,
            allocated: false,
            resident_frames: 0,
            centre: None,
            chunks: HashMap::new(),
            pending: HashSet::new(),
            stale: HashSet::new(),
            shaded: vec![false; (WINDOW_CHUNKS * WINDOW_CHUNKS) as usize],
        }
    }

    /// The image the materials bind.
    pub(crate) fn image(&self) -> Handle<Image> {
        self.image.clone()
    }

    /// Take in a tile's chunks, queueing those in the window.
    fn add_tile(&mut self, id: AssetId<AdtTile>, tile: &AdtTile) {
        for (i, chunk) in tile.chunks.iter().enumerate() {
            let Some(&nw) = chunk.positions.first() else {
                continue;
            };
            let k = chunk_of_nw(nw);
            self.chunks.insert(k, (id, i as u16));
            if self.centre.is_some_and(|c| in_window(c, k)) {
                self.pending.insert(k);
            }
        }
    }

    /// Drop a tile's chunks, queueing those in the window, which then copy as lit.
    fn remove_tile(&mut self, id: AssetId<AdtTile>) {
        let centre = self.centre;
        let pending = &mut self.pending;
        self.chunks.retain(|&k, &mut (tile, _)| {
            if tile != id {
                return true;
            }
            if centre.is_some_and(|c| in_window(c, k)) {
                pending.insert(k);
            }
            false
        });
    }

    /// Centre the window on chunk `centre` and queue the chunks it takes in: as stale where their
    /// slot holds a shadow, and where they are resident. A lit slot is already right for a chunk
    /// with no tile, so a new torus, lit everywhere, queues only resident chunks.
    fn recentre(&mut self, centre: IVec2) {
        let old = self.centre.replace(centre);
        self.pending.retain(|&k| in_window(centre, k));
        self.stale.retain(|&k| in_window(centre, k));
        let half = WINDOW_CHUNKS / 2;
        for dz in -half..half {
            for dx in -half..half {
                let k = centre + IVec2::new(dx, dz);
                if old.is_some_and(|o| in_window(o, k)) {
                    continue;
                }
                if self.shaded[slot(k)] {
                    self.stale.insert(k);
                    self.pending.insert(k);
                } else if self.chunks.contains_key(&k) {
                    self.pending.insert(k);
                }
            }
        }
    }

    /// Follow the camera's chunk `here`, recentring once it is [`RECENTRE_CHUNKS`] from the centre.
    fn follow(&mut self, here: IVec2) {
        match self.centre {
            Some(c) if (here - c).abs().max_element() < RECENTRE_CHUNKS => {}
            _ => self.recentre(here),
        }
    }

    /// Whether a chunk the shader reads still holds another chunk's block.
    fn reads_stale(&self) -> bool {
        self.centre
            .is_some_and(|c| self.stale.iter().any(|&k| in_read(c, k)))
    }

    /// Take up to [`SHADOW_WRITES_PER_FRAME`] queued chunks as blocks at their torus slots: the
    /// stale ones the shader reads first, then nearest `here`; `mcsh` answers a resident chunk's
    /// bytes, `None` lit.
    fn pop(
        &mut self,
        here: IVec2,
        mcsh: impl Fn(AssetId<AdtTile>, u16) -> Option<Vec<u8>>,
    ) -> Vec<(UVec2, Vec<u8>)> {
        let centre = self.centre.unwrap_or(here);
        let mut queue: Vec<IVec2> = self.pending.iter().copied().collect();
        queue.sort_by_key(|&k| {
            let first = self.stale.contains(&k) && in_read(centre, k);
            (!first, (k - here).length_squared(), k.y, k.x)
        });
        queue.truncate(SHADOW_WRITES_PER_FRAME);
        queue
            .into_iter()
            .map(|k| {
                self.pending.remove(&k);
                self.stale.remove(&k);
                let bytes = self
                    .chunks
                    .get(&k)
                    .and_then(|&(id, i)| mcsh(id, i))
                    .filter(|b| b.len() == CHUNK_BYTES)
                    .unwrap_or_else(|| vec![0; CHUNK_BYTES]);
                self.shaded[slot(k)] = bytes.iter().any(|&b| b != 0);
                (torus_origin(k), bytes)
            })
            .collect()
    }
}

/// The blocks a frame copies into the torus, on their way to the render world. `generation` counts
/// the frames that popped any, so the render world copies each set once; the extracted copy
/// persists there between frames that pop none.
#[derive(Resource, Clone, Default, ExtractResource)]
struct ShadowFrame {
    generation: u64,
    id: AssetId<Image>,
    writes: Arc<Vec<(UVec2, Vec<u8>)>>,
}

/// The torus: `R8Unorm`, one mip, created without data, which wgpu zero-initialises (lit).
fn torus_image() -> Image {
    let mut image = Image::new_uninit(
        Extent3d {
            width: MCSH_TEXELS,
            height: MCSH_TEXELS,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        TextureFormat::R8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
    image
}

/// Build the torus when the look turns to Improved Water and free it when it turns back, and point
/// the materials and the published pass images at the image the look binds.
fn toggle_water_shadow(
    style: Res<WaterStyle>,
    mut shadow: ResMut<WaterShadow>,
    mut window: ResMut<WaterShadowWindow>,
    mut images: ResMut<Assets<Image>>,
    tiles: Res<Assets<AdtTile>>,
    mut materials: ResMut<Assets<LiquidMaterial>>,
    pass: Option<ResMut<WaterPassImages>>,
) {
    let stylised = style.is_stylised();
    if stylised == shadow.allocated {
        return;
    }
    let placeholder = shadow.placeholder.clone();
    *shadow = WaterShadow::new(placeholder);
    *window = WaterShadowWindow::default();
    if stylised {
        shadow.image = images.add(torus_image());
        shadow.allocated = true;
        for (id, tile) in tiles.iter() {
            shadow.add_tile(id, tile);
        }
    }
    debug!(
        "water shadow: {} ({} resident chunks)",
        if stylised {
            "the 2048² torus is built"
        } else {
            "the placeholder is bound"
        },
        shadow.chunks.len()
    );
    for (_, material) in materials.iter_mut() {
        material.extension.shadow = shadow.image.clone();
    }
    if let Some(mut pass) = pass {
        pass.shadow = shadow.image.clone();
    }
}

/// Index the tiles as they load and drop them as they go.
fn track_adt_tiles(
    mut events: MessageReader<AssetEvent<AdtTile>>,
    tiles: Res<Assets<AdtTile>>,
    mut shadow: ResMut<WaterShadow>,
) {
    if !shadow.allocated {
        // The toggle indexes every resident tile when it builds the torus.
        events.clear();
        return;
    }
    for event in events.read() {
        match *event {
            AssetEvent::Added { id } | AssetEvent::LoadedWithDependencies { id } => {
                if let Some(tile) = tiles.get(id) {
                    shadow.add_tile(id, tile);
                }
            }
            AssetEvent::Removed { id } => shadow.remove_tile(id),
            AssetEvent::Modified { .. } | AssetEvent::Unused { .. } => {}
        }
    }
}

/// Follow the camera, pop this frame's blocks once the torus has been resident a frame, and
/// publish the window for the look rows.
fn drive_water_shadow(
    mut shadow: ResMut<WaterShadow>,
    mut frame: ResMut<ShadowFrame>,
    mut window: ResMut<WaterShadowWindow>,
    tiles: Res<Assets<AdtTile>>,
    cam: Query<&GlobalTransform, With<WorldCamera>>,
) {
    if !shadow.allocated {
        return;
    }
    let Some(eye) = cam.iter().next().map(GlobalTransform::translation) else {
        return;
    };
    let here = chunk_at(Vec2::new(eye.x, eye.z));
    shadow.follow(here);
    if shadow.resident_frames < 1 {
        shadow.resident_frames += 1;
    } else if !shadow.pending.is_empty() {
        let t0 = std::time::Instant::now();
        let writes = shadow.pop(here, |id, i| {
            tiles.get(id)?.chunks.get(usize::from(i))?.shadow.clone()
        });
        frame.generation += 1;
        frame.id = shadow.image.id();
        frame.writes = Arc::new(writes);
        debug!(
            "water shadow: generation {} pops {} blocks in {} us, {} queued, {} stale",
            frame.generation,
            frame.writes.len(),
            t0.elapsed().as_micros(),
            shadow.pending.len(),
            shadow.stale.len()
        );
    }
    if let Some(centre) = shadow.centre {
        window.set_if_neq(WaterShadowWindow::around(centre, !shadow.reads_stale()));
    }
}

/// The generations to copy this frame: `frame`'s if it is new (`seen` the last one taken), with
/// any still waiting for their texture; all of them once it is `ready`, none before. A generation
/// for an image no longer bound is dropped with it.
fn due_generations(
    frame: &ShadowFrame,
    seen: &mut u64,
    waiting: &mut Vec<ShadowFrame>,
    ready: bool,
) -> Vec<ShadowFrame> {
    if frame.generation != *seen {
        *seen = frame.generation;
        waiting.push(frame.clone());
    }
    waiting.retain(|f| f.id == frame.id);
    if ready {
        std::mem::take(waiting)
    } else {
        Vec::new()
    }
}

/// Render world: copy each generation's blocks into the torus once. `copied` counts every byte
/// copied, so a frame that copies nothing adds nothing (`RUST_LOG=benilla_water::mcsh=debug`).
fn upload_water_shadow(
    queue: Res<RenderQueue>,
    images: Res<RenderAssets<GpuImage>>,
    frame: Option<Res<ShadowFrame>>,
    mut seen: Local<u64>,
    mut waiting: Local<Vec<ShadowFrame>>,
    mut copied: Local<u64>,
) {
    let Some(frame) = frame else { return };
    if frame.generation == *seen && waiting.is_empty() {
        return;
    }
    let gpu = images.get(frame.id);
    let due = due_generations(&frame, &mut seen, &mut waiting, gpu.is_some());
    let Some(gpu) = gpu else {
        debug!(
            "water shadow: generation {} waits for its texture",
            frame.generation
        );
        return;
    };
    let t0 = std::time::Instant::now();
    let mut bytes = 0;
    for (origin, block) in due.iter().flat_map(|f| f.writes.iter()) {
        queue.write_texture(
            TexelCopyTextureInfo {
                texture: &gpu.texture,
                mip_level: 0,
                origin: Origin3d {
                    x: origin.x,
                    y: origin.y,
                    z: 0,
                },
                aspect: TextureAspect::All,
            },
            block,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(CHUNK_TEXELS),
                rows_per_image: Some(CHUNK_TEXELS),
            },
            Extent3d {
                width: CHUNK_TEXELS,
                height: CHUNK_TEXELS,
                depth_or_array_layers: 1,
            },
        );
        bytes += block.len() as u64;
    }
    *copied += bytes;
    debug!(
        "water shadow: generation {} copies {bytes} bytes in {} us, {} in all",
        frame.generation,
        t0.elapsed().as_micros(),
        *copied
    );
}

/// Whether the look is Improved Water, the condition every system here but the toggle runs under.
fn stylised(style: Res<WaterStyle>) -> bool {
    style.is_stylised()
}

pub(super) fn register(app: &mut App) {
    let placeholder = app
        .world_mut()
        .resource_mut::<Assets<Image>>()
        .add(Image::new_fill(
            Extent3d::default(),
            TextureDimension::D2,
            &[0],
            TextureFormat::R8Unorm,
            RenderAssetUsages::RENDER_WORLD,
        ));
    app.insert_resource(WaterShadow::new(placeholder))
        .init_resource::<WaterShadowWindow>()
        .init_resource::<ShadowFrame>()
        .add_plugins(ExtractResourcePlugin::<ShadowFrame>::default())
        // The tiles the last frame's asset events added or dropped (read before this frame's are
        // sent), then the window, before the look rows take it. Every other restamp of the
        // materials and the published pass images runs after `drive_reflection`.
        .add_systems(
            PostUpdate,
            (
                toggle_water_shadow.run_if(resource_changed::<WaterStyle>),
                track_adt_tiles
                    .run_if(stylised)
                    .before(bevy::asset::AssetEventSystems),
                drive_water_shadow.run_if(stylised),
            )
                .chain()
                .before(super::reflect::drive_reflection),
        );
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render_app.add_systems(
        Render,
        upload_water_shadow.in_set(RenderSystems::PrepareResources),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use benilla_assets::coords::wow_to_bevy;
    use benilla_formats::ChunkMesh;

    /// The torus's span in yards, 32 chunks; the shader's coordinate is the map's over it.
    const MCSH_SPAN_YD: f32 = WINDOW_CHUNKS as f32 * CHUNK_SIZE;

    /// The torus coordinate of Bevy `x`, `z` yards, as `liquid.wgsl`'s `lighting_sun_mcsh` reads
    /// it.
    fn torus_uv(xz: Vec2) -> Vec2 {
        ((xz + Vec2::splat(MAP_HALF_YD)) / MCSH_SPAN_YD).fract_gl()
    }

    /// With no map the row says so: every lane 0.
    #[test]
    fn no_map_is_an_invalid_window() {
        assert_eq!(WaterShadowWindow::default().lanes(), [0.0; 4]);
    }

    /// The shader's copies of the map's half extent and the torus's span are 32 tiles and 32
    /// chunks.
    #[test]
    fn the_shaders_spans_are_the_formats() {
        let wgsl = include_str!("../../benilla-assets/src/shaders/liquid.wgsl");
        let value = |name: &str| -> f32 {
            let decl = format!("const {name}: f32 = ");
            let at = wgsl.find(&decl).unwrap_or_else(|| panic!("no `{decl}`")) + decl.len();
            wgsl[at..]
                .split(';')
                .next()
                .unwrap()
                .trim()
                .parse()
                .unwrap()
        };
        assert!((value("MAP_HALF_YD") - 32.0 * TILE_SIZE).abs() < 1e-3);
        assert!((value("MCSH_SPAN_YD") - 32.0 * CHUNK_SIZE).abs() < 1e-3);
        assert_eq!(MAP_HALF_YD, 32.0 * TILE_SIZE);
        assert_eq!(MCSH_SPAN_YD, 32.0 * CHUNK_SIZE);
        assert_eq!(MCSH_TEXELS, 2048);
    }

    /// A chunk at global chunk `k` with `shadow` as its MCSH.
    fn chunk_at_global(k: IVec2, shadow: Option<Vec<u8>>) -> ChunkMesh {
        let nw = [
            MAP_HALF_YD - k.y as f32 * CHUNK_SIZE,
            MAP_HALF_YD - k.x as f32 * CHUNK_SIZE,
            0.0,
        ];
        ChunkMesh {
            positions: vec![nw],
            normals: Vec::new(),
            uvs: Vec::new(),
            indices: Vec::new(),
            holes: 0,
            base_texture: None,
            layer_textures: Vec::new(),
            layer_effect_ids: Vec::new(),
            alpha_map: None,
            shadow,
            pred_tex: [0; 64],
            no_effect_doodad: [false; 64],
            index_x: 0,
            index_y: 0,
            area_id: 0,
            impassable: false,
            liquids: Vec::new(),
        }
    }

    /// A deterministic stream of numbers in `0..1`.
    fn lcg(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (*state >> 40) as f32 / (1u64 << 24) as f32
    }

    /// **The torus holds a chunk's MCSH where the shader reads it.** A block written at its chunk's
    /// slot, read at the texel the shader's coordinate names for a world point, is the bit the
    /// formats' oracle (`ChunkMesh::mcsh_shadowed_at`, the texel the terrain samples) gives there,
    /// at 1000 random points over chunks on both sides of the map's centre.
    #[test]
    fn the_torus_reads_the_chunks_mcsh_where_the_terrain_does() {
        let mut rng = 0x5eed_u64;
        let mut torus = vec![0u8; (MCSH_TEXELS * MCSH_TEXELS) as usize];
        let ks = [
            IVec2::new(517, 260),
            IVec2::new(1003, 812),
            IVec2::new(31, 32),
        ];
        let chunks: Vec<ChunkMesh> = ks
            .iter()
            .map(|&k| {
                let bits = (0..CHUNK_BYTES)
                    .map(|_| if lcg(&mut rng) < 0.5 { 255 } else { 0 })
                    .collect();
                chunk_at_global(k, Some(bits))
            })
            .collect();
        for chunk in &chunks {
            let k = chunk_of_nw(chunk.positions[0]);
            let o = torus_origin(k);
            let bits = chunk.shadow.as_ref().unwrap();
            for r in 0..CHUNK_TEXELS {
                for c in 0..CHUNK_TEXELS {
                    let at = ((o.y + r) * MCSH_TEXELS + o.x + c) as usize;
                    torus[at] = bits[(r * CHUNK_TEXELS + c) as usize];
                }
            }
        }
        for n in 0..1000 {
            let chunk = &chunks[n % chunks.len()];
            let nw = chunk.positions[0];
            // Inside the chunk, clear of its edges by a hair so the oracle answers.
            let s = 0.001 + 0.998 * lcg(&mut rng);
            let e = 0.001 + 0.998 * lcg(&mut rng);
            let wow = [nw[0] - s * CHUNK_SIZE, nw[1] - e * CHUNK_SIZE, 0.0];
            let b = wow_to_bevy(wow);
            let uv = torus_uv(Vec2::new(b.x, b.z));
            let t = (uv * MCSH_TEXELS as f32).floor().as_uvec2();
            let got = torus[(t.y * MCSH_TEXELS + t.x) as usize] >= 128;
            assert_eq!(
                Some(got),
                chunk.mcsh_shadowed_at(wow),
                "point {n} at {wow:?}, texel {t}"
            );
        }
    }

    /// The chunk a window centres on is the camera's, and its middle in yards is that chunk's
    /// north-west corner. The read extent ends half a texel inside the 15th chunk on either side,
    /// so the bilinear read never reaches the window's outer ring.
    #[test]
    fn the_window_is_the_cameras_chunk() {
        let k = IVec2::new(500, 600);
        let corner = k.as_vec2() * CHUNK_SIZE - Vec2::splat(MAP_HALF_YD);
        assert_eq!(chunk_at(corner + Vec2::splat(1.0)), k);
        let w = WaterShadowWindow::around(k, true);
        assert_eq!(w.lanes(), [corner.x, corner.y, HALF_EXTENT_YD, 1.0]);
        assert!((HALF_EXTENT_YD + 0.5 * TEXEL_YD - 500.0).abs() < 1e-3);
        for d in [-HALF_EXTENT_YD + 1e-2, HALF_EXTENT_YD - 1e-2] {
            // The read point and the far edge of its bilinear footprint.
            let reach = d + d.signum() * 0.5 * TEXEL_YD;
            for at in [corner.x + d, corner.x + reach] {
                let c = chunk_at(Vec2::new(at, corner.y));
                assert!(in_read(k, c), "{at}");
                assert!(in_window(k, c));
            }
        }
        assert!(!in_read(k, k + IVec2::new(READ_CHUNKS, 0)));
        assert!(!in_read(k, k - IVec2::new(READ_CHUNKS + 1, 0)));
    }

    /// Resident chunks keyed as their tile holds them.
    fn shadow_with(chunks: &[(IVec2, AssetId<AdtTile>)]) -> WaterShadow {
        let mut s = WaterShadow::new(Handle::default());
        for (i, &(k, id)) in chunks.iter().enumerate() {
            s.chunks.insert(k, (id, i as u16));
        }
        s
    }

    /// **A one-chunk recentre queues exactly the 32 chunks the window takes in** where the torus
    /// holds shadows, the first window only the resident ones, and a move inside the hysteresis
    /// queues nothing. Over lit slots only the resident chunks it takes in are queued, none stale.
    #[test]
    fn a_one_chunk_recentre_queues_32_chunks() {
        let id = AssetId::<AdtTile>::default();
        let c = IVec2::new(512, 512);
        let east = c + IVec2::new(16, 3);
        let mut s = shadow_with(&[(c, id), (east, id)]);
        s.follow(c);
        assert_eq!(
            s.pending.len(),
            1,
            "the first window: its one resident chunk"
        );
        assert!(s.stale.is_empty(), "a new torus is lit, not stale");
        s.pending.clear();
        s.follow(c + IVec2::new(1, -1));
        assert!(s.pending.is_empty(), "inside the hysteresis");
        s.recentre(c + IVec2::X);
        assert_eq!(
            s.pending,
            HashSet::from([east]),
            "lit slots: the resident chunk"
        );
        assert!(s.stale.is_empty());
        s.pending.clear();
        s.shaded.fill(true);
        s.recentre(c + IVec2::new(2, 0));
        assert_eq!(s.pending.len(), 32);
        assert!(s.pending.iter().all(|k| k.x == c.x + 17));
        assert_eq!(s.stale, s.pending);
        s.pending.clear();
        s.stale.clear();
        s.follow(c + IVec2::new(4, 0));
        assert_eq!(s.centre, Some(c + IVec2::new(4, 0)));
        assert_eq!(s.pending.len(), 64, "two columns");
    }

    /// The pops go nearest first, at most 64 a frame, each at its chunk's slot with its bytes;
    /// a chunk with no MCSH or no tile is a lit block.
    #[test]
    fn the_pops_go_nearest_first_and_unknown_chunks_are_lit() {
        let id = AssetId::<AdtTile>::default();
        let c = IVec2::new(100, 100);
        let near = c + IVec2::new(1, 0);
        let mut s = shadow_with(&[(near, id)]);
        s.centre = Some(c);
        s.pending
            .extend((0..80).map(|i| c + IVec2::new(-15 + i % 30, -15 + i / 30)));
        s.pending.insert(near);
        let first = s.pop(c, |_, _| Some(vec![255; CHUNK_BYTES]));
        assert_eq!(first.len(), SHADOW_WRITES_PER_FRAME);
        assert_eq!(first[0], (torus_origin(near), vec![255; CHUNK_BYTES]));
        assert!(first[1..].iter().all(|(_, b)| b.iter().all(|&v| v == 0)));
        assert_eq!(s.pending.len(), 81 - 64);
        let d = |o: UVec2| {
            // The slot's offset from the centre's, unwrapped into the window's -16..16.
            let k = (o / CHUNK_TEXELS).as_ivec2() - c + IVec2::splat(16);
            (k.rem_euclid(IVec2::splat(32)) - IVec2::splat(16)).length_squared()
        };
        assert!(first.windows(2).all(|w| d(w[0].0) <= d(w[1].0)));
    }

    /// **A walk never reads a stale chunk.** A diagonal two-chunk recentre takes in 124 chunks, 59
    /// of them inside the read extent; with nearer tiles still queued those 59 go first, so they
    /// land in the recentre's own frame and the window stays valid.
    #[test]
    fn a_recentre_copies_what_the_shader_reads_first() {
        let c = IVec2::new(300, 300);
        let mut s = WaterShadow::new(Handle::default());
        s.recentre(c);
        s.shaded.fill(true);
        // A tile streaming in around the camera: 100 nearer chunks queued.
        s.pending
            .extend((0..100).map(|i| c + IVec2::new(-5 + i % 10, -5 + i / 10)));
        let next = c + IVec2::new(2, 2);
        s.follow(next);
        assert_eq!(s.stale.len(), 124);
        let read: Vec<IVec2> = s
            .stale
            .iter()
            .copied()
            .filter(|&k| in_read(next, k))
            .collect();
        assert_eq!(read.len(), 59);
        assert!(s.reads_stale());
        s.pop(next, |_, _| None);
        assert!(!s.reads_stale(), "the read chunks landed this frame");
        assert!(read.iter().all(|k| !s.pending.contains(k)));
    }

    /// **A tile's removal copies its chunks back as lit**, and leaves a later tile at the same
    /// chunks alone.
    #[test]
    fn removal_writes_zeros() {
        let mut tiles = Assets::<AdtTile>::default();
        let k = IVec2::new(700, 300);
        let tile = |shadow: u8| AdtTile {
            shading: Vec::new(),
            layer_array: Handle::default(),
            alpha_array: Handle::default(),
            shadow_array: Handle::default(),
            doodads: Vec::new(),
            wmos: Vec::new(),
            chunks: vec![chunk_at_global(k, Some(vec![shadow; CHUNK_BYTES]))],
            planar: None,
        };
        let a = tiles.add(tile(255)).id();
        let mut s = WaterShadow::new(Handle::default());
        s.recentre(k);
        s.add_tile(a, tiles.get(a).unwrap());
        let read = |s: &mut WaterShadow, tiles: &Assets<AdtTile>| {
            s.pop(k, |id, i| {
                tiles.get(id)?.chunks.get(usize::from(i))?.shadow.clone()
            })
        };
        assert_eq!(read(&mut s, &tiles)[0].1[0], 255, "the tile's shadow");
        s.remove_tile(a);
        tiles.remove(a);
        let w = read(&mut s, &tiles);
        assert_eq!(w.len(), 1);
        assert_eq!(w[0], (torus_origin(k), vec![0; CHUNK_BYTES]), "lit again");
        let b = tiles.add(tile(255)).id();
        s.add_tile(b, tiles.get(b).unwrap());
        s.remove_tile(a);
        assert_eq!(
            read(&mut s, &tiles)[0].1[0],
            255,
            "the new tile's chunk stays"
        );
        assert!(s.pending.is_empty());
    }

    /// An app with the shadow's systems, a camera at chunk `k` and a style.
    fn app_at(k: IVec2, style: WaterStyle) -> App {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, AssetPlugin::default()))
            .init_asset::<Image>()
            .init_asset::<AdtTile>()
            .init_asset::<LiquidMaterial>()
            .insert_resource(style);
        register(&mut app);
        let at = k.as_vec2() * CHUNK_SIZE - Vec2::splat(MAP_HALF_YD) + Vec2::splat(1.0);
        app.world_mut().spawn((
            GlobalTransform::from_translation(Vec3::new(at.x, 0.0, at.y)),
            WorldCamera,
        ));
        app
    }

    /// A tile of one chunk at global chunk `k`, all shadowed.
    fn shadowed_tile(k: IVec2) -> AdtTile {
        AdtTile {
            shading: Vec::new(),
            layer_array: Handle::default(),
            alpha_array: Handle::default(),
            shadow_array: Handle::default(),
            doodads: Vec::new(),
            wmos: Vec::new(),
            chunks: vec![chunk_at_global(k, Some(vec![255; CHUNK_BYTES]))],
            planar: None,
        }
    }

    /// **Nothing is copied on the torus's own frame**: the queue holds until it has been resident
    /// a frame, then pops.
    #[test]
    fn the_writes_wait_for_a_resident_frame() {
        let k = IVec2::new(400, 400);
        let mut app = app_at(k, WaterStyle::Stylised);
        let id = app
            .world_mut()
            .resource_mut::<Assets<AdtTile>>()
            .add(shadowed_tile(k));
        app.update();
        let gen = |app: &App| app.world().resource::<ShadowFrame>().generation;
        let shadow = app.world().resource::<WaterShadow>();
        assert!(shadow.allocated);
        assert_eq!(shadow.chunks.get(&k).map(|c| c.0), Some(id.id()));
        assert_eq!(
            shadow.pending.len(),
            1,
            "queued, held on the allocation frame"
        );
        assert_eq!(gen(&app), 0);
        assert!(app.world().resource::<WaterShadowWindow>().valid);
        app.update();
        assert_eq!(gen(&app), 1, "popped once resident");
        assert!(app.world().resource::<WaterShadow>().pending.is_empty());
        app.update();
        assert_eq!(gen(&app), 1, "nothing queued, no new generation");
    }

    /// **A teleport reads lit until what it reads has landed.** Over a torus holding shadows the
    /// whole window is stale: the 900 chunks of the read extent copy first, 64 a frame, and the
    /// window is valid again on the frame the last of them pops (the 15th); the outer ring follows
    /// on the 16th. Over a lit torus (the camera's first moves at boot) nothing is queued.
    #[test]
    fn a_teleport_reads_lit_until_its_chunks_land() {
        let k = IVec2::new(400, 400);
        let mut app = app_at(k, WaterStyle::Stylised);
        app.update();
        app.update();
        let mut cam = app
            .world_mut()
            .query_filtered::<&mut GlobalTransform, With<WorldCamera>>();
        let mut teleport = |app: &mut App, to: IVec2| {
            let at = to.as_vec2() * CHUNK_SIZE - Vec2::splat(MAP_HALF_YD) + Vec2::splat(1.0);
            *cam.single_mut(app.world_mut()).unwrap() =
                GlobalTransform::from_translation(Vec3::new(at.x, 0.0, at.y));
        };
        teleport(&mut app, k + IVec2::new(-60, 70));
        app.update();
        let shadow = app.world().resource::<WaterShadow>();
        assert!(shadow.pending.is_empty() && shadow.stale.is_empty());
        assert!(app.world().resource::<WaterShadowWindow>().valid);
        app.world_mut()
            .resource_mut::<WaterShadow>()
            .shaded
            .fill(true);
        let far = k + IVec2::new(100, -40);
        teleport(&mut app, far);
        let read = (2 * READ_CHUNKS * 2 * READ_CHUNKS) as usize;
        let frames = read.div_ceil(SHADOW_WRITES_PER_FRAME);
        for f in 1..=16 {
            app.update();
            let w = *app.world().resource::<WaterShadowWindow>();
            assert_eq!(w.valid, f >= frames, "frame {f}");
            assert_eq!(
                w.centre,
                far.as_vec2() * CHUNK_SIZE - Vec2::splat(MAP_HALF_YD)
            );
        }
        let shadow = app.world().resource::<WaterShadow>();
        assert!(shadow.pending.is_empty() && shadow.stale.is_empty());
    }

    /// The reference lane builds no torus and frees it on the way back.
    #[test]
    fn the_reference_lane_binds_the_placeholder() {
        let mut app = app_at(IVec2::new(10, 10), WaterStyle::Reference);
        app.update();
        let shadow = app.world().resource::<WaterShadow>();
        assert!(!shadow.allocated);
        assert_eq!(shadow.image, shadow.placeholder);
        app.insert_resource(WaterStyle::Stylised);
        app.update();
        let torus = app.world().resource::<WaterShadow>().image();
        assert!(app
            .world()
            .resource::<Assets<Image>>()
            .get(&torus)
            .is_some());
        app.insert_resource(WaterStyle::Reference);
        app.update();
        let shadow = app.world().resource::<WaterShadow>();
        assert_eq!(shadow.image, shadow.placeholder);
        assert!(shadow.chunks.is_empty() && shadow.centre.is_none());
        assert_eq!(*app.world().resource::<WaterShadowWindow>(), default());
    }

    /// **Render world: a generation is copied once**, one whose texture is missing waits for it,
    /// and one for an image no longer bound is dropped.
    #[test]
    fn a_generation_is_copied_once() {
        let (mut seen, mut waiting) = (0u64, Vec::new());
        let mut f = ShadowFrame {
            generation: 1,
            id: AssetId::default(),
            writes: Arc::new(vec![(UVec2::ZERO, vec![0; CHUNK_BYTES])]),
        };
        let mut due =
            |f: &ShadowFrame, ready| due_generations(f, &mut seen, &mut waiting, ready).len();
        assert_eq!(due(&f, false), 0, "no texture yet");
        assert_eq!(due(&f, true), 1, "copied once it is");
        assert_eq!(due(&f, true), 0, "and never again");
        f.generation = 2;
        assert_eq!(due(&f, false), 0);
        f.generation = 3;
        assert_eq!(due(&f, true), 2, "both waiting generations, in order");
        f.generation = 4;
        assert_eq!(due(&f, false), 0);
        f.id = AssetId::from(bevy::asset::uuid::Uuid::from_u128(7));
        assert_eq!(due(&f, true), 0, "generation 4 was for the old image");
    }
}
