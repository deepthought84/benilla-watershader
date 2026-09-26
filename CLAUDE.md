# CLAUDE.md

`AGENTS.md` and `docs/METHOD.md` are the project's rules; this file adds how work outside the source
tree is kept on this machine.

## Scratch work lives in `/data/scratch`, not `/tmp`

- **Use `/data/scratch` for everything worth keeping that is not source code**: analysis and
  one-off scripts, classifier dumps, rendered maps, captures, logs, gate output, notes. Water work
  goes under `/data/scratch/water/`, in a subfolder per topic (e.g. `scan/flow/` for the flow map's
  scripts and images).
- **Prefer it over `/tmp` and over the session scratchpad.** `/tmp` is a tmpfs: it is wiped when
  the machine or a crashed session goes, and a session's scratchpad with it. Use `/tmp` only for
  intermediates that cost nothing to lose.
- Anything derived from the WoW install (dumps, renders, captures) stays in `/data/scratch` and is
  never committed; nothing is ever written into the install.

## Promote what matters into the project

When a scratch script becomes important for development (it is run again and again, or it is how a
feature is verified), keep its essence in the project folder as a proper tool: an example beside
the crate it exercises (like `crates/benilla-water/examples/liquid_planar_map.rs`), a script under
`scripts/`, or a test. Keep the tool, not its output, and follow the method's rules for it (small
commit, gates green, no install data). The exploratory versions stay in `/data/scratch`.
