# Curator — Samsung Frame Curation Pipeline

Curator is a local-first studio for Samsung Frame TVs and digital displays. It
ingests photographs from local folders and mounted NAS paths, consolidates them
non-destructively, analyzes them for quality offline, proposes display
treatments, renders exact Frame-ready files, and publishes approved art through
pluggable destination adapters. It runs air-gapped by default: no mandatory
network calls during ingest, analysis, proposal, render, or a local publish.

The project is organized around **six orthogonal configuration axes** (R022):
source, intelligence provider, interface, runtime, render target, and
destination. Every valid combination travels through the same catalog, analysis
schema, manifest, renderer, validator, approval model, and job journal. There
are no bespoke code paths for specific axis combinations.

## Current status

Complete end to end: **634 automated tests**, an 8-file deterministic
`make acceptance` gate, and **27 of 27 active requirements validated**.

## Features

- **Content-addressed catalog.** SQLite + SHA-256 artifact store with
  connector-scoped source identity; renames, moves, and remote revisions never
  collide.
- **Non-destructive consolidation.** Dry-run plan → staged copy → hash verify →
  atomic promote, resumable after interruption. Sources stay untouched until you
  approve archival.
- **Local ingest.** HEIC, JPEG, PNG, WebP, TIFF from folders and NAS, with exact
  and perceptual duplicate detection plus best-original recovery. RAW and
  corrupt files report an explicit status instead of disappearing.
- **Offline analysis.** A deterministic CPU engine scores technical and
  aesthetic quality, saliency and subjects, crop safety, color story, and
  pairing affinity. No cloud required.
- **Art direction.** A policy engine proposes treatments — full-bleed,
  contain-and-matte, panoramic, square, diptych — each with rationale, captured
  in a versioned ArtDirectionManifest.
- **Deterministic rendering.** Byte-identical 1080p / 4K / custom sRGB from one
  manifest. A validator gates publishability; upscaling is never silent.
- **Approval & history.** Per-image approve, reject, undo, redo, and batch, with
  append-only history and rationale.
- **Darkroom Bench web UI.** A dependency-light, accessible browser app for the
  whole loop: browse, analyze, propose, render, validate, review. Keyboard- and
  screen-reader-first.
- **Safe publishing.** Filesystem and simulator destinations, Samsung Art Mode
  (canary upload, exact-ID replace, rollback), and Home Assistant coordination
  behind an exclusive write lease.
- **Watcher & rotation.** A durable watcher ingests stabilized photos exactly
  once. Collections and playlists rotate deterministically — intervals,
  favorites, seasons, show-now.
- **Immich connector.** Checkpointed, idempotent sync with availability
  tombstones and a disabled-by-default feedback sink that never deletes.
- **Cloud/hybrid opt-in.** Privacy-first routing with plain-language disclosure
  and per-source/per-image exclusions. A provider outage pauses only
  cloud-dependent work.
- **Crash-safe jobs.** A durable orchestrator resumes from checkpoints without
  duplicating art; every failure is classified with a recovery action.
- **Production ready.** Packaging for macOS launchd and Docker (CPU + optional
  CUDA), a prompt-free headless start, and a migration tool that imports a
  legacy Samsung SSD read-only with backups before every mutation.
- **Taste Lens.** Versioned, isolated profiles that rerank candidates as an
  explainable baseline-plus-delta, tuned by pairwise comparisons — reversible in
  one click, and it never changes approved output.
- **Taste Lens Discovery.** A federated, outage-isolated creator catalog feeding
  a Taste Deck with artist spotlights, likely/adjacent/wildcard modes, and a
  Familiar↔Surprising dial.

## Getting started

```bash
make install    # uv sync
make test       # uv run pytest -q
make lint       # uv run ruff check .
make type       # uv run mypy src
make acceptance # deterministic air-gapped acceptance gate (8 files)
make all        # install + lint + type + test
```

Requires Python 3.11+ and `uv`.

## CLI

`curator` exposes a headless interface with stable exit codes
(0 = ok, 1 = partial/warnings, 2 = fatal, 3 = no change) and `--json` output:

```bash
curator catalog init|add FILE      # create catalog / add a local file
curator ingest PATH [--resume]     # ingest a folder (dedup + cluster)
curator consolidate PATH [--dry-run|--execute] [--resume] [--archive] [--json]
curator scan PATH [--json]         # diff a folder vs catalog (exit 0/3)
curator health [--json]            # catalog status (exit 0)
curator analyze PATH [--profile] [--json]
curator propose ASSET [--target] [--json]
curator manifest ASSET [--target] [--json]
curator render SOURCE [--target 1080p|4k|WxH] [--json]
curator validate FILE --expected-sha X --target WxH [--json]
curator review [--status] [--json]          # list approvals
curator review approve|reject|undo ASSET    # change state
curator headless start [--config FILE] [--check]   # headless server (launchd/Docker entrypoint)
```

The FastAPI app is also importable (`curator.api:create_app`) and serves the
Darkroom Bench web UI plus `/docs` on loopback `127.0.0.1:8765`.

## Configuration

Configuration is read from environment variables prefixed with `CURATOR_`; nested
axis fields use a `__` delimiter. The primary knob:

| Env var              | Purpose                                 | Default     |
|----------------------|-----------------------------------------|-------------|
| `CURATOR_DATA_ROOT`  | Root directory for catalog.db + blobs    | `~/.curator`|

Per-axis overrides follow the pattern `CURATOR_<AXIS>__<FIELD>`, e.g.
`CURATOR_SOURCE__TYPE=local`. `CURATOR_NETWORK=deny` is a documented no-op
air-gap posture; the real protection is the static no-network-import audit in
the acceptance gate.

## Project layout

```
src/curator/
  analysis/     analysis contract, offline local engine, analysis pipeline
  artdirection/ ArtDirectionManifest + policy engine
  approve/      approval & history service
  catalog.py    content-addressed catalog + content store
  collections/  playlists + deterministic rotation
  connectors/   local + synthetic remote + Immich connectors
  consolidate/  consolidation plan + executor
  dest/         destination adapters (filesystem/simulator/Samsung)
  ha.py         Home Assistant coordination (exclusive write lease)
  ingest/       ingest pipeline, decode, clustering, report
  jobs/         crash-safe job orchestrator
  migrate/      legacy SSD migration tool
  providers/    cloud/hybrid provider routing + privacy
  render/       deterministic renderer + artifact validator
  taste/        Taste Lens profiles, ranking, pairwise, controls, discovery
  watch/        durable watcher
  cli.py        headless CLI (stable exit codes, --json)
  api.py        FastAPI app + Darkroom Bench web UI
```

## Notes on scope

- Live Samsung / Home Assistant / Immich / cloud transports are capability-probed
  and exercised against simulator/synthetic runtimes in the acceptance gate. The
  project stays deterministic and air-gapped in tests.
- R024 (full RAW development, ARW/CR3/NEF) is deferred; recognized RAW files
  surface an explicit unsupported/preview status rather than disappearing.
- R025 (generative outpainting/restoration), R026 (face identity), and R027
  (deleting destination art when a source disappears) are out of scope by design.

## License

Released under the [MIT License](LICENSE).
