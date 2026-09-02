# Curator — Samsung Frame Curation

Curator is a free, local-first app for Samsung Frame TVs. It takes your photo
library, cleans it up, picks the best shots, and gets them ready to display on
your Frame. It works fully offline: nothing is sent over the network unless you
turn that on yourself.

## Features

- **One safe home for your photos.** Everything is stored in a single catalog
  that knows each photo by its exact content. Copies and duplicates don't make
  extra work.
- **Clean up your SSD safely.** It finds duplicates and near-duplicates, groups
  similar shots, and helps you consolidate a messy folder. Your original files
  are never changed or deleted unless you ask.
- **Reads your formats.** Imports HEIC, JPEG, PNG, WebP, and TIFF from folders
  and NAS drives. Files it can't read (RAW, corrupt) show a clear message
  instead of silently vanishing.
- **Ranks your photos for you.** An offline engine scores each shot on things
  like sharpness, quality, composition, color, and what stands out in it. No
  cloud needed.
- **Suggests how to show them.** It proposes a layout for each photo, such as
  full-bleed, matted, panoramic, square, or a two-photo diptych, and explains
  why.
- **Packs several photos into one frame.** Hand it 2 to 9 photos and it proposes
  a triptych, a quad, or a weighted `packed` grid, with every cell's geometry and
  crop decision written down. Same photos, same weights, same seed: the same
  pixels, every time. It can also suggest which photos belong together, using
  the local image model.
- **Renders Frame-ready files.** One choice produces an exact 1080p or 4K
  image. Nothing is ever upscaled or cropped without you knowing.
- **Approve before it's used.** Approve, reject, undo, and batch-review photos.
  Full history is kept, so you can always see what you decided and why.
- **A simple web app.** Use the whole thing in your browser: browse, analyze,
  preview, and approve. Works with a keyboard and screen readers.
- **Publishes to your Frame.** Filesystem, a simulator, Samsung Art Mode, and
  Home Assistant are all supported. Updates are safe: it tests before sending,
  replaces by exact ID, and can roll back if something goes wrong.
- **Watches for new photos.** New files in your watched folders are picked up
  once, automatically.
- **Playlists and rotation.** Collections rotate through your approved shots —
  with intervals, favorites, seasons, and a "show now" option.
- **Immich support.** Sync from Immich safely, with no deletions ever. An
  optional feedback feature is off by default.
- **Optional cloud use.** If you opt in, it tells you plainly what leaves your
  machine and lets you exclude specific photos or folders. If the cloud is down,
  only cloud work pauses; your local work keeps going.
- **Survives crashes.** Long jobs resume where they stopped instead of starting
  over, and failures are explained with a next step.
- **Easy to install and run.** Ships with macOS launchd and Docker packaging,
  starts without prompts, and can import an old Samsung SSD folder safely (with
  backups first).
- **Taste Lens.** Learns your taste and gently reorders suggestions to match —
  explainable, tunable by comparing two photos, and reversible with one click.
  Approved output never changes.
- **Taste Lens Discovery.** A calm place to find new art: a feed from painters
  and photographers with artist spotlights and a Familiar ↔ Surprising dial.
- **Taste Deck.** Pick between two photos, anywhere: CLI, API, or the web app.
  Votes feed a small learned head over a local, offline image model whose only
  parameters are your votes. Every learned score opens to the votes that produced
  it, and the head-to-head against the hand-built ranking reports an interval,
  never a coin flip. The original engine is never retired automatically.

## Getting started

```bash
make install    # install
make test       # run tests
make lint       # lint
make type       # type check
make acceptance # run the acceptance checks
```

Requires Python 3.11+ and `uv`.

## CLI

The command line tool returns a clear exit code (0 = ok, 2 = error, 3 = nothing
changed) and can print JSON:

```bash
curator catalog init                 # create the catalog
curator catalog add FILE             # add a file
curator ingest PATH                  # import a folder
curator consolidate PATH             # plan a cleanup (--execute to run it)
curator scan PATH                    # compare a folder to the catalog
curator health                       # check the catalog
curator analyze PATH                 # score photos
curator propose ASSET                # suggest a layout
curator propose A B C --weights 2,1,1   # suggest a multi-photo layout (2-9 photos)
curator group SEED                   # which photos belong with this one (needs the local model)
curator manifest ASSET               # make an art-direction file
curator manifest A B C --treatment packed   # a specific multi-photo layout
curator render ASSET --target 4k     # render an image
curator validate FILE ...            # check a rendered file
curator review                       # see approvals
curator review approve ASSET         # approve a photo
curator taste vote                   # pick between two photos (also: votes, retract)
curator taste embed-status --backfill   # check or compute local image vectors
curator taste compare                # learned head vs. hand-built ranking, with uncertainty
curator headless start               # run the server
```

The web app runs at `127.0.0.1:8765` (the FastAPI app, also importable as
`curator.api`).

## Configuration

Set the `CURATOR_DATA_ROOT` environment variable to choose where data is stored
(default: `~/.curator`). Other options follow the pattern
`CURATOR_<AXIS>__<FIELD>`, for example `CURATOR_SOURCE__TYPE=local`.

## Notes

- Real devices (Samsung, Home Assistant, Immich, cloud) are tested against
  simulators, so the project stays deterministic and offline in its tests.
- Full RAW editing (ARW/CR3/NEF) isn't built yet; those files show a clear
  "unsupported" status instead of disappearing.

## License

Released under the [MIT License](LICENSE).
