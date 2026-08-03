# Curator — Samsung Frame Curation Pipeline

Curator is a single-household pipeline that ingests photographs and art from local
folders / mounted NAS paths (and, in future, other source connectors), analyzes them
for visual quality with an intelligence provider, lets you approve the best pieces as
**Art Direction Manifests**, and renders them to your Samsung Frame — either through the
Samsung Art API, an HA-coordinated Samsung, or a static filesystem/URL destination.

The project is organized around **six orthogonal configuration axes** (R022): source,
intelligence provider, interface, runtime, render target, and destination. Every valid
combination travels through the same catalog, analysis schema, manifest, renderer,
validator, approval model, and job journal — there are no bespoke code paths for specific
axis combinations.

## Current Status

This is the M001 foundation: the **catalog core**. It provides:

- A `uv`-managed `src/curator` Python package (requires Python 3.11+).
- A `CuratorConfig` six-axis configuration skeleton (pydantic-settings, `CURATOR_*` env vars).
- Typed errors (`CuratorError` hierarchy) and SHA-256 hashing helpers.
- (Upcoming in this milestone) a WAL-mode SQLite catalog, a content-addressed `ContentStore`,
  a `Catalog` API, source connector abstractions, and a minimal `curator catalog` CLI.

## Getting Started

```bash
make install   # uv sync
make test      # uv run pytest -q
make lint      # uv run ruff check .
make type      # uv run mypy src
```

## Configuration

Configuration is read from environment variables prefixed with `CURATOR_`. Nested axis
fields use a `__` delimiter. The primary knob for early development:

| Env var              | Purpose                              | Default     |
|----------------------|--------------------------------------|-------------|
| `CURATOR_DATA_ROOT`  | Root directory for catalog.db + blobs | `~/.curator` |

Per-axis overrides follow the pattern `CURATOR_<AXIS>__<FIELD>`, e.g. `CURATOR_SOURCE__TYPE=local`.

## License

Private project.
