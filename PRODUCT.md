# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Frame owners: people who own (or are considering) a Samsung Frame TV and want
their personal photo library displayed on it as art, not just screenshotted onto
a screen. They are typically not developers. Their situation: a real photo
library on a laptop, NAS, or old SSD, mixed quality and duplicates, and a Frame
that currently shows either default art or a handful of photos they set up once
and forgot. The job they are doing: get their best photos onto the Frame,
looking intentional, without an engineering project.

Secondary audience (eval): developers browsing the GitHub repo, but the page
itself leads with the owner story.

## Product Purpose

Curator turns a personal photo library into a quiet, rotating art wall on a
Samsung Frame. It consolidates the library (finds duplicates, cleans up),
picks the strongest shots (offline analysis), proposes how each should be shown
(full-bleed, matted, panoramic, diptych), renders exact Frame-ready files, and
safely publishes them. It learns taste over time and can discover new art, all
without sending photos to the cloud by default.

Success: a Frame owner goes from "I have a folder of photos" to "my Frame shows
my best photos, rotating, and it keeps getting better," without touching a
terminal if they don't want to.

## Positioning

Everything runs locally and offline by default; the tool is deterministic (same
photos in, same art out), and it learns the owner's taste rather than applying
generic defaults. The "air-gapped, private, taste-aware" combination is the
thing a neighboring photo-display tool cannot truthfully copy.

## Operating Context

- The Frame is the destination: a 16:9 display meant to look like a framed
  picture. Output must be exact 1080p/4K sRGB.
- The user's source material is a real, messy library: HEIC from iPhones, JPEG,
  PNG, WebP, TIFF; duplicates, near-duplicates, low-res and high-res copies.
- Runs on the user's own machine (macOS, Linux/Docker), not a hosted service.
- Privacy is load-bearing: originals and TV credentials stay local; cloud use is
  opt-in with plain disclosure and per-photo exclusions.
- Persona is non-technical; the CLI exists but is not the pitch.

## Capabilities and Constraints

- Ingest from local folders / NAS; HEIC/JPEG/PNG/WebP/TIFF; RAW and corrupt
  files reported, not dropped.
- Non-destructive consolidation; duplicate detection (exact + perceptual);
  best-original recovery.
- Offline, deterministic analysis: quality, saliency, crop safety, color,
  pairing affinity.
- Art-direction proposals (full-bleed, matte, panoramic, square, diptych) with
  rationale.
- Deterministic rendering + validation; never silent upscale.
- Approval/history; Darkroom Bench web UI (accessible, keyboard-first).
- Publish: filesystem, simulator, Samsung Art Mode, Home Assistant coordination.
- Watcher, collections/rotation; Immich connector (no deletes).
- Optional cloud/hybrid with privacy disclosure + exclusions.
- Crash-safe jobs; packaging (launchd, Docker, optional CUDA); legacy-SSD
  migration with backups.
- Taste Lens (profiles, pairwise learning, reversible) + Taste Lens Discovery.
- Constraint: works offline by default; deterministic; never deletes source
  without explicit action; approved output never changes when Taste Lens is on.

## Brand Commitments

- Product name: SamsungFrame Curator ("Curator" in code/docs).
- Positioning voice: calm, private, precise, taste-aware. Not hype.
- It displays art on a Samsung Frame; the art (the photos) leads; the software
  recedes.
- Open source (MIT) and public; honesty about scope (RAW editing deferred;
  generative/identity/deletion features out of scope).

## Evidence on Hand

- The shipped product (M001-M007): 634 automated tests, 8-file acceptance gate,
  27/27 requirements validated, MIT license.
- Live explainer page at GitHub Pages (current look is being replaced).
- No real user testimonials, customer logos, or benchmark numbers exist; these
  must not be fabricated.

## Product Principles

1. The art leads; the software recedes. Show the outcome (photos on a Frame),
   not the pipeline.
2. Privacy is a feature: local by default, opt-in cloud, honest disclosure.
3. Determinism builds trust: same input, same result, reversible taste.
4. Non-destructive: originals are never harmed; approved output never changes.
5. Quality over breadth: exact, verified output; no silent degradation.

## Accessibility & Inclusion

The web surface should be keyboard-operable and meet WCAG 2.1 AA (consistent
with the Darkroom Bench UI), with respect for reduced motion.
