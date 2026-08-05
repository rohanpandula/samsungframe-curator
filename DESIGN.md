---
name: SamsungFrame Curator
description: A Rietveld / De Stijl composition you operate — white and gray planes on paper, black structural frame lines, primary colors rationed to the edges that move.
colors:
  paper: "#f7f6f2"
  plane: "#eae8e1"
  plane-2: "#d8d5cb"
  ink: "#161514"
  ink-soft: "#45433d"
  mute: "#6c685d"
  red: "#d7261f"
  yellow: "#f4c21b"
  yellow-ink: "#8f6b00"
  blue: "#1f48b0"
typography:
  display:
    fontFamily: "Futura, Century Gothic, Trebuchet MS, Helvetica Neue, Arial, sans-serif"
    fontSize: "clamp(40px, 6vw, 78px)"
    fontWeight: 600
    lineHeight: 0.96
    letterSpacing: "-0.015em"
    textTransform: lowercase
  body:
    fontFamily: "Futura, Century Gothic, Trebuchet MS, Helvetica Neue, Arial, sans-serif"
    fontSize: "16px"
    fontWeight: 400
    lineHeight: 1.6
    letterSpacing: normal
  label:
    fontFamily: "Futura, Century Gothic, Trebuchet MS, Helvetica Neue, Arial, sans-serif"
    fontSize: "12px"
    letterSpacing: "0.06em"
    textTransform: lowercase
rounded:
  sm: "0px"
  md: "0px"
spacing:
  sm: "8px"
  md: "16px"
  lg: "24px"
components:
  button-primary:
    backgroundColor: "{colors.red}"
    textColor: "#ffffff"
    rounded: "{rounded.sm}"
    padding: "13px 24px"
    typography: "lowercase, 600 weight"
  button-ghost:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    rounded: "{rounded.sm}"
    padding: "13px 24px"
  card-plane:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    rounded: "{rounded.sm}"
---

# SamsungFrame Curator — Design System

## Overview

The page is a De Stijl composition you operate: one continuous room of sliding
planes inside a single black structural frame, not a hero-with-cards. Paper and
gray planes are the ground; every plane is outlined by a black structural line;
the three primary colors (red, yellow, blue) are rationed strictly to the edges
that move — the primary CTA, the live frame demo, and the section joints. The
art leads; the software recedes.

## Colors

- **Ground:** paper `#f7f6f2`, gray planes `#eae8e1` / `#d8d5cb`.
- **Structural ink:** `#161514` (borders and rules); `#45433d` body text;
  `#6c685d` small labels (≥4.5:1 on paper).
- **Primaries, used as moving edges only:** red `#d7261f` (CTA, headline word),
  yellow `#f4c21b` (joints, demo), yellow-ink `#8f6b00` (readable yellow on
  paper), blue `#1f48b0` (joints, headline word, demo).
- Color never fills whole feature planes; it marks the joint or the moving edge.

## Typography

- Lowercase geometric sans (Futura-first stack), flush to plane edges.
- Display: `clamp(40px, 6vw, 78px)`, weight 600, line-height 0.96, tight
  tracking, lowercase.
- Body: 16px/1.6, ink-soft. Small labels: 12px, letter-spaced, lowercase, mute.
- Headlines may carry one or two primary colors on key words; italic/serif
  emphasis is not used — emphasis comes from weight and color.

## Layout

- A single `frame-shell`: a 2px ink border wrapping the whole composition with a
  14px page margin.
- Planes are separated by 2px ink rules; grids are asymmetric (e.g. 3/1/2/2/1/1/3
  feature composition, two-panel taste duo) — never same-size card rows.
- Text blocks are rectangles in the composition; asymmetric balance is ruled by
  primary-color joints.
- Responsive: grids collapse to a single column under 760px; the nav collapses to
  brand-only; the hero stacks with the frame demo below the copy.

## Elevation & Depth

- Flat planes. No shadows, no gradients, no glass. Depth is expressed by plane
  boundaries (black rules) and color joints, not elevation.

## Shapes

- Zero corner radius everywhere (De Stijl planes). Buttons and planes are
  rectangles with 2px ink borders.

## Components

- **Primary CTA:** flat red plane, 2px ink border, white lowercase text; on
  hover it nudges 1px (translate) — a physical slide, not a glow.
- **Ghost CTA:** paper plane, ink border, ink text; hover fills plane gray.
- **Live frame demo:** a 16:9 plane with a slide control; artworks are flat
  De Stijl compositions that translate horizontally when advanced.
- **Steps / cells / panels:** paper planes with 2px ink rules and a colored
  joint (yellow/blue inset bar) marking the moving edge.
- **Nav:** a single continuous plane divided by the bottom ink rule; links are
  lowercase, hover underlines with a 2px ink rule.

## Do's and Don'ts

- **Do** treat every block as a rectangle plane in one composition; keep primary
  color on joints and moving edges only; use the black structural frame line to
  bind the page; set type lowercase and flush; keep zero radius.
- **Don't** add cards-with-icons as the page structure; fill whole planes with
  primary color; use gradients, glows, shadows, or glass; add an eyebrow/kicker
  above headings; use section numbers unless the sequence itself is the content;
  fabricate customer claims or metrics.
