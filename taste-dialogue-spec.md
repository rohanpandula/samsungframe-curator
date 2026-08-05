# Feature Spec: Taste Dialogue

**One-liner:** A conversational mode where the user drops in *any* images and reacts to them in plain language. Curator extracts the *why* behind those reactions into a living, human-readable **Taste Profile** that feeds every existing taste feature — and whose first-class goal is sharpening the user's own eye, not just the recommendations.

---

## Context (read this if you're new to the project)

**Curator** is a free, local-first Python app (FastAPI web UI + CLI) that manages a large personal photo library (~30k photos), scores shots offline (sharpness, composition, color), proposes display layouts (full-bleed, matted, diptych), and publishes approved art to a Samsung Frame TV. Privacy stance: fully offline by default, cloud strictly opt-in.

It already has two taste features:

- **Taste Lens** — learns preferences and reorders suggestions. Tuned by comparing two photos (pairwise A/B). Explainable and reversible.
- **Taste Lens Discovery** — a feed of painters/photographers with a swipe-style like/skip signal and a Familiar ↔ Surprising dial.

**The gap:** every existing taste signal is binary or comparative — approve/reject, A-beats-B, like/skip artist. That's ~1 bit per interaction. The system learns *what* the user likes but never *why*. Input is also closed: only the user's own library and the curated artist feed. Taste that the user has but has never articulated is invisible to the system — and stays invisible to the user too.

## Design principle (the reason this feature exists)

> **The user is the model. The app is the harness.**

The point of every interaction is to train the *user's* taste and self-knowledge. Better recommendations are the side effect, not the goal. The failure mode to avoid is the inverse — the app becomes the taste-haver and the user becomes its input device.

Corollaries:

1. **Augment, never generate.** Curator never creates art. It interrogates reactions to existing art.
2. **The user's words are the ground truth.** The profile is built from their verbatim vocabulary ("quiet", "heavy", "breathing room"), not generic aesthetic jargon.
3. **Everything the system believes about the user's taste is inspectable, editable, and disputable.** No silent scores.

---

## Feature A: Reaction Room

A drop zone plus a short conversation.

**Input:** any image or group of images — own photos, screenshots, other photographers' work, layouts, gallery walls, a Pinterest grab. NOT limited to the catalog. Dropped third-party images are ephemeral by default (kept only as thumbnails + hashes for evidence); the user can choose to save one to the catalog.

**Flow:**

1. User drops image(s).
2. User writes what they like or dislike, in their own words.
3. The assistant replies with **at most 1–2 short probing questions** — "Is it the emptiness on the left, or the symmetry?" — never a lecture, never a compliment sandwich.
4. Each exchange is distilled into structured **Taste Observations**:

```
TasteObservation {
  images: [content-hash refs or ephemeral thumbs]
  verbatim: "the fog makes it feel private"     # user's exact words, never paraphrased away
  attributes: [negative-space, muted-palette, lone-subject, ...]   # extracted, controlled vocab
  polarity: like | dislike | conflicted
  confidence: float
  session_id, created_at
}
```

**Constraints:**

- Sessions are short. 1–2 follow-up questions per drop, hard cap. This is a gym rep, not an interview.
- NL extraction requires a language model. Follow the existing cloud policy: local model if configured, otherwise cloud **opt-in** with the standard "here's what leaves your machine" disclosure. The Reaction Room is unavailable (with a clear message) if neither is enabled — it never silently degrades to keyword matching.

## Feature B: Taste Profile

The living artifact. A human-readable document the system maintains and the user can read like a page about themselves.

**Contents:**

- **Vocabulary** — the user's recurring words, mapped to visual attributes: *"quiet" → negative space, muted palette, single subject (14 uses)*.
- **Patterns** — claims with evidence: each one links the images and verbatim quotes that support it, with confidence and recency.
- **Tensions** — contradictions surfaced, not smoothed over: *"You describe your taste as minimal, but you favorite dense street scenes at night."* These are the most valuable entries.
- **Evolution** — how the profile has shifted over time.

**Interaction:**

- Every claim is traceable: click it, see the photos and quotes behind it.
- The user can **pin** (yes, that's me), **edit**, or **dispute** (no, wrong) any entry. Disputes remove the claim and mark its evidence for re-interpretation.
- After each Reaction Room session, show a **"What I learned" delta** — 2–3 lines, visible progress.

**The acceptance bar for this feature:** the user reads the profile and says *"that's accurate, and I couldn't have written it myself."*

---

## Integration: the profile becomes upstream

Existing features don't change their UX — they change their source of truth.

| Consumer | Today | With Taste Dialogue |
|---|---|---|
| Taste Lens (reorder) | pairwise-comparison weights | profile dimensions; explanations quote the profile: *"ranked up — strong negative space; you've called this 'quiet' 14 times"* |
| Taste Lens Discovery | swipe signal | artists ranked by profile fit; Familiar ↔ Surprising dial moves along profile dimensions |
| Layout proposals (mats, diptychs, pairings) | heuristics | pairing rationale may cite the profile |
| Swipes / approvals | primary taste signal | demoted to low-resolution corroborating signal |

All consumers degrade gracefully when the profile is empty — nothing existing gets gated on the new feature.

## Surface sketch

- **Web app:** new "Taste" section with two pages — Reaction Room (drop zone + chat thread) and Profile (readable doc, evidence popovers, pin/edit/dispute controls).
- **CLI:**
  - `curator taste drop PATH... --note "text"` — record a reaction non-interactively
  - `curator taste profile` — print the profile (`--json` supported)
  - `curator taste dispute CLAIM_ID`
- **Storage:** `taste_observations` and versioned `taste_profile` tables in the existing catalog DB.

## Success criteria

1. User can drop non-catalog images, react in NL, and see observations recorded with their verbatim words intact.
2. Profile page renders patterns, vocabulary, and tensions, and every claim opens its evidence.
3. Pin/edit/dispute exist and disputes actually change downstream behavior.
4. Taste Lens explanations begin citing profile entries by quote.
5. A "What I learned" delta appears after every session.
6. The subjective test in a 2-week self-trial: the profile tells the user at least one true thing about their eye they hadn't articulated.

## Anti-goals

- **No image generation.** Ever. Not even "generate an example of what you mean."
- **No silent learning.** Nothing enters the profile without appearing in a visible delta.
- **No jargon laundering.** The system never replaces "it feels lonely in a good way" with "melancholic minimalism" — it can map words to attributes, but the user's phrasing stays on top.
- **No interrogation.** Two follow-up questions max; the app never nags the user to "keep training."
- **No hard dependency.** Every existing flow works with an empty profile.

## Open questions

1. Local vs cloud LLM for extraction — which local model is good enough for attribute extraction, and is it worth shipping?
2. Cold start — seed the profile from existing approve/reject and pairwise history, or start blank so early entries are all high-provenance?
3. Retention policy for ephemeral third-party images (copyright + disk): thumbnails only? hash + crop?
4. Does the profile version with the catalog's existing history/undo system, or keep its own timeline?
