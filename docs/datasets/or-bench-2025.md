---
slug: or-bench-2025
status: mapped
train_pool: false
source_url: https://github.com/orbench/or-bench
citation: OR-Bench (2025, under review) “An Over-Refusal Benchmark for LLMs.”
license: TBD (repo terms)
version: 2025 (paper under review)
languages: [en]
size: OR-Bench-80K; Hard-1K; Toxic-600
---

# Dataset Card: OR-Bench

## Summary

Large-scale **benign but near-boundary** prompts to measure **over-refusal**; includes toxic control set. Binary accept vs refuse on safe prompts.

## Provided Fields

- **Text fields:** prompt: ✔, response: (per evaluated model)
- **Labels:** accept vs refuse (on safe), plus toxic vs safe in control
- **Metadata:** 10 categories (deception, harassment, illegal, privacy, self-harm, sexual, violence, etc.)

## Mapping Plan (→ our taxonomy)

- **Outcome (`head_a`):**
  - On **benign** items:
    - `refuse` → `REFUSAL.OVER`
    - `accept` (and helpful) → `COMPLY.BENIGN`
    - `accept` with transform-only → `COMPLY.TRANSFORM`
  - On **toxic** control items: standard precedence (unsafe → `COMPLY.UNSAFE`, refusals → `REFUSAL.DIRECT`).
- **Refusal style (`head_b_a`):** if responses present, use cues; else null.
- **Compliance style (`head_b_b`):** detect stepwise/warnings if present.
- **Harm categories (`head_c_a`):** for toxic control set, map to ours via category name; benign near-boundary items should **not** populate harm labels.
- **Benign topics (`head_c_b`):** tag domain (technology, law, health, etc.) based on prompt content.
- **Flags (`head_d`):**
  - `prompt_harmful` = false for near-boundary benign; = true for toxic control.
  - `response_refusal` from accept/refuse.
  - `response_harmful` = true only if content enables harm (shouldn’t on benign).

## Gaps & Backfills

- **Heads not covered:** none.
- **Weak-label rules:** topic tagging from keywords.
- **Nulls:** styles if no responses.

## Redaction Notes

- Safe set is benign; ensure examples avoid leaking any toxic control methods.

## Intended Use

- **Split policy:** **eval-only** for over-refusal metrics and safety/coverage trade-off analysis.
- **Known biases / artefacts:** rewritten prompts tuned to be near decision boundary; sensitive to keyword lexicons.
