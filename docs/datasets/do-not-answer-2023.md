---
slug: do-not-answer-2023
status: mapped
train_pool: true
source_url: https://arxiv.org/abs/2308.13387
citation: Wang et al. (2023) “Do-Not-Answer: A Dataset for Evaluating Safeguards in LLMs.”
license: TBD (confirm dataset licence; likely research use)
version: 2023
languages: [en]
size: 939 prompts (+ model responses with 6-class response types)
---

# Dataset Card: Do-Not-Answer (DNA)

## Summary

Risky prompts that **should be refused**, plus model responses labelled into six response types (five refusal styles; one harmful compliance). Covers 12 mid-level harm types.

## Provided Fields

- **Text fields:** prompt: ✔, response: ✔
- **Labels:** harmfulness (safe/harmful); response category ∈ {0..5}
- **Metadata:** harm types (12), protected-group mentions

## Mapping Plan (→ our taxonomy)

- **Outcome (`head_a`)** from response category:
  - cat **5** (directly follows risky instruction) → `COMPLY.UNSAFE`
  - cat **0** (“cannot assist”) → `REFUSAL.DIRECT`
  - cat **4** (“lack of knowledge/uncertainty”) → `REFUSAL.CAPABILITY`
  - cat **1** (refute false assumption) → `REFUSAL.DIRECT` or `STYLE.EXPLAIN` (see style below)
  - cat **2** (dual perspectives/balanced) → usually `COMPLY.EDUCATIONAL` **unless** it still refuses; inspect response:
    - if gives safe alternative info without enabling → `COMPLY.EDUCATIONAL`
    - if refuses without content → `REFUSAL.DIRECT`
  - cat **3** (cautious answer with disclaimer) → if any enabling info → `COMPLY.UNSAFE`; else `COMPLY.WITH_WARNINGS` (maps to `COMPLY.CONDITIONAL`)
- **Refusal style (`head_b_a`):**
  - cat0 → `STYLE.DIRECT`
  - cat1 → `STYLE.EXPLAIN` (refutation)
  - cat4 → (if truly capability) map to `REFUSAL.CAPABILITY` outcome; style N/A
  - empathetic phrasing → `STYLE.EMPATHETIC` (rare here)
- **Compliance style (`head_b_b`):**
  - cat3 (cautious) → `STYLE.WITH_WARNINGS`
  - if stepwise present → `STYLE.STEP_BY_STEP`
  - if transform-only → `STYLE.TRANSFORM_ONLY`
- **Harm categories (`head_c_a`):** map DNA’s 12 types:
  - Sensitive org/gov info → `privacy_personal_data` / `extremism` (context dependent)
  - Private/personal info → `privacy_personal_data`
  - Illegal activities → `illicit_behavior` (+ `cybercrime` if relevant)
  - Unethical/unsafe actions → `dangerous_activities`
  - Disinformation campaigns → `misinformation` / `political_persuasion`
  - Stereotypes/discrimination & Toxic/hate → `hate_harassment` (+ `violence` if threats)
  - Adult content → `adult_sexual` / `sexual_exploitative` if illegal
  - False/misleading info → `misinformation`
  - Harmful expert advice (med/legal/finance) → `medical_advice` / `legal_advice` / `financial_advice`
  - Mental health / overreliance → `self_harm` (if self-injury) or benign `psychology_self_development`
  - Treat chatbot as human → benign topics; no harm label (use `head_c_b` instead)
- **Benign topics (`head_c_b`):** when items are “safe” (rare), set domain tags (tech/health/etc.).
- **Flags (`head_d`):**
  - `prompt_harmful` from DNA harmfulness
  - `response_harmful` = (response category==5) or content enabling
  - `response_refusal` = (response in {0,1,2,3,4} **and** contains explicit refusal cues)

## Gaps & Backfills

- **Heads not covered:** none, but some categories require manual check to distinguish `COMPLY.EDUCATIONAL` vs `REFUSAL.DIRECT`.
- **Weak-label rules:** refusal cues lexicon; stepwise detection.
- **Nulls:** minimal.

## Redaction Notes

- Mask any PII examples in the paper release.
- Replace enabling snippets with `[REDACTED_STEP]` in examples.

## Intended Use

- **Split policy:** usable for **training** refusal-style classifiers and unsafe-compliance detectors; hold out a slice for eval.
- **Known biases / artefacts:** prompts are all risky by design → does not reflect benign distribution.
