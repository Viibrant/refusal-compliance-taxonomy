---
slug: sorry-bench-2024
status: mapped
train_pool: false   # primarily eval; prompts are unsafe by design
source_url: https://arxiv.org/abs/2406.14598
citation: Xie et al. (2024) “SORRY-Bench: Systematically Evaluating LLM Safety Refusal.”
license: TBD (confirm; likely research use with redistribution of labels)
version: 2024-06 (paper); base 440 instructions + mutations
languages: [multi]  # includes multiple languages; confirm exact set
size: 440 base × 20 mutations ≈ 8.8k unsafe prompts (plus eval outputs)
---

# Dataset Card: SORRY-Bench

## Summary

Benchmark of **unsafe** instructions across 44 categories with systematic linguistic mutations; labels whether a model **fulfilled** (unsafe content) or **refused**.

## Provided Fields

- **Text fields:** prompt: ✔, response: sometimes (depends on release), judge output: ✔
- **Labels:** `fulfillment` (unsafe compliance) vs `refusal` (no unsafe content)
- **Metadata:** 44-category taxonomy; mutation type; language

## Mapping Plan (→ our taxonomy)

- **Outcome (`head_a`):**
  - If `fulfillment==true` → `COMPLY.UNSAFE`.
  - If `refusal==true` → `REFUSAL.DIRECT` by default; refine to `SOFT`/`PARTIAL` if response text exists and matches cues.
- **Refusal style (`head_b_a`):** if responses available, use same cues as WildGuard; otherwise leave null.
- **Compliance style (`head_b_b`):** if responses available, derive (e.g., `STEP_BY_STEP` if instructions).
- **Harm categories (`head_c_a`):** map SORRY 44 topics → ours:
  - Hacking/System Intrusion → `cybercrime`
  - Fraud/Financial/Illegal Trade → `fraud` / `illicit_behavior`
  - Murder/Violence → `violence`
  - Terrorism → `terrorism`
  - Pornography (access/creation/distribution)/Adult → `adult_sexual`; minors/illegal → `sexual_exploitative`
  - Hate Speech domains → `hate_harassment`
  - Medical/Mental Health/Self-harm → `medical_advice` / `self_harm`
  - Legal/Financial/Investment advice → `legal_advice` / `financial_advice`
  - Conspiracy/Health/Climate/Vaccination misinformation → `misinformation`
  - PII → `privacy_personal_data`
  - Dangerous Challenges/Games/Suicide instructions → `dangerous_activities` / `self_harm`
- **Benign topics (`head_c_b`):** N/A (dataset is unsafe by construction).
- **Flags (`head_d`):**
  - `prompt_harmful` = true for all base items.
  - `response_harmful` = `fulfillment` (true iff unsafe content present).
  - `response_refusal` = `refusal`.

## Gaps & Backfills

- **Heads not covered:** `head_c_b`.
- **Weak-label:** style only if response text is present.
- **Nulls:** outcome styles when only judge labels are given.

## Redaction Notes

- Avoid publishing unsafe completions; keep judge labels and minimal snippets if needed.

## Intended Use

- **Split policy:** **eval-only** for refusal under mutations; good for robustness slices (language/encoding/persuasion).
- **Known biases / artefacts:** mutation distribution; per-category balance is synthetic.
