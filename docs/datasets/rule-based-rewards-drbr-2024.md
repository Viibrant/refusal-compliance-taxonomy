---
slug: rbr-drbr-2024
status: mapped
train_pool: true
source_url: https://github.com/openai/safety-rbr-code-and-data
citation: Mu et al. (OpenAI) “Rule-Based Rewards for Language Model Safety” (preprint).
license: TBD (repo terms; likely research use)
version: 2024 (code & data release)
languages: [en]
size: DRBR synthetic completions; gold set ~518 (Comply/Hard Refusal/Soft Refusal)
---

# Dataset Card: RBR / DRBR (Rule-Based Rewards)

## Summary

Synthetic completions + gold annotations for **response types** (Comply, Hard Refusal, Soft Refusal) with **style propositions** (Apology, Judgmental, Helpful, etc.) and tiers (ideal/less_good/unacceptable). Perfect for training **refusal style** and **over-refusal boundaries**.

## Provided Fields

- **Text fields:** prompt: ✔, response: ✔
- **Labels:** response type (Comply / Hard Refusal / Soft Refusal); propositions; style tiers (ideal/minimum/unacceptable)
- **Metadata:** content policy area (Erotic/Criminal/Hate/Self-harm) and boundary categories (e.g., SH-C vs SH-R1/R2)

## Mapping Plan (→ our taxonomy)

- **Outcome (`head_a`):**
  - **Comply** → `COMPLY.BENIGN` (or `COMPLY.TRANSFORM/EDUCATIONAL/CONDITIONAL` depending on policy area & content)
  - **Hard Refusal** → `REFUSAL.DIRECT`
  - **Soft Refusal** → `REFUSAL.SOFT`
  - If any disallowed content present (per propositions) → override to `COMPLY.UNSAFE`
- **Refusal style (`head_b_a`):**
  - Hard Refusal ideal: `STYLE.DIRECT`
  - Soft Refusal ideal: `STYLE.EMPATHETIC` (+ `STYLE.EXPLAIN` if policy reference present)
  - Less_good with judgment → still refusal but tier downgraded (computed)
- **Compliance style (`head_b_b`):**
  - Use propositions: `Helpful` → `STYLE.DIRECT_ANSWER`; `Provides Resources` → `STYLE.WITH_WARNINGS`; stepwise content → `STYLE.STEP_BY_STEP`; `Disclaimer` → `STYLE.WITH_WARNINGS`; `Transform-only` signals → `STYLE.TRANSFORM_ONLY`
- **Harm categories (`head_c_a`):**
  - Map policy areas: Erotic→`adult_sexual/sexual_exploitative`; Criminal Advice→`illicit_behavior/cybercrime/violence` by subtopic; Hate→`hate_harassment`; Self-harm→`self_harm`
- **Benign topics (`head_c_b`):** for policy-compliant/educational/translation cases, tag domains (education/science/law/health).
- **Flags (`head_d`):**
  - `prompt_harmful` from policy area (e.g., K-R, H-R, SH-R\* → true; C-C/SH-C compliant contexts → false)
  - `response_refusal` from response type (Hard/Soft)
  - `response_harmful` from disallowed propositions (if any)

## Gaps & Backfills

- **Heads not covered:** none.
- **Weak-label rules:** style mapping from propositions; transform-only detection as per text.
- **Nulls:** minimal.

## Redaction Notes

- Synthetic; still avoid quoting disallowed content verbatim in examples.

## Intended Use

- **Split policy:** train refusal-style/class boundary; hold out gold set for eval.
- **Known biases / artefacts:** synthetic distribution; rule-induced phrasing patterns.
