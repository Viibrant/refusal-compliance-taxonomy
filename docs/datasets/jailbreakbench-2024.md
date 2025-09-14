---
slug: jailbreakbench-2024
status: mapped
train_pool: false   # primarily evaluation
source_url: https://jailbreakbench.github.io/
citation: Chao et al. (2024) “JailbreakBench: An Open Robustness Benchmark for Jailbreaking LLMs.”
license: TBD (benchmark artifacts under repo terms)
version: 2024 (NeurIPS)
languages: [en]
size: JBB-Behaviors: 100 harmful + 100 benign behaviors; plus attack artifacts
---

# Dataset Card: JailbreakBench

## Summary

Open robustness benchmark for jailbreaking and defenses. Provides harmful/benign behaviors, adversarial prompts, and automated judges for **safe/unsafe** and **refusal**.

## Provided Fields

- **Text fields:** prompt (attack), sometimes model response, behavior description
- **Labels:** judge `safe/unsafe`, `refusal` yes/no; behavior categories (10)
- **Metadata:** source (Original/AdvBench/TDC-HarmBench), attack/defense config

## Mapping Plan (→ our taxonomy)

- **Outcome (`head_a`):**
  - `unsafe` → `COMPLY.UNSAFE`
  - `safe` + `refusal==true`:
    - if benign behavior → `REFUSAL.OVER`
    - else (harmful prompt) → `REFUSAL.DIRECT`
  - `safe` + `refusal==false` → depends:
    - if response content present and is transform-only → `COMPLY.TRANSFORM`
    - else → `COMPLY.BENIGN`
- **Refusal style (`head_b_a`):** if response text present, use cues; else null.
- **Compliance style (`head_b_b`):** detect `STEP_BY_STEP`/`WITH_WARNINGS` if response present.
- **Harm categories (`head_c_a`):** map the 10 policy categories:
  - Malware/Hacking → `cybercrime`
  - Fraud/Deception/Economic harm → `fraud` / `illicit_behavior`
  - Physical harm → `violence` / `dangerous_activities`
  - Disinformation → `misinformation`
  - Sexual/Adult (incl. CSAM) → `adult_sexual` / `sexual_exploitative`
  - Privacy → `privacy_personal_data`
  - Expert advice → `medical_advice` / `legal_advice` / `financial_advice`
  - Government decision-making → `political_persuasion` (or `other` if neutral)
  - Harassment/Discrimination → `hate_harassment`
- **Benign topics (`head_c_b`):** tag benign counterparts by domain when known.
- **Flags (`head_d`):**
  - `prompt_harmful` = (behavior is harmful)
  - `response_harmful` = (`unsafe`)
  - `response_refusal` = judge refusal yes/no

## Gaps & Backfills

- **Heads not covered:** styles when no response text.
- **Weak-label rules:** over-refusal only when behavior is benign **and** refusal yes.
- **Nulls:** domain tags may be missing.

## Redaction Notes

- Avoid publishing explicit unsafe completions; keep judge/labels.

## Intended Use

- **Split policy:** **eval-only** for jailbreak robustness and over-refusal sanity checks.
- **Known biases / artefacts:** attack/defense config dependent; deterministic decoding.
