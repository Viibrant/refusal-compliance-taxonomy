---
slug: wildguard-2024
status: mapped
train_pool: true
source_url: https://arxiv.org/abs/2406.18495
citation: Han et al. (2024) “WILDGUARD: Open One-stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs.”
license: TBD (confirm repo/paper; use label projections if redistribution restricted)
version: 2024-06 (paper); WGTEST ~5.3k human-labeled
languages: [en]  # may include multilingual adversarial variants; confirm
size: WGTEST ≈5.3k; plus synthetic sets (see paper)
---

# Dataset Card: WildGuard

## Summary

One-stop moderation toolkit/datasets with three orthogonal tasks: prompt harmfulness, response harmfulness, refusal detection. Includes human-labeled eval (WGTEST) and adversarial/benign distributions; explicit over-refusal stress cases.

## Provided Fields

- **Text fields:** prompt: ✔, response: ✔ (for subsets)
- **Labels:** `prompt_harmful` (yes/no), `response_harmful` (yes/no), `response_refusal` (yes/no)
- **Metadata:** harm taxonomy (4 high-level → 13 subcats), adversarial flags, benign/harmful split

## Mapping Plan (→ our taxonomy)

- **Outcome (`head_a`):**
  - Use precedence (METHODOLOGY §5.2).
  - If `response_harmful==true` → `COMPLY.UNSAFE`.
  - Else if `response_refusal==true`:
    - If `prompt_harmful==false` → `REFUSAL.OVER`
    - Else → `REFUSAL.DIRECT` vs `REFUSAL.SOFT` vs `REFUSAL.PARTIAL` via style cues.
  - Else → `COMPLY.BENIGN` or `COMPLY.TRANSFORM/CONDITIONAL/...` via cues.
- **Refusal style (`head_b_a`):** regex cues:
  - DIRECT: `\b(can't|cannot|won't)\b.*\b(help|assist)\b`, `against (policy|guidelines)`
  - EXPLAIN: mentions risk/policy reason
  - EMPATHETIC: `I'm sorry you're feeling`, `you're not alone`
  - CAPABILITY: `no web access`, `can't browse`, `lack permission` → actually routes `head_a`
- **Compliance style (`head_b_b`):** detect `STEP_BY_STEP` (numbered/imperatives), `WITH_WARNINGS` (cautionary phrases), `TRANSFORM_ONLY` (translate/summarise/classify verbs with no new steps).
- **Harm categories (`head_c_a`):** map WildGuard subcats → ours:
  - Privacy.{org,individual} → `privacy_personal_data`
  - Copyright → `intellectual_property`
  - Misinformation.{false,material_harm} → `misinformation`
  - HarmfulLanguage.{stereotypes,violence,toxic,sexual} → `hate_harassment` (+ `violence` when explicit) / `adult_sexual`
  - MaliciousUses.{cyberattacks} → `cybercrime`; {fraud/illegal} → `fraud`/`illicit_behavior`
  - {unsafe/ unethical actions} → `dangerous_activities` / `illicit_behavior`
  - {mental health & over-reliance crises} → `self_harm` (if self-injury) or `psychology_self_development` (benign)
  - Others → `other`
- **Benign topics (`head_c_b`):** derive from prompt domains (tech/education/health/etc.) when labelled; else leave null.
- **Flags (`head_d`):** direct copy:
  - `prompt_harmful` ← WildGuard prompt harm
  - `response_harmful` ← WildGuard response harm
  - `response_refusal` ← WildGuard refusal

## Gaps & Backfills

- **Heads not covered:** none structurally; styles require weak labels.
- **Weak-label rules:** style via cue lexicons; partial refusal if refusal + any task content (steps/hints).
- **Nulls:** `head_c_b` when domain tags absent.

## Redaction Notes

- If any subsets contain explicit methods (weapons/cyber/biothreat), keep labels and redact steps as `[REDACTED_STEP]` for examples.

## Intended Use

- **Split policy:** Treat WGTEST as **eval-only**; use synthetic for training if licence allows.
- **Known biases / artefacts:** Includes adversarial prompts; over-refusal stress (benign but scary-sounding).

## Examples (optional, safe)

- N/A (eval-only examples should avoid methods)
