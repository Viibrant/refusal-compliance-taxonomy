---
slug: <kebab-case-name[-year]>
status: draft | mapped | validated | eval-only
train_pool: true | false
source_url: <link>
citation: <text or BibTeX>
license: <e.g., CC-BY-4.0> # note redistribution constraints
version: <provider version/date>
languages: [en, ...]
size: <approx count or N/A>
---

# Dataset Card: <Human-readable Name>

## Summary

One paragraph: what this dataset is, where it comes from, why it’s useful for our taxonomy.

## Provided Fields

- **Text fields:** prompt: <field name or “none”>, response: <field or “none”>
- **Labels:** <list original label names and types>
- **Metadata:** <any useful columns – topic tags, spans, language codes, etc.>

## Mapping Plan (→ our taxonomy)

- **Outcome (`head_a`):** <how you derive, or “N/A (no responses)”>
- **Refusal style (`head_b_a`):** <cues or “N/A”>
- **Compliance style (`head_b_b`):** <cues or “N/A”>
- **Harm categories (`head_c_a`):** <original → ours mapping summary>
- **Benign topics (`head_c_b`):** <how you set topics>
- **Flags (`head_d`):** `prompt_harmful`: <rule/source>, `response_harmful`: <rule/source>, `response_refusal`: <rule/source>

> See detailed sketch: [`docs/templates/mapping_plan.md`](../templates/mapping_plan.md) if you need more space.

## Gaps & Backfills

- **Heads not covered:** <e.g., head_b_b>
- **Weak-label rules:** <simple heuristics you’ll apply, if any>
- **What remains null:** <be explicit>

## Redaction Notes (if any)

- <What must be masked/removed; how you’ll do it>

## Intended Use

- **Split policy:** <train-pool or eval-only; any hold-outs>
- **Known biases / artefacts:** <brief notes>

## Examples (optional, safe)

- <Up to three tiny, non-sensitive examples showing the mapping idea>
