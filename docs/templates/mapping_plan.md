# Mapping Plan: <dataset slug>

> Purpose: a concise mapping from source fields/labels to our heads and flags. Keep this to one page.

## Source Fields

- prompt: `<field or N/A>`
- response: `<field or N/A>`
- labels: `<list>`
- metadata: `<list>`

## Direct Mappings (no inference)

| Source field/value   | Target                          | Notes            |
| -------------------- | ------------------------------- | ---------------- |
| `<label or value>`   | `head_d.prompt_harmful = ...`   | map yes/no       |
| `<label or value>`   | `head_d.response_refusal = ...` | source boolean   |
| `<harm_category>`    | `head_c_a += <our_bucket>`      | many→one allowed |
| `<topic/domain tag>` | `head_c_b += <topic>`           | multilabel       |

## Outcome Derivation (precedence)

- If `response_harmful == true` → `head_a = COMPLY.UNSAFE`.
- Else if `response_refusal == true`:
  - safe prompt + policy language → `REFUSAL.OVER`
  - capability/permission language → `REFUSAL.CAPABILITY`
  - any task content → `REFUSAL.PARTIAL`
  - empathy/support → `REFUSAL.SOFT`
  - else → `REFUSAL.DIRECT`
- Else:
  - transform-only ops → `COMPLY.TRANSFORM`
  - conditional/redirect/educational/partial-safe per cues
  - else → `COMPLY.BENIGN`

## Style Cues (regex/phrases)

- **Refusal markers:** e.g., `can't help`, `against policy`, `as an AI`
- **Capability markers:** `no web access`, `can't browse`, `lack permissions`
- **Empathy markers:** `I'm sorry you're feeling`, `you're not alone`
- **Warnings/conditions:** `for educational purposes`, `I can explain ... but won't`
- **Stepwise:** numbered lists, imperative verbs, code blocks
- **Transform-only indicators:** translate/summarise/classify **without** new steps/entities

## Heads Not Covered

- `<list heads you won’t populate from this dataset>`

## Train/Eval Intention

- `<train-pool | eval-only>` and why.

## Licence Constraints

- Redistribution: `<full text | projections only>`
