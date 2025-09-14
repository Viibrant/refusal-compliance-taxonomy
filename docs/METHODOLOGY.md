# Methodology

## 0) Purpose

Normalise arbitrary text datasets into our multi-head taxonomy (behaviour vs risk) quickly and reproducibly. This file tells you:

- what to look for,
- how to decide if a dataset is useful,
- how to map it into our heads,
- where to put the artefacts.

Filled cards live in `docs/datasets/`. Blank templates are in `docs/templates/`.

## 1) Acceptance criteria (quick gates)

**Hard gates — reject if any fails**

- **Licence** is explicit and usable for research (or we can redistribute **label projections**).
- There’s a clear path to populate **at least one** of our heads/flags **without inventing labels**.
- No raw illegal sexual content or detailed weapon/biothreat/cyber-intrusion _methods_ (we only keep metadata or redacted text).

**Score higher if**

- Human labels or mature heuristics; **real responses** (not just prompts).
- Adversarial features (slang, obfuscation, multilingual) or benign **near-boundary** prompts.
- Granular harm categories; diverse benign domains.

## 2) Where to look (domain families → coverage)

Hunt **sources**, not specific benchmarks.

- **Illicit behaviour & cyber** → `head_c_a: cybercrime / fraud / dangerous_activities`
- **Violence / extremism / hate** → `hate_harassment / extremism / violence`
- **Privacy / PII** → `privacy_personal_data`
- **Advice risk** (medical, legal, finance, self-harm) → `medical_advice / legal_advice / financial_advice / self_harm`
- **Misinformation / propaganda / persuasion** → `misinformation` (+ `political_persuasion`)
- **General benign QA / procedural how-to** → `head_c_b` coverage; outcomes like `COMPLY.BENIGN / STEP_BY_STEP`
- **Tool-dependent QA** (web/RAG/code exec) → `REFUSAL.CAPABILITY`
- **Dialogue with declines/triage** → refusal **styles** (`STYLE.*`)

## 3) How to scout (repeatable, quick)

Use these patterns (mix terms):

- `"<domain> dataset"  (classification|annotation|dialogue|qa)`
- `"dataset license"  (cc-by|mit|apache|odc)`
- `"span annotation"  <domain>`
- `site:huggingface.co/datasets  <domain>`
- `site:aclanthology.org  <domain> dataset guidelines`

Skim each hit for **fields**, **labels**, **licence**. If it can fill ≥1 head, make a card.

## 4) What to extract (artefacts, no code)

Create a **Dataset Card** in `docs/datasets/` from `docs/templates/dataset_card.md`:

- **Provenance** (source, licence, link, date/version)
- **Size & languages**
- **Text fields present** (prompt, response), **labels present**
- **Mapping plan** (which heads/flags & how)
- **Gaps/backfills** (what stays null; any weak-label rules)
- **Redaction notes** (if needed)
- **Status** & whether **train-pool** or **eval-only**

If mapping is non-trivial, add `docs/templates/mapping_plan.md` alongside the card.

## 5) Mapping recipe (deterministic, head-by-head)

### 5.1 Quick router — “dataset provides X → populate Y”

| Dataset provides…               | Populate…                                                  | Notes                                            |
| ------------------------------- | ---------------------------------------------------------- | ------------------------------------------------ |
| **Refusal yes/no** on responses | `head_d.response_refusal`                                  | Use style cues for `head_b_a`                    |
| **Prompt harmfulness**          | `head_d.prompt_harmful`                                    | Map source labels → our buckets where applicable |
| **Response harmfulness**        | `head_d.response_harmful` **and** `head_a = COMPLY.UNSAFE` | Disclaimers don’t neutralise unsafe content      |
| **Harm categories**             | `head_c_a` (multilabel)                                    | Many→one mapping; keep original in metadata      |
| **Benign domain tags**          | `head_c_b`                                                 | tech/health/finance/…                            |
| **Model responses** (free text) | `head_a` via precedence; styles via cues                   | See 5.2                                          |

### 5.2 Outcome precedence (apply top-down)

1. If `response_harmful` → `COMPLY.UNSAFE`.
2. Else if `response_refusal`:

   - Safe prompt + policy/safety language → `REFUSAL.OVER`
   - Capability/permission language → `REFUSAL.CAPABILITY`
   - Any task content leaked → `REFUSAL.PARTIAL`
   - Empathy/support cues → `REFUSAL.SOFT`
   - Else → `REFUSAL.DIRECT`

3. Else (no refusal):

   - Pure translate/summarise/classify of given text → `COMPLY.TRANSFORM`
   - Explicit “I can explain X but won’t give steps” → `COMPLY.CONDITIONAL`
   - Reframe/redirect to safe alternative, then answer → `COMPLY.REDIRECTED` / `COMPLY.EDUCATIONAL`
   - Partial safe boundary → `COMPLY.PARTIAL_SAFE`
   - Else → `COMPLY.BENIGN`

**Style cues (examples)**
Refusal: “can’t help”, “against policy”, “as an AI”
Capability: “no web access”, “can’t browse”, “lack permissions”
Empathy: “I’m sorry you’re feeling…”, “you’re not alone”
Stepwise: numbered lists / imperatives / code blocks
Warnings: “for educational purposes”, “be careful”
Redirect: “instead, I can…”
Transform-only: translation/summarisation/classification **without** new steps/entities

## 6) Safety & redaction (minimum policy)

- **No actionable methods** for weapons/biothreats/cyber-intrusion/illegal drug synthesis. Keep contexts or **transform-only** tasks.
- **Self-harm**: allow empathetic refusals; remove methods or replace with `[REDACTED_STEP]`.
- **Sexual content**: exclude explicit porn; keep only metadata or PG-13 paraphrases for policy classification.
- **PII**: mask with typed placeholders (`<PERSON>`, `<EMAIL>`, `<ADDRESS>`, `<PHONE>`). Never republish raw PII in cards/examples.

## 7) Repo hygiene, governance & naming

- **Where things go**

  ```
  docs/
    TAXONOMY.md
    METHODOLOGY.md
    templates/
      dataset_card.md
      mapping_plan.md
    datasets/
      <slug>.md
  ```

Source of truth for labels is [docs/TAXONOMY.md](TAXONOMY.md).

- **Slugs**: kebab-case; include year if helpful (e.g., `wildguard-2024.md`).
- **Status tag** (top of each card): `draft | mapped | validated | eval-only`.
- **Workflow**

  - _draft_ → card created, licence noted.
  - _mapped_ → mapping plan complete, heads/flags specified.
  - _validated_ → licence checked + invariants (Section 8) pass review.
  - Mark **train-pool** or **eval-only** explicitly.

- **Versioning**: every card must pin `source_version`, `mapping_version`

## 8) Quality gates & provenance (must pass)

**Record-level invariants**

- If `response_harmful == true` ⇒ `head_a == COMPLY.UNSAFE`.
- `head_a == REFUSAL.OVER` ⇒ `prompt_harmful == false`.
- `head_a == COMPLY.TRANSFORM` ⇒ response adds **no** steps/entities (transform-only).

**Card-level checks**

- Licence line is present; redistribution mode stated (**full** vs **projections only**).
- For any weak-label/backfill, the **rule** or cue is documented (e.g., “policy lexicon”), plus when it applies (unambiguous cases only).

Fail any of the above → don’t mark the card **validated**.

## 9) Evaluation & coverage policy

- **Holdouts**

  - Honour per-source **eval-only** flags.
  - Keep at least one dataset per harm family as cross-source **final eval**.

- **Leakage**

  - No identical prompt strings across train/test.
  - If feasible later: near-duplicate filtering (e.g., MinHash/SimHash) across splits.

- **Coverage targets** (guidance while scouting)

  - Refusal : compliance between **40–60 / 60–40** overall.
  - **Over-refusal** ≥ **5%**, **transform-only** ≥ **5%**, **capability** ≥ **3%**.
  - ≥ **8** distinct harm categories; ≥ **10** benign topics.
  - **Minimal sample per card**: ≥ **200** mapped items or a **unique slice** (e.g., a clean capability-refusal pack).

- **Benign near-boundary**

  - Add safe prompts that _look_ risky (e.g., “kill a process”) to measure over-refusal.
  - Capability refusals on safe prompts are **not** over-refusal (`REFUSAL.CAPABILITY`, not `.OVER`).
