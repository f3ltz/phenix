# Academic Research Skills for Antigravity

A comprehensive suite of research skills for rigorous academic exploration, paper writing, peer review, and full pipeline orchestration, retrofitted natively for Google Antigravity.

## Skills Overview

| Skill | Purpose | Key Modes & Triggers |
|---|---|---|
| `deep-research` | Universal 13-agent research team | `full`, `quick`, `socratic`, `review`, `lit-review`, `three-way-scan`, `fact-check`, `systematic-review` |
| `academic-paper` | 12-agent academic paper writing pipeline | `full`, `plan`, `outline`, `revision`, `revision-coach`, `abstract`, `lit-review`, `format-convert`, `citation-check`, `disclosure`, `rebuttal-audit` |
| `academic-paper-reviewer` | Multi-perspective paper review (5-seat panel) | `full`, `re-review`, `quick`, `methodology-focus`, `guided`, `calibration` |
| `academic-pipeline` | Full end-to-end research orchestrator | 10-stage orchestrated workflow coordinating research -> write -> review -> revise -> passport |
| `sr-screener` | Protocol-driven systematic review screening | `protocol`, `quick`, `pilot`, `ta-screen`, `ft-screen`, `adjudicate`, `audit`, `report` |

## Antigravity Tool Equivalencies

When running ARS procedures in Antigravity:
- File Reading: Use `read_file` instead of `Read`.
- File Writing & Editing: Use `write_file` and `edit_file` instead of `Write` / `Edit` / `MultiEdit`.
- Shell Execution: Use `run_command` instead of `Bash`.
- Web Search & Retrieval: Use `search_web` instead of `WebSearch` / `WebFetch`.
- Subagent Dispatching: Use `invoke_subagent` and `define_subagent` for multi-seat review panels or isolated tasks.
- Interactive Clarification: Use `ask_question` or formatted markdown options for user decisions.

## Routing Discipline

### Step 0: Escape Hatch
If the user's message begins with `[direct-mode]`, strip the token and route directly to the explicitly requested skill or agent without intermediate clarification questions.

### Step 1: Explicit Intent
- User specifies an `/ars-*` command or clear trigger (e.g. "review this paper", "plan my paper", "screen these records"):
  -> Route directly to the matching skill/mode without an orchestrator detour.
  -> If expected inputs are missing, let the activated mode prompt for them directly.

### Step 2: Cross-Phase Materials Detected
- User provides materials spanning >= 2 phases (e.g. outline + collected references + half draft) without a clear verb:
  -> Prompt the user to clarify intent before dispatching any single-phase agent.

### Step 3: Ambiguous Intent
- Clarify candidate workflows before generating content.

## Slash Commands / Common Triggers

- `/ars-plan`: Chapter-by-chapter guided planning via Socratic dialogue (`academic-paper plan`).
- `/ars-lit-review <topic>`: Systematic literature search and synthesized review (`deep-research lit-review`).
- `/ars-reviewer`: Multi-perspective peer review simulation with a 5-seat panel (`academic-paper-reviewer full`).
- `/ars-full`: End-to-end drafting from intake to final manuscript (`academic-paper full`).
- `/ars-outline`: Transform ideas or notes into a structured paper outline (`academic-paper outline`).
- `/ars-revision`: Revise manuscript in response to reviewer comments (`academic-paper revision`).
- `/ars-revision-coach`: Generate rebuttal strategy and response letter skeleton (`academic-paper revision-coach`).
- `/ars-rebuttal-audit`: Audit existing response draft against reviewer comments (`academic-paper rebuttal-audit`).
- `/ars-abstract`: Generate structured or bilingual abstracts (`academic-paper abstract`).
- `/ars-citation-check`: Deterministic reference verification against Semantic Scholar, OpenAlex, Crossref, and arXiv (`academic-paper citation-check`).
- `/ars-disclosure`: Generate venue-compliant AI-usage disclosures (`academic-paper disclosure`).
- `/ars-format-convert`: Convert paper format to target venue template (`academic-paper format-convert`).

## Core Academic Integrity Rules

1. **Human-in-the-Loop**: AI is a copilot, never the sole pilot. All critical scientific claims, methods, and interpretations remain the author's responsibility.
2. **Every Claim Grounded**: Every empirical claim must be anchored to a cited reference with a locator anchor (`<!--anchor:<kind>:<value>-->`).
3. **Deterministic Verification**: Verify citations across the 4 bibliographic indexes (Semantic Scholar, OpenAlex, Crossref, arXiv).
4. **Evidence Hierarchy**: Meta-analyses > Systematic Reviews > RCTs > Cohort Studies > Case Reports > Expert Opinion.
5. **Full Transparency**: Acknowledge AI tools and document methodology at all stages.
