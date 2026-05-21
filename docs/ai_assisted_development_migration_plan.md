# AI-Assisted Development Migration Plan

## Purpose

This document defines the restart plan for the project after a pause. The goal is to migrate into a safer, more productive AI-assisted development workflow before resuming intense strategy backtesting.

The main priority is not to immediately add new strategy ideas. The priority is to establish a workflow that helps answer three questions:

1. What agents and workflow do I need?
2. Is the current repo structure and design good enough?
3. Can I verify the correctness of the existing code?

Only after these questions are answered should the project move into more intense backtesting.

---

## Recommended Tooling During Migration

Start simple.

```text
Cursor Pro = control room
Cursor Agent = repo mapper / first worker
ChatGPT or Codex = reviewer / planning partner
Markdown files = operating manual
```

Do not add Claude Max immediately unless Cursor feels insufficient during the first few sprints.

Later, if weekend sprint days become longer and more serious:

```text
Claude Code = stronger terminal worker
Codex = reviewer
Cursor = still the control room
```

---

## Core Workflow Principle

Every meaningful task should follow this loop:

```text
Inspect → Plan → Test → Implement → Run → Review → Memo
```

The agent should not jump directly from vague goal to code.

Default instruction:

```text
Do not edit code yet. Inspect first and propose a plan.
```

---

## Phase Overview

Estimated total workload before intense backtesting:

```text
Minimum version: ~27 hours
Careful version: ~38 hours
```

Phases:

1. Migration and workflow setup
2. Repo reacquaintance and architecture audit
3. Correctness verification audit
4. First controlled verification sprint
5. Backtesting readiness gate

---

# Phase 1 — Migration and Workflow Setup

Estimated workload: **6–8 hours**

## Block 1 — Install and Open the Repo in Cursor

Estimated time: **1 hour**

### Goal

Get the repo running inside Cursor.

### Tasks

```text
Install Cursor
Open the MomentumCVG repo
Confirm Git works
Confirm Python environment
Run existing test command
Run existing backtest or smoke command if available
```

### Output

```text
Cursor can open repo
Terminal works
Tests can run
Current baseline status is known
```

Do not use agents heavily yet. First make sure the environment works.

---

## Block 2 — Learn Cursor Basics

Estimated time: **1 hour**

### Goal

Learn the control-room workflow.

### Practice these features

```text
Chat with codebase
Ask agent to inspect files
Ask agent to explain a module
Review proposed edits
Use terminal from Cursor
Review Git diff
Reject or undo AI changes
```

### Practice prompt

```text
Inspect this repository and explain the top-level structure. Do not edit files.
```

### Output

```text
You know how to ask Cursor to inspect without editing.
You know how to review diffs before accepting changes.
```

---

## Block 3 — Create Project Instruction Files

Estimated time: **1–2 hours**

### Goal

Give AI a clear operating manual.

### Create these files

```text
AGENTS.md
docs/development_workflow.md
docs/reviewer_prompt.md
docs/backtest_safety_rules.md
```

### Starting `AGENTS.md`

```md
# AGENTS.md

This repo is for options strategy research and backtesting.

## Workflow rules

1. Inspect before editing.
2. Plan before coding.
3. Add or update tests for behavior changes.
4. Run focused tests after changes.
5. Review git diff before calling work complete.
6. Record what was verified and what was not.

## Safety rules

- Do not change strategy logic without tests.
- Do not trust mid-price-only backtests as final evidence.
- Always verify option leg type, strike, expiry, quantity sign, premium sign, payoff, and max loss.
- Do not mix refactors with research changes unless explicitly requested.
- Do not touch live execution code unless the task explicitly says so.

## Definition of done

A coding task is done only when:

- The requested behavior is implemented.
- Relevant tests pass.
- The diff has been reviewed.
- Remaining risks are documented.
```

### Output

```text
AGENTS.md exists
Basic workflow docs exist
AI has persistent project rules
```

---

## Block 4 — Create Cursor Project Rules

Estimated time: **1 hour**

### Goal

Make Cursor follow your repo rules.

### Rules to encode

```text
Always inspect before editing.
Always produce a plan first.
For financial logic, add tests before changing code.
Do not make large refactors unless requested.
After edits, summarize files changed, tests run, and remaining risks.
```

### Output

```text
Cursor has project-specific rules
Agent behavior becomes more consistent
```

---

## Block 5 — First Harmless AI-Assisted Change

Estimated time: **1–2 hours**

### Goal

Practice the full loop on a low-risk task.

### Good harmless tasks

```text
Add a docstring
Improve README wording
Add a small comment
Add one simple test that does not affect logic
```

### Workflow

```text
Ask agent to inspect
Ask for plan
Approve plan
Ask for edit
Run test
Review diff
Commit or discard
```

### Output

```text
One safe AI-assisted change completed end-to-end
```

---

# Phase 2 — Repo Reacquaintance and Architecture Audit

Estimated workload: **6–8 hours**

## Block 6 — Ask Cursor to Map the Repo

Estimated time: **1 hour**

### Goal

Understand the project again.

### Prompt

```text
Act as a senior Python engineer helping me return to this repo after a long pause.

Do not edit code.

Map the repository:
1. Main folders and files.
2. Core modules.
3. Data flow.
4. Where strategy logic lives.
5. Where backtesting logic lives.
6. Where tests live.
7. What I should read first.
```

### Output

```text
Initial repo map in notes
```

---

## Block 7 — Create `docs/repo_map.md`

Estimated time: **1 hour**

### Goal

Turn the repo explanation into a durable document.

### Prompt

```text
Create docs/repo_map.md based on your inspection. Keep it concise and practical.
```

### The document should include

```text
Top-level layout
Main abstractions
Data flow
Test layout
Important commands
```

### Output

```text
docs/repo_map.md
```

---

## Block 8 — Design Audit, Part 1

Estimated time: **1 hour**

### Goal

Assess whether the repo structure is good.

### Prompt

```text
Act as a senior quant engineering reviewer.

Do not edit code.

Review the repo design. Focus on:
1. Separation of concerns.
2. Data loading boundaries.
3. Signal generation boundaries.
4. Strategy construction boundaries.
5. Backtest execution boundaries.
6. P&L and reporting boundaries.
7. Areas that are too coupled.
8. Areas that are hard to test.
```

### Output

```text
Design strengths and weaknesses
```

---

## Block 9 — Design Audit, Part 2

Estimated time: **1 hour**

### Goal

Rank design risks.

### Prompt

```text
Based on the design audit, rank the top design risks by:
1. Probability of causing bugs.
2. Impact on backtest trustworthiness.
3. Difficulty to fix.

Return:
- High risk
- Medium risk
- Low risk
- What not to refactor yet
```

### Output

```text
Prioritized design risk list
```

---

## Block 10 — Create `docs/repo_audit.md`

Estimated time: **1–2 hours**

### Goal

Produce the first serious migration artifact.

### Document structure

```md
# Repo Audit

## Current architecture

## Data flow

## Strengths

## Design risks

## Testing gaps

## Refactor priorities

## What not to change yet

## Recommendation
```

### Output

```text
docs/repo_audit.md
```

---

## Block 11 — Reviewer Pass on Repo Audit

Estimated time: **1 hour**

Use ChatGPT or Codex as the reviewer.

### Prompt

```text
Review this repo audit as a skeptical quant engineering lead.

What is missing?
What conclusions are too strong?
What risks are understated?
What should I verify before trusting this repo for serious backtesting?
```

### Output

```text
Improved repo audit
```

---

# Phase 3 — Correctness Verification Audit

Estimated workload: **7–10 hours**

## Block 12 — Inventory Existing Tests

Estimated time: **1 hour**

### Goal

Understand what is already verified.

### Prompt

```text
Inspect the test suite.

Do not edit code.

Create an inventory:
1. What modules have tests?
2. What behavior is tested?
3. What behavior is not tested?
4. Which tests are unit tests?
5. Which tests are integration tests?
6. Which tests are fragile or shallow?
```

### Output

```text
Test inventory
```

---

## Block 13 — Identify Financial Correctness Risks

Estimated time: **1 hour**

### Goal

Find dangerous areas.

### Prompt

```text
Act as a verification engineer for an options backtesting repo.

Identify the highest-risk correctness areas:
1. Option leg sign.
2. Premium sign.
3. Expiry selection.
4. Strike selection.
5. Payoff calculation.
6. Fill price assumptions.
7. Transaction cost assumptions.
8. Capital denominator.
9. Date alignment.
10. Lookahead bias.
11. Missing data handling.
```

### Output

```text
Correctness risk list
```

---

## Block 14 — Map Code Paths for Core Logic

Estimated time: **1–2 hours**

### Goal

Know where correctness matters most.

### Prompt

```text
Trace the path from:
raw option data
→ signal calculation
→ strategy construction
→ position sizing
→ trade simulation
→ P&L
→ final metrics

Do not edit code.
```

### Output

```text
Core code path map
```

---

## Block 15 — Create `docs/correctness_audit.md`

Estimated time: **1–2 hours**

### Goal

Produce the second major artifact.

### Structure

```md
# Correctness Audit

## What is currently verified

## What is partially verified

## What is not verified

## Highest-risk correctness gaps

## Minimal tests needed before serious backtesting

## Areas safe to use

## Areas not yet safe to trust

## Recommended next verification sprint
```

### Output

```text
docs/correctness_audit.md
```

---

## Block 16 — Define Verification Test Categories

Estimated time: **1 hour**

### Goal

Turn risks into concrete tests.

### Create

```text
docs/verification_test_plan.md
```

### Sections

```text
Payoff truth-table tests
Premium sign tests
Leg sign tests
Expiry/date tests
Signal lookahead tests
Fill price tests
Capital denominator tests
Backtest smoke tests
Regression tests
```

### Output

```text
docs/verification_test_plan.md
```

---

## Block 17 — Reviewer Pass on Correctness Audit

Estimated time: **1 hour**

Use ChatGPT or Codex.

### Prompt

```text
Review this correctness audit.

Assume I may eventually trade real money based on this repo.

What tests are missing?
What risks are most dangerous?
What should block serious backtesting?
```

### Output

```text
Improved correctness audit and test plan
```

---

# Phase 4 — First Controlled Verification Sprint

Estimated workload: **5–7 hours**

This is where the first real code change should happen.

## Block 18 — Choose One Test Target

Estimated time: **30 minutes–1 hour**

Pick exactly one.

Recommended first target:

```text
Payoff truth-table test for one option structure
```

Alternative targets:

```text
Premium sign test
Expiry selection test
Date alignment test
Fill price test
```

Do not pick a large refactor.

### Output

```text
One chosen verification task
```

---

## Block 19 — Ask Agent to Plan the Test

Estimated time: **1 hour**

### Prompt

```text
We are adding one correctness test.

Do not edit code yet.

Task:
[describe the selected test]

Please inspect relevant files and produce:
1. Test plan.
2. Required fixtures.
3. Expected values.
4. Files to modify.
5. Why this test reduces real-money risk.
```

### Output

```text
Test implementation plan
```

---

## Block 20 — Add the Test

Estimated time: **1–2 hours**

### Prompt

```text
Implement only the test described in the approved plan.
Do not refactor production code.
Run the focused test afterward.
```

### Output

```text
One new or improved test
Focused test result
```

---

## Block 21 — If Test Fails, Decide Carefully

Estimated time: **1 hour if needed**

### Prompt

```text
The test failed.

Do not immediately change production code.

Explain:
1. Is the test wrong?
2. Is the implementation wrong?
3. What financial assumption is being violated?
4. What is the minimal fix?
```

### Output

```text
Clear decision: fix test or fix code
```

---

## Block 22 — Review the Diff

Estimated time: **1 hour**

Use ChatGPT or Codex.

### Prompt

```text
Review this diff as a skeptical quant engineer.

Focus on:
1. Does the test actually verify financial behavior?
2. Is it too implementation-specific?
3. Does it catch a realistic dangerous bug?
4. What edge case is still missing?
```

### Output

```text
Reviewer feedback
```

---

## Block 23 — Write Sprint Memo

Estimated time: **1 hour**

### Create

```text
docs/sprint_memos/001_migration_verification.md
```

### Template

```md
# Sprint Memo 001

## Goal

## What changed

## Tests added

## Tests run

## What is now verified

## What is still not verified

## Problems found

## Next recommended task
```

### Output

```text
First controlled verification sprint complete
```

---

# Phase 5 — Backtesting Readiness Gate

Estimated workload: **3–5 hours**

## Block 24 — Create Readiness Checklist

Estimated time: **1 hour**

### Create

```text
docs/backtesting_readiness_checklist.md
```

### Checklist

```text
Repo map exists
Repo audit exists
Correctness audit exists
Verification test plan exists
At least one verification sprint completed
Tests can run from one command
High-risk unverified areas are documented
Backtest config assumptions are documented
Fill assumptions are documented
Capital denominator assumptions are documented
```

### Output

```text
Backtesting readiness checklist
```

---

## Block 25 — Decide Whether to Add Claude Code

Estimated time: **1 hour**

Evaluate:

```text
Did Cursor feel enough?
Did I hit limits?
Did I need a stronger terminal worker?
Did the agent struggle with repo-wide reasoning?
Am I ready for longer weekend sprints?
```

Decision rule:

```text
If no pain: stay with Cursor + Codex.
If pain: add Claude Max 5x + Claude Code.
Do not jump to Max 20x yet.
```

### Output

```text
Tooling decision
```

---

## Block 26 — Decide Next Phase

Estimated time: **1–2 hours**

You are ready for more intense backtesting only if:

```text
You can run tests.
You understand the repo structure.
You know the highest-risk correctness gaps.
You have started adding verification tests.
You have a repeatable AI workflow.
```

### Output

```text
Go / no-go decision for intense backtesting
```

---

# Workload Estimate

## Minimum Version

```text
Phase 1: 6 hours
Phase 2: 6 hours
Phase 3: 7 hours
Phase 4: 5 hours
Phase 5: 3 hours

Total: ~27 hours
```

## More Careful Version

```text
Phase 1: 8 hours
Phase 2: 8 hours
Phase 3: 10 hours
Phase 4: 7 hours
Phase 5: 5 hours

Total: ~38 hours
```

This is roughly a one-month restart plan if working mostly on weekends.

---

# Suggested Calendar

## Weekend 1 — Migration

```text
Block 1: Install/open repo
Block 2: Learn Cursor basics
Block 3: Create AGENTS.md
Block 4: Create Cursor rules
Block 5: Harmless AI-assisted change
```

Expected time:

```text
5–6 hours
```

---

## Weekend 2 — Repo Audit

```text
Block 6: Repo map
Block 7: docs/repo_map.md
Block 8: Design audit
Block 9: Risk ranking
Block 10: docs/repo_audit.md
Block 11: Reviewer pass
```

Expected time:

```text
6–8 hours
```

---

## Weekend 3 — Correctness Audit

```text
Block 12: Test inventory
Block 13: Financial correctness risks
Block 14: Core logic path
Block 15: docs/correctness_audit.md
Block 16: verification_test_plan.md
Block 17: Reviewer pass
```

Expected time:

```text
7–10 hours
```

---

## Weekend 4 — First Verification Sprint

```text
Block 18: Choose one test
Block 19: Plan test
Block 20: Add test
Block 21: Handle failure if needed
Block 22: Review diff
Block 23: Sprint memo
Block 24: Readiness checklist
Block 25: Tooling decision
Block 26: Backtesting go/no-go
```

Expected time:

```text
8–10 hours
```

---

# Study Blocks

These can be mixed into the weekly plan.

## Study Block A — Cursor Workflow

Estimated time: **1 hour**

Learn:

```text
Agent mode
Chat with codebase
Project rules
Diff review
Terminal usage
How to stop/reject changes
```

Goal:

```text
You can control the agent without losing control of the repo.
```

---

## Study Block B — Agentic Development Loop

Estimated time: **1 hour**

Study and internalize:

```text
Inspect → Plan → Test → Implement → Run → Review → Memo
```

Practice prompt:

```text
Do not edit code. Inspect first and propose a plan.
```

Goal:

```text
You stop AI from jumping directly into uncontrolled edits.
```

---

## Study Block C — Repo-Level Instruction Files

Estimated time: **1 hour**

Learn the purpose of:

```text
AGENTS.md
Cursor rules
CLAUDE.md if using Claude Code
reviewer_prompt.md
development_workflow.md
```

Goal:

```text
You understand how to make the workflow persistent.
```

---

## Study Block D — Verification Mindset

Estimated time: **2 hours**

Study:

```text
unit tests
fixture tests
golden-result tests
integration smoke tests
payoff truth-table tests
lookahead bias tests
date alignment tests
```

Goal:

```text
You know how to convert anxiety into tests.
```

---

## Study Block E — Backtesting Safety

Estimated time: **2 hours**

Study:

```text
transaction costs
fill assumptions
capital denominator
margin assumptions
survivorship bias
lookahead bias
data quality
experiment reproducibility
```

Goal:

```text
You know what must be true before trusting backtest output.
```

---

# Weekly Rhythm After Migration

Once migration is complete, use this repeating sprint pattern.

## Friday Night — 1 Hour

```text
Pick one task.
Write success criteria.
Ask AI to inspect and plan.
Do not code.
```

## Saturday Morning — 2 Hours

```text
Write or update tests.
Review expected behavior.
Run focused tests.
```

## Saturday Afternoon — 2–3 Hours

```text
Implement smallest change.
Run tests.
Fix failures.
Review diff.
```

## Sunday Morning — 2 Hours

```text
Run smoke test or small backtest.
Record results.
```

## Sunday Afternoon — 1 Hour

```text
Write sprint memo.
Decide next task.
Stop cleanly.
```

Total per sprint:

```text
8–10 hours
```

---

# Final Migration Milestone

The migration phase is complete when the repo contains:

```text
AGENTS.md
Cursor rules
docs/development_workflow.md
docs/repo_map.md
docs/repo_audit.md
docs/correctness_audit.md
docs/verification_test_plan.md
docs/backtesting_readiness_checklist.md
at least one successful verification test improvement
a repeatable sprint memo process
```

At that point, the next phase becomes:

```text
run controlled backtest matrix
compare structures
stress fill assumptions
evaluate capital usage
write decision memos
```

---

# Final Principle

The plan is deliberately front-loaded on process.

Once the workflow is stable, the project can become much more aggressive without every coding session increasing hidden risk.

The goal is not simply to code faster. The goal is to build a system where AI helps inspect, plan, test, implement, review, and document progress while the human remains responsible for research judgment and trading decisions.
