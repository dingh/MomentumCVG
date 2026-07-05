Mode: Sprint004 execution guardrail.

Before starting implementation, follow this execution protocol for every commit C1–C10.

The Sprint004 sprint doc is the roadmap, not a fully justified implementation spec. Do not blindly implement every example field or JSON shape without deliberation. Each commit must begin with a small design plan and only then move to implementation after HD approval.

Global rules:

* No strategy logic changes.
* No S5/S8/ORCH changes.
* No feature branch / A4 / mom-CVG work.
* No real-data S1→S8 smoke.
* No Sharpe / parameter search / paper-live work.
* Keep docs minimal.
* Avoid documentation sprawl.
* Temporary notes must be deleted or archived before closing the commit.
* Prefer code + tests + concise updates to permanent design documents.
* If a design decision seems under-justified in the sprint doc, call it out before implementing.

Commit execution pattern:

Step A — Design plan first
For each commit, first produce a concise design plan. Do not edit files yet.

The design plan must include:

1. Commit target, e.g. C1 manifest types + snapshot_id/build_id.
2. Files expected to change.
3. Existing code/docs inspected.
4. Proposed design.
5. Key decisions and why they are justified.
6. Alternatives considered and rejected.
7. Fields or API choices that are uncertain.
8. Tests to add or update.
9. Docs to touch, if any.
10. Risks / edge cases.
11. What is explicitly out of scope for this commit.

Wait for HD approval before implementation.

Step B — Implementation after approval
After HD approves the design plan:

1. Implement only the approved commit scope.
2. Keep the diff narrow.
3. Add or update tests first when practical.
4. Run targeted tests.
5. Run broader tests if the change touches shared infrastructure.
6. Update only necessary docs.
7. Do not create permanent docs unless required by the sprint plan.
8. Remove temporary planning notes before final commit, or move truly useful historical material to `docs/archive/`.
9. Report:

   * files changed
   * tests run
   * decisions made
   * deviations from approved plan
   * unresolved questions
   * next recommended commit

Documentation hygiene:

* `docs/agenda/current_sprint.md` is the active sprint tracker.
* `docs/v1_weekly_runbook.md` is the operator runbook, created/updated when implementation makes the runbook concrete.
* `docs/sprint_memos/004_*.md` should be created only at closeout.
* Avoid creating separate design docs for every commit.
* If a temporary design note is useful during work, use `docs/tmp/` only if needed, and delete it or archive it before closing the sprint.
* Any stale or redundant information should be removed, not duplicated.
* If a doc becomes obsolete, move it to `docs/archive/` and update `docs/README.md`.

Special guardrail for C1 manifest schema:
The manifest schema in the Sprint004 doc is illustrative, not final. Before implementing C1:

* justify every required top-level field
* separate required vs optional fields
* define deterministic `snapshot_id` hash inputs precisely
* define what is excluded from the hash
* define whether audit-only commands can produce build records without canonical snapshot manifests
* decide whether reports are JSON, Markdown, or both
* avoid overfitting to fields that are not needed yet
* keep schema extensible but minimal

Special guardrail for C2/C3 CLI:
Before implementing CLI subcommands:

* define the command contract
* define exit codes
* define dry-run behavior
* define how paths are resolved
* define how missing artifacts become PASS/WARN/FAIL
* avoid wiring all scripts before plan/validate/report framework is stable

C2 plan output uses **temporary** operator copy (`Provisional`, `deferred to C3–C8`,
bracket notes on steps, C8 WARN stubs). This is intentional scaffolding — not
permanent UX. **Sprint 004 closeout (C9, blocker #13):** remove that copy from
`scripts/refresh_weekly_inputs.py` and drop the matching assertions from
`tests/unit/test_refresh_weekly_inputs_cli.py` in the same commit. Keep stable
checks (as-of fields, step names, artifact keys).

Special guardrail for data-quality commits:
For rolling liquidity, split audit, surface audit, and PIT universe:

* define PASS/WARN/FAIL before coding
* define what blocks Sprint005
* define sample vs full-cache behavior
* do not make local cache availability part of default unit tests
* real-cache checks belong in CLI audit reports, not ordinary pytest unless explicitly skipped/marked

Commit sequence:
C1 — Manifest types + snapshot_id / build_id hashing
C2 — CLI skeleton: plan, --as-of resolution, exit codes
C4 — rolling 3-month liquidity panel
C5 — split golden tests + adjusted-liquid backfill + audit_adjusted_liquid ✓
C6 — surface tests T1–T6 + surface-audit
C7 — PIT universe harness (tests + audit module)
C8 — refresh --dry-run + bounded refresh subprocess wiring
C3 — validate + default report paths + umbrella inventory (after C4–C8; post-artifact trust check)
C9 — runbook + v1_universe_protocol + data-contract drift
C10 — closeout docs only after C1–C9 are accepted

Do not merge adjacent commits unless HD explicitly approves.
Do not begin implementation of a later commit before the earlier commit is accepted, except C4/C5 may be parallel after C1–C2.
C5/C6/C7 own their audit markdown reports; C3 consolidates and shares default report-path conventions afterward.
