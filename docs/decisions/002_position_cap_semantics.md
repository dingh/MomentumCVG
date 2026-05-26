# Decision 002: Max concurrent position cap semantics

**Status:** Accepted  
**Date:** 2026-05-25

---

## Context

The v1 spec pins max concurrent positions at **50**, but the implementation currently uses `max_names_per_side`, which can be interpreted as a per-side cap. If set to 50, that could allow up to 50 long straddles plus 50 short structures on the same rebalance date.

That is not the intended v1 risk contract.

---

## Decision

For v1, **50 max concurrent positions means 50 total positions across long and short sides combined**.

The exact number can be revisited later, but until another decision changes it, code and tests should treat the cap as total portfolio positions, not per-side capacity.

---

## Consequences

- `SurfaceRunner` must not use `max_names_per_side=50` as a substitute for the v1 cap.
- Future config naming should distinguish total portfolio cap from any optional side-specific caps.
- Session B / Sprint 002 tests should expose the current mismatch between total cap semantics and `max_names_per_side`.
- Long/short allocation policy remains open: once the total cap is enforced, a future build sprint should decide whether selection is purely rank-based across both sides or uses side budgets within the total cap.

---

## References

- `docs/v1_spec_pins.md`
- `docs/surface_runner_data_flow.md`
- `docs/sprint_memos/001_repo_audit_verification.md`

