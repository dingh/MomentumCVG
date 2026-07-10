# C6.1C — Calendar-Paired Weekly Expiry Implementation

**Verdict: PASS**

**Commit baseline (C6.1B follow-up):** `7049037174ac30b735e1328a7d465d8dc14a64bb`

C6.1C changes weekly producer expiry behavior from permissive nearest-DTE selection to strict calendar-paired target-expiry selection.

C6.1C does not change strategy logic, S5, S8, ORCH, or backtest logic.

C6.1C does not run full backfill.

C6.1C does not touch raw ORATS roots or the old legacy adjusted mirror.

No producer monthly expiry behavior was changed.

---

## Scope

### Files changed

| File | Change |
|------|--------|
| `src/features/option_surface_analyzer.py` | Strict weekly expiry resolver; weekly `process_single_entry` path; optional soft failure tag for body legs |
| `scripts/precompute_option_surface.py` | Build weekly schedule with successor tail; pass schedule into workers |
| `tests/unit/test_option_surface_weekly_expiry.py` | **New** — C6.1C unit coverage |
| `docs/tmp/c6_1c_calendar_paired_weekly_expiry_report.md` | This report |

### Producer behavior changed

- Weekly mode (`frequency="weekly"`) now resolves expiry via `target_weekly_expiry_from_schedule(entry_date, weekly_schedule)`.
- Exact target must be listed on the entry-date chain.
- Missing exact target → metadata failure row; **no nearest-DTE fallback**.
- Weekly body-leg soft failure may set `failure_reason="target_weekly_body_not_quotable"` while preserving `surface_valid` semantics.

### Producer behavior not changed

- `_find_best_expiry` implementation (still available; used by monthly mode).
- Monthly expiry selection path.
- `surface_valid == has_body_call AND has_body_put AND n_surface_quotes > 0`
- Dry-run / overwrite / output-root safety from C6.1A.
- Strategy / S5 / S8 / ORCH / backtest logic.
- Raw ORATS roots and legacy adjusted mirror.

---

## Policy implemented

Strict calendar-paired weekly expiry:

```text
entry_date = last available ORATS trading day of week i
target_weekly_expiry = last available ORATS trading day of week i+1
```

- No fallback to nearest-DTE expiry.
- Missing exact target expiry means no weekly trade / not weekly-ready.
- Non-weekly names become explicit no-trade cases (`target_weekly_expiry_not_listed`), not silent substitutions.

---

## Failure behavior

| Tag | When |
|-----|------|
| `no_target_weekly_expiry` | Entry date has no `schedule[i+1]` (or schedule missing) |
| `no_expiries_on_entry_chain` | Ticker has no listed expiries on entry date |
| `target_weekly_expiry_not_listed` | Exact target missing; nearby expiries ignored |
| `target_weekly_body_not_quotable` | Target listed but body call/put not quotable (weekly soft tag) |

Hard failures emit schema-compatible metadata rows with empty quote lists.

---

## Tests

```powershell
C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/unit/test_option_surface_weekly_expiry.py tests/unit/test_precompute_option_surface_cli.py tests/unit/test_diagnose_weekly_expiry_policy.py -q
```

**Result:** `41 passed`

```powershell
C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/unit/test_straddle_analyzer.py -q
```

**Result:** `12 passed` (legacy permissive `_find_best_expiry` helper unchanged).

Coverage includes:

1. Exact target expiry selected when listed
2. No fallback when target missing (nearby expiry present)
3. Empty expiry list → `no_expiries_on_entry_chain`
4. Last schedule entry → `no_target_weekly_expiry`
5. Target listed / body not quotable → `surface_valid=False` + `target_weekly_body_not_quotable`
6. Monthly path still uses `_find_best_expiry`
7. Dry-run safety unchanged (existing CLI tests)

---

## Smoke evidence

Bounded dry-run only (no parquet writes, no full backfill, no strategy/backtest):

```powershell
C:/MomentumCVG_env/venv/Scripts/python.exe scripts/precompute_option_surface.py `
  --frequency weekly --start-year 2024 --end-year 2024 `
  --start-date 2024-01-01 --end-date 2024-01-31 `
  --tickers AAPL MSFT `
  --output-root C:/MomentumCVG_env/cache/c6_1c_smoke `
  --dry-run
```

Confirms schedule generation + dry-run plan still work with the C6.1C schedule-tail wiring.

No full historical surface regeneration was run.

---

## Remaining limitations

- Broader soft-failure vocabulary / T6 cleanup remains **C6.1D**.
- Artifact join/grain audit remains **C6.2 / C6.3**.
- Full-universe weekly opportunity capacity is not claimed; Sample B from C6.1B remains informational.
- Canonical production surface artifacts were not regenerated.
- PIT universe harness remains **C7**.

C6.1C is ready for review as a **weekly expiry semantics** change only.
