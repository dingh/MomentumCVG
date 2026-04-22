# Option surface implementation for MomentumCVG

This package adds a surface-first workflow so you can vary:

- iron fly wing delta rules
- iron condor short/long delta rules
- fill assumptions
- friction assumptions
- leg-level tradability thresholds

without regenerating precompute files.

## Files included

- `src/features/option_surface_analyzer.py`
- `scripts/precompute_option_surface.py`
- `src/backtest/option_surface.py`

## What changes in your workflow

### Old primary artifact
`ironfly_condor_history_*.parquet`

This is still useful for quick exploratory analysis, but it is already strategy-shaped.

### New primary artifact
Two new parquet files:

- `option_surface_meta_<frequency>_<start>_<end>.parquet`
- `option_surface_quotes_<frequency>_<start>_<end>.parquet`

These hold:

- one selected expiry per `(ticker, entry_date)`
- entry spot / exit spot / body strike
- body call / body put / OTM call ladder / OTM put ladder

Then you assemble iron flies and iron condors later in the backtest.

## Example usage

### 1. Precompute the surface

```bash
python scripts/precompute_option_surface.py --frequency monthly --start-year 2020 --end-year 2024
```

### 2. Load the surface

```python
from src.backtest.option_surface import OptionSurfaceDB, FillAssumption, build_ironfly_from_surface, build_ironcondor_from_surface

surface = OptionSurfaceDB.load(
    "C:/MomentumCVG_env/cache/option_surface_meta_monthly_2020_2024.parquet",
    "C:/MomentumCVG_env/cache/option_surface_quotes_monthly_2020_2024.parquet",
)
```

### 3. Build an iron fly

```python
fly = build_ironfly_from_surface(
    surface_db=surface,
    ticker="AAPL",
    entry_date=date(2023, 1, 6),
    wing_target_delta=0.15,
    fill=FillAssumption.mid(),
    max_leg_spread_pct=0.50,
)
```

### 4. Build an iron condor

```python
condor = build_ironcondor_from_surface(
    surface_db=surface,
    ticker="AAPL",
    entry_date=date(2023, 1, 6),
    short_delta_target=0.30,
    long_delta_target=0.10,
    fill=FillAssumption.cross(),
    max_leg_spread_pct=0.50,
)
```

### 5. Settle at expiry

```python
position = fly.settle(exit_spot=Decimal("184.25"))
print(position.pnl, position.pnl_pct)
```

## Why this is the right abstraction for your goal

The precompute layer now answers:

- what expiry was traded?
- what body strike was ATM?
- what call/put quotes existed in the relevant OTM delta band?
- what was the exit spot?

The backtest layer answers:

- which legs should I choose?
- how should I fill them?
- what friction model should I assume?
- what spread threshold should I enforce?
- what ROC definition should I use?

That is exactly the separation you want for flexible fly/condor research.
