# Backtest Configuration Files

This directory contains JSON configuration files for backtests. Each config file completely defines a backtest run, including:

- Strategy parameters
- Execution settings (dates, capital, DTE)
- Universe filter (S&P 500, All, etc.)
- Data provider settings
- Optimizer configuration
- Output options

## Quick Start

Run a backtest from config:

```bash
python scripts/run_backtest.py configs/baseline_sp500.json
```

## Available Configs

### baseline_sp500.json
- **Description:** Baseline weekly momentum strategy on S&P 500
- **Universe:** S&P 500 only
- **Strategy:** 60-week lookback, 8-week skip
- **DTE:** 7 days (weekly options)
- **Period:** 2024-10-01 to 2024-12-31

### all_tickers.json
- **Description:** Same strategy but on all tickers (no universe filter)
- **Universe:** All tickers with sufficient data
- **Strategy:** Same as baseline (60/8 momentum)

### momentum_30_4.json
- **Description:** Faster momentum (30-week lookback, 4-week skip)
- **Universe:** S&P 500
- **Strategy:** 30/4 momentum (minimal config, uses defaults for everything else)

## Config Structure

### Required Sections

```json
{
  "config_name": "unique_identifier",
  "execution": {
    "start_date": "2024-01-01",
    "end_date": "2024-12-31"
  },
  "features": {
    "path": "path/to/features.parquet"
  }
}
```

### Optional Sections (use defaults if not specified)

- `universe`: Default is "All"
- `strategy`: Default is MomentumCVG with 60/8 params
- `optimizer`: Default is EqualWeight with $10k per side
- `data_provider`: Default is ORATS with standard filters
- `executor`: Default is mid execution
- `output`: Default saves trades, equity curve, summary
- `logging`: Default is INFO level

See `src/backtest/config.py` for all defaults.

## Creating New Configs

### Option 1: Copy and Edit
```bash
cp configs/baseline_sp500.json configs/my_experiment.json
# Edit my_experiment.json
python scripts/run_backtest.py configs/my_experiment.json
```

### Option 2: Minimal Config
Create a minimal config with only required fields, let defaults handle the rest:

```json
{
  "config_name": "quick_test",
  "execution": {
    "start_date": "2024-10-01",
    "end_date": "2024-12-31"
  },
  "features": {
    "path": "c:/MomentumCVG_env/cache/momen_cvg_weekly_2018_2025.parquet"
  },
  "strategy": {
    "params": {
      "max_lag": 30
    }
  }
}
```

This overrides only `max_lag`, uses defaults for everything else.

## Parameter Sweeps

To run multiple backtests with different parameters:

```python
import json
from pathlib import Path

base_config = json.load(open('configs/baseline_sp500.json'))

for max_lag in [30, 60, 90]:
    config = base_config.copy()
    config['config_name'] = f'momentum_{max_lag}_8'
    config['strategy']['params']['max_lag'] = max_lag
    
    # Save new config
    with open(f'configs/momentum_{max_lag}_8.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Run backtest
    import subprocess
    subprocess.run(['python', 'scripts/run_backtest.py', f'configs/momentum_{max_lag}_8.json'])
```

## Output Structure

Results are saved to `results/{config_name}/`:

```
results/
  baseline_sp500_weekly_7dte/
    trades.csv              # Trade-by-trade log
    equity_curve.csv        # Daily capital snapshots
    summary.json            # Performance metrics
```

## Config Validation

The config system validates:
- Required fields are present
- Date formats are correct (ISO: YYYY-MM-DD)
- start_date < end_date
- Feature file exists
- Strategy type is valid
- Universe type is valid

Validation errors are shown immediately when loading the config.

## Tips

1. **Use descriptive config_name**: It becomes the output directory name
2. **Add description field**: Document what the config tests
3. **Start minimal**: Let defaults handle most settings
4. **Version control**: Commit config files to track experiments
5. **Reproducibility**: Config + feature file = complete reproduction

## Advanced: Config Inheritance (Future)

Phase 2 may support config inheritance:

```json
{
  "extends": "configs/baseline_sp500.json",
  "config_name": "baseline_with_tight_filters",
  "data_provider": {
    "params": {
      "min_volume": 100,
      "max_spread_pct": 0.20
    }
  }
}
```

This would load baseline_sp500.json and override only the data_provider params.
