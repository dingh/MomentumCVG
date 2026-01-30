**Status:** 🚧 Active Development 

Modular backtesting framework for option momentum strategies using interface-based architecture.

## Quick Start

```bash
# Install
pip install -e .

# Configure data paths
cp configs/default.yaml configs/local.yaml
# Edit local.yaml with your ORATS data paths

# Run backtest
python scripts/run_backtest.py --config configs/local.yaml

# Run tests
pytest tests/ -v