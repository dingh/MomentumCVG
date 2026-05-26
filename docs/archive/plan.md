# Plan: Clean Room Repository Migration

**TL;DR:** Build production-ready repo from scratch in new location (`ORATS_v2/`), selectively copying and validating each component with comprehensive tests and documentation. Keep large data/results in separate external directory (`ORATS_data/`) linked via config. Estimated 20-30 hours over 2-3 weeks, but results in industrial-grade codebase.

---

## Phase 1: Setup New Repository Structure (2-3 hours)

### 1.1 Create Directory Layout

```bash
# Create new clean repo (outside current workspace)
mkdir ~/Projects/ORATS_v2
cd ~/Projects/ORATS_v2

# Create production repo structure
mkdir -p src/{core,data,features,strategy,portfolio,execution,backtest}
mkdir -p tests/{unit,integration,fixtures}
mkdir -p scripts
mkdir -p configs
mkdir -p docs/{architecture,api,user_guide}

# Create separate data/results directory (NOT in repo)
mkdir ~/ORATS_data
cd ~/ORATS_data
mkdir -p {raw,processed,cache,results,logs}
```

**Key Decision:** Data and results live **outside Git repo** entirely.

```
~/Projects/
  └── ORATS_v2/              # Git repo (code only)
      ├── src/
      ├── tests/
      ├── docs/
      └── configs/

~/ORATS_data/                # External (large files)
  ├── raw/                   # ORATS parquet files
  │   └── ORATS_Adjusted/
  ├── processed/             # Pre-computed features
  │   └── features_all.parquet
  ├── cache/                 # Temporary computation artifacts
  ├── results/               # Backtest outputs
  │   └── [run_id]/
  └── logs/                  # Application logs
```

### 1.2 Initialize Git with Best Practices

```bash
cd ~/Projects/ORATS_v2

# Initialize repo
git init
git branch -m main

# Create comprehensive .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*.so
*.egg-info/
.pytest_cache/
.coverage
htmlcov/

# IDEs
.vscode/
.idea/
*.swp

# Notebooks
.ipynb_checkpoints/

# Environment
.env
venv/
.venv/

# Never commit data or results
cache/
results/
logs/
*.parquet
*.csv
!tests/fixtures/*.csv  # Except test fixtures

# OS
.DS_Store
Thumbs.db
EOF

# Create initial README
cat > README.md << 'EOF'
# ORATS Option Momentum Backtesting System

**Status:** 🚧 Active Development - Production Quality

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
```

## Architecture

- **Interface-based design**: Protocol pattern for component swapping
- **Immutable domain models**: Frozen dataclasses prevent state bugs
- **Test-first development**: 100% test coverage goal
- **Comprehensive documentation**: Every module/function documented

See [docs/architecture/](docs/architecture/) for details.

## Project Status

**Phase 1 - Core Infrastructure** (Current)
- [x] Domain models
- [ ] Data provider
- [ ] Feature calculators
- [ ] Strategy framework
- [ ] Portfolio optimizer
- [ ] Execution simulator
- [ ] Backtest engine

**Phase 2 - Validation**
- [ ] Parameter optimization
- [ ] Walk-forward testing
- [ ] Risk management

**Phase 3 - Live Trading**
- [ ] IB integration
- [ ] Real-time execution

## Documentation

- [Architecture Overview](docs/architecture/system_design.md)
- [API Reference](docs/api/)
- [User Guide](docs/user_guide/)
- [Contributing](CONTRIBUTING.md)

## Requirements

- Python 3.10+
- See [requirements.txt](requirements.txt)

## License

[Your license here]
EOF
```

### 1.3 Connect to GitHub

```bash
# 1. Create GitHub repository (via web browser)
# Go to: https://github.com/new
# - Repository name: ORATS_v2 (or MomentumCVG)
# - Description: "Production-grade option momentum backtesting framework"
# - Private repository (recommended initially)
# - Do NOT initialize with README, .gitignore, or license (we already have these)
# - Click "Create repository"

# 2. Add remote and push initial commit
cd ~/Projects/ORATS_v2

# Make first commit
git add .
git commit -m "chore: initial repository setup

- Directory structure for modular architecture
- Testing infrastructure (pytest, mypy, ruff, black)
- Comprehensive .gitignore excluding data/cache/results
- README with project overview and quick start
- Requirements files for production and development
"

# Add GitHub remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/ORATS_v2.git

# Push to GitHub
git push -u origin main

# Verify remote
git remote -v
```

**GitHub Authentication Options:**

**Option A: Personal Access Token (Recommended)**
```bash
# Generate token at: https://github.com/settings/tokens
# Scopes needed: repo (full control of private repositories)
# When prompted for password, paste the token instead

# Cache credentials (so you don't re-enter every time)
git config --global credential.helper cache
# Or for Windows:
git config --global credential.helper wincred
```

**Option B: SSH Key (More Secure)**
```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your_email@example.com"
# Press Enter to accept default location
# Enter passphrase (optional but recommended)

# Add SSH key to ssh-agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# Copy public key
cat ~/.ssh/id_ed25519.pub
# Windows: type %USERPROFILE%\.ssh\id_ed25519.pub

# Add to GitHub: https://github.com/settings/keys
# Click "New SSH key", paste public key

# Change remote to SSH
git remote set-url origin git@github.com:YOUR_USERNAME/ORATS_v2.git
```

### 1.4 Setup Branch Protection (Optional but Recommended)

**Via GitHub Web Interface:**

1. Go to repository → Settings → Branches
2. Click "Add branch protection rule"
3. Branch name pattern: `main`
4. Enable:
   - ✅ Require a pull request before merging
   - ✅ Require status checks to pass before merging
   - ✅ Require branches to be up to date before merging
   - ✅ Do not allow bypassing the above settings (if working solo, you might skip this)

**Benefits:**
- Forces code review (even for solo dev, creates discipline)
- Prevents accidental force pushes to main
- Can integrate with CI/CD status checks later

### 1.5 Setup Testing Infrastructure

```bash
# Create pytest configuration
cat > pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "orats-momentum"
version = "0.1.0"
description = "Option momentum backtesting framework"
requires-python = ">=3.10"
dependencies = [
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "pyarrow>=12.0.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "mypy>=1.0.0",
    "ruff>=0.1.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "-v",
    "--cov=src",
    "--cov-report=html",
    "--cov-report=term-missing",
]

[tool.black]
line-length = 100
target-version = ["py310"]

[tool.ruff]
line-length = 100
target-version = "py310"
select = ["E", "F", "I", "N", "W", "UP"]

[tool.mypy]
python_version = "3.10"
strict = true
warn_return_any = true
warn_unused_configs = true
EOF

# Create requirements files
cat > requirements.txt << 'EOF'
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0
pyyaml>=6.0
EOF

cat > requirements-dev.txt << 'EOF'
-r requirements.txt
pytest>=7.0.0
pytest-cov>=4.0.0
black>=23.0.0
mypy>=1.0.0
ruff>=0.1.0
jupyter>=1.0.0
notebook>=7.0.0
EOF

# Create test utilities
cat > tests/conftest.py << 'EOF'
"""
Pytest configuration and shared fixtures.
"""
import pytest
from pathlib import Path
from decimal import Decimal
from datetime import date

# Test data paths
FIXTURES_DIR = Path(__file__).parent / "fixtures"

@pytest.fixture
def sample_config():
    """Sample backtest configuration for testing."""
    return {
        'initial_capital': Decimal('100000'),
        'start_date': date(2023, 1, 1),
        'end_date': date(2023, 12, 31),
        'target_dte': 30,
        'max_positions': 10
    }

@pytest.fixture
def fixtures_dir():
    """Path to test fixtures directory."""
    return FIXTURES_DIR

# Add more shared fixtures as needed
EOF
```

---

## Phase 2: Migration Order & Standards (Core Framework)

**Principle:** Migrate in dependency order (models → data → features → strategy → execution → engine)

### 2.1 Component Migration Template

For **each component** being migrated:

```markdown
## Migration Checklist: [Component Name]

### Pre-Migration
- [ ] Read original code (`old_workspace/src/.../file.py`)
- [ ] Identify all dependencies
- [ ] List all functions/classes
- [ ] Document current behavior (what works, what's broken)

### Copy & Review
- [ ] Create new file in `ORATS_v2/src/.../`
- [ ] Add comprehensive module docstring
- [ ] Review and refactor each function:
  - [ ] Add type hints (all parameters and returns)
  - [ ] Add docstring (Google style)
  - [ ] Fix any code smells
  - [ ] Remove dead code
- [ ] Add `__all__` export list

### Documentation
- [ ] Write module-level docs in `docs/api/`
- [ ] Add usage examples
- [ ] Document assumptions and limitations

### Testing
- [ ] Write unit tests (`tests/unit/test_[component].py`)
  - [ ] Happy path tests
  - [ ] Edge case tests
  - [ ] Error handling tests
- [ ] Aim for >90% coverage
- [ ] All tests passing

### Validation
- [ ] Run `pytest tests/unit/test_[component].py -v`
- [ ] Run `mypy src/[path]/[component].py`
- [ ] Run `ruff check src/[path]/[component].py`
- [ ] Manual smoke test (if applicable)

### Integration
- [ ] Update `src/__init__.py` exports
- [ ] Add to integration test suite
- [ ] Update README with component status
- [ ] Git commit with descriptive message
```

---

## Phase 3: Component-by-Component Migration Plan

### Week 1: Foundation (8-10 hours)

#### Day 1-2: Core Models (`src/core/models.py`)

**Source:** `old_workspace/src/core/models.py`

**Copy Checklist:**
```python
# BEFORE copying, document what you're bringing:
"""
Migrating from old_workspace/src/core/models.py:
- OptionQuote ✅ (stable, well-tested)
- OptionLeg ✅ (stable)
- OptionStrategy ✅ (stable)
- StrategyType enum ✅
- Signal ✅ (stable)
- Position ✅ (stable)

Changes during migration:
- Add stricter type hints
- Enhance docstrings with examples
- Add validation methods
"""
```

**Test Requirements:**
- Test immutability (all frozen dataclasses)
- Test validation (negative prices, invalid dates)
- Test equality and hashing
- Test serialization (to/from dict for JSON)

**Example Test Structure:**
```python
# tests/unit/test_models.py

import pytest
from decimal import Decimal
from datetime import date
from src.core.models import OptionQuote, OptionLeg, OptionStrategy

class TestOptionQuote:
    """Test suite for OptionQuote model."""
    
    def test_create_valid_quote(self):
        """Should create valid option quote with all fields."""
        quote = OptionQuote(
            ticker='AAPL',
            option_type='call',
            strike=Decimal('150.00'),
            expiry=date(2024, 3, 15),
            quote_date=date(2024, 1, 15),
            bid=Decimal('5.20'),
            ask=Decimal('5.40'),
            mid=Decimal('5.30'),
            iv=0.25,
            delta=0.55,
            gamma=0.02,
            theta=-0.05,
            vega=0.15,
            volume=1000,
            open_interest=5000
        )
        
        assert quote.ticker == 'AAPL'
        assert quote.mid == Decimal('5.30')
    
    def test_immutability(self):
        """Should prevent modification after creation."""
        quote = OptionQuote(...)  # full params
        
        with pytest.raises(AttributeError):
            quote.bid = Decimal('5.50')  # Should fail
    
    def test_negative_price_validation(self):
        """Should reject negative prices."""
        with pytest.raises(ValueError, match="bid must be non-negative"):
            OptionQuote(..., bid=Decimal('-1.00'), ...)
    
    def test_bid_ask_consistency(self):
        """Should validate bid <= ask."""
        with pytest.raises(ValueError, match="bid cannot exceed ask"):
            OptionQuote(..., bid=Decimal('5.50'), ask=Decimal('5.00'), ...)
    
    # ... 15-20 more tests for edge cases

class TestPosition:
    """Test suite for Position model."""
    
    def test_pnl_calculation(self):
        """Should calculate P&L correctly."""
        position = Position(
            ticker='AAPL',
            quantity=Decimal('2.0'),
            entry_cost=Decimal('1060.00'),  # $530 per straddle × 2
            exit_value=Decimal('1200.00'),
            entry_date=date(2024, 1, 15),
            exit_date=date(2024, 2, 15),
            strategy=...,  # OptionStrategy
            metadata={}
        )
        
        assert position.pnl == Decimal('140.00')
        assert position.return_pct == pytest.approx(0.132, rel=0.001)
    
    # ... 10-15 more tests
```

**Documentation (`docs/api/core_models.md`):**
```markdown
# Core Domain Models

## Overview

Immutable dataclasses representing core domain concepts.

## OptionQuote

Represents a single option contract at a point in time.

**Fields:**
- `ticker` (str): Underlying symbol
- `option_type` (Literal['call', 'put']): Option type
- `strike` (Decimal): Strike price
- `expiry` (date): Expiration date
- `quote_date` (date): Quote snapshot date
- `bid` (Decimal): Bid price
- `ask` (Decimal): Ask price
- `mid` (Decimal): Mid price (bid+ask)/2
- `iv` (float): Implied volatility
- `delta` (float): Delta greek
- ... [document all fields]

**Validation:**
- All prices must be non-negative
- bid <= mid <= ask
- expiry must be >= quote_date
- volume and open_interest must be non-negative

**Example:**
```python
from decimal import Decimal
from datetime import date
from src.core.models import OptionQuote

quote = OptionQuote(
    ticker='AAPL',
    option_type='call',
    strike=Decimal('150.00'),
    ...
)
```

[Continue for all models...]
```

**Time Estimate:** 8-10 hours (models are already good, mostly adding tests/docs)

#### Day 3-4: Data Provider (`src/data/orats_provider.py`)

**Source:** `old_workspace/src/data/orats_provider.py`

**Pre-Migration Review:**
- LRU cache - is it safe? Document reasoning
- Error handling - improve logging
- Point-in-time guarantees - verify no lookahead

**Test Requirements:**
- Mock parquet file loading (use fixtures)
- Test filtering logic (volume, spread, etc.)
- Test date range handling
- Test missing data scenarios

**Example Fixture:**
```python
# tests/fixtures/sample_option_chain.csv
ticker,quoteDate,expirDate,strike,adj_strike,cBid,cAsk,pBid,pAsk,delta,smoothSmvVol
AAPL,2024-01-15,2024-02-16,150.00,150.00,5.20,5.40,2.10,2.30,0.55,0.25
AAPL,2024-01-15,2024-02-16,155.00,155.00,3.50,3.70,3.80,4.00,0.42,0.26
```

**Time Estimate:** 6-8 hours

---

### Week 2: Features & Strategy (10-12 hours)

#### Day 5-6: Feature Calculators

**Components:**
1. `src/features/straddle_builder.py` (rename from straddle_analyzer)
2. `src/features/momentum_calculator.py`
3. `src/features/cvg_calculator.py`

**Critical Review Points:**
- Verify rolling window calculations don't leak future data
- Test with missing data (< min_count_pct threshold)
- Document formula precisely

**Test Strategy:**
- Use small synthetic DataFrames (easier to verify by hand)
- Test edge cases: single data point, all NaN, boundary dates

**Time Estimate:** 8-10 hours

#### Day 7-8: Strategy (`src/strategy/momentum_cvg.py`)

**Test Requirements:**
- Test signal generation with known feature values
- Test filtering thresholds
- Test empty/no-signal scenarios
- Integration test with real features

**Time Estimate:** 6-8 hours

---

### Week 3: Execution & Engine (10-12 hours)

#### Day 9-10: Portfolio Optimizer & Executor

**Components:**
1. `src/portfolio/optimizer.py`
2. `src/execution/backtest_executor.py`

**Time Estimate:** 8-10 hours

#### Day 11-12: Backtest Engine

**Source:** `old_workspace/src/backtest/engine.py`

**Critical Test:** Full integration test
```python
# tests/integration/test_full_backtest.py

def test_end_to_end_backtest(tmp_path):
    """Full backtest with known inputs produces expected outputs."""
    # Use fixture data
    # Run backtest
    # Assert key metrics
    assert results['num_trades'] == 52  # Expected number
    assert results['sharpe_ratio'] > 0.5
    # ... more assertions
```

**Time Estimate:** 8-10 hours

---

## Phase 4: External Data Configuration (2-3 hours)

### 4.1 Config System with External Paths

```yaml
# configs/default.yaml

data:
  # External data directory (not in repo)
  external_root: "~/ORATS_data"  # User edits this
  
  sources:
    orats:
      path: "${data.external_root}/raw/ORATS_Adjusted"
      required: true
    
  cache:
    path: "${data.external_root}/processed"
    features_file: "features_all.parquet"
  
  output:
    results_dir: "${data.external_root}/results"
    logs_dir: "${data.external_root}/logs"

# ... rest of config
```

### 4.2 Path Resolution Utility

```python
# src/utils/config.py

from pathlib import Path
import os
import yaml

def load_config(config_path: str) -> dict:
    """Load YAML config and resolve environment variables."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Resolve ${...} references
    config = _resolve_references(config)
    
    # Expand ~ and environment variables
    config = _expand_paths(config)
    
    return config

def _expand_paths(config: dict) -> dict:
    """Recursively expand paths with ~ and env vars."""
    for key, value in config.items():
        if isinstance(value, str) and ('/' in value or '\\' in value):
            config[key] = os.path.expanduser(os.path.expandvars(value))
        elif isinstance(value, dict):
            config[key] = _expand_paths(value)
    return config
```

---

## Phase 5: Continuous Validation (Ongoing)

### 5.1 Git Workflow

```bash
# After each component migration:
git add [files]
git commit -m "feat: add [Component] with tests and docs

- Migrated from old codebase
- Added comprehensive unit tests (coverage: X%)
- Documented API in docs/api/
- All tests passing

Tests: pytest tests/unit/test_[component].py
Coverage: pytest --cov=src.[module]
"

# Create migration tracking branch
git checkout -b migration/[component-name]
# Work, test, commit
git checkout main
git merge migration/[component-name]
```

### 5.2 Pre-Commit Hooks

```bash
# .git/hooks/pre-commit (make executable)
#!/bin/bash

echo "Running pre-commit checks..."

# Format code
black src/ tests/
ruff check src/ tests/ --fix

# Type check
mypy src/

# Run tests
pytest tests/ -v

if [ $? -ne 0 ]; then
    echo "❌ Tests failed. Commit aborted."
    exit 1
fi

echo "✅ All checks passed"
```

---

## Phase 6: Documentation Strategy

### 6.1 Documentation Standards

**Every module must have:**
```python
"""
Module: [module name]

Description: [1-2 sentence overview]

Key Components:
- ClassName1: [brief description]
- ClassName2: [brief description]

Dependencies:
- Module1 (for X)
- Module2 (for Y)

Example:
    ```python
    from src.module import Class
    obj = Class(...)
    result = obj.method()
    ```

Notes:
    - Any important caveats
    - Performance considerations
    - Thread safety
"""
```

**Every function must have:**
```python
def function_name(param1: Type1, param2: Type2) -> ReturnType:
    """
    One-line summary.
    
    Longer description explaining what this does, why it exists,
    and any important behavior.
    
    Args:
        param1: Description of param1. Include valid ranges/constraints.
        param2: Description of param2.
    
    Returns:
        Description of return value. Include type and meaning.
    
    Raises:
        ValueError: When param1 is negative.
        FileNotFoundError: When data file doesn't exist.
    
    Example:
        >>> result = function_name(10, "test")
        >>> print(result)
        42
    
    Note:
        Any caveats, performance notes, or related functions.
    """
```

---

## Timeline & Milestones

| Week | Focus | Deliverables | Validation |
|------|-------|--------------|------------|
| 1 | Setup + Core Models | Repo structure, models.py, 20+ tests | `pytest tests/unit/test_models.py` all passing |
| 2 | Data & Features | Data provider, 3 feature calculators, 30+ tests | Integration test with real ORATS file |
| 3 | Strategy & Execution | Strategy, optimizer, executor, engine, 40+ tests | End-to-end backtest runs |
| 4 | CLI & Documentation | run_backtest.py script, complete API docs | One-command backtest from config |

**Total Estimate:** 20-30 hours over 3-4 weeks

---

## Success Criteria

**After migration, you should have:**

✅ Clean Git repo with meaningful commit history  
✅ 100+ unit tests with >90% coverage  
✅ Every module fully documented  
✅ One-command backtest: `python scripts/run_backtest.py --config configs/local.yaml`  
✅ All data/results external to repo  
✅ CI/CD ready (GitHub Actions can run tests)  
✅ Code passes: pytest, mypy, ruff, black  
✅ Reproducible: another developer can clone and run  

**Confidence Boost:** You can explain every line of code because you rewrote it thoughtfully.

---

## Key Principles

1. **Quality over speed** - Take time to understand and validate each component
2. **Test-first mindset** - Write tests as you migrate, not after
3. **Document as you go** - Fresh understanding = best documentation
4. **Clean commits** - Each component is a logical, working unit
5. **External data** - Never commit large files to Git
6. **Reproducibility** - Someone else should be able to clone and run
7. **Industrial grade** - Code you'd be proud to show in an interview

---

## Next Steps

1. Create the directory structure (Phase 1.1)
2. Initialize Git and testing infrastructure (Phase 1.2-1.3)
3. Start with Core Models migration (Week 1, Day 1-2)
4. Follow the component migration template for each piece
5. Maintain migration log tracking progress
