"""
Unified backtest configuration system.

Single JSON config file defines a complete backtest:
- Strategy parameters
- Execution settings
- Universe filters
- Data provider settings
- Output options

Example usage:
    config = BacktestConfig.from_json('configs/baseline.json')
    components = config.instantiate_components()
    engine = BacktestEngine(**components)
    results = engine.run(config.to_execution_config())
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd

from src.data.orats_provider import ORATSDataProvider, IDataProvider
from src.strategy.momentum_cvg import MomentumCVGStrategy
from src.strategy.base import IStrategy
from src.strategy.universe_filter import SP500UniverseFilter, AllTickersUniverseFilter, IUniverseFilter
from src.portfolio.optimizer import EqualWeightOptimizer
from src.portfolio.base import IPortfolioOptimizer
from src.execution.backtest_executor import BacktestExecutor, IExecutor

logger = logging.getLogger(__name__)


# Default configuration values
DEFAULT_CONFIG = {
    "config_version": "1.0",
    "execution": {
        "initial_capital": 100000.0,
        "target_dte": 30,
        "max_positions": 10,
        "capital_mode": "compound"
    },
    "features": {
        "required_columns": ["date", "ticker"]
    },
    "universe": {
        "type": "All",
        "params": {}
    },
    "strategy": {
        "type": "MomentumCVG",
        "params": {
            "max_lag": 60,
            "min_lag": 8,
            "momentum_long_pct": 0.10,
            "momentum_short_pct": 0.10,
            "cvg_filter_pct": 0.50,
            "min_count_pct": 0.80
        }
    },
    "optimizer": {
        "type": "EqualWeight",
        "params": {
            "notional_per_side": 10000.0
        }
    },
    "data_provider": {
        "type": "ORATS",
        "params": {
            "data_root": "c:/ORATS/data/ORATS_Adjusted",
            "min_volume": 10,
            "min_open_interest": 0,
            "min_bid": 0.05,
            "max_spread_pct": 0.50,
            "cache_size": 5
        }
    },
    "executor": {
        "type": "BacktestExecutor",
        "params": {
            "execution_mode": "mid"
        }
    },
    "output": {
        "results_dir": "results",
        "save_trades": True,
        "save_daily_portfolio": False,
        "save_equity_curve": True,
        "save_summary_json": True,
        "trades_filename": "trades.csv",
        "equity_curve_filename": "equity_curve.csv",
        "summary_filename": "summary.json"
    },
    "logging": {
        "level": "INFO",
        "log_file": None,
        "console_output": True
    }
}


@dataclass
class BacktestConfig:
    """
    Unified configuration for a complete backtest run.
    
    Load from JSON and instantiate all components automatically.
    Supports defaults for optional sections.
    
    Example:
        >>> config = BacktestConfig.from_json('configs/baseline.json')
        >>> components = config.instantiate_components()
        >>> engine = BacktestEngine(**components)
        >>> results = engine.run(config.to_execution_config())
    """
    
    # Meta
    config_version: str = "1.0"
    config_name: str = "unnamed"
    description: str = ""
    created_date: Optional[str] = None
    author: Optional[str] = None
    
    # Execution (required)
    execution: Dict[str, Any] = field(default_factory=dict)
    
    # Features (required)
    features: Dict[str, Any] = field(default_factory=dict)
    
    # Universe (optional - defaults to All)
    universe: Dict[str, Any] = field(default_factory=dict)
    
    # Strategy (optional - defaults to MomentumCVG)
    strategy: Dict[str, Any] = field(default_factory=dict)
    
    # Optimizer (optional - defaults to EqualWeight)
    optimizer: Dict[str, Any] = field(default_factory=dict)
    
    # Data provider (optional - defaults to ORATS)
    data_provider: Dict[str, Any] = field(default_factory=dict)
    
    # Executor (optional - defaults to BacktestExecutor)
    executor: Dict[str, Any] = field(default_factory=dict)
    
    # Output (optional)
    output: Dict[str, Any] = field(default_factory=dict)
    
    # Logging (optional)
    logging: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Apply defaults and validate configuration"""
        # Merge with defaults
        self._apply_defaults()
        
        # Validate required fields
        self._validate()
    
    def _apply_defaults(self):
        """Merge user config with defaults"""
        for section, defaults in DEFAULT_CONFIG.items():
            if section not in ['config_version']:
                current = getattr(self, section, {})
                if isinstance(defaults, dict) and isinstance(current, dict):
                    # Deep merge
                    merged = self._deep_merge(defaults.copy(), current)
                    setattr(self, section, merged)
    
    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        """Deep merge two dictionaries"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = BacktestConfig._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def _validate(self):
        """Validate configuration"""
        errors = []
        
        # Check execution section
        if 'start_date' not in self.execution:
            errors.append("execution.start_date is required")
        if 'end_date' not in self.execution:
            errors.append("execution.end_date is required")
        
        # Check dates
        if 'start_date' in self.execution and 'end_date' in self.execution:
            start = date.fromisoformat(self.execution['start_date'])
            end = date.fromisoformat(self.execution['end_date'])
            if start >= end:
                errors.append("execution.start_date must be before end_date")
        
        # Check features path
        if 'path' not in self.features:
            errors.append("features.path is required")
        elif not Path(self.features['path']).exists():
            errors.append(f"Features file not found: {self.features['path']}")
        
        # Check universe type
        valid_universes = ['SP500', 'All', 'Custom']
        if self.universe.get('type') not in valid_universes:
            errors.append(f"universe.type must be one of {valid_universes}")
        
        # Check strategy type
        if self.strategy.get('type') not in ['MomentumCVG']:
            errors.append(f"Unknown strategy type: {self.strategy.get('type')}")
        
        if errors:
            raise ValueError(f"Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
    
    @classmethod
    def from_json(cls, json_path: str) -> 'BacktestConfig':
        """
        Load config from JSON file.
        
        Args:
            json_path: Path to JSON config file
            
        Returns:
            BacktestConfig instance with defaults applied
        """
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Config file not found: {json_path}")
        
        logger.info(f"Loading config from {json_path}")
        
        with open(json_path) as f:
            data = json.load(f)
        
        return cls(**data)
    
    def to_json(self, json_path: str):
        """
        Save config to JSON file.
        
        Args:
            json_path: Path to save JSON config
        """
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(json_path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        
        logger.info(f"Saved config to {json_path}")
    
    def instantiate_components(self) -> Dict[str, Any]:
        """
        Factory method: create all backtest components from config.
        
        Returns:
            Dictionary with instantiated components:
                - features: pd.DataFrame (filtered by universe)
                - data_provider: IDataProvider
                - strategy: IStrategy
                - optimizer: IPortfolioOptimizer
                - executor: IExecutor
        """
        logger.info("Instantiating components from config")
        
        # 1. Load features
        logger.info(f"Loading features from {self.features['path']}")
        features = pd.read_parquet(self.features['path'])
        logger.info(f"Loaded {len(features):,} feature rows")
        
        # 2. Apply universe filter
        universe_filter = self._create_universe_filter()
        features = universe_filter.filter_features(features)
        logger.info(
            f"Filtered to {len(features):,} rows, "
            f"{features['ticker'].nunique()} tickers ({self.universe['type']} universe)"
        )
        
        # 3. Create data provider
        data_provider = self._create_data_provider()
        
        # 4. Create strategy
        strategy = self._create_strategy()
        
        # 5. Create optimizer
        optimizer = self._create_optimizer()
        
        # 6. Create executor
        executor = self._create_executor()
        
        return {
            'features': features,
            'data_provider': data_provider,
            'strategy': strategy,
            'optimizer': optimizer,
            'executor': executor
        }
    
    def _create_universe_filter(self) -> IUniverseFilter:
        """Create universe filter from config"""
        universe_type = self.universe['type']
        params = self.universe.get('params', {})
        
        if universe_type == 'SP500':
            if 'sp500_csv_path' not in params:
                raise ValueError("SP500 universe requires sp500_csv_path parameter")
            return SP500UniverseFilter(params['sp500_csv_path'])
        
        elif universe_type == 'All':
            return AllTickersUniverseFilter()
        
        else:
            raise ValueError(f"Unknown universe type: {universe_type}")
    
    def _create_data_provider(self) -> IDataProvider:
        """Create data provider from config"""
        provider_type = self.data_provider['type']
        params = self.data_provider['params']
        
        if provider_type == 'ORATS':
            return ORATSDataProvider(
                data_root=params['data_root'],
                min_volume=params.get('min_volume', 10),
                min_open_interest=params.get('min_open_interest', 0),
                min_bid=params.get('min_bid', 0.05),
                max_spread_pct=params.get('max_spread_pct', 0.50),
                cache_size=params.get('cache_size', 5)
            )
        else:
            raise ValueError(f"Unknown data provider type: {provider_type}")
    
    def _create_strategy(self) -> IStrategy:
        """Create strategy from config"""
        strategy_type = self.strategy['type']
        params = self.strategy['params']
        
        if strategy_type == 'MomentumCVG':
            return MomentumCVGStrategy(
                max_lag=params['max_lag'],
                min_lag=params['min_lag'],
                momentum_long_pct=params['momentum_long_pct'],
                momentum_short_pct=params['momentum_short_pct'],
                cvg_filter_pct=params['cvg_filter_pct'],
                min_count_pct=params.get('min_count_pct', 0.80)
            )
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    def _create_optimizer(self) -> IPortfolioOptimizer:
        """Create optimizer from config"""
        optimizer_type = self.optimizer['type']
        params = self.optimizer['params']
        
        if optimizer_type == 'EqualWeight':
            return EqualWeightOptimizer(
                notional_per_side=Decimal(str(params['notional_per_side']))
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    def _create_executor(self) -> IExecutor:
        """Create executor from config"""
        executor_type = self.executor['type']
        params = self.executor['params']
        
        if executor_type == 'BacktestExecutor':
            return BacktestExecutor(
                execution_mode=params.get('execution_mode', 'mid')
            )
        else:
            raise ValueError(f"Unknown executor type: {executor_type}")
    
    def to_execution_config(self):
        """
        Convert to engine's BacktestExecutionConfig format.
        
        Returns:
            BacktestExecutionConfig object for engine.run()
        """
        from src.backtest.engine import BacktestExecutionConfig
        
        return BacktestExecutionConfig(
            initial_capital=Decimal(str(self.execution['initial_capital'])),
            start_date=date.fromisoformat(self.execution['start_date']),
            end_date=date.fromisoformat(self.execution['end_date']),
            target_dte=self.execution['target_dte'],
            max_positions=self.execution['max_positions'],
            capital_mode=self.execution['capital_mode']
        )
    
    @property
    def output_dir(self) -> Path:
        """Get output directory for this config"""
        base = Path(self.output['results_dir'])
        return base / self.config_name
    
    def setup_output_dir(self):
        """Create output directory structure"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")
    
    def setup_logging(self):
        """Setup logging based on config"""
        level = getattr(logging, self.logging['level'])
        
        # Configure root logger
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[]
        )
        
        # Console handler
        if self.logging['console_output']:
            console = logging.StreamHandler()
            console.setLevel(level)
            logging.getLogger().addHandler(console)
        
        # File handler
        if self.logging['log_file']:
            log_path = self.logging['log_file'].format(
                config_name=self.config_name,
                timestamp=date.today().isoformat()
            )
            log_path = Path(log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(level)
            logging.getLogger().addHandler(file_handler)
            logger.info(f"Logging to file: {log_path}")
