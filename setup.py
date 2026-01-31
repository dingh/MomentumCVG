"""
Option Momentum Trading System
Setup configuration for package installation
"""

from setuptools import setup, find_packages

setup(
    name="option-momentum",
    version="0.1.0",
    description="Option momentum trading system for backtesting and live trading",
    author="",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests", "tests.*", "notebook", "old_src"]),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "pyarrow>=12.0.0",
        "python-dateutil>=2.8.2",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.11.0",
            "black>=23.0.0",
            "ruff>=0.1.0", 
            "mypy>=1.5.0",
            "ipython>=8.14.0",
            "jupyter>=1.0.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ]
    },
)
