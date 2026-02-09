"""
Root conftest.py - makes pytest aware of project structure.

This file must exist at the project root to enable:
1. Importing from src/ without sys.path manipulation
2. Shared fixtures available to all test files
3. Test collection from tests/ directory
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
