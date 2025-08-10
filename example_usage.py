#!/usr/bin/env python3
"""
Example script showing how to use the new organized project structure.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def example_usage():
    """Demonstrate how to use the different modules."""
    
    print("=" * 60)
    print("EXAMPLE USAGE OF THE ORGANIZED PROJECT STRUCTURE")
    print("=" * 60)
    
    # Example 1: Using visualization functions
    print("\n1. Using visualization functions:")
    print("   from src.visualization import graficar_roc_auc")
    print("   from src.visualization import crear_reporte_completo")
    
    # Example 2: Running Optuna optimizations
    print("\n2. Running Optuna optimizations:")
    print("   python -m src.scripts.run_optuna_fe")
    print("   python -m src.scripts.run_optuna_fe_balanced")
    
    # Example 3: Using utility functions
    print("\n3. Using utility functions:")
    print("   from src.utils import test_optuna_setup")
    print("   from src.utils import compare_models")
    
    # Example 4: Using feature engineering
    print("\n4. Using feature engineering:")
    print("   from src import SimpleSensorFE")
    print("   from src import TimeSeriesFeatureExtractor")
    
    # Example 5: Direct script execution
    print("\n5. Direct script execution:")
    print("   cd src/scripts && python run_optuna_fe.py")
    print("   cd src/utils && python test_optuna_setup.py")
    
    print("\n" + "=" * 60)
    print("PROJECT STRUCTURE:")
    print("=" * 60)
    print("src/")
    print("├── __init__.py              # Main package exports")
    print("├── feature_engineering.py   # Feature engineering classes")
    print("├── utils.py                 # General utilities")
    print("├── models/                  # ML models and optimization")
    print("│   ├── __init__.py")
    print("│   ├── xgb_fe_optuna.py")
    print("│   ├── xgb_fe_optuna_balanced.py")
    print("│   └── xgb_sin_se_optuna.py")
    print("├── scripts/                 # Executable scripts")
    print("│   ├── __init__.py")
    print("│   ├── run_optuna_fe.py")
    print("│   └── run_optuna_fe_balanced.py")
    print("├── utils/                   # Utility scripts")
    print("│   ├── __init__.py")
    print("│   ├── test_optuna_setup.py")
    print("│   └── compare_optuna_results.py")
    print("└── visualization/           # Visualization functions")
    print("    ├── __init__.py")
    print("    └── model_viz.py")

if __name__ == "__main__":
    example_usage()
