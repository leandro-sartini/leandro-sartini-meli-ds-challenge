#!/usr/bin/env python3
"""
Script para ejecutar la optimización de Optuna para el modelo con feature engineering (versión balanceada).
"""

import sys
from pathlib import Path

# Agregar el directorio src al path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

if __name__ == "__main__":
    print("=" * 60)
    print("OPTIMIZACIÓN DE XGBOOST CON FEATURE ENGINEERING (BALANCEADA)")
    print("=" * 60)
    print("Objetivo: Maximizar detección de fallas (F3-score)")
    print("=" * 60)
    
    try:
        # Importar y ejecutar el script de optimización
        import models.xgb_fe_optuna_balanced
        
        print("\n✅ Optimización completada exitosamente!")
        print("\n📊 Archivos generados:")
        print(f"  - Modelo: production/model/xgb_fe_optuna_balanced_best.joblib")
        print(f"  - Estudio: production/model/xgb_fe_optuna_balanced_study.joblib")
        print(f"  - Umbral: production/model/xgb_fe_optuna_balanced_threshold.json")
        print(f"  - Base de datos: production/pipeline/optuna_study_fe_balanced.db")
        
    except Exception as e:
        print(f"\n❌ Error durante la optimización: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
