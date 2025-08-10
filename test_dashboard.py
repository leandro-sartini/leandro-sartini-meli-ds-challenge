#!/usr/bin/env python3
"""
Script de prueba para el Dashboard de Predicción de Fallas

Este script verifica que todos los componentes del dashboard funcionen correctamente
antes de ejecutar la aplicación principal.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json

# Agregar src al path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Prueba que todas las importaciones funcionen correctamente."""
    print("🔍 Probando importaciones...")
    
    try:
        import streamlit as st
        print("✅ Streamlit importado correctamente")
    except ImportError as e:
        print(f"❌ Error importando Streamlit: {e}")
        return False
    
    try:
        import plotly.graph_objects as go
        print("✅ Plotly importado correctamente")
    except ImportError as e:
        print(f"❌ Error importando Plotly: {e}")
        return False
    
    try:
        from feature_engineering import SimpleSensorFE
        print("✅ Feature Engineering importado correctamente")
    except ImportError as e:
        print(f"❌ Error importando Feature Engineering: {e}")
        return False
    
    return True

def test_model_loading():
    """Prueba la carga del modelo."""
    print("\n🔍 Probando carga del modelo...")
    
    model_paths = [
        Path(__file__).parent / "production" / "model" / "xgb_fe_optuna_best.joblib",
        Path(__file__).parent / "production" / "model" / "best_model_fe.pkl"
    ]
    
    model_loaded = False
    for model_path in model_paths:
        if model_path.exists():
            try:
                model = joblib.load(model_path)
                print(f"✅ Modelo cargado desde: {model_path.name}")
                model_loaded = True
                break
            except Exception as e:
                print(f"❌ Error cargando modelo desde {model_path.name}: {e}")
    
    if not model_loaded:
        print("❌ No se pudo cargar ningún modelo")
        return False
    
    return True

def test_metrics_loading():
    """Prueba la carga de métricas del modelo."""
    print("\n🔍 Probando carga de métricas...")
    
    metrics_path = Path(__file__).parent / "production" / "model" / "xgb_fe_optuna_threshold.json"
    
    if metrics_path.exists():
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            print("✅ Métricas cargadas correctamente")
            print(f"   - Threshold: {metrics.get('threshold', 'N/A')}")
            print(f"   - Recall: {metrics.get('recall_test', 'N/A')}")
            print(f"   - Precision: {metrics.get('precision_test', 'N/A')}")
            return True
        except Exception as e:
            print(f"❌ Error cargando métricas: {e}")
            return False
    else:
        print("⚠️ Archivo de métricas no encontrado")
        return True  # No es crítico

def test_feature_engineering():
    """Prueba el feature engineering con datos sintéticos."""
    print("\n🔍 Probando feature engineering...")
    
    try:
        from feature_engineering import SimpleSensorFE
        
        # Crear datos sintéticos
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        data = []
        
        for i, date in enumerate(dates):
            row = {
                'date': date,
                'device': 'TEST_DEVICE',
                'failure': 0,
                'attribute1': 150000000 + np.random.normal(0, 1000000),
                'attribute2': 50 + np.random.normal(0, 5),
                'attribute3': 5 + np.random.normal(0, 1),
                'attribute4': np.random.choice([0, 1, 2], p=[0.9, 0.08, 0.02]),
                'attribute5': 10 + np.random.normal(0, 2),
                'attribute6': 250000 + np.random.normal(0, 10000),
                'attribute7': np.random.choice([0, 1], p=[0.95, 0.05]),
                'attribute9': np.random.choice([0, 1], p=[0.98, 0.02])
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Aplicar feature engineering
        fe = SimpleSensorFE(
            do_diff=True,
            do_roll_mean=True,
            do_roll_std=False,
            do_roll_min_max=False,
            do_lag=False,
            roll_windows=[3, 7],
            date_col="date",
            device_col="device",
            attr_prefix="attribute"
        )
        
        df_features = fe.fit_transform(df)
        
        print(f"✅ Feature engineering completado")
        print(f"   - Registros originales: {len(df)}")
        print(f"   - Registros procesados: {len(df_features)}")
        print(f"   - Características generadas: {len([col for col in df_features.columns if col not in ['date', 'device', 'failure']])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en feature engineering: {e}")
        return False

def test_data_generation():
    """Prueba la generación de datos sintéticos."""
    print("\n🔍 Probando generación de datos sintéticos...")
    
    try:
        # Simular la generación de datos del dashboard
        from datetime import datetime, timedelta
        
        device_id = "TEST_DEVICE_001"
        days = 30
        
        # Fecha base para la simulación
        base_date = datetime.now() - timedelta(days=days)
        dates = [base_date + timedelta(days=i) for i in range(days)]
        
        # Valores base para cada atributo
        base_values = {
            'attribute1': 150000000,
            'attribute2': 50,
            'attribute3': 5,
            'attribute4': 0,
            'attribute5': 10,
            'attribute6': 250000,
            'attribute7': 0,
            'attribute9': 0
        }
        
        # Generar datos
        data = []
        for i, date in enumerate(dates):
            row = {'date': date.strftime('%Y-%m-%d'), 'device': device_id, 'failure': 0}
            
            trend_factor = 1 + 0.1 * np.sin(i * 0.1)
            noise_factor = np.random.normal(1, 0.05)
            
            for attr, base_val in base_values.items():
                if base_val == 0:
                    value = np.random.choice([0, 1, 2], p=[0.9, 0.08, 0.02])
                else:
                    value = int(base_val * trend_factor * noise_factor)
                    value = max(0, value)
                
                row[attr] = value
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        print(f"✅ Datos sintéticos generados")
        print(f"   - Dispositivo: {device_id}")
        print(f"   - Período: {days} días")
        print(f"   - Registros: {len(df)}")
        print(f"   - Columnas: {list(df.columns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generando datos sintéticos: {e}")
        return False

def main():
    """Función principal de pruebas."""
    print("🧪 Iniciando pruebas del Dashboard de Predicción de Fallas")
    print("=" * 60)
    
    tests = [
        ("Importaciones", test_imports),
        ("Carga del Modelo", test_model_loading),
        ("Carga de Métricas", test_metrics_loading),
        ("Feature Engineering", test_feature_engineering),
        ("Generación de Datos", test_data_generation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ Prueba '{test_name}' falló")
        except Exception as e:
            print(f"❌ Error en prueba '{test_name}': {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Resultados: {passed}/{total} pruebas pasaron")
    
    if passed == total:
        print("🎉 ¡Todas las pruebas pasaron! El dashboard está listo para usar.")
        print("\n🚀 Para ejecutar el dashboard:")
        print("   python run_dashboard.py")
        print("   o")
        print("   streamlit run src/dashboard.py")
    else:
        print("⚠️ Algunas pruebas fallaron. Revisa los errores antes de ejecutar el dashboard.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
