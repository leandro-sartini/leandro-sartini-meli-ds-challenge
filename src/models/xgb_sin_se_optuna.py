# src/models/xgb_sin_se_optuna.py
"""
Modelo XGBoost con optimización de hiperparámetros usando Optuna para predicción de fallas.
Utiliza datos balanceados sin early stopping para compatibilidad.
"""

from pathlib import Path
import numpy as np
import joblib
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix, f1_score
import warnings
warnings.filterwarnings('ignore')

# ---------- Configuración de rutas ----------
HERE = Path(__file__).resolve()
PIPE = HERE.parents[2] / "production" / "pipeline"   # ../../production/pipeline
MODEL_DIR = HERE.parents[2] / "production" / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Configuración de Optuna
STORAGE = f"sqlite:///{(PIPE / 'optuna_study_sin_se.db').as_posix()}"
STUDY_NAME = "xgb_failure_pred_sin_se"

# ---------- Carga de datos ----------
print("Cargando datos...")
X_train_res = joblib.load(PIPE / "X_train_sin_se.joblib")  # Datos balanceados
y_train_res = joblib.load(PIPE / "y_train_sin_se.joblib")  # Datos balanceados
X_test = joblib.load(PIPE / "X_test_sin_se.joblib")
y_test = joblib.load(PIPE / "y_test_sin_se.joblib")

print(f"Forma de los datos:")
print(f"X_train_res (balanceado): {X_train_res.shape}")
print(f"y_train_res (balanceado): {y_train_res.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_test: {y_test.shape}")
print(f"Distribución de clases en train balanceado: {np.bincount(y_train_res)}")
print(f"Distribución de clases en test: {np.bincount(y_test)}")

# ---------- Split para validación ----------
print("\nCreando split train/validation para optimización...")
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_res, y_train_res, test_size=0.2, stratify=y_train_res, random_state=42
)

print(f"Split final:")
print(f"X_tr: {X_tr.shape}, y_tr: {y_tr.shape}")
print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
print(f"Distribución de clases en validation: {np.bincount(y_val)}")

# ---------- Utilidades ----------
def best_threshold_fbeta(y_true, y_proba, beta=2.0, min_precision=0.0):
    """
    Encuentra el mejor umbral basado en F-beta score con restricción de precisión mínima.
    
    Parámetros:
    -----------
    y_true : array-like
        Valores reales
    y_proba : array-like
        Probabilidades predichas
    beta : float
        Peso para recall vs precisión (beta=2 da más peso al recall)
    min_precision : float
        Precisión mínima requerida
        
    Retorna:
    --------
    tuple : (threshold, f_beta_score, precision, recall)
    """
    p, r, t = precision_recall_curve(y_true, y_proba)
    f = (1 + beta**2) * p[:-1] * r[:-1] / (beta**2 * p[:-1] + r[:-1] + 1e-12)
    
    # Aplicar restricción de precisión mínima
    if min_precision > 0:
        f[p[:-1] < min_precision] = -1.0
    
    i = int(np.argmax(f)) if len(f) else 0
    thr = float(t[i]) if len(t) else 0.5
    
    return thr, float(f[i] if len(f) else 0.0), float(p[i] if len(p) else 0.0), float(r[i] if len(r) else 0.0)

# Configuración de validación cruzada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def objective(trial: optuna.Trial):
    """
    Función objetivo para Optuna que optimiza hiperparámetros usando train/validation split.
    
    Parámetros:
    -----------
    trial : optuna.Trial
        Objeto de prueba de Optuna
        
    Retorna:
    --------
    float : F1-score promedio (balance entre precisión y recall, priorizando detección de fallas)
    """
    # Hiperparámetros a optimizar
    params = {
        # Parámetros básicos
        "objective": "binary:logistic",
        
        # Hiperparámetros de aprendizaje
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),  # Optimizar número de árboles
        
        # Hiperparámetros de complejidad del árbol
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        
        # Hiperparámetros de muestreo
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
        
        # Hiperparámetros de regularización
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        
        # Otros parámetros
        "random_state": 42,
        "scale_pos_weight": 1.0,  # Datos ya balanceados
        "verbosity": 0
    }
    
    # Crear y entrenar modelo
    clf = XGBClassifier(**params)
    
    # Entrenar en train split
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Predicciones en validation
    proba = clf.predict_proba(X_val)[:, 1]
    
    # Encontrar umbral que maximice F1-score en validation
    from sklearn.metrics import f1_score
    thresholds = np.linspace(0.01, 0.99, 99)  # Rango amplio de umbrales
    best_f1 = 0
    best_threshold = 0.5
    
    for thresh in thresholds:
        y_pred_thresh = (proba >= thresh).astype(int)
        f1 = f1_score(y_val, y_pred_thresh, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    # Reportar progreso
    trial.report(best_f1, 0)
    
    return best_f1

if __name__ == "__main__":
    print("Iniciando optimización de hiperparámetros con Optuna...")
    
    # Crear estudio de Optuna
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )
    
    # Optimizar hiperparámetros
    print("Ejecutando optimización...")
    study.optimize(objective, n_trials=100, n_jobs=1, show_progress_bar=True)
    
    print(f"\nMejor F1-score (Validation): {study.best_value:.4f}")
    print("Mejores hiperparámetros:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # ---------- Entrenamiento del modelo final ----------
    print("\nEntrenando modelo final con los mejores hiperparámetros...")
    
    # Parámetros del modelo final
    best_params = {
        **study.best_params,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "random_state": 42,
        "scale_pos_weight": 1.0,
        "verbosity": 0
    }
    
    # Entrenar modelo final en train balanceado completo
    final_model = XGBClassifier(**best_params)
    final_model.fit(
        X_train_res, y_train_res,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # ---------- Optimización del umbral final ----------
    print("Optimizando umbral de clasificación final...")
    
    # Predicciones en validation (usar validation para optimizar umbral)
    proba_val = final_model.predict_proba(X_val)[:, 1]
    
    # Encontrar mejor umbral que maximice F1-score en validation
    from sklearn.metrics import recall_score, precision_score, f1_score
    thresholds = np.linspace(0.01, 0.99, 99)  # Rango amplio de umbrales
    best_f1 = 0
    best_threshold = 0.5
    best_precision = 0
    best_recall = 0
    
    for thresh in thresholds:
        y_pred_thresh = (proba_val >= thresh).astype(int)
        recall = recall_score(y_val, y_pred_thresh, zero_division=0)
        precision = precision_score(y_val, y_pred_thresh, zero_division=0)
        f1 = f1_score(y_val, y_pred_thresh, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
            best_precision = precision
            best_recall = recall
    
    thr = best_threshold
    rec_val = best_recall
    prec_val = best_precision
    f1_val = best_f1
    
    print(f"Umbral optimizado: {thr:.4f}")
    print(f"Métricas en validation:")
    print(f"  Recall: {rec_val:.4f}")
    print(f"  Precisión: {prec_val:.4f}")
    print(f"  F1-score: {f1_val:.4f}")
    
    # ---------- Evaluación final en test ----------
    print("\nEvaluación final en test con umbral optimizado...")
    
    # Predicciones en test
    proba_test = final_model.predict_proba(X_test)[:, 1]
    y_pred_test = (proba_test >= thr).astype(int)
    
    # Métricas en test
    from sklearn.metrics import recall_score, precision_score, f1_score
    rec_test = recall_score(y_test, y_pred_test, zero_division=0)
    prec_test = precision_score(y_test, y_pred_test, zero_division=0)
    f1_test = f1_score(y_test, y_pred_test, zero_division=0)
    
    # Calcular F2-score manualmente
    f2_test = (1 + 2**2) * (prec_test * rec_test) / (2**2 * prec_test + rec_test) if (2**2 * prec_test + rec_test) > 0 else 0
    
    print(f"Métricas finales en test:")
    print(f"  Recall: {rec_test:.4f}")
    print(f"  Precisión: {prec_test:.4f}")
    print(f"  F1-score: {f1_test:.4f}")
    print(f"  F2-score: {f2_test:.4f}")
    
    # Reporte de clasificación
    print("\nReporte de clasificación (test):")
    print(classification_report(y_test, y_pred_test, digits=4))
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred_test)
    print("Matriz de confusión (test):")
    print(cm)
    
    # ---------- Guardar modelo y estudio ----------
    print("\nGuardando modelo y estudio...")
    
    # Guardar modelo final
    model_path = MODEL_DIR / "xgb_sin_se_optuna_best.joblib"
    joblib.dump(final_model, model_path)
    print(f"Modelo guardado en: {model_path}")
    
    # Guardar estudio
    study_path = MODEL_DIR / "xgb_sin_se_optuna_study.joblib"
    joblib.dump(study, study_path)
    print(f"Estudio guardado en: {study_path}")
    
    # Guardar umbral
    threshold_info = {
        "threshold": thr,
        "f1_score_val": f1_val,
        "f2_score_val": (1 + 2**2) * (prec_val * rec_val) / (2**2 * prec_val + rec_val) if (2**2 * prec_val + rec_val) > 0 else 0,
        "recall_val": rec_val,
        "precision_val": prec_val,
        "f1_score_test": f1_test,
        "f2_score_test": f2_test,
        "recall_test": rec_test,
        "precision_test": prec_test
    }
    threshold_path = MODEL_DIR / "xgb_sin_se_optuna_threshold.json"
    
    import json
    with open(threshold_path, 'w') as f:
        json.dump(threshold_info, f, indent=2)
    print(f"Información del umbral guardada en: {threshold_path}")
    
    print("\n¡Optimización completada exitosamente!")
