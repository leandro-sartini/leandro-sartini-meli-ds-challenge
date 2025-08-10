# src/models/xgb_fe_optuna_balanced.py
"""
Modelo XGBoost con optimización de hiperparámetros usando Optuna para predicción de fallas.
Versión balanceada que prioriza fuertemente la detección de fallas.
"""

from pathlib import Path
import numpy as np
import joblib
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix, f1_score, recall_score, precision_score
import warnings
warnings.filterwarnings('ignore')

# ---------- Configuración de rutas ----------
HERE = Path(__file__).resolve()
PIPE = HERE.parents[2] / "production" / "pipeline"   # ../../production/pipeline
MODEL_DIR = HERE.parents[2] / "production" / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Configuración de Optuna
STORAGE = f"sqlite:///{(PIPE / 'optuna_study_fe_balanced.db').as_posix()}"
STUDY_NAME = "xgb_failure_pred_fe_balanced"

# ---------- Carga de datos ----------
print("Cargando datos con feature engineering...")
X_train_res = joblib.load(PIPE / "X_train.joblib")  # Datos balanceados con FE
y_train_res = joblib.load(PIPE / "y_train.joblib")  # Datos balanceados con FE
X_test = joblib.load(PIPE / "X_test.joblib")
y_test = joblib.load(PIPE / "y_test.joblib")

print(f"Forma de los datos:")
print(f"X_train_res (balanceado con FE): {X_train_res.shape}")
print(f"y_train_res (balanceado con FE): {y_train_res.shape}")
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

# ---------- Función de métrica personalizada balanceada ----------
def balanced_failure_score(y_true, y_pred, y_proba=None):
    """
    Métrica personalizada que prioriza fuertemente la detección de fallas.
    Usa F3-score (aún más peso al recall) y penalización adaptativa.
    """
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    
    # F3-score (da aún más peso al recall que F2)
    f3 = (1 + 3**2) * (precision * recall) / (3**2 * precision + recall) if (3**2 * precision + recall) > 0 else 0
    
    # Penalización por falsos positivos (más suave)
    fp_penalty = 0
    if y_proba is not None:
        # Penalización más suave para permitir más detección de fallas
        false_positive_confidence = np.mean(y_proba[y_true == 0])
        fp_penalty = false_positive_confidence * 0.05  # Penalización más baja
    
    # Bonus por alto recall
    recall_bonus = recall * 0.1 if recall > 0.7 else 0
    
    # Score final: F3-score + bonus por recall - penalización por falsos positivos
    final_score = f3 + recall_bonus - fp_penalty
    
    return final_score, f3, recall, precision

# Configuración de validación cruzada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def objective(trial: optuna.Trial):
    """
    Función objetivo para Optuna que optimiza hiperparámetros usando train/validation split.
    Se enfoca en maximizar la detección de fallas con penalización mínima.
    
    Parámetros:
    -----------
    trial : optuna.Trial
        Objeto de prueba de Optuna
        
    Retorna:
    --------
    float : Score personalizado que prioriza fuertemente detección de fallas
    """
    # Hiperparámetros a optimizar
    params = {
        # Parámetros básicos
        "objective": "binary:logistic",
        
        # Hiperparámetros de aprendizaje
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        
        # Hiperparámetros de complejidad del árbol
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        
        # Hiperparámetros de muestreo
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        
        # Hiperparámetros de regularización
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        
        # Otros parámetros
        "random_state": 42,
        "scale_pos_weight": 1.0,  # Datos ya balanceados
        "verbosity": 0,
        "tree_method": "hist"  # Método más rápido
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
    
    # Encontrar umbral que maximice nuestro score personalizado
    thresholds = np.linspace(0.01, 0.99, 99)
    best_score = -1
    best_threshold = 0.5
    best_f3 = 0
    best_recall = 0
    best_precision = 0
    
    for thresh in thresholds:
        y_pred_thresh = (proba >= thresh).astype(int)
        custom_score, f3, recall, precision = balanced_failure_score(y_val, y_pred_thresh, proba)
        
        if custom_score > best_score:
            best_score = custom_score
            best_threshold = thresh
            best_f3 = f3
            best_recall = recall
            best_precision = precision
    
    # Reportar progreso
    trial.report(best_score, 0)
    
    # Almacenar métricas adicionales
    trial.set_user_attr("threshold", best_threshold)
    trial.set_user_attr("f3_score", best_f3)
    trial.set_user_attr("recall", best_recall)
    trial.set_user_attr("precision", best_precision)
    
    return best_score

if __name__ == "__main__":
    print("Iniciando optimización de hiperparámetros con Optuna...")
    print("Objetivo: Maximizar detección de fallas (versión balanceada)")
    
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
    study.optimize(objective, n_trials=200, n_jobs=1, show_progress_bar=True)
    
    print(f"\nMejor score personalizado (Validation): {study.best_value:.4f}")
    print("Mejores hiperparámetros:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Métricas del mejor trial
    best_trial = study.best_trial
    print(f"\nMétricas del mejor trial:")
    print(f"  Threshold: {best_trial.user_attrs['threshold']:.4f}")
    print(f"  F3-score: {best_trial.user_attrs['f3_score']:.4f}")
    print(f"  Recall: {best_trial.user_attrs['recall']:.4f}")
    print(f"  Precision: {best_trial.user_attrs['precision']:.4f}")
    
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
    
    # Encontrar mejor umbral que maximice nuestro score personalizado
    thresholds = np.linspace(0.01, 0.99, 99)
    best_score = -1
    best_threshold = 0.5
    best_f3 = 0
    best_recall = 0
    best_precision = 0
    
    for thresh in thresholds:
        y_pred_thresh = (proba_val >= thresh).astype(int)
        custom_score, f3, recall, precision = balanced_failure_score(y_val, y_pred_thresh, proba_val)
        
        if custom_score > best_score:
            best_score = custom_score
            best_threshold = thresh
            best_f3 = f3
            best_recall = recall
            best_precision = precision
    
    thr = best_threshold
    rec_val = best_recall
    prec_val = best_precision
    f3_val = best_f3
    
    print(f"Umbral optimizado: {thr:.4f}")
    print(f"Métricas en validation:")
    print(f"  Recall: {rec_val:.4f}")
    print(f"  Precisión: {prec_val:.4f}")
    print(f"  F3-score: {f3_val:.4f}")
    print(f"  Score personalizado: {best_score:.4f}")
    
    # ---------- Evaluación final en test ----------
    print("\nEvaluación final en test con umbral optimizado...")
    
    # Predicciones en test
    proba_test = final_model.predict_proba(X_test)[:, 1]
    y_pred_test = (proba_test >= thr).astype(int)
    
    # Métricas en test
    custom_score_test, f3_test, rec_test, prec_test = balanced_failure_score(y_test, y_pred_test, proba_test)
    
    print(f"Métricas finales en test:")
    print(f"  Recall: {rec_test:.4f}")
    print(f"  Precisión: {prec_test:.4f}")
    print(f"  F3-score: {f3_test:.4f}")
    print(f"  Score personalizado: {custom_score_test:.4f}")
    
    # Reporte de clasificación
    print("\nReporte de clasificación (test):")
    print(classification_report(y_test, y_pred_test, digits=4))
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred_test)
    print("Matriz de confusión (test):")
    print(cm)
    
    # Análisis de falsos positivos y negativos
    tn, fp, fn, tp = cm.ravel()
    print(f"\nAnálisis detallado:")
    print(f"  Verdaderos Negativos (TN): {tn}")
    print(f"  Falsos Positivos (FP): {fp}")
    print(f"  Falsos Negativos (FN): {fn}")
    print(f"  Verdaderos Positivos (TP): {tp}")
    print(f"  Tasa de Falsos Positivos: {fp/(fp+tn):.4f}")
    print(f"  Tasa de Falsos Negativos: {fn/(fn+tp):.4f}")
    
    # ---------- Guardar modelo y estudio ----------
    print("\nGuardando modelo y estudio...")
    
    # Guardar modelo final
    model_path = MODEL_DIR / "xgb_fe_optuna_balanced_best.joblib"
    joblib.dump(final_model, model_path)
    print(f"Modelo guardado en: {model_path}")
    
    # Guardar estudio
    study_path = MODEL_DIR / "xgb_fe_optuna_balanced_study.joblib"
    joblib.dump(study, study_path)
    print(f"Estudio guardado en: {study_path}")
    
    # Guardar umbral y métricas
    threshold_info = {
        "threshold": thr,
        "custom_score_val": best_score,
        "f3_score_val": f3_val,
        "recall_val": rec_val,
        "precision_val": prec_val,
        "custom_score_test": custom_score_test,
        "f3_score_test": f3_test,
        "recall_test": rec_test,
        "precision_test": prec_test,
        "confusion_matrix": cm.tolist(),
        "false_positive_rate": fp/(fp+tn),
        "false_negative_rate": fn/(fn+tp)
    }
    threshold_path = MODEL_DIR / "xgb_fe_optuna_balanced_threshold.json"
    
    import json
    with open(threshold_path, 'w') as f:
        json.dump(threshold_info, f, indent=2)
    print(f"Información del umbral guardada en: {threshold_path}")
    
    print("\n¡Optimización completada exitosamente!")
    print(f"\nResumen del modelo (versión balanceada):")
    print(f"  - Detecta {rec_test:.1%} de las fallas reales")
    print(f"  - Tiene una precisión de {prec_test:.1%} en las predicciones de falla")
    print(f"  - F3-score de {f3_test:.4f} (prioriza fuertemente detección de fallas)")
    print(f"  - Umbral optimizado: {thr:.4f}")
