# src/visualization/model_viz.py
"""
Módulo de visualización para evaluación de modelos de machine learning.
Contiene funciones para graficar métricas de evaluación como ROC curves, confusion matrices, etc.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo de matplotlib
plt.style.use('default')
sns.set_palette("husl")

def graficar_roc_auc(y_real, y_prob, titulo="Curva ROC", save_path=None):
    """
    Grafica la curva ROC y calcula métricas asociadas.
    
    Parámetros:
    -----------
    y_real : array-like de tamaño (n_muestras,)
        Etiquetas verdaderas {0,1}
    y_prob : array-like de tamaño (n_muestras,)
        Probabilidades predichas para la clase 1
    titulo : str, default="Curva ROC"
        Título del gráfico
    save_path : str or Path, optional
        Ruta para guardar el gráfico
        
    Retorna:
    --------
    dict : Diccionario con métricas calculadas
    """
    fpr, tpr, thr = roc_curve(y_real, y_prob)
    auc = roc_auc_score(y_real, y_prob)

    # Índice de Youden J para encontrar el umbral óptimo
    j = tpr - fpr
    idx_opt = np.argmax(j)
    umbral_opt = thr[idx_opt]
    fpr_opt, tpr_opt = fpr[idx_opt], tpr[idx_opt]

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", lw=1, color='gray', label="Azar (50%)")
    plt.scatter([fpr_opt], [tpr_opt], s=40, zorder=3, color='red',
                label=f"Óptimo @ umbral={umbral_opt:.3f}\nTPR={tpr_opt:.3f}, FPR={fpr_opt:.3f}")

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Tasa de Falsos Positivos (FPR)")
    plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
    plt.title(titulo)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

    return {
        "AUC": auc,
        "Umbral óptimo": float(umbral_opt),
        "TPR óptimo": float(tpr_opt),
        "FPR óptimo": float(fpr_opt)
    }

def graficar_confusion_matrix(y_true, y_pred, titulo="Matriz de Confusión", save_path=None):
    """
    Grafica la matriz de confusión con anotaciones.
    
    Parámetros:
    -----------
    y_true : array-like
        Etiquetas verdaderas
    y_pred : array-like
        Predicciones del modelo
    titulo : str, default="Matriz de Confusión"
        Título del gráfico
    save_path : str or Path, optional
        Ruta para guardar el gráfico
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Falla', 'Falla'],
                yticklabels=['No Falla', 'Falla'])
    plt.title(titulo)
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Predicción')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return cm

def graficar_metricas_comparacion(metricas_dict, titulo="Comparación de Métricas", save_path=None):
    """
    Grafica una comparación de métricas entre diferentes modelos.
    
    Parámetros:
    -----------
    metricas_dict : dict
        Diccionario con métricas por modelo
    titulo : str, default="Comparación de Métricas"
        Título del gráfico
    save_path : str or Path, optional
        Ruta para guardar el gráfico
    """
    modelos = list(metricas_dict.keys())
    metricas = list(metricas_dict[modelos[0]].keys())
    
    fig, axes = plt.subplots(1, len(metricas), figsize=(5*len(metricas), 6))
    if len(metricas) == 1:
        axes = [axes]
    
    for i, metrica in enumerate(metricas):
        valores = [metricas_dict[modelo][metrica] for modelo in modelos]
        axes[i].bar(modelos, valores, color=sns.color_palette("husl", len(modelos)))
        axes[i].set_title(metrica)
        axes[i].set_ylabel('Valor')
        axes[i].tick_params(axis='x', rotation=45)
        
        # Agregar valores en las barras
        for j, v in enumerate(valores):
            axes[i].text(j, v + max(valores)*0.01, f'{v:.3f}', 
                        ha='center', va='bottom')
    
    plt.suptitle(titulo)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def graficar_curvas_precision_recall(y_true, y_prob, titulo="Curva Precision-Recall", save_path=None):
    """
    Grafica la curva Precision-Recall.
    
    Parámetros:
    -----------
    y_true : array-like
        Etiquetas verdaderas
    y_prob : array-like
        Probabilidades predichas
    titulo : str, default="Curva Precision-Recall"
        Título del gráfico
    save_path : str or Path, optional
        Ruta para guardar el gráfico
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, lw=2, label=f'AP = {avg_precision:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(titulo)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return {
        "Average Precision": avg_precision,
        "Precision": precision,
        "Recall": recall,
        "Thresholds": thresholds
    }

def crear_reporte_completo(y_true, y_pred, y_prob, titulo="Reporte Completo del Modelo", 
                          save_dir=None):
    """
    Crea un reporte completo con múltiples visualizaciones.
    
    Parámetros:
    -----------
    y_true : array-like
        Etiquetas verdaderas
    y_pred : array-like
        Predicciones del modelo
    y_prob : array-like
        Probabilidades predichas
    titulo : str, default="Reporte Completo del Modelo"
        Título general del reporte
    save_dir : str or Path, optional
        Directorio para guardar las figuras
    """
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Crear figura con subplots
    fig = plt.figure(figsize=(20, 12))
    
    # ROC Curve
    plt.subplot(2, 3, 1)
    roc_metrics = graficar_roc_auc(y_true, y_prob, "Curva ROC", 
                                  save_path=save_dir/"roc_curve.png" if save_dir else None)
    
    # Precision-Recall Curve
    plt.subplot(2, 3, 2)
    pr_metrics = graficar_curvas_precision_recall(y_true, y_prob, "Curva Precision-Recall",
                                                 save_path=save_dir/"pr_curve.png" if save_dir else None)
    
    # Confusion Matrix
    plt.subplot(2, 3, 3)
    cm = graficar_confusion_matrix(y_true, y_pred, "Matriz de Confusión",
                                  save_path=save_dir/"confusion_matrix.png" if save_dir else None)
    
    # Classification Report
    plt.subplot(2, 3, 4)
    report = classification_report(y_true, y_pred, output_dict=True)
    plt.text(0.1, 0.9, f"Classification Report:\n\n", fontsize=12, fontweight='bold')
    plt.text(0.1, 0.7, f"Precision: {report['1']['precision']:.3f}\n", fontsize=10)
    plt.text(0.1, 0.6, f"Recall: {report['1']['recall']:.3f}\n", fontsize=10)
    plt.text(0.1, 0.5, f"F1-Score: {report['1']['f1-score']:.3f}\n", fontsize=10)
    plt.text(0.1, 0.4, f"Support: {report['1']['support']}", fontsize=10)
    plt.axis('off')
    plt.title("Métricas de Clasificación")
    
    # Threshold Analysis
    plt.subplot(2, 3, 5)
    thresholds = np.linspace(0.01, 0.99, 99)
    precisions = []
    recalls = []
    
    for thresh in thresholds:
        y_pred_thresh = (y_prob >= thresh).astype(int)
        from sklearn.metrics import precision_score, recall_score
        precisions.append(precision_score(y_true, y_pred_thresh, zero_division=0))
        recalls.append(recall_score(y_true, y_pred_thresh, zero_division=0))
    
    plt.plot(thresholds, precisions, label='Precision', lw=2)
    plt.plot(thresholds, recalls, label='Recall', lw=2)
    plt.xlabel('Umbral')
    plt.ylabel('Valor')
    plt.title('Análisis de Umbral')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Feature Importance (placeholder)
    plt.subplot(2, 3, 6)
    plt.text(0.5, 0.5, 'Feature Importance\n(Requiere modelo entrenado)', 
             ha='center', va='center', fontsize=12)
    plt.title('Importancia de Features')
    plt.axis('off')
    
    plt.suptitle(titulo, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(save_dir/"reporte_completo.png", dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return {
        "roc_metrics": roc_metrics,
        "pr_metrics": pr_metrics,
        "confusion_matrix": cm,
        "classification_report": report
    }
