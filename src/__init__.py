"""
Main package for the ML project.
"""

# Import key functions from modules
from .feature_engineering import SimpleSensorFE, TimeSeriesFeatureExtractor
from .DFutils import *

# Import from submodules
from .models import *
from .visualization import *
from .utils import *

__all__ = [
    # Feature Engineering
    'SimpleSensorFE',
    'TimeSeriesFeatureExtractor',
    
    # Visualization
    'graficar_roc_auc',
    'graficar_confusion_matrix',
    'graficar_metricas_comparacion',
    'graficar_curvas_precision_recall',
    'crear_reporte_completo',
]
__version__ = '1.0.0'
