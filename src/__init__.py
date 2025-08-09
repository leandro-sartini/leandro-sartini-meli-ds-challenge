"""
Módulo principal del proyecto leandro-sartini-meli-ds-challenge.

Este módulo contiene utilidades y herramientas para el análisis de datos,
procesamiento de DataFrames e ingeniería de características para series temporales.
"""

from .utils import DataFrameUtils, VisualizationUtils
from .feature_engineering import SimpleSensorFE, TimeSeriesFeatureExtractor

__all__ = [
    'DataFrameUtils', 
    'VisualizationUtils',
    'SimpleSensorFE',
    'TimeSeriesFeatureExtractor'
]
__version__ = '1.0.0'
