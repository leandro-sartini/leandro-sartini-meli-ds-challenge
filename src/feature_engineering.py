"""
Módulo de ingeniería de características para series temporales.

Este módulo contiene clases y utilidades para la creación de características
especializadas en datos de series temporales, especialmente para mantenimiento
predictivo y análisis de sensores.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Union
from sklearn.base import BaseEstimator, TransformerMixin
import warnings


class SimpleSensorFE(BaseEstimator, TransformerMixin):
    """
    Ingeniería de características mínima por dispositivo con opciones:
      - attr_diff1: valor actual - valor anterior (por dispositivo)
      - attr_roll_mean{W}: media móvil sobre una ventana W aplicada a shift(1) para evitar fuga de información
      - attr_roll_std{W}: desviación estándar móvil
      - attr_roll_min{W}, attr_roll_max{W}: valores mínimo y máximo móviles
      - attr_lag{W}: valores con lag W

    Parámetros
    ----------
    do_diff : bool
        Si es True, crea attr_diff1 para cada atributo.
    do_roll_mean : bool
        Si es True, crea características attr_roll_mean{W} para cada atributo.
    do_roll_std : bool
        Si es True, crea características attr_roll_std{W} para cada atributo.
    do_roll_min_max : bool
        Si es True, crea características attr_roll_min{W} y attr_roll_max{W} para cada atributo.
    do_lag : bool
        Si es True, crea características attr_lag{W} para cada atributo.
    roll_windows : List[int]
        Tamaños de ventana para las características móviles.
    lag_windows : List[int]
        Tamaños de lag para las características de retraso.
    date_col, device_col, attr_prefix : str
        Nombres de las columnas y prefijo de los atributos.
    sort_in_transform : bool
        Ordenar por (dispositivo, fecha) antes de calcular las características.
    diff_first_zero : bool
        Si es True, la primera diferencia por dispositivo se convierte en 0; si es False, se mantiene como NaN.
    roll_fill_current : bool
        Si es True, los NaN restantes en las medias móviles se rellenan con el valor actual.
    """
    
    def __init__(
        self,
        do_diff: bool = True,
        do_roll_mean: bool = True,
        do_roll_std: bool = False,
        do_roll_min_max: bool = False,
        do_lag: bool = False,
        roll_windows: Optional[List[int]] = None,
        lag_windows: Optional[List[int]] = None,
        date_col: str = "date",
        device_col: str = "device",
        attr_prefix: str = "attribute",
        sort_in_transform: bool = True,
        diff_first_zero: bool = True,
        roll_fill_current: bool = True,
    ):
        self.do_diff = do_diff
        self.do_roll_mean = do_roll_mean
        self.do_roll_std = do_roll_std
        self.do_roll_min_max = do_roll_min_max
        self.do_lag = do_lag
        self.roll_windows = roll_windows if roll_windows is not None else [3, 7]
        self.lag_windows = lag_windows if lag_windows is not None else [1, 2, 3]
        self.date_col = date_col
        self.device_col = device_col
        self.attr_prefix = attr_prefix
        self.sort_in_transform = sort_in_transform
        self.diff_first_zero = diff_first_zero
        self.roll_fill_current = roll_fill_current

        self._attr_cols: List[str] = []
        self._fitted = False
        self._feature_names: List[str] = []

    def _infer_attr_cols(self, df: pd.DataFrame) -> List[str]:
        """Infiere las columnas de atributos basándose en el prefijo."""
        return [c for c in df.columns if c.startswith(self.attr_prefix)]

    def fit(self, df: pd.DataFrame):
        """Ajusta el transformador identificando las columnas de atributos."""
        if self.date_col not in df.columns or self.device_col not in df.columns:
            raise ValueError(f"Se necesitan las columnas '{self.device_col}' y '{self.date_col}'.")
        
        self._attr_cols = self._infer_attr_cols(df)
        if not self._attr_cols:
            raise ValueError(f"No se encontraron columnas que empiecen con '{self.attr_prefix}'.")
        
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforma el DataFrame aplicando la ingeniería de características."""
        if not self._fitted:
            self.fit(df)

        out = df.copy()
        
        # Convertir columna de fecha si es necesario
        if not pd.api.types.is_datetime64_any_dtype(out[self.date_col]):
            out[self.date_col] = pd.to_datetime(out[self.date_col], errors="coerce")

        if self.sort_in_transform:
            out.sort_values([self.device_col, self.date_col], inplace=True, kind="mergesort")

        creadas = []

        for attr in self._attr_cols:
            g = out.groupby(self.device_col)[attr]

            # Diferencia primera
            if self.do_diff:
                diff_col = f"{attr}_diff1"
                diff_vals = g.diff()
                if self.diff_first_zero:
                    diff_vals = diff_vals.fillna(0)
                out[diff_col] = diff_vals
                creadas.append(diff_col)

            # Características de lag
            if self.do_lag:
                for lag in self.lag_windows:
                    lag_col = f"{attr}_lag{lag}"
                    out[lag_col] = g.shift(lag)
                    creadas.append(lag_col)

            # Media móvil en shift(1) para evitar fuga de información
            if self.do_roll_mean:
                base = g.shift(1)
                for W in self.roll_windows:
                    mcol = f"{attr}_roll_mean{W}"
                    out[mcol] = base.rolling(W, min_periods=1).mean()
                    if self.roll_fill_current:
                        out[mcol] = out[mcol].fillna(out[attr])
                    creadas.append(mcol)

            # Desviación estándar móvil
            if self.do_roll_std:
                base = g.shift(1)
                for W in self.roll_windows:
                    std_col = f"{attr}_roll_std{W}"
                    out[std_col] = base.rolling(W, min_periods=1).std()
                    if self.roll_fill_current:
                        out[std_col] = out[std_col].fillna(0)
                    creadas.append(std_col)

            # Mínimo y máximo móvil
            if self.do_roll_min_max:
                base = g.shift(1)
                for W in self.roll_windows:
                    min_col = f"{attr}_roll_min{W}"
                    max_col = f"{attr}_roll_max{W}"
                    out[min_col] = base.rolling(W, min_periods=1).min()
                    out[max_col] = base.rolling(W, min_periods=1).max()
                    if self.roll_fill_current:
                        out[min_col] = out[min_col].fillna(out[attr])
                        out[max_col] = out[max_col].fillna(out[attr])
                    creadas.extend([min_col, max_col])

        self._feature_names = creadas
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajusta el transformador y aplica la transformación."""
        return self.fit(df).transform(df)

    @property
    def feature_names_(self) -> List[str]:
        """Retorna la lista de nombres de características creadas."""
        return list(self._feature_names)

    def get_feature_importance_summary(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Calcula un resumen de importancia de características basado en correlación con la variable objetivo.
        
        Parámetros
        ----------
        df : pd.DataFrame
            DataFrame con las características creadas y la variable objetivo.
            
        Retorna
        -------
        Dict[str, Dict[str, float]]
            Diccionario con estadísticas de correlación por atributo.
        """
        if not self._fitted:
            raise ValueError("El transformador debe estar ajustado antes de usar este método.")
        
        # Aplicar transformación si no se ha hecho
        if not any(col in df.columns for col in self._feature_names):
            df = self.transform(df)
        
        # Buscar columna objetivo (failure o similar)
        target_cols = [col for col in df.columns if 'failure' in col.lower() or 'target' in col.lower()]
        if not target_cols:
            warnings.warn("No se encontró columna objetivo. Usando la primera columna numérica.")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            target_cols = [numeric_cols[0]] if len(numeric_cols) > 0 else []
        
        if not target_cols:
            return {}
        
        target_col = target_cols[0]
        summary = {}
        
        for attr in self._attr_cols:
            attr_features = [col for col in self._feature_names if col.startswith(attr)]
            if attr_features:
                correlations = {}
                for feat in attr_features:
                    if feat in df.columns and target_col in df.columns:
                        corr = df[feat].corr(df[target_col])
                        if not pd.isna(corr):
                            correlations[feat] = abs(corr)
                
                if correlations:
                    summary[attr] = {
                        'max_correlation': max(correlations.values()),
                        'mean_correlation': np.mean(list(correlations.values())),
                        'feature_count': len(correlations)
                    }
        
        return summary


class TimeSeriesFeatureExtractor:
    """
    Extractores de características adicionales para series temporales.
    
    Esta clase proporciona métodos estáticos para crear características
    específicas de series temporales que pueden ser útiles para
    mantenimiento predictivo.
    """
    
    @staticmethod
    def create_trend_features(
        df: pd.DataFrame,
        device_col: str = "device",
        date_col: str = "date",
        attr_cols: Optional[List[str]] = None,
        windows: List[int] = [7, 14, 30]
    ) -> pd.DataFrame:
        """
        Crea características de tendencia usando regresión lineal en ventanas móviles.
        
        Parámetros
        ----------
        df : pd.DataFrame
            DataFrame con los datos.
        device_col : str
            Nombre de la columna de dispositivo.
        date_col : str
            Nombre de la columna de fecha.
        attr_cols : List[str], opcional
            Lista de columnas de atributos. Si es None, se infieren automáticamente.
        windows : List[int]
            Tamaños de ventana para calcular las tendencias.
            
        Retorna
        -------
        pd.DataFrame
            DataFrame con las características de tendencia añadidas.
        """
        from scipy import stats
        
        if attr_cols is None:
            attr_cols = [col for col in df.columns if col.startswith('attribute')]
        
        out = df.copy()
        
        for attr in attr_cols:
            for window in windows:
                trend_col = f"{attr}_trend_{window}"
                
                def calculate_trend(group):
                    if len(group) < window:
                        return pd.Series([np.nan] * len(group), index=group.index)
                    
                    trends = []
                    for i in range(len(group)):
                        start_idx = max(0, i - window + 1)
                        window_data = group.iloc[start_idx:i+1]
                        if len(window_data) >= 2:
                            x = np.arange(len(window_data))
                            y = window_data.values
                            slope, _, _, _, _ = stats.linregress(x, y)
                            trends.append(slope)
                        else:
                            trends.append(np.nan)
                    
                    return pd.Series(trends, index=group.index)
                
                out[trend_col] = out.groupby(device_col)[attr].transform(calculate_trend)
        
        return out
    
    @staticmethod
    def create_cyclical_features(
        df: pd.DataFrame,
        date_col: str = "date",
        periods: List[str] = ["day", "week", "month"]
    ) -> pd.DataFrame:
        """
        Crea características cíclicas basadas en la fecha.
        
        Parámetros
        ----------
        df : pd.DataFrame
            DataFrame con los datos.
        date_col : str
            Nombre de la columna de fecha.
        periods : List[str]
            Períodos para crear características cíclicas.
            
        Retorna
        -------
        pd.DataFrame
            DataFrame con las características cíclicas añadidas.
        """
        out = df.copy()
        
        if not pd.api.types.is_datetime64_any_dtype(out[date_col]):
            out[date_col] = pd.to_datetime(out[date_col])
        
        for period in periods:
            if period == "day":
                out[f"{period}_sin"] = np.sin(2 * np.pi * out[date_col].dt.dayofyear / 365.25)
                out[f"{period}_cos"] = np.cos(2 * np.pi * out[date_col].dt.dayofyear / 365.25)
            elif period == "week":
                out[f"{period}_sin"] = np.sin(2 * np.pi * out[date_col].dt.dayofweek / 7)
                out[f"{period}_cos"] = np.cos(2 * np.pi * out[date_col].dt.dayofweek / 7)
            elif period == "month":
                out[f"{period}_sin"] = np.sin(2 * np.pi * out[date_col].dt.month / 12)
                out[f"{period}_cos"] = np.cos(2 * np.pi * out[date_col].dt.month / 12)
        
        return out
    
    @staticmethod
    def create_statistical_features(
        df: pd.DataFrame,
        device_col: str = "device",
        attr_cols: Optional[List[str]] = None,
        windows: List[int] = [7, 14, 30]
    ) -> pd.DataFrame:
        """
        Crea características estadísticas adicionales.
        
        Parámetros
        ----------
        df : pd.DataFrame
            DataFrame con los datos.
        device_col : str
            Nombre de la columna de dispositivo.
        attr_cols : List[str], opcional
            Lista de columnas de atributos.
        windows : List[int]
            Tamaños de ventana para las estadísticas.
            
        Retorna
        -------
        pd.DataFrame
            DataFrame con las características estadísticas añadidas.
        """
        if attr_cols is None:
            attr_cols = [col for col in df.columns if col.startswith('attribute')]
        
        out = df.copy()
        
        for attr in attr_cols:
            for window in windows:
                # Coeficiente de variación
                cv_col = f"{attr}_cv_{window}"
                out[cv_col] = out.groupby(device_col)[attr].rolling(
                    window, min_periods=1
                ).apply(lambda x: x.std() / x.mean() if x.mean() != 0 else 0).reset_index(0, drop=True)
                
                # Rango intercuartil
                iqr_col = f"{attr}_iqr_{window}"
                out[iqr_col] = out.groupby(device_col)[attr].rolling(
                    window, min_periods=1
                ).quantile(0.75).reset_index(0, drop=True) - out.groupby(device_col)[attr].rolling(
                    window, min_periods=1
                ).quantile(0.25).reset_index(0, drop=True)
        
        return out
