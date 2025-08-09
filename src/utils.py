import pandas as pd
from typing import Union


class DataFrameUtils:
    """
    Clase utilitaria para operaciones comunes con DataFrames de pandas.
    
    Esta clase contiene métodos para manipular y analizar DataFrames,
    especialmente útiles para análisis de datos y procesamiento de fechas.
    """
    
    @staticmethod
    def compute_duration_hours(df: pd.DataFrame, start_col: str, end_col: str) -> pd.Series:
        """
        Calcula la duración en horas entre dos columnas de tipo datetime en un DataFrame.

        Parámetros:
            df (pd.DataFrame): DataFrame que contiene las columnas de fecha y hora.
            start_col (str): Nombre de la columna con la fecha/hora de inicio.
            end_col (str): Nombre de la columna con la fecha/hora de fin.

        Retorna:
            pd.Series: Serie de pandas con la duración en horas (tipo float) para cada fila.
        
        Lanza:
            TypeError: Si alguna de las columnas no es de tipo datetime.
            KeyError: Si alguna de las columnas no existe en el DataFrame.
        """
        # Verificar que las columnas existen
        if start_col not in df.columns:
            raise KeyError(f"La columna '{start_col}' no existe en el DataFrame.")
        if end_col not in df.columns:
            raise KeyError(f"La columna '{end_col}' no existe en el DataFrame.")
        
        # Verificar que las columnas son de tipo datetime
        if not pd.api.types.is_datetime64_any_dtype(df[start_col]):
            raise TypeError(f"La columna '{start_col}' debe ser de tipo datetime.")
        if not pd.api.types.is_datetime64_any_dtype(df[end_col]):
            raise TypeError(f"La columna '{end_col}' debe ser de tipo datetime.")

        return (df[end_col] - df[start_col]).dt.total_seconds() / 3600
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: list = None) -> bool:
        """
        Valida que un DataFrame cumple con ciertos criterios básicos.
        
        Parámetros:
            df (pd.DataFrame): DataFrame a validar.
            required_columns (list, opcional): Lista de columnas requeridas.
            
        Retorna:
            bool: True si el DataFrame es válido, False en caso contrario.
        """
        if not isinstance(df, pd.DataFrame):
            return False
        
        if df.empty:
            return False
            
        if required_columns:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                return False
                
        return True
