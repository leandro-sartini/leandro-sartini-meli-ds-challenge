import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.axes as axes
from typing import Union, Optional, Callable, Any


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


class VisualizationUtils:
    """
    Clase utilitaria para operaciones de visualización y gráficos.
    
    Esta clase contiene métodos para mejorar y personalizar visualizaciones
    creadas con matplotlib y seaborn, proporcionando funcionalidades comunes
    para el análisis exploratorio de datos y presentación de resultados.
    """
    
    @staticmethod
    def add_bar_labels(
        ax: axes.Axes,
        fmt: str = "{:,.0f}",
        fontsize: int = 10,
        rotation: int = 0,
        offset_y: int = 3,
        color: Optional[str] = None,
        weight: str = "normal",
        min_value_threshold: Optional[float] = None,
        custom_formatter: Optional[Callable[[float], str]] = None
    ) -> None:
        """
        Añade etiquetas de valores sobre las barras de un gráfico de barras.
        
        Esta función itera sobre todos los patches (barras) en el eje especificado
        y añade anotaciones con los valores correspondientes. Es especialmente útil
        para gráficos de barras donde se desea mostrar los valores exactos de cada
        barra para mejorar la legibilidad y comprensión de los datos.
        
        Parámetros:
            ax (matplotlib.axes.Axes): El eje de matplotlib donde se encuentran las barras.
            fmt (str, opcional): Formato de cadena para los valores numéricos.
                Por defecto "{:,.0f}" (números enteros con separadores de miles).
            fontsize (int, opcional): Tamaño de fuente para las etiquetas.
                Por defecto 10.
            rotation (int, opcional): Rotación de las etiquetas en grados.
                Por defecto 0 (horizontal).
            offset_y (int, opcional): Desplazamiento vertical de las etiquetas
                desde la parte superior de las barras en puntos.
                Por defecto 3.
            color (str, opcional): Color de las etiquetas. Si es None, usa el
                color por defecto del texto. Por defecto None.
            weight (str, opcional): Peso de la fuente ('normal', 'bold', 'light').
                Por defecto 'normal'.
            min_value_threshold (float, opcional): Valor mínimo para mostrar
                etiquetas. Si el valor de la barra es menor que este umbral,
                no se mostrará la etiqueta. Por defecto None (muestra todas).
            custom_formatter (callable, opcional): Función personalizada para
                formatear los valores. Debe aceptar un float y retornar un str.
                Si se proporciona, ignora el parámetro fmt. Por defecto None.
        
        Retorna:
            None: La función modifica el eje directamente.
        
        Lanza:
            TypeError: Si ax no es un objeto matplotlib.axes.Axes válido.
            ValueError: Si los parámetros de formato no son válidos.
        
        Ejemplos:
            # Uso básico
            fig, ax = plt.subplots()
            ax.bar(['A', 'B', 'C'], [1000, 2000, 3000])
            VisualizationUtils.add_bar_labels(ax)
            
            # Con formato personalizado
            VisualizationUtils.add_bar_labels(ax, fmt="{:.1f}%", fontsize=12)
            
            # Solo mostrar etiquetas para valores > 1000
            VisualizationUtils.add_bar_labels(ax, min_value_threshold=1000)
            
            # Con formateador personalizado
            def custom_format(val):
                return f"${val:,.0f}K" if val >= 1000 else f"${val:,.0f}"
            
            VisualizationUtils.add_bar_labels(ax, custom_formatter=custom_format)
        
        Notas:
            - La función solo añade etiquetas a barras con valores positivos.
            - Las etiquetas se posicionan centradas horizontalmente sobre cada barra.
            - El desplazamiento vertical ayuda a evitar que las etiquetas se
              superpongan con las barras.
            - Para gráficos con muchas barras o valores muy pequeños, considera
              ajustar fontsize o min_value_threshold para mejorar la legibilidad.
        """
        # Validación de parámetros
        if not isinstance(ax, axes.Axes):
            raise TypeError("El parámetro 'ax' debe ser un objeto matplotlib.axes.Axes válido.")
        
        if fontsize <= 0:
            raise ValueError("El parámetro 'fontsize' debe ser un número positivo.")
        
        if not isinstance(fmt, str):
            raise ValueError("El parámetro 'fmt' debe ser una cadena de texto.")
        
        # Iterar sobre todos los patches (barras) en el eje
        for patch in ax.patches:
            # Obtener el valor de la barra
            value = patch.get_height()
            
            # Verificar si el valor cumple con el umbral mínimo
            if min_value_threshold is not None and value < min_value_threshold:
                continue
            
            # Solo añadir etiquetas para valores positivos
            if value > 0:
                # Determinar el texto a mostrar
                if custom_formatter is not None:
                    try:
                        text = custom_formatter(value)
                    except Exception as e:
                        raise ValueError(f"Error en el formateador personalizado: {e}")
                else:
                    try:
                        text = fmt.format(value)
                    except (ValueError, TypeError) as e:
                        raise ValueError(f"Error al formatear el valor {value} con el formato '{fmt}': {e}")
                
                # Calcular la posición de la etiqueta
                x_pos = patch.get_x() + patch.get_width() / 2
                y_pos = value
                
                # Crear la anotación
                ax.annotate(
                    text,
                    xy=(x_pos, y_pos),
                    ha="center",  # Alineación horizontal centrada
                    va="bottom",  # Alineación vertical en la parte inferior
                    fontsize=fontsize,
                    rotation=rotation,
                    color=color,
                    weight=weight,
                    xytext=(0, offset_y),  # Desplazamiento desde la posición
                    textcoords="offset points"  # Unidades en puntos
                )
