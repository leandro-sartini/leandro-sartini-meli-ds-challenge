"""
Dashboard de Predicción de Fallas de Dispositivos

Este módulo contiene la aplicación Streamlit para el sistema de mantenimiento predictivo
de dispositivos en galpones de Full de Mercado Libre. Permite simular datos de dispositivos
y obtener predicciones de fallas en tiempo real.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Importar módulos del proyecto
import sys
sys.path.append(str(Path(__file__).parent))

from feature_engineering import SimpleSensorFE


class DeviceFailurePredictor:
    """
    Clase principal para la predicción de fallas de dispositivos.
    
    Esta clase maneja la carga del modelo, la ingeniería de características
    y las predicciones de fallas para dispositivos simulados.
    """
    
    def __init__(self):
        """Inicializa el predictor cargando el modelo y configuraciones."""
        self.model = None
        self.feature_engineer = None
        self.threshold = 0.5
        self.model_metrics = {}
        self._load_model()
        self._load_metrics()
        self._initialize_feature_engineer()
    
    def _load_model(self):
        """Carga el modelo entrenado desde el archivo guardado."""
        try:
            # Priorizar el modelo best_model_fe.pkl (Regresión Logística)
            model_path = Path(__file__).parent.parent / "production" / "model" / "best_model_fe.pkl"
            if model_path.exists():
                self.model = joblib.load(model_path)
                st.success("✅ Modelo de Regresión Logística (best_model_fe.pkl) cargado exitosamente")
                # Usar threshold de 0.3 para este modelo
                self.threshold = 0.3
            else:
                # Intentar cargar el modelo alternativo XGBoost
                alt_model_path = Path(__file__).parent.parent / "production" / "model" / "xgb_fe_optuna_best.joblib"
                if alt_model_path.exists():
                    self.model = joblib.load(alt_model_path)
                    st.success("✅ Modelo XGBoost con Feature Engineering cargado exitosamente")
                else:
                    st.error("❌ No se encontró ningún modelo válido")
        except Exception as e:
            st.error(f"❌ Error al cargar el modelo: {str(e)}")
    
    def _load_metrics(self):
        """Carga las métricas del modelo desde el archivo JSON."""
        try:
            metrics_path = Path(__file__).parent.parent / "production" / "model" / "xgb_fe_optuna_threshold.json"
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    self.model_metrics = json.load(f)
                # Solo actualizar threshold si no se ha establecido específicamente (para regresión logística)
                if self.threshold == 0.5:  # valor por defecto
                    self.threshold = self.model_metrics.get('threshold', 0.5)
        except Exception as e:
            st.warning(f"⚠️ No se pudieron cargar las métricas del modelo: {str(e)}")
    
    def _initialize_feature_engineer(self):
        """Inicializa el ingeniero de características."""
        # Configurar feature engineering para coincidir con el modelo entrenado
        # El modelo espera: 8 atributos originales + 10 características específicas
        self.feature_engineer = SimpleSensorFE(
            do_diff=True,
            do_roll_mean=True,
            do_roll_std=False,
            do_roll_min_max=False,
            do_lag=False,
            roll_windows=[3],  # Solo ventana 3 para coincidir con el modelo
            date_col="date",
            device_col="device",
            attr_prefix="attribute"
        )
    
    def generate_synthetic_device_data(self, device_id: str, days: int = 30) -> pd.DataFrame:
        """
        Genera datos sintéticos de un dispositivo para simulación.
        
        Parámetros
        ----------
        device_id : str
            ID del dispositivo
        days : int
            Número de días de datos a generar
            
        Retorna
        -------
        pd.DataFrame
            DataFrame con datos sintéticos del dispositivo
        """
        # Fecha base para la simulación
        base_date = datetime.now() - timedelta(days=days)
        dates = [base_date + timedelta(days=i) for i in range(days)]
        
        # Generar datos sintéticos basados en patrones reales
        np.random.seed(hash(device_id) % 2**32)  # Seed determinístico basado en device_id
        
        # Valores base para cada atributo (basados en análisis del dataset real)
        base_values = {
            'attribute1': 150000000,  # Valores altos típicos
            'attribute2': 50,         # Valores bajos
            'attribute3': 5,          # Valores muy bajos
            'attribute4': 0,          # Mayormente ceros
            'attribute5': 10,         # Valores bajos
            'attribute6': 250000,     # Valores medios
            'attribute7': 0,          # Mayormente ceros
            'attribute9': 0           # Mayormente ceros
        }
        
        # Generar datos con tendencias y ruido
        data = []
        for i, date in enumerate(dates):
            row = {'date': date.strftime('%Y-%m-%d'), 'device': device_id, 'failure': 0}
            
            # Agregar tendencia temporal y ruido
            trend_factor = 1 + 0.1 * np.sin(i * 0.1)  # Tendencia sinusoidal
            noise_factor = np.random.normal(1, 0.05)   # Ruido gaussiano
            
            for attr, base_val in base_values.items():
                if base_val == 0:
                    # Para atributos que son mayormente ceros
                    value = np.random.choice([0, 1, 2], p=[0.9, 0.08, 0.02])
                else:
                    # Para atributos con valores significativos
                    value = int(base_val * trend_factor * noise_factor)
                    value = max(0, value)  # No valores negativos
                
                row[attr] = value
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def add_failure_pattern(self, df: pd.DataFrame, failure_day: int = None) -> pd.DataFrame:
        """
        Agrega un patrón de falla a los datos del dispositivo.
        
        Parámetros
        ----------
        df : pd.DataFrame
            DataFrame con datos del dispositivo
        failure_day : int
            Día en el que ocurre la falla (None para falla aleatoria)
            
        Retorna
        -------
        pd.DataFrame
            DataFrame con patrón de falla agregado
        """
        df = df.copy()
        
        if failure_day is None:
            failure_day = np.random.randint(15, len(df) - 5)  # Fallo entre día 15 y 5 días antes del final
        
        # Marcar el día de falla
        df.loc[failure_day, 'failure'] = 1
        
        # Agregar patrones de degradación antes de la falla
        degradation_start = max(0, failure_day - 7)  # 7 días antes de la falla
        
        for i in range(degradation_start, failure_day):
            # Incrementar valores de atributos que indican problemas
            df.loc[i, 'attribute2'] = int(df.loc[i, 'attribute2'] * 1.5)  # Aumentar attribute2
            df.loc[i, 'attribute3'] = int(df.loc[i, 'attribute3'] * 2.0)  # Aumentar attribute3
            df.loc[i, 'attribute5'] = int(df.loc[i, 'attribute5'] * 1.3)  # Aumentar attribute5
            
            # Decrementar attribute6 (indicador de rendimiento)
            df.loc[i, 'attribute6'] = int(df.loc[i, 'attribute6'] * 0.8)
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara las características para la predicción usando feature engineering.
        
        Parámetros
        ----------
        df : pd.DataFrame
            DataFrame con datos del dispositivo
            
        Retorna
        -------
        pd.DataFrame
            DataFrame con características procesadas
        """
        # Convertir fecha a datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Crear DataFrame con las características exactas que espera el modelo
        df_features = df.copy()
        
        # Características exactas que espera el modelo
        expected_features = [
            'attribute1', 'attribute2', 'attribute3', 'attribute4', 'attribute5',
            'attribute6', 'attribute7', 'attribute9', 'attribute1_diff1',
            'attribute1_roll_mean3', 'attribute2_diff1', 'attribute3_diff1',
            'attribute4_diff1', 'attribute5_diff1', 'attribute6_diff1',
            'attribute7_diff1', 'attribute7_roll_mean3', 'attribute9_diff1'
        ]
        
        # Crear características diff1 (diferencia de 1 día)
        for attr in ['attribute1', 'attribute2', 'attribute3', 'attribute4', 'attribute5', 'attribute6', 'attribute7', 'attribute9']:
            diff_col = f'{attr}_diff1'
            df_features[diff_col] = df_features[attr].diff().fillna(0)
        
        # Crear características roll_mean3 (media móvil de 3 días)
        # Solo attribute1 y attribute7 según el modelo, en el orden específico
        df_features['attribute1_roll_mean3'] = df_features['attribute1'].rolling(window=3, min_periods=1).mean().fillna(0)
        df_features['attribute7_roll_mean3'] = df_features['attribute7'].rolling(window=3, min_periods=1).mean().fillna(0)
        
        # Reordenar las características para que coincidan exactamente con el orden esperado por el modelo
        expected_order = [
            'attribute1', 'attribute2', 'attribute3', 'attribute4', 'attribute5',
            'attribute6', 'attribute7', 'attribute9', 'attribute1_diff1',
            'attribute1_roll_mean3', 'attribute2_diff1', 'attribute3_diff1',
            'attribute4_diff1', 'attribute5_diff1', 'attribute6_diff1',
            'attribute7_diff1', 'attribute7_roll_mean3', 'attribute9_diff1'
        ]
        
        # Mantener las columnas originales (date, device, failure) y reordenar las características
        original_cols = ['date', 'device', 'failure']
        final_cols = original_cols + expected_order
        
        return df_features[final_cols]
    
    def predict_failure(self, df: pd.DataFrame) -> tuple:
        """
        Realiza predicción de falla para los datos del dispositivo.
        
        Parámetros
        ----------
        df : pd.DataFrame
            DataFrame con características del dispositivo
            
        Retorna
        -------
        tuple
            (probabilidades, predicciones)
        """
        if self.model is None:
            return None, None
        
        try:
            # Características exactas que espera el modelo en el orden correcto
            expected_features = [
                'attribute1', 'attribute2', 'attribute3', 'attribute4', 'attribute5',
                'attribute6', 'attribute7', 'attribute9', 'attribute1_diff1',
                'attribute1_roll_mean3', 'attribute2_diff1', 'attribute3_diff1',
                'attribute4_diff1', 'attribute5_diff1', 'attribute6_diff1',
                'attribute7_diff1', 'attribute7_roll_mean3', 'attribute9_diff1'
            ]
            
            # Seleccionar solo las características esperadas en el orden correcto
            X = df[expected_features].fillna(0)
            
            # Realizar predicción
            probabilities = self.model.predict_proba(X)[:, 1]
            predictions = (probabilities >= self.threshold).astype(int)
            
            return probabilities, predictions
            
        except Exception as e:
            st.error(f"Error en predicción: {str(e)}")
            return None, None


class DashboardVisualizer:
    """
    Clase para la visualización de datos y resultados en el dashboard.
    
    Esta clase maneja todas las visualizaciones usando Plotly para crear
    gráficos interactivos y atractivos.
    """
    
    def __init__(self):
        """Inicializa el visualizador con configuraciones de tema."""
        self.color_scheme = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8'
        }
    
    def plot_device_telemetry(self, df: pd.DataFrame, device_id: str) -> go.Figure:
        """
        Crea gráfico de telemetría del dispositivo.
        
        Parámetros
        ----------
        df : pd.DataFrame
            DataFrame con datos del dispositivo
        device_id : str
            ID del dispositivo
            
        Retorna
        -------
        go.Figure
            Gráfico de telemetría
        """
        # Crear subplots para diferentes atributos
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=['Attribute1', 'Attribute2', 'Attribute3', 'Attribute4', 
                          'Attribute5', 'Attribute6', 'Attribute7', 'Attribute9'],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # Atributos a graficar
        attributes = ['attribute1', 'attribute2', 'attribute3', 'attribute4', 
                     'attribute5', 'attribute6', 'attribute7', 'attribute9']
        
        for i, attr in enumerate(attributes):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            # Línea de telemetría
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df[attr],
                    mode='lines+markers',
                    name=attr,
                    line=dict(color=self.color_scheme['primary'], width=2),
                    marker=dict(size=4),
                    showlegend=False
                ),
                row=row, col=col
            )
            
            # Marcar fallas
            failure_dates = df[df['failure'] == 1]['date']
            failure_values = df[df['failure'] == 1][attr]
            
            if len(failure_dates) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=failure_dates,
                        y=failure_values,
                        mode='markers',
                        marker=dict(
                            color=self.color_scheme['danger'],
                            size=10,
                            symbol='x'
                        ),
                        name='Falla',
                        showlegend=False
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title=f'Telemetría del Dispositivo {device_id}',
            height=800,
            showlegend=False,
            template='plotly_white'
        )
        
        return fig
    
    def plot_failure_prediction(self, df: pd.DataFrame, probabilities: np.ndarray, 
                               predictions: np.ndarray, threshold: float) -> go.Figure:
        """
        Crea gráfico de predicción de fallas.
        
        Parámetros
        ----------
        df : pd.DataFrame
            DataFrame con datos del dispositivo
        probabilities : np.ndarray
            Probabilidades de falla
        predictions : np.ndarray
            Predicciones binarias
        threshold : float
            Umbral de clasificación
            
        Retorna
        -------
        go.Figure
            Gráfico de predicción
        """
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Probabilidad de Falla', 'Predicción vs Realidad'],
            vertical_spacing=0.15,
            shared_xaxes=True
        )
        
        # Gráfico de probabilidades
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=probabilities,
                mode='lines+markers',
                name='Probabilidad de Falla',
                line=dict(color=self.color_scheme['primary'], width=3),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        # Línea de umbral
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color=self.color_scheme['warning'],
            annotation_text=f"Umbral: {threshold:.3f}",
            row=1, col=1
        )
        
        # Gráfico de predicciones vs realidad
        colors = []
        for i, (pred, real) in enumerate(zip(predictions, df['failure'])):
            if pred == 1 and real == 1:
                colors.append(self.color_scheme['success'])  # TP
            elif pred == 1 and real == 0:
                colors.append(self.color_scheme['warning'])  # FP
            elif pred == 0 and real == 1:
                colors.append(self.color_scheme['danger'])   # FN
            else:
                colors.append(self.color_scheme['info'])     # TN
        
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=predictions,
                mode='markers',
                name='Predicción',
                marker=dict(
                    color=colors,
                    size=10,
                    symbol='circle'
                )
            ),
            row=2, col=1
        )
        
        # Marcadores de fallas reales
        real_failures = df[df['failure'] == 1]
        if len(real_failures) > 0:
            fig.add_trace(
                go.Scatter(
                    x=real_failures['date'],
                    y=real_failures['failure'],
                    mode='markers',
                    name='Falla Real',
                    marker=dict(
                        color=self.color_scheme['danger'],
                        size=12,
                        symbol='x'
                    )
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title='Análisis de Predicción de Fallas',
            height=600,
            template='plotly_white',
            showlegend=True
        )
        
        return fig
    



def main():
    """
    Función principal del dashboard Streamlit.
    
    Configura la interfaz de usuario y maneja la interacción
    con el usuario para la simulación y predicción de fallas.
    """
    # Configuración de la página
    st.set_page_config(
        page_title="Predicción de Fallas - Mercado Libre",
        page_icon="🔧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Título principal
    st.title("🔧 Sistema de Predicción de Fallas de Dispositivos")
    st.markdown("---")
    
    # Inicializar clases
    predictor = DeviceFailurePredictor()
    visualizer = DashboardVisualizer()
    
    # Sidebar para configuración
    st.sidebar.header("⚙️ Configuración de Simulación")
    
    # Parámetros de simulación
    # Lista de dispositivos predefinidos para simulación
    device_options = [
        "SIM_DEVICE_001",
        "SIM_DEVICE_002", 
        "SIM_DEVICE_003",
        "SIM_DEVICE_004",
        "SIM_DEVICE_005",
        "SIM_DEVICE_006",
        "SIM_DEVICE_007",
        "SIM_DEVICE_008",
        "SIM_DEVICE_009",
        "SIM_DEVICE_010"
    ]
    
    device_id = st.sidebar.selectbox(
        "Seleccionar Dispositivo",
        options=device_options,
        index=0,
        help="Selecciona el dispositivo a simular"
    )
    
    days = st.sidebar.slider(
        "Días de Datos",
        min_value=7,
        max_value=90,
        value=30,
        help="Número de días de datos telemetría a generar"
    )
    
    add_failure = st.sidebar.checkbox(
        "Agregar Patrón de Falla",
        value=True,
        help="Simular una falla en el dispositivo"
    )
    
    failure_day = None
    if add_failure:
        failure_day = st.sidebar.slider(
            "Día de Falla",
            min_value=5,
            max_value=days-5,
            value=days//2,
            help="Día en el que ocurre la falla"
        )
    
    # Botón para generar simulación
    if st.sidebar.button("🚀 Generar Simulación", type="primary"):
        with st.spinner("Generando datos de simulación..."):
            # Generar datos sintéticos
            df = predictor.generate_synthetic_device_data(device_id, days)
            
            if add_failure:
                df = predictor.add_failure_pattern(df, failure_day)
            
            # Preparar características
            df_features = predictor.prepare_features(df)
            
            # Realizar predicción
            probabilities, predictions = predictor.predict_failure(df_features)
            
            if probabilities is not None:
                # Mostrar resultados
                st.success("✅ Simulación completada exitosamente!")
                
                # Métricas de la simulación
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Probabilidad Máxima",
                        f"{np.max(probabilities):.3f}",
                        delta=f"{np.max(probabilities) - predictor.threshold:.3f}"
                    )
                
                with col2:
                    st.metric(
                        "Predicciones Positivas",
                        f"{np.sum(predictions)}",
                        delta=f"{np.sum(predictions) - np.sum(df['failure'])}"
                    )
                
                with col3:
                    st.metric(
                        "Fallas Reales",
                        f"{np.sum(df['failure'])}",
                        delta="0"
                    )
                
                with col4:
                    st.metric(
                        "Threshold del Modelo",
                        f"{predictor.threshold:.3f}",
                        delta="0"
                    )
                
                # Gráficos
                tab1, tab2, tab3 = st.tabs(["📊 Telemetría", "🔮 Predicción", "📈 Métricas del Modelo"])
                
                with tab1:
                    fig_telemetry = visualizer.plot_device_telemetry(df, device_id)
                    st.plotly_chart(fig_telemetry, use_container_width=True)
                
                with tab2:
                    fig_prediction = visualizer.plot_failure_prediction(
                        df, probabilities, predictions, predictor.threshold
                    )
                    st.plotly_chart(fig_prediction, use_container_width=True)
                
                with tab3:
                    st.info("📊 Aquí se mostrarían métricas adicionales del modelo si estuvieran disponibles.")
                
                # Información adicional
                st.markdown("---")
                st.subheader("📋 Información del Modelo")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Configuración del Modelo:**")
                    st.write(f"- **Tipo:** Regresión Logística con Feature Engineering")
                    st.write(f"- **Umbral de Clasificación:** {predictor.threshold:.3f}")
                    st.write(f"- **Características Generadas:** {len([col for col in df_features.columns if col not in ['date', 'device', 'failure']])}")
                
                with col2:
                    st.markdown("**Características del Dataset:**")
                    st.write(f"- **Período:** {days} días")
                    st.write(f"- **Registros:** {len(df)}")
                    st.write(f"- **Fallas Simuladas:** {np.sum(df['failure'])}")
    
    # Información del modelo en la sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Información del Modelo")
    
    # Mostrar información específica del modelo de Regresión Logística
    st.sidebar.markdown("**Modelo Actual:**")
    st.sidebar.markdown("• Regresión Logística")
    st.sidebar.markdown("• Threshold: 0.3")
    st.sidebar.markdown("• 18 características")
    
    st.sidebar.markdown("**Pipeline:**")
    st.sidebar.markdown("• StandardScaler")
    st.sidebar.markdown("• LogisticRegression")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Desarrollado por:** Leandro Sartini")
    st.sidebar.markdown("**Mercado Libre DS Challenge**")


if __name__ == "__main__":
    main()
