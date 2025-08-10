# Documentación del Sistema de Predicción de Fallas - Mercado Libre DS Challenge

## Resumen Ejecutivo

Este documento describe el sistema completo de **mantenimiento predictivo** desplegado para la predicción de fallas de dispositivos en galpones de Full de Mercado Libre. El sistema incluye un dashboard interactivo de Streamlit, un modelo de machine learning optimizado, y una arquitectura modular escalable.

### Objetivo Principal
Desarrollar un sistema que pueda predecir fallas de dispositivos antes de que ocurran, permitiendo mantenimiento preventivo y reduciendo tiempos de inactividad en las operaciones de Mercado Libre.

---

## Arquitectura del Sistema

### Estructura de Carpetas

```
leandro-sartini-meli-ds-challenge/
├── 📁 data/                           # Datos del proyecto
│   ├── 00_raw/                        # Archivos CSV originales
│   ├── 01_processed/                  # Datos procesados (Parquet)
│   └── 02_external/                   # Datos externos adicionales
│
├── 📁 notebooks/                      # Análisis y desarrollo
│   ├── 01_Ofertas Relampago/          # Ejercicio 1: Análisis completo
│   └── 02_Previsión de Falla/         # Ejercicio 3: Análisis completo
│
├── 📁 production/                     # Componentes de producción
│   ├── model/                         # Modelos entrenados
│   └── pipeline/                      # Pipelines de procesamiento
│
├── 📁 src/                           # Código fuente modular
│   ├── dashboard.py                   # Dashboard Streamlit principal
│   ├── feature_engineering.py         # Ingeniería de características
│   ├── models/                        # Modelos de ML
│   ├── scripts/                       # Scripts de entrenamiento
│   ├── utils/                         # Utilidades
│   └── visualization/                 # Visualizaciones
│
├── Dockerfile                      # Configuración Docker
├── docker-compose.yml              # Orquestación Docker
└── requirements.txt                # Dependencias
```

---

## Dashboard Streamlit - Componente Principal

### Descripción General
El dashboard es una aplicación web interactiva desarrollada con **Streamlit** que permite simular dispositivos y obtener predicciones de fallas en tiempo real. Está completamente dockerizado para facilitar el despliegue.

### Cómo Ejecutar el Dashboard

#### Opción 1: Docker (Recomendado)
```bash
# Construir y ejecutar el contenedor
docker-compose up --build

# Para ejecutar en segundo plano
docker-compose up -d --build
```

#### Opción 2: Streamlit Directo
```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar dashboard
streamlit run src/dashboard.py
```

#### Acceso al Dashboard
- **URL Principal**: http://localhost:8501
- **URL Alternativa**: http://127.0.0.1:8501

### Funcionalidades del Dashboard

#### 1. **Simulador de Dispositivos**
- **10 dispositivos predefinidos** para selección
- **Generación de datos sintéticos** realistas
- **Configuración flexible** de días (7-90 días)
- **Patrones de falla personalizables**

#### 2. **Configuración de Simulación**
- **Selección de dispositivo**: SIM_DEVICE_001 a SIM_DEVICE_010
- **Período de datos**: 7 a 90 días
- **Patrón de falla**: Activación/desactivación y día específico
- **Generación en tiempo real**

#### 3. **Visualizaciones Interactivas**
- **Telemetría**: 8 gráficos de atributos en tiempo real
- **Predicción**: Probabilidades y predicciones vs realidad
- **Métricas**: Rendimiento del modelo y estadísticas

---

## Modelo de Machine Learning

### Modelo Principal: Regresión Logística

#### Características del Modelo
- **Tipo**: Regresión Logística con Feature Engineering
- **Archivo**: `production/model/best_model_fe.pkl`
- **Threshold de clasificación**: 0.3
- **Características**: 18 características específicas
- **Pipeline**: StandardScaler + LogisticRegression

#### Métricas de Rendimiento
- **Recall**: 76.2% (detección de fallas)
- **ROC AUC**: 0.826
- **Precision**: Optimizada para el contexto de mantenimiento
- **F1-Score**: Balance entre precision y recall

### Modelos Alternativos Disponibles

#### 1. **XGBoost con Feature Engineering**
- **Archivo**: `xgb_fe_optuna_best.joblib`
- **Threshold**: 0.14
- **Optimización**: Optuna para hiperparámetros
- **Métricas**: Recall 42.9%, Precision 4.8%

#### 2. **XGBoost Balanceado**
- **Archivo**: `xgb_optuna_balanced_best.joblib`
- **Balanceo**: SMOTE + Undersampling
- **Optimización**: Optuna

#### 3. **XGBoost Sin Feature Engineering**
- **Archivo**: `xgb_sin_se_optuna_best.joblib`
- **Comparación**: Baseline sin características adicionales

### Selección del Modelo
El sistema prioriza el **modelo de Regresión Logística** (`best_model_fe.pkl`) por:
- **Simplicidad**: Menor complejidad computacional
- **Interpretabilidad**: Coeficientes lineales claros
- **Rendimiento**: Mejor balance precision-recall
- **Estabilidad**: Menos propenso a overfitting

---

## Ingeniería de Características

### Clase Principal: `SimpleSensorFE`

#### Ubicación
```python
src/feature_engineering.py
```

#### Características Generadas (18 total)

##### 1. **Atributos Originales (8)**
- `attribute1` - attribute9 (excluyendo attribute8)

##### 2. **Diferencias Temporales (8)**
- `attribute1_diff1` - Diferencia de 1 día
- `attribute2_diff1` - Diferencia de 1 día
- `attribute3_diff1` - Diferencia de 1 día
- `attribute4_diff1` - Diferencia de 1 día
- `attribute5_diff1` - Diferencia de 1 día
- `attribute6_diff1` - Diferencia de 1 día
- `attribute7_diff1` - Diferencia de 1 día
- `attribute9_diff1` - Diferencia de 1 día

##### 3. **Medias Móviles (2)**
- `attribute1_roll_mean3` - Media móvil de 3 días
- `attribute7_roll_mean3` - Media móvil de 3 días

#### Configuración del Feature Engineering
```python
SimpleSensorFE(
    do_diff=True,           # Generar diferencias
    do_roll_mean=True,      # Generar medias móviles
    do_roll_std=False,      # No desviación estándar
    do_roll_min_max=False,  # No min/max móviles
    do_lag=False,           # No características de lag
    roll_windows=[3],       # Solo ventana de 3 días
    date_col="date",
    device_col="device",
    attr_prefix="attribute"
)
```

---

## Clases y Componentes Principales

### 1. **DeviceFailurePredictor** (`src/dashboard.py`)

#### Responsabilidades
- **Carga del modelo**: Gestión de modelos entrenados
- **Generación de datos sintéticos**: Simulación realista
- **Feature engineering**: Preparación de características
- **Predicción**: Inferencia en tiempo real

#### Métodos Principales
```python
class DeviceFailurePredictor:
    def __init__(self):
        # Inicialización automática del modelo
    
    def generate_synthetic_device_data(self, device_id: str, days: int):
        # Genera datos sintéticos realistas
    
    def add_failure_pattern(self, df: pd.DataFrame, failure_day: int):
        # Agrega patrones de degradación
    
    def prepare_features(self, df: pd.DataFrame):
        # Prepara 18 características específicas
    
    def predict_failure(self, df: pd.DataFrame):
        # Realiza predicciones con probabilidades
```

### 2. **DashboardVisualizer** (`src/dashboard.py`)

#### Responsabilidades
- **Visualizaciones interactivas**: Gráficos con Plotly
- **Telemetría**: 8 subplots de atributos
- **Predicciones**: Análisis de probabilidades
- **Métricas**: Rendimiento del modelo

#### Métodos Principales
```python
class DashboardVisualizer:
    def plot_device_telemetry(self, df: pd.DataFrame, device_id: str):
        # Gráfico de telemetría con 8 atributos
    
    def plot_failure_prediction(self, df, probabilities, predictions, threshold):
        # Análisis de predicciones vs realidad
```

### 3. **SimpleSensorFE** (`src/feature_engineering.py`)

#### Responsabilidades
- **Feature engineering temporal**: Diferencias y medias móviles
- **Transformación de datos**: Pipeline scikit-learn compatible
- **Análisis de importancia**: Correlación con variable objetivo

#### Métodos Principales
```python
class SimpleSensorFE(BaseEstimator, TransformerMixin):
    def fit(self, df: pd.DataFrame):
        # Identifica columnas de atributos
    
    def transform(self, df: pd.DataFrame):
        # Aplica ingeniería de características
    
    def get_feature_importance_summary(self, df: pd.DataFrame):
        # Análisis de correlación con target
```

### 4. **TimeSeriesFeatureExtractor** (`src/feature_engineering.py`)

#### Responsabilidades
- **Características adicionales**: Tendencias, cíclicas, estadísticas
- **Métodos estáticos**: Utilidades para series temporales
- **Flexibilidad**: Configuración de ventanas y períodos

---

## Configuración Docker

### Dockerfile
```dockerfile
# Imagen base: Python 3.9
FROM python:3.9-slim

# Dependencias del sistema
RUN apt-get update && apt-get install -y gcc g++

# Instalación de dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código fuente y modelos
COPY src/ ./src/
COPY production/ ./production/

# Configuración Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
```

### Docker Compose
```yaml
services:
  failure-prediction-dashboard:
    build: .
    ports:
      - "8501:8501"
    environment:
      - PYTHONPATH=/app
      - STREAMLIT_SERVER_PORT=8501
    volumes:
      - ./data:/app/data:ro
    restart: unless-stopped
```

---

## Datos y Simulación

### Generación de Datos Sintéticos

#### Patrones Base
```python
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
```

#### Patrones de Degradación
- **7 días antes de la falla**: Incremento gradual de atributos problemáticos
- **attribute2**: Incremento 1.5x
- **attribute3**: Incremento 2.0x
- **attribute5**: Incremento 1.3x
- **attribute6**: Decremento 0.8x (indicador de rendimiento)

### Realismo de los Datos
- **Seed determinístico**: Basado en device_id
- **Tendencias temporales**: Patrones sinusoidales
- **Ruido gaussiano**: Variabilidad realista
- **Patrones de falla**: Degradación gradual

---

## Análisis de Rendimiento

### Métricas del Modelo Principal

#### Regresión Logística (best_model_fe.pkl)
- **Threshold**: 0.3
- **Recall**: 76.2%
- **ROC AUC**: 0.826
- **Características**: 18 específicas
- **Pipeline**: StandardScaler + LogisticRegression

#### Comparación con Otros Modelos
| Modelo | Threshold | Recall | Precision | ROC AUC |
|--------|-----------|--------|-----------|---------|
| **Regresión Logística** | 0.3 | **76.2%** | **Optimizada** | **0.826** |
| XGBoost FE | 0.14 | 42.9% | 4.8% | 0.826 |

### Optimización de Threshold
- **Objetivo**: Maximizar recall (detección de fallas)
- **Contexto**: Mantenimiento predictivo
- **Costo**: Falsos positivos vs fallas no detectadas
- **Resultado**: Threshold 0.3 para Regresión Logística

---

## Utilidades y Herramientas

### Visualización (`src/visualization/model_viz.py`)

#### Funciones Disponibles
```python
def graficar_roc_auc(y_real, y_prob, titulo="Curva ROC"):
    # Curva ROC con métricas

def graficar_confusion_matrix(y_true, y_pred, titulo="Matriz de Confusión"):
    # Matriz de confusión con anotaciones

def graficar_metricas_comparacion(metricas_dict, titulo="Comparación de Métricas"):
    # Comparación entre modelos

def crear_reporte_completo(y_true, y_pred, y_prob, titulo="Reporte Completo"):
    # Reporte completo con múltiples visualizaciones
```

### Scripts de Entrenamiento (`src/scripts/`)

#### Scripts Disponibles
- `run_optuna_fe.py`: Entrenamiento con feature engineering
- `run_optuna_fe_balanced.py`: Entrenamiento con balanceo de clases

### Utilidades (`src/utils/`)

#### Funciones Disponibles
- `compare_optuna_results.py`: Comparación de resultados Optuna
- `test_optuna_setup.py`: Validación de configuración Optuna

---

## Flujo de Trabajo del Sistema

### 1. **Inicialización**
```python
# Carga automática del modelo
predictor = DeviceFailurePredictor()
visualizer = DashboardVisualizer()
```

### 2. **Configuración de Simulación**
- Usuario selecciona dispositivo
- Define período de datos (7-90 días)
- Configura patrón de falla
- Ejecuta simulación

### 3. **Generación de Datos**
```python
# Datos sintéticos realistas
df = predictor.generate_synthetic_device_data(device_id, days)

# Patrón de falla (opcional)
if add_failure:
    df = predictor.add_failure_pattern(df, failure_day)
```

### 4. **Feature Engineering**
```python
# Preparación de 18 características
df_features = predictor.prepare_features(df)
```

### 5. **Predicción**
```python
# Inferencia en tiempo real
probabilities, predictions = predictor.predict_failure(df_features)
```

### 6. **Visualización**
```python
# Gráficos interactivos
fig_telemetry = visualizer.plot_device_telemetry(df, device_id)
fig_prediction = visualizer.plot_failure_prediction(df, probabilities, predictions, threshold)
```

---

## Monitoreo y Métricas

### Métricas en Tiempo Real
- **Probabilidad máxima**: Valor más alto de predicción
- **Predicciones positivas**: Número de alertas generadas
- **Fallas reales**: Número de fallas simuladas
- **Threshold del modelo**: Umbral de clasificación

### Información del Modelo
- **Tipo de modelo**: Regresión Logística
- **Threshold**: 0.3
- **Características**: 18 específicas
- **Pipeline**: StandardScaler + LogisticRegression

---

## Despliegue y Escalabilidad

### Configuración de Producción
- **Docker**: Contenedor optimizado
- **Streamlit**: Configuración headless
- **Puerto**: 8501
- **Restart**: unless-stopped

### Escalabilidad
- **Arquitectura modular**: Fácil extensión
- **Modelos intercambiables**: Múltiples modelos disponibles
- **Configuración flexible**: Parámetros ajustables
- **Docker Compose**: Orquestación simple

### Monitoreo
- **Logs**: Docker logs en tiempo real
- **Métricas**: Rendimiento del modelo
- **Visualizaciones**: Gráficos interactivos
- **Alertas**: Predicciones de falla

---

## Mantenimiento y Actualizaciones

### Actualización de Modelos
1. **Entrenar nuevo modelo**: Scripts en `src/scripts/`
2. **Guardar en `production/model/`**: Formato .pkl o .joblib
3. **Actualizar threshold**: Archivo JSON de métricas
4. **Reconstruir Docker**: `docker-compose build --no-cache`

### Configuración de Entorno
```bash
# Variables de entorno
PYTHONPATH=/app
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

### Logs y Debugging
```bash
# Ver logs en tiempo real
docker-compose logs -f

# Acceso al contenedor
docker exec -it meli-failure-dashboard bash

# Reconstruir sin caché
docker-compose build --no-cache
```

---

## Recursos Adicionales

### Documentación
- **README.md**: Guía completa del proyecto
- **Notebooks**: Análisis detallado en Jupyter
- **Comentarios en código**: Documentación inline

### Dependencias Principales
```
streamlit>=1.25.0    # Dashboard web
plotly>=5.0.0        # Visualizaciones interactivas
scikit-learn>=1.0.0  # Machine learning
pandas>=2.0.0        # Manipulación de datos
numpy>=1.19.5        # Computación numérica
joblib>=1.1.0        # Persistencia de modelos
```

### Contacto y Soporte
- **Autor**: Leandro Sartini
- **Proyecto**: Mercado Libre DS Challenge
- **Repositorio**: GitHub con código completo

---

## Conclusiones

Este sistema de predicción de fallas representa una solución completa y robusta para el mantenimiento predictivo en Mercado Libre, con:

- **Dashboard interactivo** para simulación y análisis
- **Modelo optimizado** con 76.2% de recall
- **Arquitectura modular** y escalable
- **Despliegue containerizado** para fácil distribución
- **Visualizaciones profesionales** con Plotly
- **Predicciones en tiempo real** con feature engineering automático

El sistema está listo para producción y puede ser fácilmente extendido para nuevos dispositivos, características adicionales o modelos más complejos según las necesidades del negocio.
