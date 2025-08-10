# Desafío de Ciencia de Datos - Mercado Libre

Este repositorio contiene mi solución al Desafío de Ciencia de Datos propuesto por el equipo de Data & Analytics de Mercado Libre. El proyecto incluye análisis completos de **Ofertas Relámpago** y **Previsión de Falla** de dispositivos, estructurado siguiendo las mejores prácticas de Data Science.

## 🚀 Dashboard de Predicción de Fallas

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
# Crear y activar entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar dashboard
streamlit run src/dashboard.py
```

#### Acceso al Dashboard
Una vez ejecutado, abre tu navegador en:
- **URL Principal**: http://localhost:8501
- **URL Alternativa**: http://127.0.0.1:8501

### Características del Dashboard
- ✅ **Simulador de Dispositivos**: 10 dispositivos predefinidos para selección
- ✅ **Configuración Flexible**: Días de datos (7-90) y día de falla personalizable
- ✅ **Visualizaciones Interactivas**: Gráficos de telemetría y predicciones con Plotly
- ✅ **Predicciones en Tiempo Real**: Modelo de Regresión Logística con threshold 0.3
- ✅ **Feature Engineering Automático**: 18 características específicas del modelo
- ✅ **Dockerizado**: Fácil despliegue y distribución

### Cómo Usar el Simulador
1. **Seleccionar Dispositivo**: Elige uno de los 10 dispositivos disponibles
2. **Configurar Días**: Define cuántos días de datos generar (7-90 días)
3. **Configurar Falla**: Activa/desactiva el patrón de falla y elige el día
4. **Generar Simulación**: Haz clic en "🚀 Generar Simulación"
5. **Analizar Resultados**: Revisa telemetría, predicciones y métricas

### Solución de Problemas Docker
```bash
# Ver logs en tiempo real
docker-compose logs -f

# Detener el contenedor
docker-compose down

# Reconstruir sin caché
docker-compose build --no-cache
docker-compose up
```

## Objetivos del Proyecto

### Ejercicio 1: Análisis de Ofertas Relámpago
Analizar el comportamiento de las **Ofertas Relámpago** para identificar patrones de éxito, optimizar campañas y generar insights accionables para mejorar la conversión y rentabilidad.

### Ejercicio 3: Previsión de Falla
Desarrollar un sistema de **mantenimiento predictivo** para dispositivos en galpones de Full, utilizando técnicas de machine learning para predecir fallas antes de que ocurran.

## Resumen Ejecutivo

### Ejercicio 1: Ofertas Relámpago
- **Dataset**: 48,746 registros de ofertas relámpago
- **Período**: Julio 2021 (concentrado en 30/07/2021)
- **Insights principales**:
  - **Beauty & Health**: Domina en volumen de ventas con tendencia creciente
  - **Consumer Electronics**: Mayor ticket medio (2.16x superior)
  - **Duración promedio**: 6 horas por oferta
  - **Tasa de conversión**: ~31% del stock involucrado se vende

### Ejercicio 3: Previsión de Falla
- **Dataset**: 124,494 registros de telemetría de dispositivos
- **Período**: 11 meses de datos históricos
- **Desafío**: Desbalance extremo (0.94% de fallas)
- **Mejor modelo**: Regresión Logística con threshold 0.3
- **Resultados**:
  - **Recall**: 76.2% de detección de fallas
  - **ROC AUC**: 0.826
  - **Feature Engineering**: Mejoró significativamente el rendimiento

## Estructura del Proyecto

```
leandro-sartini-meli-ds-challenge/
├── data/                           # Datos del proyecto
│   ├── 00_raw/                     # Archivos CSV originales
│   ├── 01_processed/               # Datos procesados (Parquet)
│   └── 02_external/                # Datos externos adicionales
│
├── notebooks/                      # Notebooks de análisis
│   ├── Ofertas Relampago/          # Ejercicio 1: Análisis completo
│   │   ├── 0-Primeras-analises.ipynb
│   │   ├── 1-Creando-Variables.ipynb
│   │   ├── 2-EDA_Verticales.ipynb
│   │   ├── 3-EDA_Nulos_Sin_Movimiento.ipynb
│   │   ├── 4-EDA_Nulos_Con_Movimiento.ipynb
│   │   └── 5-Hypothesis_Testing.ipynb
│   │
│   ├── Previsión de Falla/         # Ejercicio 3: Análisis completo
│   │   ├── 0-EDA.ipynb
│   │   ├── 1-Feature-Engineering.ipynb
│   │   ├── 2-Model-Building-Sin-FE.ipynb
│   │   └── 3-Model-Building-FE.ipynb
│   │
│   └── Resumen Ejercicios/         # Resúmenes ejecutivos
│       ├── Ejercicio-1-Ofertas-Relampago.ipynb
│       └── Ejercicio-3-Previsión-de-Falla.ipynb
│
├── production/                     # Componentes de producción
│   ├── model/                      # Modelos entrenados
│   ├── pipeline/                   # Pipelines de procesamiento
│   └── inference/                  # Scripts de inferencia
│
├── src/                           # Código fuente modular
│   ├── config/                    # Configuraciones
│   ├── data/                      # Procesamiento de datos
│   ├── dashboard.py               # 🎯 Dashboard Streamlit principal
│   ├── feature_engineering.py     # Feature engineering para fallas
│   ├── models/                    # Modelos de ML
│   ├── scripts/                   # Scripts de entrenamiento
│   ├── utils/                     # Utilidades
│   ├── utils.py                   # Utilidades generales
│   └── visualization/             # Visualizaciones
│
├── github/                        # Configuraciones de CI/CD
│   └── workflows/
│       └── lint-check.yml         # Workflow de linting
│
├── Dockerfile                     # 🐳 Configuración Docker
├── docker-compose.yml            # 🐳 Orquestación Docker
├── pyproject.toml                 # Configuración del proyecto
├── requirements.txt               # Dependencias
└── README.md                      # Este archivo
```

## Componentes Principales

### Utilidades del Proyecto
- **DataFrameUtils**: Manipulación de DataFrames y validaciones
- **VisualizationUtils**: Utilidades para visualización profesional
- **SimpleSensorFE**: Feature engineering para datos de telemetría

### Modelos de Machine Learning
- **Regresión Logística**: Mejor rendimiento para previsión de fallas
- **XGBoost**: Optimizado con Optuna
- **Árboles de Decisión**: Comparación de modelos

### Dashboard de Predicción de Fallas
- **Streamlit App**: Interfaz web interactiva
- **Simulador de Dispositivos**: Generación de datos sintéticos realistas
- **Visualizaciones**: Gráficos interactivos con Plotly
- **Predicciones en Tiempo Real**: Modelo de Regresión Logística
- **Feature Engineering Automático**: 18 características específicas
- **Dockerizado**: Fácil despliegue y distribución

## Notebooks de Análisis

### Ejercicio 1: Ofertas Relámpago

#### Notebooks de Análisis Detallado
1. **0-Primeras-analises.ipynb** - Análisis inicial y estructura
2. **1-Creando-Variables.ipynb** - Feature engineering
3. **2-EDA_Verticales.ipynb** - Análisis por categorías
4. **3-EDA_Nulos_Sin_Movimiento.ipynb** - Ofertas sin ventas
5. **4-EDA_Nulos_Con_Movimiento.ipynb** - Inconsistencias
6. **5-Hypothesis_Testing.ipynb** - Pruebas estadísticas

#### Resumen Ejecutivo
- **Ejercicio-1-Ofertas-Relampago.ipynb**: Resumen completo con visualizaciones

### Ejercicio 3: Previsión de Falla

#### Notebooks de Análisis Detallado
1. **0-EDA.ipynb** - Análisis exploratorio de datos
2. **1-Feature-Engineering.ipynb** - Creación de features temporales
3. **2-Model-Building-Sin-FE.ipynb** - Modelos sin feature engineering
4. **3-Model-Building-FE.ipynb** - Modelos con feature engineering

#### Resumen Ejecutivo
- **Ejercicio-3-Previsión-de-Falla.ipynb**: Resumen completo con visualizaciones

## Instalación y Configuración

### Prerrequisitos
- Python 3.10+
- pip

### Instalación

```bash
# Clonar el repositorio
git clone https://github.com/leandro-sartini/leandro-sartini-meli-ds-challenge.git
cd leandro-sartini-meli-ds-challenge

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt
```

### Uso Rápido

```python
# Ejercicio 1: Análisis de Ofertas Relámpago
import sys
sys.path.append('src')
from utils import DataFrameUtils, VisualizationUtils

# Ejercicio 3: Previsión de Falla
from feature_engineering import SimpleSensorFE
from models.xgb_fe_optuna import XGBOptunaModel
```

## 📊 Metodología de Análisis

### Ejercicio 1: Ofertas Relámpago
1. **Exploración Inicial**: Estructura, tipos, valores nulos
2. **Feature Engineering**: Duración, métricas por hora, ticket medio
3. **Análisis por Verticales**: Comportamiento diferenciado
4. **Análisis de Nulos**: Ofertas sin ventas e inconsistencias
5. **Pruebas Estadísticas**: Validación de hipótesis
6. **Insights y Recomendaciones**: Conclusiones accionables

### Ejercicio 3: Previsión de Falla
1. **EDA**: Análisis del desbalance extremo (0.94% fallas)
2. **Feature Engineering**: Rolling statistics y diferencias temporales
3. **Balanceo de Clases**: SMOTE + Undersampling
4. **Evaluación de Modelos**: Regresión Logística, Árboles, XGBoost
5. **Optimización de Threshold**: Maximización de recall
6. **Validación**: Matriz de confusión y métricas de rendimiento

## 🔍 Principales Hallazgos

### Ejercicio 1: Ofertas Relámpago
- **Beauty & Health** lidera en volumen con tendencia creciente
- **Consumer Electronics** tiene el ticket medio más alto (2.16x superior)
- **Duración promedio**: 6 horas por oferta
- **Tasa de conversión**: ~31% del stock involucrado se vende
- **Envío gratuito**: Impacto significativo en Beauty & Health

### Ejercicio 3: Previsión de Falla
- **Desbalance extremo**: Solo 0.94% de registros con fallas
- **Feature Engineering crucial**: Mejoró significativamente el recall
- **Modelos simples superiores**: Regresión Logística mejor que modelos complejos
- **Threshold crítico**: 0.3 optimizado para maximizar detección
- **Recall de 76.2%**: Detecta el 76.2% de las fallas potenciales

## 🛠️ Desarrollo y Contribución

### Linting y Formato
```bash
# Formatear código
black .

# Verificar estilo
flake8 .
```

### Workflow de Desarrollo
- **Ramas de características**: `feature/nombre-caracteristica`
- **Pull Requests**: Revisión obligatoria antes del merge
- **CI/CD**: GitHub Actions para linting automático

## Estado del Proyecto

### Ejercicio 1: Ofertas Relámpago
- ✅ **Análisis Exploratorio Completo**
- ✅ **Feature Engineering** implementado
- ✅ **Utilidades personalizadas** (DataFrameUtils, VisualizationUtils)
- ✅ **Análisis de nulos** y casos especiales
- ✅ **Pruebas estadísticas** y validación de hipótesis
- ✅ **Resumen ejecutivo** con visualizaciones

### Ejercicio 3: Previsión de Falla
- ✅ **Análisis Exploratorio de Datos**
- ✅ **Feature Engineering** temporal
- ✅ **Balanceo de clases** con SMOTE
- ✅ **Evaluación de múltiples modelos**
- ✅ **Optimización de hiperparámetros** con Optuna
- ✅ **Resumen ejecutivo** con visualizaciones
- ✅ **Modelo de producción** (Regresión Logística)
- ✅ **Dashboard Streamlit** con simulador de dispositivos
- ✅ **Dockerización completa** para despliegue fácil

## Contacto

- **Autor**: Leandro Sartini
- **Proyecto**: Desafío de Ciencia de Datos - Mercado Libre
- **Repositorio**: [GitHub](https://github.com/leandro-sartini/leandro-sartini-meli-ds-challenge)

---

*Este proyecto demuestra habilidades avanzadas en análisis exploratorio de datos, machine learning, feature engineering, visualización y desarrollo de sistemas de mantenimiento predictivo para proyectos de Data Science.*
