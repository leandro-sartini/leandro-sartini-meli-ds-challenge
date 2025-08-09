# Desafío de Ciencia de Datos - Mercado Libre

Este repositorio contiene mi solución al Desafío de Ciencia de Datos propuesto por el equipo de Data & Analytics de Mercado Libre. El proyecto incluye un análisis exploratorio completo de las Ofertas Relámpago y está estructurado siguiendo las mejores prácticas de Data Science.

## 🎯 Objetivo del Proyecto

Analizar el comportamiento de las **Ofertas Relámpago** de Mercado Libre para identificar patrones de éxito, optimizar campañas y generar insights accionables para mejorar la conversión y rentabilidad.

## 📊 Análisis de Ofertas Relámpago - Resumen Ejecutivo

### Dataset Analizado
- **48,746 registros** de ofertas relámpago
- **Período**: Julio 2021 (concentrado en 30/07/2021 - 1,300 ofertas)
- **Variables clave**: Stock involucrado, ventas, duración, categorías, envío

### Principales Insights Descubiertos

#### 🏆 Comportamiento por Verticales
- **Beauty & Health**: Domina en volumen de ventas con tendencia creciente
- **Consumer Electronics**: Mayor ticket medio (2.16x superior a la segunda categoría)
- **Home & Industry** y **Entertainment**: Generan los mayores montos totales

#### ⏰ Patrones Temporales
- **Duración promedio**: 6 horas por oferta relámpago
- **Picos de ventas**: Consistente en Beauty & Health, irregular en CE
- **Evolución diaria**: Patrones similares entre cantidad y monto vendido por hora

#### 💰 Análisis Monetario
- **Ticket medio promedio**: $51 por unidad
- **Stock promedio**: 35 unidades por oferta
- **Tasa de conversión**: ~31% del stock involucrado se vende

#### 🚚 Logística
- **Envío gratuito** es el tipo predominante
- **Origen**: Principalmente "Otros" (categoría agregada para valores nulos)

### 🎯 Recomendaciones Estratégicas
1. **Beauty & Health**: Expandir campañas por su consistencia y volumen
2. **CE**: Enfocar en productos premium con alto ticket medio
3. **Optimización**: Ajustar duración de ofertas según categoría
4. **Stock**: Mejorar estimación basada en patrones históricos por vertical

## 📁 Estructura del Proyecto

```
leandro-sartini-meli-ds-challenge/
├── data/                           # Datos del proyecto
│   ├── 00_raw/                     # Archivos CSV originales
│   ├── 01_processed/               # Datos procesados (Parquet)
│   ├── 02_external/                # Datos externos adicionales
│   └── 03_reports/                 # Reportes y documentación
│
├── notebooks/                      # Notebooks de análisis
│   └── exploratory/
│       └── Ofertas Relampago/      # Análisis completo de ofertas relámpago
│           ├── 0-Primeras-analises.ipynb           # Análisis inicial y estructura
│           ├── 1-Creando-Variables.ipynb           # Feature engineering
│           ├── 2-EDA_Verticales.ipynb              # Análisis por categorías
│           ├── 3-EDA_Nulos_Sin_Movimiento.ipynb    # Análisis de ofertas sin ventas
│           ├── 4-EDA_Nulos_Con_Movimiento.ipynb    # Análisis de inconsistencias
│           └── 5-Hypothesis_Testing.ipynb          # Pruebas estadísticas
│
├── production/                     # Componentes de producción
│   ├── model/                      # Modelos entrenados
│   ├── pipeline/                   # Pipelines de procesamiento
│   └── inference/                  # Scripts de inferencia
│
├── src/                           # Código fuente modular
│   ├── utils.py                   # Utilidades para DataFrames y visualización
│   └── __init__.py                # Configuración del paquete
│
├── github/                        # Configuraciones de CI/CD
│   └── workflows/
│       └── lint-check.yml         # Workflow de linting
│
├── pyproject.toml                 # Configuración del proyecto
├── requirements.txt               # Dependencias
└── README.md                      # Este archivo
```

## 🔧 Utilidades del Proyecto

### DataFrameUtils
Clase con utilidades para manipulación de DataFrames:

```python
from src.utils import DataFrameUtils

# Calcular duración en horas entre columnas datetime
duration_hours = DataFrameUtils.compute_duration_hours(df, 'start_time', 'end_time')

# Validar DataFrame
is_valid = DataFrameUtils.validate_dataframe(df, required_columns=['col1', 'col2'])
```

### VisualizationUtils
Clase con utilidades para visualización:

```python
from src.utils import VisualizationUtils

# Añadir etiquetas a gráficos de barras
VisualizationUtils.add_bar_labels(ax, fmt="{:,.0f}", fontsize=12)

# Con formateador personalizado
def custom_format(val):
    return f"${val:,.0f}K" if val >= 1000 else f"${val:,.0f}"

VisualizationUtils.add_bar_labels(ax, custom_formatter=custom_format)
```

## 📈 Notebooks de Análisis

### 1. **0-Primeras-analises.ipynb** - Análisis Inicial
- Estructura del dataset (48,746 registros, 13 columnas)
- Identificación de tipos de datos y valores nulos
- Resumen estadístico general
- Preparación para feature engineering

### 2. **1-Creando-Variables.ipynb** - Feature Engineering
- Conversión de fechas a datetime
- Creación de variables temporales (duración, día de semana, mes)
- Limpieza de datos (eliminación de columnas redundantes)
- Uso de utilidades personalizadas del proyecto

### 3. **2-EDA_Verticales.ipynb** - Análisis por Categorías
- Evolución diaria de ventas por verticales
- Análisis de comportamiento por categorías
- Visualizaciones de tendencias temporales
- Comparación de métricas entre verticales

### 4. **3-EDA_Nulos_Sin_Movimiento.ipynb** - Ofertas Sin Ventas
- Análisis de ~23k ofertas sin ventas
- Distribución de duración de ofertas sin éxito
- Patrones por categorías y dominios
- Identificación de factores asociados a la falta de ventas

### 5. **4-EDA_Nulos_Con_Movimiento.ipynb** - Inconsistencias
- Análisis de ofertas con movimiento de stock pero sin ventas registradas
- Detección de casos sospechosos
- Validación de integridad de datos
- Identificación de posibles errores en el sistema

### 6. **5-Hypothesis_Testing.ipynb** - Pruebas Estadísticas
- **Test de hipótesis sobre envío gratis**: Impacto en ventas por vertical
- **ANOVA**: Diferencias significativas entre verticales
- **Análisis de efectos**: Tamaño del efecto (Cohen's d)
- **Conclusiones estadísticas**: Beauty & Health muestra diferencias significativas

## 🚀 Instalación y Configuración

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

### Uso de las Utilidades

```python
# En un notebook Jupyter
import sys
sys.path.append('src')

from utils import DataFrameUtils, VisualizationUtils

# Ejemplo de uso
import pandas as pd
import matplotlib.pyplot as plt

# Crear datos de ejemplo
df = pd.DataFrame({
    'start_time': pd.to_datetime(['2023-01-01 10:00:00', '2023-01-01 14:00:00']),
    'end_time': pd.to_datetime(['2023-01-01 12:00:00', '2023-01-01 18:00:00'])
})

# Calcular duración
duration = DataFrameUtils.compute_duration_hours(df, 'start_time', 'end_time')
print(duration)

# Crear gráfico con etiquetas
fig, ax = plt.subplots()
ax.bar(['A', 'B'], [1000, 2000])
VisualizationUtils.add_bar_labels(ax, fmt="{:,.0f}")
plt.show()
```

## 📊 Metodología de Análisis

### Herramientas Utilizadas
- **Python**: pandas, numpy, matplotlib, seaborn
- **Análisis Estadístico**: scipy, statsmodels
- **Visualización**: Gráficos interactivos y estáticos
- **Feature Engineering**: Variables temporales y métricas de rendimiento

### Proceso de Análisis
1. **Exploración Inicial**: Estructura, tipos, valores nulos
2. **Limpieza**: Conversión de fechas, manejo de valores faltantes
3. **Feature Engineering**: Duración, métricas por hora, ticket medio
4. **Análisis por Verticales**: Comportamiento diferenciado
5. **Análisis de Nulos**: Ofertas sin ventas e inconsistencias
6. **Pruebas Estadísticas**: Validación de hipótesis
7. **Insights y Recomendaciones**: Conclusiones accionables

## 🔍 Principales Hallazgos

### Comportamiento de Ventas
- **Beauty & Health** lidera en volumen con tendencia creciente
- **Consumer Electronics** tiene el ticket medio más alto
- **Home & Industry** y **Entertainment** generan mayores montos totales

### Patrones Temporales
- Duración promedio de 6 horas por oferta
- Picos de ventas consistentes en Beauty & Health
- Evolución diaria similar entre cantidad y monto

### Análisis de Nulos
- ~47% de ofertas no tuvieron ventas
- Casos sospechosos con movimiento de stock pero sin ventas registradas
- Patrones específicos por categorías y dominios

### Impacto del Envío Gratis
- Diferencias significativas por vertical
- Beauty & Health: 36 unidades menos con envío gratis
- Entertainment: No hay diferencia estadísticamente significativa

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

## 📈 Estado del Proyecto

- ✅ **Estructura de proyecto organizada**
- ✅ **Análisis Exploratorio Completo** de Ofertas Relámpago
- ✅ **Feature Engineering** implementado
- ✅ **Utilidades personalizadas** (DataFrameUtils, VisualizationUtils)
- ✅ **Análisis de nulos** y casos especiales
- ✅ **Pruebas estadísticas** y validación de hipótesis
- ✅ **Visualizaciones** y reportes
- 🔄 **Componentes de producción** - En desarrollo

## 📞 Contacto

- **Autor**: Leandro Sartini
- **Proyecto**: Desafío de Ciencia de Datos - Mercado Libre
- **Repositorio**: [GitHub](https://github.com/leandro-sartini/leandro-sartini-meli-ds-challenge)

---

*Este proyecto demuestra habilidades avanzadas en análisis exploratorio de datos, feature engineering, visualización y desarrollo de utilidades reutilizables para proyectos de Data Science.*
