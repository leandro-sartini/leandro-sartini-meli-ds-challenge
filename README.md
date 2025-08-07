# Desafío de Ciencia de Datos - Mercado Libre

Este repositorio contiene mi solución al Desafío de Ciencia de Datos propuesto por el equipo de Data & Analytics de Mercado Libre. El desafío incluye ejercicios independientes que abarcan exploración de datos, modelado y preparación de componentes productivos en machine learning.

## Estructura del Proyecto

```
meli_ds_challenge/
├── data/                # Datos del desafío y reportes
│   ├── 00_raw/          # Archivos CSV originales (e.g. ofertas_relampago.csv, full_devices.csv)
│   ├── 01_processed/    # Datos procesados o con features agregadas
│   ├── 02_external/     # Datos externos adicionales (si aplica)
│   └── 03_reports/      # PDF del enunciado, presentaciones, papers
│
├── notebooks/           # Notebooks principales de análisis y experimentación
│   └── exploratory/     # Notebooks exploratorios o preliminares
│
├── production/          # Componentes preparados para producción
│   ├── model/           # Modelos entrenados en formato binario
│   ├── pipeline/        # Código de preprocesamiento y transformación
│   └── inference/       # Scripts de inferencia (opcional)
│
├── src/                 # Código fuente modular principal
│   ├── data/            # Scripts de carga, limpieza y generación de features
│   ├── models/          # Entrenamiento, evaluación y tuning de modelos
│   └── config/          # Configuraciones en YAML o JSON para reproducibilidad
│
├── github/              # Configuraciones de workflows para GitHub Actions
│   └── workflows/       # CI para linting y formato automático (Black, flake8)
│
├── tests/               # Pruebas unitarias (opcional)
├── .pre-commit-config.yaml
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Ejercicios Abordados

### 1. Análisis Exploratorio de Ofertas Relámpago
EDA para obtener insights sobre este tipo de ofertas: consumo de stock, duración, comportamiento por categoría u horario, etc.

### 3. Predicción de Fallas de Dispositivos
Entrenamiento de un modelo predictivo para estimar la probabilidad de falla de un dispositivo con un día de anticipación, utilizando telemetría diaria.

## Instalación y Entorno

```bash
# Clonar el repositorio
git clone https://github.com/leandro-sartini/leandro-sartini-meli-ds-challenge.git
cd leandro-sartini-meli-ds-challenge.git

# (Opcional) Crear y activar entorno virtual
python -m venv venv
source venv/bin/activate   # o venv\Scripts\activate en Windows

# Instalar dependencias
pip install -r requirements.txt
```

## Linting y Formato de Código

GitHub Actions ejecuta automáticamente, todavia no esta configurado! Necesitas de una GitHub Org:
- `black` para formato de código
- `flake8` para cumplimiento con PEP8

También puedes ejecutarlos localmente:

```bash
black .
flake8 .
```

## Workflow y Estrategia de Desarrollo

Este proyecto sigue una estrategia de desarrollo basada en **Git Flow** con las siguientes prácticas:

### Estrategia de Ramas (Branching Strategy)
- **`main`**: Rama principal que contiene el código estable y listo para producción
- **`feature/*`**: Ramas de características para desarrollo de nuevas funcionalidades
- **Pull Requests**: Todas las características se desarrollan en ramas separadas y se integran a `main` mediante Pull Requests

### Flujo de Trabajo
1. Crear una nueva rama desde `main` para cada característica: `git checkout -b feature/nombre-caracteristica`
2. Desarrollar y probar la funcionalidad en la rama de característica
3. Crear un Pull Request hacia `main` con descripción detallada de los cambios
4. Revisión de código y aprobación del PR
5. Merge a `main` una vez aprobado

### Beneficios de esta Estrategia
- **Trazabilidad**: Cada característica tiene su propia rama y PR
- **Revisión de Código**: Proceso de revisión obligatorio antes del merge
- **Estabilidad**: El código en `main` siempre está en estado estable
- **Colaboración**: Facilita el trabajo en equipo y la revisión de código

## Estado del Proyecto

- Estructura de carpetas organizada
- Workflow de desarrollo establecido con ramas de características y Pull Requests
