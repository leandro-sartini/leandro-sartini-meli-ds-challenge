# Dockerfile para el Dashboard de Predicción de Fallas
# Mercado Libre DS Challenge

# Usar imagen base de Python 3.9
FROM python:3.9-slim

# Establecer directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos de dependencias
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY src/ ./src/
COPY production/ ./production/

# Crear directorio para datos si no existe
RUN mkdir -p data/00_raw

# Exponer puerto de Streamlit
EXPOSE 8501

# Configurar variables de entorno
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Comando para ejecutar la aplicación
CMD ["streamlit", "run", "src/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Metadatos
LABEL maintainer="Leandro Sartini"
LABEL description="Dashboard de Predicción de Fallas - Mercado Libre DS Challenge"
LABEL version="1.0.0"
