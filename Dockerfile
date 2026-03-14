# Usamos una imagen base oficial de Python y ligera (slim)
FROM python:3.11-slim

# Configuramos variables de entorno para python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Instalar las librerías del sistema necesarias para OpenCV y PlantCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Crear y movernos al directorio de trabajo en el contenedor
WORKDIR /app

# Copiar el archivo de requerimientos y cachear esta capa
COPY requirements.txt /app/

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el código de tu proyecto al contenedor
COPY . /app/

# Exponer el puerto por el que tu aplicación web/API corre
# Cloud Run por defecto inyecta la variable de entorno $PORT (suele ser 8080)
EXPOSE 8080

# Comando para correr la aplicación FastAPI con Uvicorn
# Usamos el formato exec para que pueda recibir señales de apagado correctamente
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
