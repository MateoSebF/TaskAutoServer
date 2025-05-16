FROM python:3.9.21-slim

# Instalación de dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar archivos del proyecto
COPY . /app

# Instalar dependencias de Python
RUN pip install --no-cache-dir \
    flask \
    opencv-python-headless \
    numpy \
    tensorflow==2.19.0 \
    openai \
    scikit-learn \
    pytesseract

# Puerto expuesto por Flask (puedes cambiar si es necesario)
EXPOSE 5000

# Comando por defecto
CMD ["python", "main.py"]
