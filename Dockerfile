FROM python:3.9.21-slim

# Instalar dependencias mínimas del sistema (incluyendo solo lo necesario para OpenCV y Tesseract)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Establecer directorio de trabajo
WORKDIR /app

# Copiar solo lo necesario del proyecto (usa .dockerignore para excluir basura)
COPY . /app

# Instalar dependencias de Python de forma limpia
RUN pip install --no-cache-dir \
    flask \
    opencv-python-headless \
    numpy \
    tensorflow-cpu==2.19.0 \
    openai \
    scikit-learn \
    pytesseract

# Exponer el puerto de Flask
EXPOSE 5000

# Comando por defecto
CMD ["python", "TaskAutomationServer.py"]
