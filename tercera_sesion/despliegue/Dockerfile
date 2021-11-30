# Especificamos en que imagen nos basamos para crear nuestra imagen propia.
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9-slim

# Especificando el directorio
WORKDIR /app

# El puerto que vamos a exponer
EXPOSE 80

# comando COPY para traer el archivo requirements.txt 
COPY requirements.txt .

# comando RUN para instalar nuestras dependencias
RUN ["pip", "install", "-r","./requirements.txt"]

#Obteniendo toda la carpeta de despliegue para ponerla en la imagen docker(carpeta app)
COPY . .

# comando CMD para iniciar el servicio
CMD ["python3","main.py"]

