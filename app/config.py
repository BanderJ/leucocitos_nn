import os

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'cambia_esta_clave')

    # Directorio base de este archivo (app/)
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))

    # Carpeta de uploads: ../uploads respecto a app/
    UPLOAD_FOLDER = os.path.join(BASE_DIR, os.pardir, 'uploads')

    # Extensiones permitidas para subida
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

    # Tamaño en píxeles al que redimensionar las imágenes (ancho, alto)
    IMG_SIZE = (575, 575)

