import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'cambia_esta_clave')
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    IMG_SIZE = (575, 575)
