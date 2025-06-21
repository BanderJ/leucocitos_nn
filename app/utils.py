import os
from PIL import Image
import numpy as np
from flask import current_app

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

def load_and_prep_image(path):
    img = Image.open(path).convert('RGB').resize(current_app.config['IMG_SIZE'])
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)
