import os
import tensorflow as tf
from flask import current_app
import logging

_model = None
_class_names = ['Basophil','Eosinophil','Lymphocyte','Monocyte','Neutrophil']

def load_model():
    global _model
    if _model is None:
        # Ruta absoluta a tu modelo entrenado
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'models', 'best_model.h5'
        )
        _model = tf.keras.models.load_model(model_path)
    return _model

def predict(image_array):
    model = load_model()
    preds = model.predict(image_array)[0]        # array de 5 probabilidades
    logging.debug(f"Preds raw: {preds}")         # se verán en la consola de Flask
    idx = preds.argmax()
    return _class_names[idx], float(preds[idx]), preds

