import tensorflow as tf
from flask import current_app

_model = None
_class_names = None

def load_model():
    global _model, _class_names
    if _model is None:
        # Ruta al modelo preentrenado guardado tras entrenamiento
        _model = tf.keras.models.load_model('best_model.h5')
        # Asegúrate de que el orden coincide con test_ds.class_indices
        _class_names = ['Basofilo','Eosinofilo','Linfocito','Monocito','Neutrofilos']
    return _model

def predict(image_array):
    model = load_model()
    preds = model.predict(image_array)[0]
    idx = preds.argmax()
    return _class_names[idx], float(preds[idx])
