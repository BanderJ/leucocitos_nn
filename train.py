import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, callbacks
import tensorflow as tf

# 1. Directorios
BASE_DIR = os.getcwd()                     # e.g. C:\Users\ander\Downloads\leucocitos_nn
TRAIN_DIR = os.path.join(BASE_DIR, 'dataset', 'Train')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# 2. Parámetros
IMG_SIZE = (575, 575)
BATCH = 32
EPOCHS = 1
NUM_CLASSES = 5

# 3. Generadores con split de validación
gen = ImageDataGenerator(
    rescale=1/255.,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

train_ds = gen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH,
    class_mode='categorical', subset='training'
)
val_ds = gen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH,
    class_mode='categorical', subset='validation'
)

# 4. Modelo (ejemplo simple, o sustituye por tu arquitectura preferida)
base = tf.keras.applications.EfficientNetB0(
    input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet'
)
base.trainable = False

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Callbacks
early = callbacks.EarlyStopping(patience=3, restore_best_weights=True)
checkpoint = callbacks.ModelCheckpoint(
    os.path.join(MODELS_DIR, 'best_model.h5'),
    save_best_only=True, monitor='val_loss'
)

# 6. Entrenamiento
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early, checkpoint]
)

print("¡Entrenamiento completado! Modelo guardado en models/best_model.h5")