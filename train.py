import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, callbacks
import tensorflow as tf

# 1. Directorios
BASE_DIR = os.getcwd()
TRAIN_DIR = os.path.join(BASE_DIR, 'dataset', 'Train')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# 2. Parámetros optimizados
IMG_SIZE = (224, 224)
BATCH = 64  # Aumentado para mejor generalización
EPOCHS = 50  # Más épocas con early stopping
NUM_CLASSES = 5
INIT_LR = 1e-4  # Tasa de aprendizaje inicial más baja

# 3. Generadores con aumento de datos mejorado
train_gen = ImageDataGenerator(
    rescale=1/255.,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
    validation_split=0.2
)

val_gen = ImageDataGenerator(rescale=1/255., validation_split=0.2)

train_ds = train_gen.flow_from_directory(
    TRAIN_DIR, 
    target_size=IMG_SIZE, 
    batch_size=BATCH,
    class_mode='categorical', 
    subset='training',
    shuffle=True
)

val_ds = val_gen.flow_from_directory(
    TRAIN_DIR, 
    target_size=IMG_SIZE, 
    batch_size=BATCH,
    class_mode='categorical', 
    subset='validation',
    shuffle=False
)

# 4. Modelo optimizado
base = tf.keras.applications.EfficientNetB3(  # Versión más potente
    input_shape=IMG_SIZE + (3,), 
    include_top=False, 
    weights='imagenet',
    pooling='avg'  # Reemplaza GlobalAveragePooling
)

# Descongelar las últimas capas para fine-tuning
base.trainable = True
for layer in base.layers[:-10]:
    layer.trainable = False

model = models.Sequential([
    base,
    layers.Dropout(0.5),  # Mayor dropout para evitar overfitting
    layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# Optimizador con schedule de aprendizaje
optimizer = tf.keras.optimizers.Adam(
    learning_rate=INIT_LR,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07
)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy', 
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)

# 5. Callbacks mejorados
early = callbacks.EarlyStopping(
    patience=10,
    restore_best_weights=True,
    monitor='val_loss',
    min_delta=0.001
)

checkpoint = callbacks.ModelCheckpoint(
    os.path.join(MODELS_DIR, 'best_model.h5'),
    save_best_only=True, 
    monitor='val_accuracy',
    mode='max'
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# 6. Entrenamiento con más épocas
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early, checkpoint, reduce_lr],
    verbose=1
)

print("¡Entrenamiento completado! Modelo guardado en models/best_model.h5")