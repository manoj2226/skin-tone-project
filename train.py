# train.py
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# ---- Config ----
DATA_DIR = "dataset"
MODEL_DIR = "models"
MODEL_NAME = "SkinTone_MobileNetV2_v1.h5"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15       # start small, increase later
LEARNING_RATE = 1e-4

os.makedirs(MODEL_DIR, exist_ok=True)

# ---- Data generators (with augmentation) ----
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=(0.8, 1.2),
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

num_classes = len(train_gen.class_indices)
print("Classes found:", train_gen.class_indices)

# ---- Build model (Transfer Learning) ----
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base.trainable = False   # freeze base for initial training

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base.input, outputs=outputs)
model.compile(optimizer=Adam(LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ---- Train ----
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# ---- Optionally fine-tune: unfreeze some layers ----
base.trainable = True
# unfreeze last N layers
fine_tune_at = len(base.layers) - 30
for layer in base.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer=Adam(LEARNING_RATE/10), loss='categorical_crossentropy', metrics=['accuracy'])
# add more epochs
fine_history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5
)

# ---- Save model and class indices mapping ----
model_path = os.path.join(MODEL_DIR, MODEL_NAME)
model.save(model_path)
print("Saved model to", model_path)

# Save class indices mapping for prediction script
import json
mapping = {v:k for k,v in train_gen.class_indices.items()}  # index->label
with open(os.path.join(MODEL_DIR, "class_indices.json"), "w") as f:
    json.dump(mapping, f)
print("Saved class mapping.")
