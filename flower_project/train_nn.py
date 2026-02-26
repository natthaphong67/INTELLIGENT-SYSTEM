import tensorflow as tf
import os

# -----------------
# Path
# -----------------

train_path = "flowers/train"
val_path = "flowers/validation"


# -----------------
# Load Dataset
# -----------------

IMG_SIZE = (224,224)
BATCH_SIZE = 32

train_data = tf.keras.preprocessing.image_dataset_from_directory(

    train_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE

)

val_data = tf.keras.preprocessing.image_dataset_from_directory(

    val_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE

)

class_names = train_data.class_names

print(class_names)


# -----------------
# ⭐ IMPORTANT: Preprocess for MobileNetV2
# -----------------

preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

train_data = train_data.map(lambda x, y: (preprocess(x), y))
val_data = val_data.map(lambda x, y: (preprocess(x), y))


# -----------------
# Data Augmentation
# -----------------

data_augmentation = tf.keras.Sequential([

    tf.keras.layers.RandomFlip("horizontal"),

    tf.keras.layers.RandomRotation(0.15),

    tf.keras.layers.RandomZoom(0.15),

])


# -----------------
# Load MobileNetV2
# -----------------

base_model = tf.keras.applications.MobileNetV2(

    input_shape=(224,224,3),
    include_top=False,
    weights="imagenet"

)


# Freeze ก่อน
base_model.trainable = False


# -----------------
# Build Model
# -----------------

x = data_augmentation(base_model.output)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Dense(256, activation="relu")(x)

x = tf.keras.layers.Dropout(0.5)(x)

output = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)


model = tf.keras.Model(base_model.input, output)


# -----------------
# Compile
# -----------------

model.compile(

    optimizer=tf.keras.optimizers.Adam(0.0001),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)


# -----------------
# EarlyStopping
# -----------------

early_stop = tf.keras.callbacks.EarlyStopping(

    monitor="val_accuracy",

    patience=5,

    restore_best_weights=True

)


# -----------------
# Train Phase 1
# -----------------

print("Training Phase 1...")

model.fit(

    train_data,

    validation_data=val_data,

    epochs=20,

    callbacks=[early_stop]

)


# -----------------
# ⭐ Fine Tuning (เฉพาะ layer บน)
# -----------------

base_model.trainable = True


for layer in base_model.layers[:-50]:
    layer.trainable = False


model.compile(

    optimizer=tf.keras.optimizers.Adam(0.00001),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)


print("Fine tuning...")

model.fit(

    train_data,

    validation_data=val_data,

    epochs=15,

    callbacks=[early_stop]

)


# -----------------
# Save
# -----------------

model.save("flower_nn.keras")

print("Training Complete ✅")