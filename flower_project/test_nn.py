import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------
# Load trained model
# -----------------

model = tf.keras.models.load_model("flower_nn.keras")

print("Model loaded ✅")


# -----------------
# Load test dataset
# -----------------

test_path = "flowers/test"

IMG_SIZE = (224,224)
BATCH_SIZE = 32

test_data = tf.keras.preprocessing.image_dataset_from_directory(

    test_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False   # สำคัญมาก

)

class_names = test_data.class_names

print("Classes:", class_names)


# -----------------
# Preprocess (เหมือนตอน train)
# -----------------

preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

test_data = test_data.map(lambda x, y: (preprocess(x), y))


# -----------------
# Predict
# -----------------

y_true = []
y_pred = []

for images, labels in test_data:

    preds = model.predict(images, verbose=0)

    pred_classes = np.argmax(preds, axis=1)

    y_pred.extend(pred_classes)

    y_true.extend(labels.numpy())


y_true = np.array(y_true)
y_pred = np.array(y_pred)


# -----------------
# Accuracy
# -----------------

acc = accuracy_score(y_true, y_pred)

print("\nAccuracy:", round(acc*100,2), "%")


# -----------------
# Classification Report
# -----------------

print("\nClassification Report:\n")

print(classification_report(y_true, y_pred, target_names=class_names))


# -----------------
# Confusion Matrix
# -----------------

print("\nConfusion Matrix:\n")

print(confusion_matrix(y_true, y_pred))