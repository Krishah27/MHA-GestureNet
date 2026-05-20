import numpy as np
import pickle

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from tensorflow.keras.models import load_model

# ==========================================
# LOAD DATA
# ==========================================

X = np.load(
    "Dataset/processed/X_healthcare.npy"
)

y = np.load(
    "Dataset/processed/y_healthcare.npy"
)

# ==========================================
# LOAD MODEL
# ==========================================

model = load_model(
    "weights/mha_gesturenet.keras"
)

# ==========================================
# LOAD LABEL ENCODER
# ==========================================

with open(
    "weights/label_encoder.pkl",
    "rb"
) as f:

    label_encoder = pickle.load(f)

# ==========================================
# PREDICTIONS
# ==========================================

print("\nGenerating Predictions...")

y_pred_probs = model.predict(X)

y_pred = np.argmax(
    y_pred_probs,
    axis=1
)

# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\n===================================")
print("CLASSIFICATION REPORT")
print("===================================")

report = classification_report(
    y,
    y_pred,
    target_names=label_encoder.classes_
)

print(report)

# ==========================================
# CONFUSION MATRIX
# ==========================================

print("\n===================================")
print("CONFUSION MATRIX")
print("===================================")

cm = confusion_matrix(
    y,
    y_pred
)

print(cm)