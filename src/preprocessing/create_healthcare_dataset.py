import numpy as np
import os
import pickle

from sklearn.preprocessing import LabelEncoder

# ==========================================
# LOAD DATASET
# ==========================================

print("\nLoading Dataset...")

X = np.load("Dataset/processed/X.npy")

y = np.load("Dataset/processed/y.npy")

print("\nOriginal Shapes")

print("X shape:", X.shape)

print("y shape:", y.shape)

# ==========================================
# HEALTHCARE GESTURES
# ==========================================

HEALTHCARE_CLASSES = [

    "call",
    "stop",
    "palm",
    "fist",
    "like",
    "dislike",
    "mute",
    "peace",
    "ok"
]

# ==========================================
# FILTER DATASET
# ==========================================

print("\nFiltering Healthcare Classes...")

X_filtered = []
y_filtered = []

for i in range(len(y)):

    label = y[i]

    if label in HEALTHCARE_CLASSES:

        X_filtered.append(X[i])

        y_filtered.append(label)

# ==========================================
# CONVERT TO NUMPY
# ==========================================

X_filtered = np.array(
    X_filtered,
    dtype=np.float32
)

y_filtered = np.array(
    y_filtered
)

# ==========================================
# LABEL ENCODING
# ==========================================

print("\nEncoding Labels...")

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(
    y_filtered
)

# ==========================================
# SAVE LABEL ENCODER
# ==========================================

os.makedirs(
    "weights",
    exist_ok=True
)

with open(
    "weights/label_encoder.pkl",
    "wb"
) as f:

    pickle.dump(
        label_encoder,
        f
    )

# ==========================================
# FINAL SHAPES
# ==========================================

print("\n===================================")
print("FINAL HEALTHCARE DATASET")
print("===================================")

print("\nX shape:", X_filtered.shape)

print("y shape:", y_encoded.shape)

print("\nHealthcare Classes:")

for idx, label in enumerate(
    label_encoder.classes_
):

    print(f"{idx} ---> {label}")

# ==========================================
# SAVE DATASET
# ==========================================

os.makedirs(
    "Dataset/processed",
    exist_ok=True
)

np.save(
    "Dataset/processed/X_healthcare.npy",
    X_filtered
)

np.save(
    "Dataset/processed/y_healthcare.npy",
    y_encoded
)

print("\n===================================")
print("HEALTHCARE DATASET SAVED")
print("===================================")

print("\nSaved Files:")

print("Dataset/processed/X_healthcare.npy")

print("Dataset/processed/y_healthcare.npy")

print("weights/label_encoder.pkl")
