import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    Attention,
    GlobalAveragePooling1D
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)

# ==========================================
# LOAD DATA
# ==========================================

print("\nLoading Healthcare Dataset...")

X = np.load(
    "Dataset/processed/X_healthcare.npy"
)

y = np.load(
    "Dataset/processed/y_healthcare.npy"
)

print("\nDataset Shapes")

print("X shape:", X.shape)

print("y shape:", y.shape)

# ==========================================
# ONE HOT ENCODING
# ==========================================

num_classes = len(np.unique(y))

y = to_categorical(y, num_classes)

print("\nOne-hot Labels Shape:")

print(y.shape)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Shape:")

print(X_train.shape)

print("\nTesting Shape:")

print(X_test.shape)

# ==========================================
# MODEL INPUT
# ==========================================

inputs = Input(shape=(10, 42))

# ==========================================
# 1D CNN
# ==========================================

x = Conv1D(
    filters=64,
    kernel_size=3,
    activation='relu',
    padding='same'
)(inputs)

x = MaxPooling1D(pool_size=2)(x)

# ==========================================
# BiLSTM
# ==========================================

x = Bidirectional(
    LSTM(
        64,
        return_sequences=True
    )
)(x)

x = Dropout(0.3)(x)

# ==========================================
# ATTENTION
# ==========================================

attention = Attention()([x, x])

# ==========================================
# GLOBAL POOLING
# ==========================================

x = GlobalAveragePooling1D()(attention)

# ==========================================
# DENSE LAYERS
# ==========================================

x = Dense(
    64,
    activation='relu'
)(x)

x = Dropout(0.3)(x)

outputs = Dense(
    num_classes,
    activation='softmax'
)(x)

# ==========================================
# BUILD MODEL
# ==========================================

model = Model(
    inputs=inputs,
    outputs=outputs
)

# ==========================================
# COMPILE
# ==========================================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ==========================================
# SUMMARY
# ==========================================

print("\n===================================")
print("MHA-GestureNet Summary")
print("===================================")

model.summary()

# ==========================================
# CALLBACKS
# ==========================================

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "weights/mha_gesturenet.h5",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# ==========================================
# TRAIN
# ==========================================

print("\n===================================")
print("TRAINING STARTED")
print("===================================")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=64,
    callbacks=[
        early_stopping,
        checkpoint
    ]
)

# ==========================================
# EVALUATE
# ==========================================

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print("\n===================================")
print("FINAL RESULTS")
print("===================================")

print("\nTest Accuracy:", accuracy)

print("\nTest Loss:", loss)

print("\n===================================")
print("MODEL TRAINING COMPLETED")
print("===================================")