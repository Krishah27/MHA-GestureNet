import json
import os
import numpy as np

# ==========================================
# DATASET PATH
# ==========================================

DATASET_PATH = "Dataset/ann_train_val"

# ==========================================
# STORAGE
# ==========================================

X = []
y = []

# ==========================================
# PROCESS EACH JSON FILE
# ==========================================

for filename in os.listdir(DATASET_PATH):

    # Skip non-json files
    if not filename.endswith(".json"):
        continue

    json_path = os.path.join(DATASET_PATH, filename)

    print(f"\nProcessing: {filename}")

    # ==========================================
    # LOAD JSON
    # ==========================================

    with open(json_path, "r") as file:
        data = json.load(file)

    # ==========================================
    # LOOP THROUGH SAMPLES
    # ==========================================

    for sample_id, sample_data in data.items():

        try:

            # ==========================================
            # LABEL
            # ==========================================

            label = sample_data["labels"][0]

            # ==========================================
            # LANDMARK SEQUENCE
            # ==========================================

            landmarks_sequence = sample_data["landmarks"]

            sequence = []

            # ==========================================
            # EACH FRAME
            # ==========================================

            for frame in landmarks_sequence:

                frame_landmarks = []

                try:

                    # Validate frame
                    if not isinstance(frame, list):
                        continue

                    # ==========================================
                    # EACH LANDMARK POINT
                    # ==========================================

                    for point in frame:

                        # Validate landmark point
                        if (
                            not isinstance(point, list)
                            or len(point) < 2
                        ):
                            continue

                        x = point[0]
                        y_coord = point[1]

                        # Validate coordinates
                        if (
                            not isinstance(x, (int, float))
                            or not isinstance(y_coord, (int, float))
                        ):
                            continue

                        frame_landmarks.extend(
                            [x, y_coord]
                        )

                    # ==========================================
                    # KEEP ONLY VALID FRAMES
                    # ==========================================

                    if len(frame_landmarks) == 42:

                        sequence.append(
                            frame_landmarks
                        )

                except:
                    continue

            # ==========================================
            # SKIP EMPTY SEQUENCES
            # ==========================================

            if len(sequence) == 0:
                continue

            # ==========================================
            # SAVE SAMPLE
            # ==========================================

            X.append(sequence)

            y.append(label)

        except Exception as e:

            print(f"Skipped sample: {sample_id}")
            print(e)

# ==========================================
# BASIC DATASET CHECK
# ==========================================

print("\n===================================")
print("DATASET SUMMARY")
print("===================================")

print("\nTotal Samples:", len(X))
print("Total Labels:", len(y))

# ==========================================
# CHECK SAMPLE SHAPES
# ==========================================

print("\nChecking First 10 Sample Shapes:\n")

for i in range(10):

    try:

        print(
            f"Sample {i}:",
            np.array(X[i]).shape
        )

    except:

        print(f"Sample {i}: INVALID")

# ==========================================
# FIXED LENGTH PADDING
# ==========================================

print("\n===================================")
print("APPLYING SEQUENCE PADDING")
print("===================================")

SEQ_LEN = 10

X_padded = []
y_clean = []

for idx, sequence in enumerate(X):

    try:

        sequence = np.array(
            sequence,
            dtype=np.float32
        )

        # Skip invalid sequences
        if len(sequence.shape) != 2:
            continue

        # Skip wrong feature dimensions
        if sequence.shape[1] != 42:
            continue

        current_len = sequence.shape[0]

        # ==========================================
        # PAD SHORT SEQUENCES
        # ==========================================

        if current_len < SEQ_LEN:

            padding = np.zeros(
                (SEQ_LEN - current_len, 42),
                dtype=np.float32
            )

            sequence = np.vstack(
                [sequence, padding]
            )

        # ==========================================
        # TRUNCATE LONG SEQUENCES
        # ==========================================

        elif current_len > SEQ_LEN:

            sequence = sequence[:SEQ_LEN]

        # ==========================================
        # STORE CLEAN SAMPLE
        # ==========================================

        X_padded.append(sequence)

        y_clean.append(y[idx])

    except:
        continue

# ==========================================
# CONVERT TO NUMPY
# ==========================================

X = np.array(
    X_padded,
    dtype=np.float32
)

y = np.array(
    y_clean
)

# ==========================================
# FINAL SHAPES
# ==========================================

print("\n===================================")
print("FINAL DATASET SHAPES")
print("===================================")

print("X shape:", X.shape)
print("y shape:", y.shape)

# ==========================================
# CREATE OUTPUT FOLDER
# ==========================================

os.makedirs(
    "Dataset/processed",
    exist_ok=True
)

# ==========================================
# SAVE DATASET
# ==========================================

np.save(
    "Dataset/processed/X.npy",
    X
)

np.save(
    "Dataset/processed/y.npy",
    y
)

print("\n===================================")
print("DATASET SAVED SUCCESSFULLY")
print("===================================")

print("\nSaved Files:")

print("Dataset/processed/X.npy")
print("Dataset/processed/y.npy")