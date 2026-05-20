import numpy as np
from collections import Counter

# ==========================================
# LOAD DATA
# ==========================================

X = np.load("Dataset/processed/X.npy")

y = np.load("Dataset/processed/y.npy")

# ==========================================
# BASIC INFO
# ==========================================

print("\n===================================")
print("DATASET ANALYSIS")
print("===================================")

print("\nX shape:", X.shape)

print("y shape:", y.shape)

# ==========================================
# UNIQUE CLASSES
# ==========================================

unique_classes = np.unique(y)

print("\nTotal Classes:")

print(len(unique_classes))

# ==========================================
# CLASS DISTRIBUTION
# ==========================================

class_counts = Counter(y)

print("\nTop 20 Classes:\n")

for label, count in class_counts.most_common(20):

    print(f"{label}: {count}")

# ==========================================
# MIN / MAX SAMPLES
# ==========================================

print("\n===================================")

print("Largest Class:")

print(max(class_counts.values()))

print("\nSmallest Class:")

print(min(class_counts.values()))

print("\n===================================")