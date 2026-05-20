import json

# Path to one gesture file
json_path = "Dataset/ann_train_val/call.json"

# Load JSON
with open(json_path, "r") as file:
    data = json.load(file)

# Print type
print(type(data))

# Print first keys
if isinstance(data, dict):
    print("\nKeys:")
    print(data.keys())

# Print sample data
print("\nFirst Entry:")
print(data[list(data.keys())[0]])