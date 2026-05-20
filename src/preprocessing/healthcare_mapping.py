HEALTHCARE_MAP = {

    "call": "nurse_call",

    "stop": "emergency_stop",

    "palm": "attention",

    "fist": "pain",

    "like": "stable",

    "dislike": "discomfort",

    "mute": "cannot_speak",

    "peace": "assistance",

    "ok": "confirmation"
}

print("\nHealthcare Gesture Mapping:\n")

for k, v in HEALTHCARE_MAP.items():

    print(f"{k}  --->  {v}")