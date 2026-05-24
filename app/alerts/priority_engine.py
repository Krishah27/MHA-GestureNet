def get_gesture_info(gesture):

    gesture_map = {

        "call": {
            "message": "🩺 Nurse Call Required",
            "priority": "Attention"
        },

        "stop": {
            "message": "🚨 Emergency Stop",
            "priority": "Critical"
        },

        "palm": {
            "message": "✋ Attention Needed",
            "priority": "Urgent"
        },

        "fist": {
            "message": "⚠ Pain / Distress Detected",
            "priority": "Urgent"
        },

        "like": {
            "message": "✅ Patient Stable",
            "priority": "Normal"
        },

        "dislike": {
            "message": "😟 Patient Discomfort",
            "priority": "Attention"
        },

        "mute": {
            "message": "🔇 Cannot Speak",
            "priority": "Critical"
        },

        "peace": {
            "message": "🙏 Assistance Required",
            "priority": "Attention"
        },

        "ok": {
            "message": "👌 Confirmation Received",
            "priority": "Normal"
        }
    }

    return gesture_map.get(
        gesture,
        {
            "message": "Unknown Gesture",
            "priority": "Unknown"
        }
    )