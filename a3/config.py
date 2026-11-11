from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP_VERSION = "a3-1.0"
N_MAIN = 20
N_AI_WRONG = 4
N_ATTENTION = 2
ATTENTION_FAILS_TO_TERMINATE = 2
AI_WORKFLOW = "two_stage"
SHOW_AI_PROBS = False
IMAGE_DISPLAY_PX = 240
LABELS = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]
CUES = [
    "No clear expression; face relaxed",
    "Smile; cheeks raised",
    "Eyebrows raised, eyes wide, mouth often open",
    "Inner brows raised, lip corners pulled down",
    "Brows lowered and drawn together; lips pressed or teeth bared",
    "Nose wrinkled; upper lip raised",
    "Brows raised and pulled together; eyes wide; lips stretched",
    "One lip corner tightened or raised (asymmetric smirk)",
]
