import random
from app.services.preprocess import load_image

CLASSES = ["hongo_comestible", "hongo_toxico", "desconocido"]


async def classify_image(path: str):
    _ = load_image(path)  # todavía no usamos modelo

    label = random.choice(CLASSES)
    confidence = round(random.uniform(0.6, 0.99), 3)

    return {
        "label": label,
        "confidence": confidence,
        "top": [
            {"label": label, "confidence": confidence}
        ]
    }
