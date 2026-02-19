import numpy as np

from app.core.config import UNKNOWN_THRESHOLD
from app.services.preprocess import load_image


async def classify_image(path: str, model, labels: list[str]):
    if model is None or labels is None:
        return {"error": "Modelo no entrenado/cargado. Corre POST /train y luego /reload-model."}

    x = load_image(path)
    x = np.expand_dims(x, axis=0)

    probs = model.predict(x, verbose=0)[0]
    probs = probs.astype(float)

    top_idx = int(np.argmax(probs))
    confidence = float(probs[top_idx])
    label = labels[top_idx]

    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [{"label": labels[i], "confidence": float(probs[i])} for i in top3_idx]

    if confidence < UNKNOWN_THRESHOLD:
        return {"label": "unknown", "confidence": confidence, "top": top3}

    return {"label": label, "confidence": confidence, "top": top3}
