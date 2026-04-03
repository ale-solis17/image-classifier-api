import numpy as np

from app.core.config import MIN_CONFIDENCE_MARGIN, UNKNOWN_LABEL, UNKNOWN_THRESHOLD
from app.services.preprocess import load_image


async def classify_image(path: str, model, labels: list[str]):
    if model is None or labels is None:
        return {"error": "Modelo no entrenado/cargado. Corre POST /train y reinicia la API para recargar el modelo."}

    x = load_image(path)
    x = np.expand_dims(x, axis=0)

    probs = model.predict(x, verbose=0)[0]
    probs = probs.astype(float)

    sorted_idx = np.argsort(probs)[::-1]
    top_idx = int(sorted_idx[0])
    confidence = float(probs[top_idx])
    label = labels[top_idx]
    runner_up_confidence = float(probs[sorted_idx[1]]) if len(sorted_idx) > 1 else 0.0
    confidence_margin = confidence - runner_up_confidence

    top3_idx = sorted_idx[:3]
    top3 = [{"label": labels[i], "confidence": float(probs[i])} for i in top3_idx]

    if confidence < UNKNOWN_THRESHOLD or confidence_margin < MIN_CONFIDENCE_MARGIN:
        return {
            "label": UNKNOWN_LABEL,
            "confidence": confidence,
            "top": top3,
        }

    return {"label": label, "confidence": confidence, "top": top3}
