from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "training" / "model.h5"
METRICS_PATH = BASE_DIR / "artifacts" / "evaluation" / "metrics.json"
DATASET_PATH = BASE_DIR / "artifacts" / "data_ingestion" / "Chest_CT_scan_DATA"
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def load_model():
    if not MODEL_PATH.exists():
        return None

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(
        optimizer=tf.keras.optimizers.SGD(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def load_metrics():
    if not METRICS_PATH.exists():
        return None

    with open(METRICS_PATH, encoding="utf-8") as metrics_file:
        return json.load(metrics_file)


def load_class_names():
    if not DATASET_PATH.exists():
        return ["adenocarcinoma", "normal"]

    return sorted([item.name for item in DATASET_PATH.iterdir() if item.is_dir()])


def preprocess_image(image_path: Path):
    image = tf.io.read_file(str(image_path))
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32) / 255.0
    return tf.expand_dims(image, axis=0)


MODEL = load_model()
CLASS_NAMES = load_class_names()


def predict(image_path: Path):
    if MODEL is None:
        return None

    batch = preprocess_image(image_path)
    probabilities = MODEL.predict(batch, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities) * 100)
    label = CLASS_NAMES[predicted_index] if predicted_index < len(CLASS_NAMES) else str(predicted_index)
    return {
        "label": label,
        "confidence": round(confidence, 2),
        "probabilities": [round(float(value) * 100, 2) for value in probabilities.tolist()],
    }


@app.route("/", methods=["GET", "POST"])
def index():
    metrics = load_metrics()
    prediction = None
    uploaded_image = None
    error_message = None

    if request.method == "POST":
        uploaded_file = request.files.get("image")
        if uploaded_file and uploaded_file.filename:
            filename = f"{uuid4().hex}_{secure_filename(uploaded_file.filename)}"
            image_path = UPLOAD_FOLDER / filename
            uploaded_file.save(image_path)
            uploaded_image = f"uploads/{filename}"
            prediction = predict(image_path)
            if prediction is None:
                error_message = "Model file is not available yet. Run training first."
        else:
            error_message = "Choose an image before submitting the form."

    return render_template(
        "index.html",
        metrics=metrics,
        prediction=prediction,
        uploaded_image=uploaded_image,
        error_message=error_message,
        class_names=CLASS_NAMES,
    )


if __name__ == "__main__":
    app.run(debug=True)