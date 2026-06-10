from flask import Flask, request, jsonify
import requests
import cv2
import numpy as np
import traceback
import threading
import pickle
import os
from queue import Queue

app = Flask(__name__)

# =========================
# MODEL STORAGE
# =========================
MODEL_PATH = "tfr_model.pkl"

if os.path.exists(MODEL_PATH):
    model = pickle.load(open(MODEL_PATH, "rb"))
else:
    model = {
        "w": np.array([1.0, 1.0, 1.0, 1.0]),
        "b": 0.0
    }

# =========================
# QUEUE (NON-BLOCK TRAINING)
# =========================
train_queue = Queue()


# =========================
# DOWNLOAD IMAGE
# =========================
def download_image(url):
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return None

        return cv2.imdecode(
            np.frombuffer(r.content, np.uint8),
            cv2.IMREAD_COLOR
        )
    except:
        return None


# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(img):

    img = cv2.resize(img, (640, 480))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = (
        cv2.inRange(hsv, (10, 30, 60), (40, 255, 255)) |
        cv2.inRange(hsv, (0, 50, 50), (10, 255, 255)) |
        cv2.inRange(hsv, (160, 50, 50), (180, 255, 255)) |
        cv2.inRange(hsv, (90, 40, 40), (130, 255, 255))
    )

    h, w = mask.shape
    roi = mask[int(h*0.08):int(h*0.95), int(w*0.02):int(w*0.98)]

    if roi.size == 0:
        return np.zeros(4)

    area = np.mean(roi > 0)

    top = roi[:int(roi.shape[0]*0.3), :]
    mid = roi[int(roi.shape[0]*0.3):int(roi.shape[0]*0.7), :]
    bottom = roi[int(roi.shape[0]*0.7):, :]

    return np.array([
        area,
        np.mean(top > 0),
        np.mean(mid > 0),
        np.mean(bottom > 0)
    ])


# =========================
# PREDICT MODEL
# =========================
def predict(features):
    return float(np.dot(model["w"], features) + model["b"])


# =========================
# TRAIN MODEL
# =========================
def train_model(features, actual, pred, lr=0.03):

    error = actual - pred

    model["w"] += lr * error * features
    model["b"] += lr * error


# =========================
# SAVE MODEL
# =========================
def save_model():
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)


# =========================
# BACKGROUND TRAIN WORKER
# =========================
def train_worker():

    while True:

        item = train_queue.get()

        if item is None:
            break

        features, actual, pred = item

        try:
            train_model(features, actual, pred)
            save_model()
        except:
            pass

        train_queue.task_done()


# start worker thread
threading.Thread(target=train_worker, daemon=True).start()


# =========================
# API
# =========================
@app.route("/predict", methods=["POST"])
def predict_api():

    try:
        data = request.get_json(silent=True)

        image_url = data.get("link")
        row_id = data.get("id")

        if not image_url or not row_id:
            return jsonify({"error": "missing data"}), 400

        # =========================
        # LOAD IMAGE
        # =========================
        img = download_image(image_url)
        if img is None:
            return jsonify({"error": "image fail"}), 400

        # =========================
        # FEATURES
        # =========================
        features = extract_features(img)

        # =========================
        # PREDICT
        # =========================
        pred = predict(features)
        pred = max(0, min(100, pred))
        pred_out = int(round(pred / 5) * 5)

        # =========================
        # NON-BLOCK LEARNING
        # =========================
        actual = data.get("actual")

        if actual is not None:
            try:
                train_queue.put_nowait(
                    (features, float(actual), pred)
                )
            except:
                pass

        return jsonify({
            "status": "success",
            "pred": pred_out
        })

    except:
        print(traceback.format_exc())
        return jsonify({"error": "server error"}), 500


# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=10000,
        threaded=True
    )
