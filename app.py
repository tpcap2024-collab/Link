from flask import Flask, request, jsonify
import requests
import cv2
import numpy as np
import traceback
import time
import threading

app = Flask(__name__)

# =========================
# APPSHEET CONFIG
# =========================
APP_ID = "5ebec09a-62dd-4fa9-8f14-830fb104518f"
ACCESS_KEY = "V2-2ZX8p-jmYBx-bH09l-nFTYW-cvV8W-7wNy3-zqOQQ-JvMrp"
TABLE_NAME = "Data TFR"

# =========================
# LOCK
# =========================
processed_ids = set()
lock = threading.Lock()


# =========================
# DOWNLOAD IMAGE
# =========================
def download_image(url):
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return None

        img = cv2.imdecode(
            np.frombuffer(r.content, np.uint8),
            cv2.IMREAD_COLOR
        )
        return img
    except:
        return None


# =========================
# 🔥 BALANCED VOLUME MODEL
# =========================
def gen_volume(img):

    img = cv2.resize(img, (640, 480))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (5, 5), 0)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # =========================
    # COLOR MASK (balanced)
    # =========================
    mask = (
        cv2.inRange(hsv, (10, 35, 60), (40, 255, 255)) |
        cv2.inRange(hsv, (0, 60, 50), (10, 255, 255)) |
        cv2.inRange(hsv, (160, 60, 50), (180, 255, 255)) |
        cv2.inRange(hsv, (90, 50, 50), (130, 255, 255))
    )

    # =========================
    # MORPH CLEAN
    # =========================
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    h, w = mask.shape

    # =========================
    # ROI (clean truck interior only)
    # =========================
    roi = mask[int(h*0.18):int(h*0.90), int(w*0.04):int(w*0.96)]

    if roi.size == 0:
        return 0

    # =========================
    # AREA DENSITY
    # =========================
    area_density = np.count_nonzero(roi) / roi.size
    area_density = np.clip(area_density, 0, 1)

    # =========================
    # VERTICAL SIGNAL
    # =========================
    v_proj = np.sum(roi, axis=1)
    v_norm = v_proj / (np.max(v_proj) + 1e-6)
    v_score = np.mean(v_norm > 0.10)

    # =========================
    # HORIZONTAL SIGNAL
    # =========================
    h_proj = np.sum(roi, axis=0)
    h_norm = h_proj / (np.max(h_proj) + 1e-6)
    h_score = np.mean(h_norm > 0.10)

    # =========================
    # BASE VOLUME (BALANCED WEIGHTS)
    # =========================
    volume = (
        area_density * 100 * 0.80 +
        v_score * 100 * 0.12 +
        h_score * 100 * 0.08
    )

    # =========================
    # NON-LINEAR CALIBRATION
    # =========================
    if volume < 30:
        volume *= 1.10
    elif volume > 75:
        volume *= 0.95

    # =========================
    # LOW DENSITY FIX
    # =========================
    if area_density < 0.12:
        volume *= 0.85

    # =========================
    # FINAL NORMALIZE
    # =========================
    volume = int(round(volume / 5) * 5)
    volume = max(0, min(100, volume))

    return volume


# =========================
# UPDATE APPSHEET
# =========================
def update_appsheet(row_id, volume_text):

    url = f"https://api.appsheet.com/api/v2/apps/{APP_ID}/tables/{TABLE_NAME}/Action"

    headers = {
        "ApplicationAccessKey": ACCESS_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "Action": "Edit",
        "Rows": [
            {
                "id": row_id,
                "TFR AI": volume_text,
                "status": "Done"
            }
        ]
    }

    try:
        requests.post(url, json=payload, headers=headers, timeout=20)
    except:
        pass


# =========================
# API ENDPOINT
# =========================
@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json(silent=True)

        if not data:
            return jsonify({"error": "no json"}), 400

        image_url = data.get("link")
        row_id = data.get("id")

        if not image_url or not row_id:
            return jsonify({"error": "missing data"}), 400

        # =========================
        # DUPLICATE LOCK
        # =========================
        with lock:
            if row_id in processed_ids:
                return jsonify({"status": "skipped"}), 200
            processed_ids.add(row_id)

        # =========================
        # IMAGE LOAD
        # =========================
        img = download_image(image_url)

        if img is None:
            return jsonify({"error": "image fail"}), 400

        # =========================
        # AI PROCESS
        # =========================
        volume = gen_volume(img)
        volume_text = f"{volume}%"

        print("VOLUME:", volume_text)

        # =========================
        # UPDATE SHEET
        # =========================
        update_appsheet(row_id, volume_text)

        return jsonify({
            "status": "success",
            "id": row_id,
            "volume": volume_text
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
