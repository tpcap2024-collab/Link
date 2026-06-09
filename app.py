from flask import Flask, request, jsonify
import requests
import cv2
import numpy as np
import traceback
import time
import threading

app = Flask(__name__)

# =========================
# APP SHEET CONFIG
# =========================
APP_ID = "5ebec09a-62dd-4fa9-8f14-830fb104518f"
ACCESS_KEY = "V2-2ZX8p-jmYBx-bH09l-nFTYW-cvV8W-7wNy3-zqOQQ-JvMrp"
TABLE_NAME = "Data TFR"

# =========================
# MEMORY LOCK
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

        return cv2.imdecode(
            np.frombuffer(r.content, np.uint8),
            cv2.IMREAD_COLOR
        )

    except:
        return None


# =========================
# 🔥 AI CORE (FAST VOLUME)
# =========================
def gen_volume(img):

    img = cv2.resize(img, (640, 480))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (3, 3), 0)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mask = (
        cv2.inRange(hsv, (10, 30, 60), (45, 255, 255)) |
        cv2.inRange(hsv, (0, 70, 50), (10, 255, 255)) |
        cv2.inRange(hsv, (160, 70, 50), (180, 255, 255)) |
        cv2.inRange(hsv, (90, 50, 50), (130, 255, 255)) |
        cv2.inRange(hsv, (40, 40, 40), (80, 255, 255))
    )

    edges = cv2.Canny(gray, 50, 120)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(edges))

    projection = np.sum(mask, axis=1)
    norm = projection / (np.max(projection) + 1e-6)

    active = np.where(norm > 0.08)[0]

    h, w = mask.shape

    if len(active) > 0:
        top = int(np.percentile(active, 5))
        bottom = int(np.percentile(active, 95))
    else:
        top = int(h * 0.2)
        bottom = int(h * 0.8)

    roi = mask[top:bottom, int(w * 0.03):int(w * 0.97)]

    if roi.shape[0] > 10:
        roi[:int(roi.shape[0] * 0.1), :] = 0

    roi = cv2.morphologyEx(
        roi,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8)
    )

    fill = cv2.countNonZero(roi)
    total = roi.size

    volume = (fill / total) * 100

    volume = int(round(volume / 5) * 5)
    volume = max(0, min(100, volume))

    if volume >= 85:
        volume = 100

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

    for _ in range(3):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=30)

            if r.status_code == 200:
                return True

        except:
            time.sleep(1)

    return False


# =========================
# API
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
        # LOCK (กันยิงซ้ำ)
        # =========================
        with lock:
            if row_id in processed_ids:
                return jsonify({"status": "skipped"}), 200
            processed_ids.add(row_id)

        time.sleep(1)

        # =========================
        # IMAGE
        # =========================
        img = download_image(image_url)

        if img is None:
            return jsonify({"error": "image fail"}), 400

        # =========================
        # AI
        # =========================
        volume = gen_volume(img)
        volume_text = f"{volume}%"

        print("VOLUME:", volume_text)

        # =========================
        # 🔥 UPDATE APPSHEET
        # =========================
        update_appsheet(row_id, volume_text)

        # =========================
        # RESPONSE
        # =========================
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
