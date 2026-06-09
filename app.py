from flask import Flask, request, jsonify
import requests
import cv2
import numpy as np
import traceback
import time
import threading

app = Flask(__name__)

APP_ID = "5ebec09a-62dd-4fa9-8f14-830fb104518f"
ACCESS_KEY = "V2-2ZX8p-jmYBx-bH09l-nFTYW-cvV8W-7wNy3-zqOQQ-JvMrp"
TABLE_NAME = "Data TFR"

processed_ids = set()
lock = threading.Lock()

# =========================
# DOWNLOAD IMAGE
# =========================
def download_image(url):
    try:
        r = requests.get(url, timeout=60)

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
# 🔥 AI CORE (FROM COLAB → GEN)
# =========================
def gen_evaluate_image(img):

    # =========================
    # PREPROCESS
    # =========================
    img = cv2.resize(img, (800, 600))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (5, 5), 0)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # =========================
    # MASK (NO WHITE)
    # =========================
    mask_egg = cv2.inRange(hsv, (10, 30, 60), (45, 255, 255))

    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, (0, 70, 50), (10, 255, 255)),
        cv2.inRange(hsv, (160, 70, 50), (180, 255, 255))
    )

    mask_blue = cv2.inRange(hsv, (90, 50, 50), (130, 255, 255))
    mask_green = cv2.inRange(hsv, (40, 40, 40), (80, 255, 255))
    mask_black = cv2.inRange(hsv, (0, 0, 0), (180, 255, 50))

    # =========================
    # REMOVE NOISE
    # =========================
    edges = cv2.Canny(gray, 50, 150)

    smooth_dark = cv2.bitwise_and(
        mask_black,
        cv2.bitwise_not(edges)
    )

    combined = mask_egg | mask_red | mask_blue | mask_green | mask_black
    combined = cv2.bitwise_and(combined, cv2.bitwise_not(smooth_dark))

    # =========================
    # DYNAMIC ROI
    # =========================
    projection = np.sum(combined, axis=1)
    norm = projection / (np.max(projection) + 1e-6)

    active = np.where(norm > 0.08)[0]

    h, w = combined.shape

    if len(active) > 0:
        ceiling_y = int(np.percentile(active, 5))
        floor_y = int(np.percentile(active, 95))
    else:
        ceiling_y = int(h * 0.2)
        floor_y = int(h * 0.8)

    margin = int((floor_y - ceiling_y) * 0.05)

    y1 = max(0, ceiling_y + margin)
    y2 = min(h, floor_y - margin)

    x1 = int(w * 0.03)
    x2 = int(w * 0.97)

    roi = combined[y1:y2, x1:x2]

    # =========================
    # CLEAN ROI
    # =========================
    if roi.shape[0] > 10:
        roi[:int(roi.shape[0]*0.12), :] = 0

    roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    roi = cv2.dilate(roi, np.ones((7, 7), np.uint8), 1)

    # =========================
    # VOLUME (%)
    # =========================
    fill = cv2.countNonZero(roi)
    total = roi.size

    volume = (fill / total) * 100

    # =========================
    # HEIGHT (%)
    # =========================
    projection_roi = np.sum(roi, axis=1)
    norm_roi = projection_roi / (np.max(projection_roi) + 1e-6)

    active_roi = np.where(norm_roi > 0.08)[0]

    if len(active_roi) > 0:
        highest = int(np.percentile(active_roi, 15))
    else:
        highest = roi.shape[0]

    height = ((roi.shape[0] - highest) / roi.shape[0]) * 100

    # =========================
    # NORMALIZE
    # =========================
    volume = int(round(volume / 5) * 5)
    height = int(round(height / 5) * 5)

    volume = max(0, min(100, volume))
    height = max(0, min(100, height))

    if volume >= 85:
        volume = 100
    if height >= 85:
        height = 100

    return {
        "volume": volume,
        "height": height
    }


# =========================
# APP SHEET UPDATE
# =========================
def update_appsheet(row_id, volume_text, height_text):

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
                "Height AI": height_text,
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

        time.sleep(2)

        # =========================
        # IMAGE
        # =========================
        img = download_image(image_url)

        if img is None:
            return jsonify({"error": "image fail"}), 400

        # =========================
        # AI GEN CORE
        # =========================
        result = gen_evaluate_image(img)

        volume = result["volume"]
        height = result["height"]

        volume_text = f"{volume}%"
        height_text = f"{height}%"

        print("RESULT:", volume_text, height_text)

        # =========================
        # UPDATE APP SHEET
        # =========================
        update_appsheet(row_id, volume_text, height_text)

        return jsonify({
            "status": "success",
            "id": row_id,
            "volume": volume_text,
            "height": height_text
        })

    except:
        print(traceback.format_exc())
        return jsonify({"error": "server error"}), 500


# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
