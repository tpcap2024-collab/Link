from flask import Flask, request, jsonify
import requests
import cv2
import numpy as np
import traceback

app = Flask(__name__)

# =========================
# APP SETTINGS (ใส่ของคุณตรงนี้)
# =========================
APP_ID = "5ebec09a-62dd-4fa9-8f14-830fb104518f"
ACCESS_KEY = "V2-2ZX8p-jmYBx-bH09l-nFTYW-cvV8W-7wNy3-zqOQQ-JvMrp"
TABLE_NAME = "Data TFR"

# =========================
# UPDATE APP SHEET FUNCTION
# =========================
def update_appsheet(row_id, volume, height):

    try:
        url = f"https://api.appsheet.com/api/v2/apps/{APP_ID}/tables/{TABLE_NAME}/Action"

        headers = {
            "ApplicationAccessKey": ACCESS_KEY,
            "Content-Type": "application/json"
        }

        payload = {
            "Action": "Edit",
            "Properties": {},
            "Rows": [
                {
                    "ID": row_id,          # KEY column
                    "TFR AI": volume,      # column M
                }
            ]
        }

        r = requests.post(url, json=payload, headers=headers)

        print("APP SHEET UPDATE:", r.status_code, r.text)

    except Exception as e:
        print("UPDATE ERROR:", str(e))


# =========================
# HOME
# =========================
@app.route("/")
def home():
    return "TFR AI Running"


# =========================
# MAIN PREDICT API
# =========================
@app.route("/predict", methods=["POST"])
def predict():

    try:

        print("=" * 80)

        data = request.get_json(silent=True)
        print("JSON:", data)

        if not data:
            return jsonify({"status": "error", "message": "No JSON"}), 400

        image_url = data.get("link")
        row_id = data.get("id")

        print("IMAGE URL:", image_url)
        print("ROW ID:", row_id)

        if not image_url or not row_id:
            return jsonify({"status": "error", "message": "missing data"}), 400

        # =========================
        # DOWNLOAD IMAGE
        # =========================
        response = requests.get(
            image_url,
            timeout=60,
            allow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        if response.status_code != 200:
            return jsonify({"status": "error", "message": "download failed"}), 400

        img_array = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"status": "error", "message": "decode failed"}), 400

        print("IMAGE OK")

        # =========================
        # RESIZE
        # =========================
        img_resized = cv2.resize(img, (800, 600))

        h, w = img_resized.shape[:2]

        # =========================
        # HSV PROCESS
        # =========================
        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (5,5), 0)

        # =========================
        # COLOR MASKS
        # =========================
        mask = cv2.inRange(hsv, np.array([10,30,60]), np.array([45,255,255]))

        mask_red = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0,70,50]), np.array([10,255,255])),
            cv2.inRange(hsv, np.array([160,70,50]), np.array([180,255,255]))
        )

        mask_blue = cv2.inRange(hsv, np.array([90,50,50]), np.array([130,255,255]))
        mask_green = cv2.inRange(hsv, np.array([40,40,40]), np.array([80,255,255]))
        mask_white = cv2.inRange(hsv, np.array([0,0,160]), np.array([180,70,255]))

        combined = cv2.bitwise_or(mask, mask_red)
        combined = cv2.bitwise_or(combined, mask_blue)
        combined = cv2.bitwise_or(combined, mask_green)
        combined = cv2.bitwise_or(combined, mask_white)

        # =========================
        # ROI
        # =========================
        x1 = int(w * 0.05)
        x2 = int(w * 0.95)
        y1 = int(h * 0.18)
        y2 = int(h * 0.80)

        roi = combined[y1:y2, x1:x2]

        # =========================
        # CLEAN TOP AREA
        # =========================
        roi[:int(roi.shape[0]*0.12), :] = 0

        # =========================
        # MORPHOLOGY
        # =========================
        kernel = np.ones((5,5), np.uint8)
        roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel)

        roi = cv2.dilate(roi, np.ones((7,7), np.uint8), 1)

        # =========================
        # CALCULATE VOLUME
        # =========================
        fill_pixels = cv2.countNonZero(roi)
        total_pixels = roi.shape[0] * roi.shape[1]

        volume_percent = int((fill_pixels / total_pixels) * 100)

        if volume_percent >= 85:
            volume_percent = 100

        print("VOLUME:", volume_percent)

        # =========================
        # UPDATE APP SHEET (REALTIME)
        # =========================
        update_appsheet(
            row_id=row_id,
            volume=volume_percent,
            height=0
        )

        # =========================
        # RESPONSE
        # =========================
        return jsonify({
            "status": "success",
            "volume": volume_percent
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
