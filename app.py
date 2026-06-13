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

    # =========================
    # DETECT VIEW TYPE
    # =========================
    orig_h, orig_w = img.shape[:2]

    if orig_h > orig_w:
        view_type = "rear"
    else:
        view_type = "side"

    # =========================
    # RESIZE
    # =========================
    img = cv2.resize(img, (640, 480))

    # =========================
    # SIDE VIEW
    # 4:3 -> 16:9
    # =========================
    if view_type == "side":

        h, w = img.shape[:2]

        target_h = int(w * 9 / 16)

        top = (h - target_h) // 2

        img = img[
            top:top + target_h,
            :
        ]

    h, w = img.shape[:2]

    print(
        f"VIEW={view_type} "
        f"SIZE={w}x{h}"
    )

    # =========================
    # ROI
    # =========================
   if view_type == "rear":

    # ไม่แก้
    roi = img[
        int(h * 0.18):int(h * 0.82),
        int(w * 0.15):int(w * 0.85)
    ]

else:

    # แนวนอน ตัดแคบลง
    roi = img[
        int(h * 0.25):int(h * 0.75),
        int(w * 0.15):int(w * 0.85)
    ]

    if roi.size == 0:
        return 0

    # =========================
    # DEBUG ROI
    # =========================
    debug = img.copy()

    x1 = int(w * 0.15) if view_type == "rear" else int(w * 0.08)
    x2 = int(w * 0.85) if view_type == "rear" else int(w * 0.92)

    y1 = int(h * 0.18) if view_type == "rear" else int(h * 0.15)
    y2 = int(h * 0.82) if view_type == "rear" else int(h * 0.85)

    cv2.rectangle(
        debug,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3
    )

    cv2.imwrite(
        f"debug_roi_{view_type}.jpg",
        debug
    )

    # =========================
    # GRAYSCALE
    # =========================
    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # =========================
    # EDGE DENSITY
    # =========================
    edges = cv2.Canny(
        gray,
        50,
        150
    )

    kernel = np.ones((3, 3), np.uint8)

    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel
    )

    edge_density = (
        np.count_nonzero(edges)
        / edges.size
    )

    # =========================
    # TEXTURE DENSITY
    # =========================
    lap = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    texture_density = (
        np.mean(np.abs(lap))
        / 255.0
    )

    # =========================
    # SCORE
    # =========================
   score = (
    edge_density * 0.75 +
    texture_density * 0.25
    )

if view_type == "rear":

    # แนวตั้ง ใช้สูตรเดิม
    volume = int(score * 800)

else:

    # แนวนอน ลด Scale ลง
    volume = int(score * 280)

    volume = int(round(volume / 5) * 5)

    volume = max(0, min(100, volume))

    print(
        f"VIEW={view_type} "
        f"EDGE={edge_density:.4f} "
        f"TEXTURE={texture_density:.4f} "
        f"SCORE={score:.4f} "
        f"VOL={volume}%"
    )

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
