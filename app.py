from flask import Flask, request
import requests
import numpy as np
import cv2

app = Flask(__name__)

@app.route("/")
def home():
    return "TFR AI Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    img_url = data.get("photo")

    # โหลดรูปจาก URL
    img_bytes = requests.get(img_url).content
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # -----------------------------
    # AI STEP (เริ่มต้นแบบ baseline)
    # -----------------------------

    # แปลงเป็น grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # วัดความ “หนาแน่นของภาพ” แบบง่าย (proxy fill rate)
    non_zero = cv2.countNonZero(gray)
    total = gray.shape[0] * gray.shape[1]

    fill_rate = int((non_zero / total) * 100)

    # ปรับให้อยู่ 0–100
    fill_rate = max(0, min(fill_rate, 100))

    return {
        "status": "ok",
        "fill_rate": fill_rate
    }
