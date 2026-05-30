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

    img_url = data["photo"]

    # โหลดรูปจาก URL
    img_bytes = requests.get(img_url).content
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # TODO: ใส่ AI วิเคราะห์จริงตรงนี้ก่อน
    fill_rate = 78

    return {
        "status": "ok",
        "fill_rate": fill_rate
    }
