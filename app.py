from flask import Flask, request, jsonify
import requests
import numpy as np
import cv2

app = Flask(**name**)

@app.route("/")
def home():
return "TFR AI Running"

@app.route("/predict", methods=["POST"])
def predict():

```
try:

    # ======================================
    # RECEIVE DATA
    # ======================================

    data = request.json or {}

    print("================================")
    print("DATA =", data)

    img_url = data.get("link")

    print("URL =", img_url)

    if not img_url:
        return jsonify({
            "status": "error",
            "message": "no link"
        })

    # ======================================
    # DOWNLOAD IMAGE
    # ======================================

    response = requests.get(img_url, timeout=30)

    print("HTTP STATUS =", response.status_code)
    print("FILE SIZE =", len(response.content))

    img_bytes = response.content

    img_array = np.frombuffer(
        img_bytes,
        np.uint8
    )

    img = cv2.imdecode(
        img_array,
        cv2.IMREAD_COLOR
    )

    if img is None:

        print("DECODE FAILED")

        return jsonify({
            "status": "error",
            "message": "cannot decode image"
        })

    print("IMAGE SHAPE =", img.shape)

    # ======================================
    # RESIZE
    # ======================================

    img_resized = cv2.resize(
        img,
        (800, 600)
    )

    h, w = img_resized.shape[:2]

    # ======================================
    # HSV
    # ======================================

    hsv = cv2.cvtColor(
        img_resized,
        cv2.COLOR_BGR2HSV
    )

    hsv = cv2.GaussianBlur(
        hsv,
        (5,5),
        0
    )

    # ======================================
    # DARK MASK
    # ======================================

    lower_dark = np.array([0,0,0])
    upper_dark = np.array([180,255,60])

    mask_dark = cv2.inRange(
        hsv,
        lower_dark,
        upper_dark
    )

    # ======================================
    # ROI
    # ======================================

    x1 = int(w * 0.05)
    x2 = int(w * 0.95)

    y1 = int(h * 0.18)
    y2 = int(h * 0.80)

    crop_mask = mask_dark[
        y1:y2,
        x1:x2
    ]

    # ======================================
    # REMOVE CEILING
    # ======================================

    ceiling_cut = int(
        crop_mask.shape[0] * 0.12
    )

    crop_mask[0:ceiling_cut, :] = 0

    # ======================================
    # MORPHOLOGY
    # ======================================

    kernel = np.ones(
        (5,5),
        np.uint8
    )

    crop_mask = cv2.morphologyEx(
        crop_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    crop_mask = cv2.dilate(
        crop_mask,
        kernel,
        iterations=1
    )

    # ======================================
    # FIND CONTOURS
    # ======================================

    contours, _ = cv2.findContours(
        crop_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    total_area = 0
    valid_y = []

    roi_area = (
        crop_mask.shape[0]
        *
        crop_mask.shape[1]
    )

    for c in contours:

        area = cv2.contourArea(c)

        if area > 1000:

            x, y, w_box, h_box = cv2.boundingRect(c)

            total_area += area

            valid_y.append(y)

    # ======================================
    # HEIGHT CALC
    # ======================================

    if len(valid_y) > 0:

        highest_y = int(
            np.percentile(
                valid_y,
                15
            )
        )

    else:

        highest_y = crop_mask.shape[0]

    height_fill = (
        (
            crop_mask.shape[0]
            -
            highest_y
        )
        /
        crop_mask.shape[0]
    ) * 100

    volume_fill = (
        total_area
        /
        roi_area
    ) * 100

    # ======================================
    # ROUND
    # ======================================

    height_fill = int(
        round(height_fill / 5) * 5
    )

    volume_fill = int(
        round(volume_fill / 5) * 5
    )

    height_fill = max(
        0,
        min(100, height_fill)
    )

    volume_fill = max(
        0,
        min(100, volume_fill)
    )

    if height_fill >= 85:
        height_fill = 100

    if volume_fill >= 85:
        volume_fill = 100

    # ======================================
    # RESULT
    # ======================================

    result = {
        "status": "ok",
        "fill_rate_height": height_fill,
        "fill_rate_volume": volume_fill
    }

    print("RESULT =", result)
    print("================================")

    return jsonify(result)

except Exception as e:

    print("ERROR =", str(e))

    return jsonify({
        "status": "error",
        "message": str(e)
    })
```

if **name** == "**main**":
app.run(
host="0.0.0.0",
port=10000
)
