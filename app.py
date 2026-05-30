```python
from flask import Flask, request, jsonify
import requests
import cv2
import numpy as np
import traceback

app = Flask(__name__)

@app.route("/")
def home():
    return "TFR AI Running"

@app.route("/predict", methods=["POST"])
def predict():

    try:

        print("=" * 60)

        print("HEADERS:", dict(request.headers))

        print("RAW DATA:", request.data)

        data = request.get_json(silent=True)

        print("JSON:", data)

        if not data:
            return jsonify({
                "status": "error",
                "message": "No JSON received"
            }), 400

        image_url = data.get("linkapp")

        print("IMAGE URL:", image_url)

        if not image_url:
            return jsonify({
                "status": "error",
                "message": "link is empty"
            }), 400

        response = requests.get(
            image_url,
            timeout=30,
            allow_redirects=True
        )

        print("DOWNLOAD STATUS:", response.status_code)

        print(
            "CONTENT TYPE:",
            response.headers.get("Content-Type")
        )

        print(
            "CONTENT LENGTH:",
            len(response.content)
        )

        if response.status_code != 200:

            return jsonify({
                "status": "error",
                "message": f"download failed {response.status_code}"
            }), 400

        img_array = np.asarray(
            bytearray(response.content),
            dtype=np.uint8
        )

        img = cv2.imdecode(
            img_array,
            cv2.IMREAD_COLOR
        )

        print(
            "IMAGE OBJECT:",
            img is not None
        )

        if img is None:

            return jsonify({
                "status": "error",
                "message": "cannot decode image"
            }), 400

        h, w = img.shape[:2]

        print(
            "IMAGE SIZE:",
            w,
            "x",
            h
        )

        return jsonify({
            "status": "success",
            "width": int(w),
            "height": int(h)
        })

    except Exception as e:

        print("EXCEPTION:")
        print(traceback.format_exc())

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=10000
    )
```
