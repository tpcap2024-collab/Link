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

        print("=" * 80)

        data = request.get_json(silent=True)

        print("JSON:")
        print(data)

        if not data:
            return jsonify({
                "status": "error",
                "message": "No JSON received"
            }), 400

        image_url = data.get("link")

        print("=" * 80)
        print("IMAGE URL:")
        print(image_url)

        if not image_url:
            return jsonify({
                "status": "error",
                "message": "link empty"
            }), 400

        response = requests.get(
            image_url,
            timeout=60,
            allow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        print("=" * 80)
        print("STATUS:", response.status_code)
        print("FINAL URL:", response.url)
        print("CONTENT TYPE:", response.headers.get("Content-Type"))
        print("SIZE:", len(response.content))

        if response.status_code != 200:

            return jsonify({
                "status": "error",
                "message": f"download failed {response.status_code}"
            }), 400

        img_array = np.frombuffer(
            response.content,
            np.uint8
        )

        img = cv2.imdecode(
            img_array,
            cv2.IMREAD_COLOR
        )

        print("=" * 80)
        print("IMAGE OBJECT:", img is not None)

        if img is None:

            return jsonify({
                "status": "error",
                "message": "decode failed"
            }), 400

        h, w = img.shape[:2]

        print("IMAGE SIZE:", w, "x", h)

        return jsonify({
            "status": "success",
            "width": int(w),
            "height": int(h)
        })

    except Exception as e:

        print("=" * 80)
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
