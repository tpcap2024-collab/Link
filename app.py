from flask import Flask, request, jsonify
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

        print("JSON:", data)

        if not data:
            return jsonify({
                "status": "error",
                "message": "No JSON"
            }), 400

        photo = data.get("photo")
        datetime_value = data.get("datetime")

        print("PHOTO:", photo)
        print("DATETIME:", datetime_value)

        if not photo:
            return jsonify({
                "status": "error",
                "message": "photo empty"
            }), 400

        return jsonify({
            "status": "success",
            "photo": photo,
            "datetime": datetime_value
        })

    except Exception as e:

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
