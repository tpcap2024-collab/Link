from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return "TFR AI Running"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    print("=" * 80)
    print(data)

    return jsonify({
        "status": "success"
    })

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=10000
    )
