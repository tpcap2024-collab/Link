from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
return "TFR AI Running"

@app.route("/predict", methods=["POST"])
def predict():
return jsonify({
"status": "ok"
})

if **name** == "**main**":
app.run(
host="0.0.0.0",
port=10000
)
