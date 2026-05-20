from flask import Flask, request, jsonify
from src.components.prediction import PredictionPipeline

app = Flask(__name__)

prediction_pipeline = PredictionPipeline()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    try:
        result = prediction_pipeline.predict(file.read())
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)