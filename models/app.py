# app.py
from fastapi import FastAPI, request, jsonify
from data_processing import process_float_record

app = FastAPI(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON with ARGO float fields.
    Example:
    {
        "latitude": -40.3,
        "longitude": 73.4,
        "depth": 980,
        "temperature": 3.8,
        "salinity": 34.5,
        "oxygen": 210,
        "chlorophyll": 0.4
    }
    """
    try:
        data = request.get_json()
        result = process_float_record(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
