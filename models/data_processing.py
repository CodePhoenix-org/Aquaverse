# data_processing.py
from models.disaster_prediction import predict_disaster

def process_float_record(record: dict):
    """
    Processes a single ARGO float record by running disaster prediction.
    
    Example record:
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
    result = predict_disaster(record)

    # Attach prediction to record
    record["disaster_prediction"] = result["prediction"]
    record["prediction_confidence"] = result["confidence"]

    return record


def process_multiple_records(records: list):
    """
    Process a list of ARGO float records.
    
    Parameters:
        records (list): list of dicts
    
    Returns:
        list of dicts with prediction fields
    """
    return [process_float_record(r) for r in records]


if __name__ == "__main__":
    # Test with sample float data
    float_sample = {
        "latitude": -40.3,
        "longitude": 73.4,
        "depth": 980,
        "temperature": 3.8,
        "salinity": 34.5,
        "oxygen": 210,
        "chlorophyll": 0.4
    }

    processed = process_float_record(float_sample)
    print(processed)
