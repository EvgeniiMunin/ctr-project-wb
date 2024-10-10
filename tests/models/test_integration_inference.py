import numpy as np
import logging
import sys
from fastapi.testclient import TestClient

from src.data.make_dataset import read_data
from app import app


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "it is entry point of our predictor"


def test_inference_model(processed_dataset_path: str):
    data = read_data(processed_dataset_path)
    request_data = [
        x.item() if isinstance(x, np.generic) else x for x in data.iloc[0].tolist()
    ]

    logger.info(f"request_data: {request_data}")

    response = client.post(
        "/predict/", json={"data": [request_data], "features": list(data.columns)}
    )

    assert response.status_code == 200
    assert response.json() is not None
    assert response.json()[0]['device_ip'] is not None
    assert response.json()[0]['click_proba'] is not None
    assert 0 <= response.json()[0]['click_proba'] < 1


def test_inference_model_invalid_data(processed_dataset_path: str):
    data = read_data(processed_dataset_path)
    data.drop(["site_id"], axis=1, inplace=True)
    request_data = [
        x.item() if isinstance(x, np.generic) else x for x in data.iloc[0].tolist()
    ]

    response = client.post(
        "/predict/", json={"data": [request_data], "features": list(data.columns)}
    )

    assert response.status_code == 400
    assert "Missing features in schema" in response.json()["detail"]
