# To run, first start the FastAPI server by running the LightFM_API.py script. 
# In a new terminal run the test cases using the following command: 
# pytest tests/test_api_endpoints.py

import pytest
import requests

API_BASE_URL = "http://172.30.253.127:8000"

def test_predict_endpoint():
    # Test case for /predict endpoint
    payload = {
        "user_id": "136336",
        "start_index": 0,
        "no_recommendations": 5
    }
    response = requests.post(f"{API_BASE_URL}/predict", json=payload)
    assert response.status_code == 200
    assert "news" in response.json()

def test_all_news_endpoint():
    # Test case for /all endpoint
    payload = {
        "start_index": 0,
        "no_recommendations": 10
    }
    response = requests.post(f"{API_BASE_URL}/all", json=payload)
    assert response.status_code == 200
    assert "news" in response.json()

def test_root_endpoint():
    # Test case for root endpoint
    response = requests.get(API_BASE_URL)
    assert response.status_code == 200
    assert response.json()["message"] == "You get used to it, I don't even see the code, all I see is blonde, brunette, redhead."
