import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from fastapi.testclient import TestClient
from src.api.app import app
from src.data.build_graph import load_and_build_graph

# Initialize FastAPI TestClient
client = TestClient(app)

def test_home_route():
    """Test if the home route is live and returns the correct message."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "GNN Fraud Detection Pipeline is live and running!"}

def test_predict_safe_transaction():
    """Test a clean transaction with 0 historical fraud to ensure safe baseline."""
    payload = {
        "sender_id": "Acc_Safe",
        "receiver_id": "Acc_Receiver",
        "amount": 250.0,
        "sender_historical_fraud_count": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert data["sender"] == "Acc_Safe"
    assert data["status"] == "APPROVED"
    assert data["fraud_probability"] <= 10.0 

def test_predict_risky_transaction():
    """Test a high-risk transaction with massive fraud history."""
    payload = {
        "sender_id": "Acc_BadActor",
        "receiver_id": "Acc_Victim",
        "amount": 8000.0,
        "sender_historical_fraud_count": 25
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert data["sender"] == "Acc_BadActor"
    assert data["status"] == "FRAUD_ALERT"
    assert data["fraud_probability"] > 50.0 

def test_graph_builder_pipeline():
    """Test if the graph builder successfully processes CSV data into a PyG Data object."""
    graph_data = load_and_build_graph("data/raw/transactions.csv")
    assert graph_data.x is not None
    assert graph_data.edge_index is not None
    assert graph_data.y is not None
    assert graph_data.edge_index.shape[0] == 2 