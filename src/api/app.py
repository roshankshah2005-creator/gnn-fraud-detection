import torch
from fastapi import FastAPI
from pydantic import BaseModel
from src.models.gnn_model import FraudGNN

app=FastAPI(title="GNN Fraud Detection API",version="1.0")

class TransactionRequest(BaseModel):
    sender_id:str
    receiver_id:str
    amount:float
    sender_historical_fraud_count: int = 0

#Load the trained model architecture and weights upon startup
in_channels=2
hidden_channels=16
out_channels=2

model=FraudGNN(in_channels=in_channels,hidden_channels=hidden_channels,out_channels=out_channels)

#Load saved weights safely
checkpoint_path = "saved_models/fraud_gnn_checkpoint.pt"
try:
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    print("Loaded trained GNN model weights successfully!")
except Exception as e:
    print(f"Warning:Could not load checkpoint({e}). Running with random weeights.")

@app.get("/")
def home():
    return{"message":"GNN Fraud Detection Pipeline is live and running!"}

@app.post("/predict")
def predict_fraud(transaction: TransactionRequest):
    #For interface, we simulate a small neighbourhood tensor context for the incoming transaction
    #In a full production pipeline,this pulls live node features from your graph/feature store.
    user_amount=float(transaction.amount)
    fraud_history = float(transaction.sender_historical_fraud_count)

    scaled_amount = min(user_amount / 10000.0, 1.0)       
    scaled_history = min(fraud_history / 50.0, 1.0)
    
    input_node_features = torch.tensor([[scaled_amount, scaled_history]], dtype=torch.float)
    mock_edge_index = torch.tensor([[0],[0]],dtype=torch.long)

    with torch.no_grad():
        outputs=model(input_node_features,mock_edge_index)
        probabilities=torch.softmax(outputs,dim=1)
        base_prob=probabilities[0][1].item()

    model_score = base_prob * 30.0
    history_boost = min(fraud_history * 2, 60.0)     
    amount_boost = min(user_amount / 2000.0, 15.0)
    final_fraud_probability = model_score + history_boost + amount_boost
    final_fraud_probability = max(0.0,min(final_fraud_probability,100.0))

    if fraud_history == 0:
        final_fraud_probability = min(user_amount / 2000.0, 5.0)

    final_fraud_probability = max(0.0, min(final_fraud_probability, 100.0))

    is_risky=final_fraud_probability > 50.0

    return {
        "sender": transaction.sender_id,
        "receiver":transaction.receiver_id,
        "amount":transaction.amount,
        "fraud_probability":round(final_fraud_probability,2),
        "status":"FRAUD_ALERT" if is_risky else "APPROVED"
    }