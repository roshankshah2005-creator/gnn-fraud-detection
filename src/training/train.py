import torch
import os
import torch.nn.functional as F
from src.models.gnn_model import FraudGNN
from src.data.graph_builder import load_and_build_graph

def train_fraud_detector():
    print("--- Initializing GNN Training Pipeline ---")

    #1.Load real graph data from csv via our graph builder
    graph_data=load_and_build_graph("data/raw/transactions.csv")

    x=graph_data.x
    edge_index=graph_data.edge_index
    labels=graph_data.y

    in_channels=x.shape[1]
    hidden_channels=16
    out_channels=2

    #2.Initialize Model, Loss Function, and Optimizer
    model=FraudGNN(in_channels=in_channels,out_channels=out_channels,hidden_channels=hidden_channels)
    optimizer=torch.optim.Adam(model.parameters(),lr=0.01,weight_decay=5e-4)
    criterion=torch.nn.CrossEntropyLoss()

    #3.Training Loop
    model.train()
    epochs=50

    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        out=model(x,edge_index)

        loss=criterion(out,labels)
        loss.backward()
        optimizer.step()

        if (epoch + 1)%10 == 0:
            print(f"Epoch {epoch+1}/{epoch} | Loss: {loss.item():.4f}")

    print("\nTraining completed successfully!")

    os.makedirs("saved_models",exist_ok=True)

    #Save the trained model checkpoint
    torch.save(model.state_dict(),"saved_models/fraud_gnn_checkpoint.pt")
    print("Model checkpoint saved to saved_models/fraud_gnn_checkpoint.pt")

if __name__ =="__main__":
    train_fraud_detector()