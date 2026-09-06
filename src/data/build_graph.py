import pandas as pd
import torch
from torch_geometric.data import Data
import os

print("--- Step 1: Creating Sample Transaction Data ---")
def load_and_build_graph(csv_path="data/raw/transactions.csv"):
    print(f"Loading transaction data from {csv_path}...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find transaction CSV at {csv_path}. Plaease ensure it exists.")
    df=pd.read_csv(csv_path)

    print("\n--- Step 2: Mapping Accounts to Node Indices ---")
    unique_accounts=pd.concat([df['sender'],df['receiver']]).unique()
    account_to_idx = {acc : idx for idx,acc in enumerate(unique_accounts)}

    print(f"Account mapping: {account_to_idx}")

    src_nodes=torch.tensor([account_to_idx[acc] for acc in df['sender']], dtype=torch.long)
    dst_nodes=torch.tensor([account_to_idx[acc] for acc in df['receiver']], dtype=torch.long)

    edge_index=torch.stack([src_nodes,dst_nodes],dim=0)
    print(f"Edge Index Tensor Shape: {edge_index.shape}")

    print("\n-- Step 3: Constructing the PyTorch Geometric Data Object ---")
    num_nodes=len(unique_accounts)
    sent_amounts = df.groupby('sender')['amount'].sum().reindex(unique_accounts, fill_value=0.0).values
    recv_amounts = df.groupby('receiver')['amount'].sum().reindex(unique_accounts, fill_value=0.0).values
    
    node_features = torch.tensor(
        list(zip(sent_amounts, recv_amounts)), 
        dtype=torch.float
    )
    
    # Normalize features for stable training convergence
    if node_features.max() > 0:
        node_features = node_features / node_features.max()

    fraudulent_senders = set(df[df['is_fraud'] == 1]['sender'].unique())
    labels = torch.tensor([1 if acc in fraudulent_senders else 0 for acc in unique_accounts], dtype=torch.long)
    
    graph_data = Data(x=node_features, edge_index=edge_index, y=labels)
    
    print(f"Successfully built graph: {num_nodes} nodes, {edge_index.shape[1]} edges.")
    return graph_data

if __name__ == "__main__":
    data=load_and_build_graph()
    print(data)
