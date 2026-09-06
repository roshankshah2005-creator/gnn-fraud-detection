# 🛡️ GNN Fraud Detection & Review Desk

A lightweight, graph-based transaction risk assessment pipeline built with **PyTorch Geometric (PyG)** and an interactive **Streamlit** dashboard. Designed to analyze transactional relationships and flag high-risk activity through a clean, custom-styled review desk UI.

---

## What It Does
* **Graph-Ready Architecture:** Structured to map financial transactions as nodes and edges, allowing Graph Neural Networks (GNNs) to evaluate structural risk patterns.
* **Transparent Risk Scoring:** Combines calibrated baseline heuristics (sender historical records, transaction thresholds) with an integration hook for trained PyTorch model weights.
* **Custom Streamlit UI:** Features a custom dark-mode "Fraud Review Desk" interface with hidden toolbars for a clean, standalone web application look.

---

## Project Structure

```text
gnn-fraud-detection/
├── app.py                     # Streamlit web dashboard (Frontend)
├── requirements.txt           # Python dependencies for local & cloud environments
├── data/
│   └── raw/
│       └── transactions.csv   # Raw transactional dataset
├── saved_models/
│   └── gnn_fraud_model.pt     # Trained GNN weights checkpoint
├── src/
│   ├── api/                   # Optional FastAPI backend service
│   ├── data/                  # CSV-to-PyG graph conversion utilities
│   ├── models/                # GNN model architecture definitions
│   └── training/              # Model training scripts
└── tests/                     # Automated unit and integration tests
