import torch
import torch_geometric
import networkx as x
import sklearn

print("PyTorch Version:",torch.__version__)
print("PyTorch Geometric Version:",torch_geometric.__version__)
print("Is CUDA available:",torch.cuda.is_available())