import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

class FraudGNN(nn.Module):
    def __init__(self,in_channels,hidden_channels,out_channels):
        super(FraudGNN,self).__init__()

        self.conv1=SAGEConv(in_channels,hidden_channels)
        self.conv2=SAGEConv(hidden_channels,out_channels)
        self.relu=nn.ReLU()

    def forward(self,x,edge_index):

        x=self.conv1(x,edge_index)
        x=self.relu(x)

        x=self.conv2(x,edge_index)
        return x

if __name__ == "__main__":
    x_dummy=torch.randn((5,2),dtype=torch.float)

    edge_index_dummy=torch.tensor([
        [0,1,2,3],
        [1,2,0,4]
    ],dtype=torch.long)

    model = FraudGNN(in_channels=2,hidden_channels=16,out_channels=2)

    out=model(x_dummy,edge_index_dummy)
    print("Successfully ran GNN forward pass!")
    print("Output embedding shape:",out.shape)