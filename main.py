import torch.nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.datasets import Planetoid

print(Planetoid)

dataset = Planetoid(root="dataset",name="Cora")

# print(dataset.num_classes)
# print(dataset.num_node_features)
# print(dataset.num_edge_features)
# print(dataset.data.x.shape)
# print(dataset[0])

class Net(nn.module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv = SAGEConv(dataset.num_features,
                             dataset.num_classes,
                             aggr="max")
    def forward(self):
        x = self.conv(data.x,data.edge_index)
        return F.log_softmax(x,dim=1)


import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(dataset.num_features, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.classifier = Linear(2, dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()  # Final GNN embedding space.
        
        # Apply a final (linear) classifier.
        out = self.classifier(h)

        return out, h














