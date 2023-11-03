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

:



