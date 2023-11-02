import torch_geometric
from torch_geometric.datasets import Planetoid

print(Planetoid)

dataset = Planetoid(root="dataset",name="Cora")


