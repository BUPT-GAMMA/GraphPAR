import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layer, activation='leakyrelu'):
        super().__init__()
        self.num_layer = num_layer
        self.gnns = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()

        for layer in range(self.num_layer):
            if layer == 0:
                gcn = GCNConv(input_dim, hidden_dim, cached=True)
            else:
                gcn = GCNConv(hidden_dim, hidden_dim, cached=True)
            self.gnns.append(gcn)

            if activation == 'relu':
                self.activations.append(nn.ReLU())
            elif activation == 'leakyrelu':
                self.activations.append(nn.LeakyReLU())
            elif activation == 'prelu':
                self.activations.append(nn.PReLU())

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        h = x
        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index)
            h = self.activations[layer](h)
        return h


class Adapter(torch.nn.Module):
    def __init__(self, input_dim, activation):
        super(Adapter, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(input_dim, input_dim // 2),
                                 nn.PReLU() if activation == "prelu" else nn.LeakyReLU(),
                                 nn.Linear(input_dim // 2, input_dim))

        self.output_layer = nn.BatchNorm1d(input_dim, affine=False)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        z = self.mlp(x)
        return self.output_layer(z)


class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, num_classes)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        ret = self.fc1(x)
        return ret
