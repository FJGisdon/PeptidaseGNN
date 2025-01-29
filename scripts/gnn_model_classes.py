import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv

# Define the GCN model classes
class GCN_h1(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels_1, out_channels, dropout_rate_1):
        super().__init__()
        self.dropout_rate_1 = dropout_rate_1
        self.conv1 = GCNConv(in_channels, hidden_channels_1)
        self.conv2 = GCNConv(hidden_channels_1, out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate_1, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return F.log_softmax(x, dim=1)

class GCN_h2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels_1, hidden_channels_2, out_channels, dropout_rate_1, dropout_rate_2):
        super().__init__()
        self.dropout_rate_1 = dropout_rate_1
        self.dropout_rate_2 = dropout_rate_2
        self.conv1 = GCNConv(in_channels, hidden_channels_1)
        self.conv2 = GCNConv(hidden_channels_1, hidden_channels_2)
        self.conv3 = GCNConv(hidden_channels_2, out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate_1, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate_2, training=self.training)
        x = self.conv3(x, edge_index, edge_attr)
        return F.log_softmax(x, dim=1) 
