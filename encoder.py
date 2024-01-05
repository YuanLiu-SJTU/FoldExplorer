import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool, SAGPooling
from einops import rearrange
import torch.nn.functional as F


class GATModel(nn.Module):
    def __init__(self, in_channels, graph_out_channels, esm_out_channels, heads=8, use_gat=True, use_esm=True, use_ss=False):
        super(GATModel, self).__init__()
        hidden_channels = [64, 128, 256, 512]
        self.hidden_channels = hidden_channels
        self.use_gat = use_gat
        self.use_esm = use_esm
        self.use_ss = use_ss
        self.CNN_conv2d = nn.Sequential(
            nn.Conv2d(1, hidden_channels[0], kernel_size=(1, in_channels)),
            nn.BatchNorm2d(hidden_channels[0]),
            nn.LeakyReLU(1e-2, inplace=True),
        )
        self.GAT_conv = nn.ModuleList()
        self.pool = nn.ModuleList()
        self.graph_mlp = nn.ModuleList()
        self.relu = nn.ModuleList()
        for i in range(len(hidden_channels)-1):
            self.GAT_conv.append(GATConv(hidden_channels[i], hidden_channels[i+1], heads, concat=False, dropout=0.2))
            self.relu.append(nn.LeakyReLU(1e-2, inplace=True))
            self.pool.append(SAGPooling(in_channels=hidden_channels[i+1], ratio=0.5))
            self.graph_mlp.append(nn.Linear(hidden_channels[i+1], hidden_channels[-1]))
        self.fc_mlp = nn.Sequential(
            nn.Linear(graph_out_channels, graph_out_channels),
        )
        self.esm_mlp = nn.Sequential(
            nn.Linear(1280, esm_out_channels),
        )
        
    def forward(self, data):
        graph_out, esm_out = 0, 0
        if self.use_gat:
            graph_fea_list = []
            x, edge_index, batch = data.x, data.edge_index, data.batch
            x = rearrange(x, 'n d -> n () () d')
            x = self.CNN_conv2d(x).squeeze().squeeze()
            for i in range(len(self.GAT_conv)):
                x = self.GAT_conv[i](x, edge_index)
                graph_x = global_mean_pool(x, batch)
                graph_fea = self.graph_mlp[i](graph_x)
                graph_fea_list.append(graph_fea)
                if i < (len(self.GAT_conv)-1):
                    x = self.relu[i](x)
                    x, edge_index, _, batch, _, _ = self.pool[i](x, edge_index, None, batch)
            graph_out = torch.stack(graph_fea_list)
            graph_out = torch.sum(graph_out, dim=0)
            graph_out = self.fc_mlp(graph_out)
        if self.use_esm:
            esm_out = self.esm_mlp(data.esm_feature)
        out = graph_out + esm_out
        return out
