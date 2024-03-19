import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from .configs.basic.build_layer import build_activation_layer, build_norm_layer
from .configs.basic.gcn import Graph
from .configs.common import BaseModule, Attention_Layer, Frame_Att, Joint_Att, ST_Joint_Att, CrissCrossAttention
class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, max_graph_distance):
        super(GraphConv, self).__init__()
        self.s_kernel_size = max_graph_distance + 1
        self.gcn = nn.Conv2d(in_channels, out_channels*self.s_kernel_size, 1)

    def forward(self, x, A):
        x = self.gcn(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc//self.s_kernel_size, t, v)
        x = torch.einsum('nkctv, kvw->nctw', (x, A[:self.s_kernel_size])).contiguous()
        return x


class Spatial_Layer(BaseModule):
    def __init__(self, in_channels, out_channels, max_graph_distance, act_cfg, dims, reduction=4, **kwargs):
        super(Spatial_Layer, self).__init__()
        inter_channels = out_channels // reduction
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
        )
        self.map = nn.Identity()
        self.gcn_res = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 1),
            nn.BatchNorm2d(inter_channels),
        )
        self.down_sample = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        self.gcn = GraphConv(inter_channels, inter_channels, max_graph_distance)
        self.gcn_bn = nn.BatchNorm2d(inter_channels)
        self.up_sample = nn.Sequential(
            nn.Conv2d(inter_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.act = build_activation_layer(act_cfg)
        
    def forward(self, x, A):
        final_res = self.residual(x)
        x = self.down_sample(x)
        gcn_res = self.act(self.gcn_res(x))
        gcn_map = self.map(x)
        x = self.gcn(x, A)
        x = self.gcn_bn(x) + gcn_res + gcn_map
        x = self.up_sample(x)
        x = x + final_res
        return x

class Temporal_Layer(BaseModule):
    def __init__(self, channels, temporal_window_size, act_cfg, stride=1, reduction=4, **kwargs):
        super(Temporal_Layer, self).__init__()
        padding = ((temporal_window_size - 1) // 2, 0)
        inter_channels = channels // reduction
        self.residual = nn.Sequential(
            nn.Conv2d(channels, channels, 1, (stride, 1)),
            nn.BatchNorm2d(channels),
        )
        self.down_sample = nn.Sequential(
            nn.Conv2d(channels, inter_channels, 1),
            nn.BatchNorm2d(inter_channels),
        )
        self.plane = nn.Sequential(
                nn.Conv2d(inter_channels, inter_channels, (temporal_window_size, 1), (stride, 1), padding),
                nn.BatchNorm2d(inter_channels),
            )
        self.up_sample = nn.Sequential(
                nn.Conv2d(inter_channels, channels, 1),
                nn.BatchNorm2d(channels),
            )
        self.act = build_activation_layer(act_cfg)
        self.attn = Frame_Att()
    def forward(self, x):
        final_res = self.residual(x)
        x = self.act(self.down_sample(x))
        x = self.attn(x)
        x = self.act(self.plane(x))
        x = self.attn(x)
        x = self.act(self.up_sample(x))
        x = self.act(x + final_res)
        return x


class RepGCNBlock(BaseModule):
    def __init__(self, in_channels, out_channels, temp_act_cfg, spa_act_cfg, dims, kernel_size, stride=1, **kwargs):
        super(RepGCNBlock, self).__init__()

        A = torch.tensor(Graph(layout='coco', strategy='spatial').A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        if not len(kernel_size) == 2:
            logging.info('')
            logging.error('Error: Please check whether len(kernel_size) == 2')
            raise ValueError()
        if not kernel_size[0] % 2 == 1:
            logging.info('')
            logging.error('Error: Please check whether kernel_size[0] % 2 == 1')
            raise ValueError()
        temporal_window_size, max_graph_distance = kernel_size
        self.scn = Spatial_Layer(in_channels, out_channels, max_graph_distance, spa_act_cfg, dims, **kwargs)
        self.attn = CrissCrossAttention(in_dim=out_channels, A=A)
        self.tcn = Temporal_Layer(out_channels, temporal_window_size, temp_act_cfg, stride, **kwargs)
        self.edge = nn.Parameter(torch.ones_like(A))

    def forward(self, x):
        x = self.scn(x, self.A * self.edge)
        x = self.attn(x)
        x = self.tcn(x)
        return x


class RepGCN(BaseModule):
    arch_settings={
        's72': {
                'kernel_size': [[7, 2], [7, 2], [7, 2], [7, 2], [7, 2]],
                'stride': [1, 2, 1, 2, 1],
                'temporal_length': [60, 30, 30, 15, 15]
            },
        's92': {
            'kernel_size': [[9, 2], [9, 2], [9, 2], [9, 2], [9, 2]],
            'stride': [1, 2, 1, 2, 1]
        },
        's9_32': {
            'kernel_size': [[3, 2], [9, 2], [3, 2], [9, 2], [9, 2]],
            'stride': [1, 2, 2, 2, 1]
        },
    }
    
    parts={
        "cdut_gait": [(1, 0), (2, 1), (3, 2), (4, 2), (5, 4), (6, 5), (7, 6), (8, 7), (9, 8), (10, 7), (11, 2), (12, 11),
                             (13, 12), (14, 13), (15, 14), (16, 15), (17, 14), (18, 0), (19, 18), (20, 19), (21, 20), (22, 0), (23, 22), (24, 23), (25, 24),
                             (26, 3), (27, 26), (28, 26), (29, 26), (30, 26), (31, 26)],
        "coco": [(0,1), (0,2), (1,3), (2,4), (3,5), (4,6), (5,6),
                             (5,7), (7,9), (6,8), (8,10), (5,11), (6, 12), (11, 12),
                             (11, 13), (13, 15), (12, 14), (14, 16)]
    }

    def __init__(self,
                 in_channels,
                 temp_act_cfg,
                 spa_act_cfg,
                 attn_cfg,
                 arch='s72',
                 **kwargs
                 ):
        super(RepGCN, self).__init__()
        self.arch = self.arch_settings[arch]
        self.attn = attn_cfg
        kernel_size = self.arch['kernel_size']
        stride = self.arch['stride']
        temporal_length = self.arch['temporal_length']
        network = [RepGCNBlock(in_channels=in_channels, out_channels=64, temp_act_cfg=temp_act_cfg, spa_act_cfg=spa_act_cfg, dims=60, kernel_size=kernel_size[0], stride=stride[0])]
        
        network += [RepGCNBlock(in_channels=64, out_channels=64, temp_act_cfg=temp_act_cfg, spa_act_cfg=spa_act_cfg, dims=60, kernel_size=kernel_size[1], stride=stride[1])]
 
        network += [RepGCNBlock(in_channels=64, out_channels=32, temp_act_cfg=temp_act_cfg, spa_act_cfg=spa_act_cfg, dims=30, kernel_size=kernel_size[2], stride=stride[2])]
        
        network += [RepGCNBlock(in_channels=32, out_channels=128, temp_act_cfg=temp_act_cfg, spa_act_cfg=spa_act_cfg, dims=30, kernel_size=kernel_size[3], stride=stride[3])]

        network += [RepGCNBlock(in_channels=128, out_channels=128, temp_act_cfg=temp_act_cfg, spa_act_cfg=spa_act_cfg, dims=15, kernel_size=kernel_size[4], stride=stride[4])]
        
        self.network = nn.ModuleList(network)
        self.gap = nn.AdaptiveAvgPool2d(1)
    def forward_backbone(self, x):
        x = x.permute(0, 3, 1, 2)
        for layer in self.network:
            x = layer(x)
        return x

    def forward(self, x):
        fc = self.forward_backbone(x)
        fc = self.gap(fc)
        fc = fc.squeeze(2).squeeze(2)
        fc = F.normalize(fc, dim=1, p=2)
        return fc