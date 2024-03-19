import torch
from torch import nn
from .base_module import BaseModule
from .depthwise_separable_conv_module import DepthwiseSeparableConvModule as DWConv
from .triplet_attn import BasicConv, ZPool, AttentionGate
from einops import rearrange, repeat
from torch import einsum
class Attention_Layer(nn.Module):
    def __init__(self, out_channels, att_type, act, **kwargs):
        super(Attention_Layer, self).__init__()

        __attention = {
            'stja': ST_Joint_Att,
            'pa': Part_Att,
            'ca': Channel_Att,
            'fa': Frame_Att,
            'ja': Joint_Att,
            'oa': A_Attn
        }

        self.att = __attention[att_type](channel=out_channels, **kwargs)
        # self.bn = nn.BatchNorm2d(out_channels)
        # self.act = act

    def forward(self, x):
        res = x
        x = x * self.att(x)
        return x + res

class Part_Att(BaseModule):
    def __init__(self, channel, parts, **kwargs):
        super(Part_Att, self).__init__()

        self.parts = parts
        self.joints = get_corr_joints(parts)
        inter_channel = channel // 4

        self.fcn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, inter_channel, kernel_size=1),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channel, channel*len(self.parts), kernel_size=1),
        )

        self.softmax = nn.Softmax(dim=-1)
        self.bn = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        N, C, T, V = x.size()
        res = x
        x_att = self.softmax(self.fcn(x).view(N, C, len(self.parts)))
        x_att = torch.split(x_att, 1, dim=-1)
        x_att = [x_att[self.joints[i]].expand_as(x[:, :, :, i]) for i in range(V)]
        x_att = torch.stack(x_att, dim=-1)
        return self.relu(self.bn(x * x_att) + res)


class Part_Share_Att(nn.Module):
    def __init__(self, channel, parts, **kwargs):
        super(Part_Share_Att, self).__init__()

        self.parts = parts
        self.joints = get_corr_joints(parts)

        inter_channel = channel // 4

        self.part_pool = nn.Sequential(
            nn.Conv2d(channel, inter_channel, kernel_size=1),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.fcn = nn.Sequential(
            nn.Conv2d(inter_channel, inter_channel, kernel_size=1),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channel, channel*len(self.parts), kernel_size=1),
        )

        self.softmax = nn.Softmax(dim=-1)
        self.bn = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        N, C, T, V = x.size()
        res = x

        x_split = [self.part_pool(x[:,:,:,part]) for part in self.parts]
        x_att = self.softmax(self.fcn(sum(x_split)).view(N, C, len(self.parts)))
        x_att = torch.split(x_att, 1, dim=-1)
        x_att = [x_att[self.joints[i]].expand_as(x[:, :, :, i]) for i in range(V)]
        x_att = torch.stack(x_att, dim=-1)
        return self.relu(self.bn(x * x_att) + res)

class Part_Conv_Att(nn.Module):
    def __init__(self, channel, parts, **kwargs):
        super(Part_Conv_Att, self).__init__()

        self.parts = parts
        self.joints = get_corr_joints(parts)

        inter_channel = channel // 4

        self.part_pool = nn.ModuleList([nn.Sequential(
            nn.Conv2d(channel, inter_channel, kernel_size=1),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        ) for _ in range(len(self.parts))])

        self.fcn = nn.Sequential(
            nn.Conv2d(inter_channel, inter_channel, kernel_size=1),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channel, channel*len(self.parts), kernel_size=1),
        )

        self.softmax = nn.Softmax(dim=-1)
        self.bn = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        N, C, T, V = x.size()
        res = x

        x_split = [pool(x[:,:,:,part]) for part, pool in zip(self.parts, self.part_pool)]
        x_att = self.softmax(self.fcn(sum(x_split)).view(N, C, len(self.parts)))
        x_att = torch.split(x_att, 1, dim=-1)
        x_att = [x_att[self.joints[i]].expand_as(x[:,:,:,i]) for i in range(V)]
        x_att = torch.stack(x_att, dim=-1)
        return self.relu(self.bn(x * x_att) + res)

class Channel_Att(nn.Module):
    def __init__(self, channel, **kwargs):
        super(Channel_Att, self).__init__()

        self.fcn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel//4, kernel_size=1),
            nn.BatchNorm2d(channel//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//4, channel, kernel_size=1),
            nn.Softmax(dim=1)
        )

        self.bn = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x
        x_att = self.fcn(x).squeeze()
        return self.relu(self.bn(x * x_att[:, :, None, None]) + res)




class Frame_Att(nn.Module):
    def __init__(self, **kwargs):
        super(Frame_Att, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fcn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=(7, 1), padding=(3, 0), dilation=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=(9, 1), padding=(4, 0), dilation=1),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        orig = x
        x = x.transpose(1, 2)
        x = torch.cat([self.avg_pool(x), self.max_pool(x)], dim=2).transpose(1, 2)
        x_att = self.fcn(x).squeeze()
        res = orig * x_att[:, None, :, None] + orig
        return res

class ST_Joint_Att(nn.Module):
    def __init__(self, channel=48, reduct_ratio=4, bias=False, **kwargs):
        super(ST_Joint_Att, self).__init__()

        inner_channel = channel // reduct_ratio

        self.fcn = nn.Sequential(
            nn.Conv2d(channel, inner_channel, kernel_size=1, bias=bias),
            nn.BatchNorm2d(inner_channel),
            nn.Hardswish(),
        )
        self.conv_t = nn.Conv2d(inner_channel, channel, kernel_size=1)
        self.conv_v = nn.Conv2d(inner_channel, channel, kernel_size=1)

    def forward(self, x):
        N, C, T, V = x.size()
        x_t = x.mean(3, keepdims=True)
        x_v = x.mean(2, keepdims=True).transpose(2, 3)
        x_att = self.fcn(torch.cat([x_t, x_v], dim=2))
        x_t, x_v = torch.split(x_att, [T, V], dim=2)
        x_t_att = self.conv_t(x_t).sigmoid()
        x_v_att = self.conv_v(x_v.transpose(2, 3)).sigmoid()
        x_att = x_t_att * x_v_att
        return x_att * x

def get_corr_joints(parts):
    num_joints = max([max(part) for part in parts]) + 1
    res = []
    for i in range(num_joints):
        for j in range(len(parts)):
            if i in parts[j]:
                res.append(j)
                break
    return torch.Tensor(res).long()


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class SA(nn.Module):
    def __init__(self, in_channels, hidden_dim, n_heads):
        super(SA, self).__init__()
        self.scale = hidden_dim ** -0.5
        inner_dim = hidden_dim * n_heads
        self.to_qk = nn.Linear(in_channels, inner_dim*2)
        self.n_heads = n_heads
        self.ln = nn.LayerNorm(in_channels)
        nn.init.normal_(self.to_qk.weight, 0, 1)

    def forward(self, x):
        y = rearrange(x, 'n c t v -> n t v c').contiguous()
        y = self.ln(y)
        y = self.to_qk(y)
        qk = y.chunk(2, dim=-1)
        q, k = map(lambda t: rearrange(t, 'b t v (h d) -> (b t) h v d', h=self.n_heads), qk)

        # attention
        dots = einsum('b h i d, b h j d -> b h i j', q, k)*self.scale
        attn = dots.softmax(dim=-1).float()
        return attn


class Joint_Att(nn.Module):
    def __init__(self, num_joint, channel, **kwargs):
        super(Joint_Att, self).__init__()
        self.num_joint = num_joint
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fcn1 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1,),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.fcn = nn.Sequential(
            nn.Conv2d(self.num_joint, self.num_joint//2, kernel_size=1),
            nn.BatchNorm2d(self.num_joint//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.num_joint//2, self.num_joint, kernel_size=1),
            nn.Softmax(dim=1)
        )
        self.bn = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)
        self.sa = SA(in_channels=num_joint, hidden_dim=32, n_heads=17)

    def forward(self, x):
        res = x
        x_attn = torch.cat([self.avg_pool(torch.transpose(x, 1, 3)), self.max_pool(torch.transpose(x, 1, 3))], dim=2).transpose(1, 2)
        x_attn = self.fcn1(x_attn).transpose(1, 2)
        x_attn = self.sa(x_attn)
        x_attn = self.fcn(x_attn).squeeze()
        return self.relu(self.bn(x * x_attn[:, None, None, :]) + res)

    
class Channel_Joint_Att(nn.Module):
    def __init__(self, channel, **kwargs):
        super(Channel_Joint_Att, self).__init__()
        inner_channel = channel // 2
        self.fcn = nn.Sequential(
            nn.Conv2d(channel, inner_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_channel),
            nn.ReLU(),
        )
        self.conv_c = nn.Conv2d(inner_channel, channel, kernel_size=1)
        self.conv_v = nn.Conv2d(inner_channel, channel, kernel_size=1)
        self.bn = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        N, C, T, V = x.size()
        x_v = x.mean(3, keepdims=True).transpose(1, 2) ##(N, T, C, 1)
        x_c = x.mean(1, keepdims=True).transpose(1, 3).transpose(1, 2) ## (N, T, V, 1)
        x_attn = self.fcn(torch.cat([x_c, x_v], dim=2))
        x_c, x_v = torch.split(x_attn, [C, V], dim=2)
        x_c_attn = self.conv_c(x_c)
        x_v_attn = self.conv_v(x_v).transpose(2, 3).sigmoid()
        x_attn = (x_c_attn * x_v_attn).transpose(1, 2)
        return x_attn * x

def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class GraphAttnConv(nn.Module):
    def __init__(self, in_channels, out_channels, max_graph_distance, A):
        super(GraphAttnConv, self).__init__()
        self.s_kernel_size = max_graph_distance
        self.gcn = nn.Conv2d(in_channels, out_channels * self.s_kernel_size, 1)
        self.register_buffer('A', A)

    def forward(self, x):
        x = self.gcn(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc // self.s_kernel_size, t, v)
        x = torch.einsum('nkctv, kvw->nctw', (x, self.A[:self.s_kernel_size])).contiguous()
        return x

class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""

    def __init__(self, in_dim, A, **kwargs):
        super(CrissCrossAttention, self).__init__()        
        #self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8 ,kernel_size=1)
        self.query_conv = GraphAttnConv(in_channels=in_dim, out_channels=in_dim // 8, max_graph_distance=1, A=A)

        #self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8 ,kernel_size=1)
        self.key_conv = GraphAttnConv(in_channels=in_dim, out_channels=in_dim // 8, max_graph_distance=1, A=A)

        #self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv = GraphAttnConv(in_channels=in_dim, out_channels=in_dim, max_graph_distance=1, A=A)
        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        N, _, T, V = x.size() ## N, C, T, V
        proj_query = self.query_conv(x)
        proj_query_T = proj_query.permute(0, 3, 1, 2).contiguous().view(N * V, -1, T).permute(0, 2, 1)
        proj_query_V = proj_query.permute(0, 2, 1, 3).contiguous().view(N * T, -1, V).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_T = proj_key.permute(0, 3, 1, 2).contiguous().view(N * V, -1, T)
        proj_key_V = proj_key.permute(0, 2, 1, 3).contiguous().view(N * T, -1, V)
        proj_value = self.value_conv(x)
        proj_value_T = proj_value.permute(0, 3, 1, 2).contiguous().view(N * V, -1, T)
        proj_value_V = proj_value.permute(0, 2, 1, 3).contiguous().view(N * T, -1, V)
        energy_T = (torch.bmm(proj_query_T, proj_key_T) + self.INF(N, T, V).to('cuda:0')).view(N, V, T, T).permute(0, 2, 1, 3)
        energy_V = torch.bmm(proj_query_V, proj_key_V).view(N, T, V, V)
        concate = self.softmax(torch.cat([energy_T, energy_V], 3))

        att_T = concate[:, :, :, 0:T].permute(0,2,1,3).contiguous().view(N * V, T, T)
        att_V = concate[:, :, :, T:T + V].contiguous().view(N * T, V, V)
        out_T = torch.bmm(proj_value_T, att_T.permute(0, 2, 1)).view(N, V, -1, T).permute(0, 2, 3, 1)
        out_V = torch.bmm(proj_value_V, att_V.permute(0, 2, 1)).view(N, T, -1, V).permute(0, 2, 1, 3)

        return self.gamma * (out_V + out_T) + x
