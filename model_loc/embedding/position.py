import torch.nn as nn
import torch
import math


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class PolarPostionEmbedding(nn.Module):
    def __init__(self, voc_num, d_model):
        super().__init__()
        # log(r*2)
        # self.radius_emb = nn.Sequential(nn.Linear(d_model, 1), nn.ReLU())
        self.radius_emb = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU())
        self.period_emb = nn.Linear(d_model, d_model)
    
    def forward(self, we):
        # * 得到平均影响力r, 和word embedding有关
        r = self.radius_emb(we) + 1e-8  # [b, l, d] --> [b, l, 1]
        # * 得到周期 T = (2*pi)/w, 和word embedding有关
        t = self.period_emb(we)  # [b, l, d] --> [b, l, d]
        pos_seq = torch.arange(1, we.shape[1]+1, 1.0, device=we.device)
        pos_seq = pos_seq.unsqueeze(0).unsqueeze(-1)
        pos_seq = pos_seq.repeat([we.shape[0], 1, we.shape[-1]])

        phase = torch.mul(pos_seq, t) % (2*math.pi)

        return r, t, phase, r*torch.add(torch.cos(phase), torch.sin(phase))

'''
class PolarRotate(nn.Module):
    def __init__(self, d_model):
        self.phase_rotate = nn.Linear(d_model, d_model)
        self.radius_flex = nn.Linear(d_model, 1)
    
    # ! 这里应该有2种方案
    """
    * 1: 所有的伸缩旋转都基于初始值 （合理）
    * 2: 所有的伸缩和旋转都基于上一步的值
    """
    # * 方案1
    def forward(self, radius, phase, multi_h_out):
        phase_new = self.phase_rotate(multi_h_out) + phase  # ([b, l, d] --> [b, l, d]) + [b, l, d]
        radius_new = self.radius_flex(multi_h_out)*radius  # ([b, l, d] --> [b, l, 1])*[b, l, 1]
        return radius_new, phase_new, radius_new*torch.mul(torch.cos(phase), torch.sin(phase))

    # * 方案2
    #def forward(self, init_radius, init_phase, multi_h_out):
    #    phase_new = self.phase_rotate(multi_h_out) + init_phase  # ([b, l, d] --> [b, l, d]) + [b, l, d]
    #    radius_new = self.radius_flex(multi_h_out)*init_phase  # ([b, l, d] --> [b, l, 1])*[b, l, 1]
         # 不改变原始的
    #    return radius_new, phase_new, radius_new*torch.cos(phase), radius_new*torch.sin(phase)
'''
