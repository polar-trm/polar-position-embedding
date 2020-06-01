import torch.nn as nn
from .single import Attention
import torch
import math
from ..utils.layer_norm import LayerNorm
from ..utils.gelu import GELU

class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)


        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query, key, value = self.q_linear(query), self.k_linear(key), self.v_linear(value)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class MultiHeadedPolarRotateAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1, rotate_lr=1.0):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.output_linear_2 = nn.Linear(d_model, d_model)

        self.attention = Attention()
        self.polar_rotate = PolarRotate(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.rotate_lr = rotate_lr
        self.position_LN = LayerNorm(d_model)

        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, radius, phase, old_position_e, mask=None):
        batch_size = query.size(0)
        head_dim = query.size(-1)
        query, key, value = self.query_linear(query), self.key_linear(key), self.value_linear(value)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        
        out_x = self.output_linear(x)
        radius_new, phase_new, out_position_rotate = self.polar_rotate(radius, phase, out_x)
        return (out_x + self.rotate_lr*(out_position_rotate - old_position_e), radius_new, phase_new, out_position_rotate)
        
        # out_x = self.output_linear(x)
        # radius_new, phase_new, out_position_rotate = self.polar_rotate(radius, phase, out_x)
        # out_x = self.output_linear_2(out_x + self.rotate_lr*(out_position_rotate - old_position_e))
        # return (out_x, radius_new, phase_new, out_position_rotate)

        # radius_new, phase_new, out_position_rotate = self.polar_rotate(radius, phase, x)
        # out_x = self.output_linear(x + self.rotate_lr*(out_position_rotate - old_position_e))
        # return (out_x, radius_new, phase_new, out_position_rotate)

class PolarRotate(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.phase_rotate = nn.Linear(d_model, d_model)
        self.radius_flex = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU())
    
    # ! 这里应该有2种方案
    """
    * 1: 所有的伸缩旋转都基于初始值 （合理） (加一个梯度值进去(new-old))
    * 2: 所有的伸缩和旋转都基于上一步的值
    """
    def forward(self, radius, phase, multi_h_out):
        # 旋转角度
        phase_new = (self.phase_rotate(multi_h_out) + phase) % (2*math.pi)  # ([b, l, d] --> [b, l, d]) + [b, l, d]
        # 缩放半径
        radius_new = self.radius_flex(multi_h_out)*radius  # ([b, l, d] --> [b, l, 1])*[b, l, 1]
        return radius_new, phase_new, radius_new*torch.add(torch.cos(phase_new), torch.sin(phase_new))

