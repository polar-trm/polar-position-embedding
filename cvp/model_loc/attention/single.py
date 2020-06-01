import torch.nn as nn
import torch.nn.functional as F
import torch

import math


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, query_i, key_i, value_i, mask=None, dropout=None):
        
        real_s = torch.matmul(query, key.transpose(-2, -1))
        img_s = torch.matmul(query_i, key_i.transpose(-2, -1))

        scores = torch.sqrt(torch.mul(real_s, real_s) + torch.mul(img_s, img_s) + 1e-8) \
                 / math.sqrt(query.size(-1))

        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)



        if dropout is not None:
            p_attn = dropout(p_attn)

        real = torch.matmul(p_attn, value)
        img = torch.matmul(p_attn, value_i)

        return real, img