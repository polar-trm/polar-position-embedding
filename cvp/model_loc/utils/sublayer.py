import torch.nn as nn
from .layer_norm import LayerNorm


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = LayerNorm(size)
    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return self.norm2(x + self.dropout(sublayer(self.norm(x))))


class SublayerPolarConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerPolarConnection, self).__init__()
        # self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, real, img, sublayer):
        "Apply residual connection to any sublayer with the same size."
        real_, img_ = sublayer(real, img)
        return (real + self.dropout(real_), img + self.dropout2(img_))
