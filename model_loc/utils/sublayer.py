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

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        new_x = sublayer(x)
        new_x = x + self.dropout(new_x)
        new_x = self.norm(new_x)
        return new_x


class SublayerPolarConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerPolarConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.radius_norm = LayerNorm(size)
        self.phase_norm = LayerNorm(size)
        self.postion_norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.postion_drop = nn.Dropout(dropout)

    def forward(self, x, radius, phase, old_position_e, sublayer):
        "Apply residual connection to any sublayer with the same size."
        new_x, radius_new, phase_new, position_e = sublayer(x, radius, phase, old_position_e)
        new_x = x + self.dropout(new_x)
        # new_x = x + new_x
        new_x = self.norm(new_x)
        return (new_x, radius_new, phase_new, position_e)