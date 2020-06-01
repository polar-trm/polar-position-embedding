import torch.nn as nn
from .token import TokenEmbedding
from .position import PositionalEmbedding, PolarPostionEmbedding
from .segment import SegmentEmbedding


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, dropout=0.1, single_segment = True):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        if (single_segment):
            pass
        else:
            self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence, segment_label = None):
        if (segment_label == None):
            x = self.token(sequence) + self.position(sequence)
        else:
            x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        return self.dropout(x)


class BERTpolarEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, dropout=0.1, single_segment=True):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PolarPostionEmbedding(voc_num=vocab_size, d_model=embed_size)
        if (single_segment):
            pass
        else:
            self.segment = SegmentEmbedding(embed_size=embed_size)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence, segment_label=None):
        token_e = self.token(sequence)
        # 获得初始半径 周期 初始相位 位置embedding
        init_radius, period, init_phase, position_e = self.position(token_e)
        if (segment_label is None):
            x = token_e + position_e
        else:
            segment_e = self.segment(segment_label)
            x = token_e + position_e + segment_e
        return self.dropout(x), init_radius, period, init_phase, position_e

