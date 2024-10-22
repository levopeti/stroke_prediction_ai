import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Linear,Dropout, LayerNorm

def d(tensor=None):
    if tensor is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cuda' if tensor.is_cuda else 'cpu'

class AttentionError(Exception):
    pass


class RelPosMultiHeadedAttention(nn.Module):
    """
    Narrow multiheaded attention. Each attention head inspects a
    fraction of the embedding space and expresses attention vectors for each sequence position as a weighted average of all (earlier) positions.
    """

    def __init__(self, d_model, heads, seq_length, dropout=0.1, relative_pos=True):
        super().__init__()
        self.batch_first = True
        if d_model % heads != 0:
            raise AttentionError("Number of heads does not divide model dimension")
        self.d_model = d_model
        self.heads = heads
        s = d_model // heads
        self.linears = torch.nn.ModuleList([nn.Linear(s, s, bias=False) for _ in range(3)])
        self.recombine_heads = nn.Linear(heads * s, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.max_length = seq_length
        # relative positional embeddings
        self.relative_pos = relative_pos
        if relative_pos:
            self.Er = torch.randn([heads, self.max_length, s],
                                  device=d())
        else:
            self.Er = None

    def forward(self, x, mask=None):
        # batch size, sequence length, embedding dimension
        b, t, e = x.size()
        h = self.heads
        # each head inspects a fraction of the embedded space
        # head dimension
        s = e // h
        # start index of position embedding
        embedding_start = self.max_length - t
        x = x.view(b, t, h, s)
        queries, keys, values = [w(x).transpose(1, 2) for w, x in zip(self.linears, (x, x, x))]
        if self.relative_pos:
            # apply same position embeddings across the batch
            # Is it possible to apply positional self-attention over
            # only half of all relative distances?
            Er = self.Er[:, embedding_start:, :].unsqueeze(0)
            QEr = torch.matmul(queries, Er.transpose(-1, -2))
            QEr = self._mask_positions(QEr)
            # Get relative position attention scores
            # combine batch with head dimension
            SRel = self._skew(QEr).contiguous().view(b * h, t, t)
        else:
            SRel = torch.zeros([b * h, t, t], device=d())
        queries, keys, values = map(lambda x: x.contiguous().view(b * h, t, s), (queries, keys, values))
        # Compute scaled dot-product self-attention
        # scale pre-matrix multiplication
        queries = queries / (e ** (1 / 4))
        keys = keys / (e ** (1 / 4))

        scores = torch.bmm(queries, keys.transpose(1, 2))
        scores = scores + SRel
        # (b*h, t, t)

        subsequent_mask = torch.triu(torch.ones(1, t, t, device=d()),
                                     1)
        scores = scores.masked_fill(subsequent_mask == 1, -1e9)
        if mask is not None:
            mask = mask.repeat_interleave(h, 0)
            wtf = (mask == 0).nonzero().transpose(0, 1)
            scores[wtf[0], wtf[1], :] = -1e9

        # Convert scores to probabilities
        attn_probs = F.softmax(scores, dim=2)
        attn_probs = self.dropout(attn_probs)
        # use attention to get a weighted average of values
        out = torch.bmm(attn_probs, values).view(b, h, t, s)
        # transpose and recombine attention heads
        out = out.transpose(1, 2).contiguous().view(b, t, s * h)
        # last linear layer of weights
        return self.recombine_heads(out)

    @staticmethod
    def _mask_positions(qe):
        # QEr is a matrix of queries (absolute position) dot distance embeddings (relative pos).
        # Mask out invalid relative positions: e.g. if sequence length is L, the query at
        # L-1 can only attend to distance r = 0 (no looking backward).
        L = qe.shape[-1]
        mask = torch.triu(torch.ones(L, L, device=d()), 1).flip(1)
        return qe.masked_fill((mask == 1), 0)

    @staticmethod
    def _skew(qe):
        # pad a column of zeros on the left
        padded_qe = F.pad(qe, [1, 0])
        s = padded_qe.shape
        padded_qe = padded_qe.view(s[0], s[1], s[3], s[2])
        # take out first (padded) row
        return padded_qe[:, :, 1:, :]


class RelPosTransformerEncoderLayer(Module):
    def __init__(self, d_model, nhead, seq_length, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = RelPosMultiHeadedAttention(d_model, nhead, seq_length, dropout=dropout)
        self.linear_1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear_2 = Linear(dim_feedforward, d_model)

        # input shape: [n, c, l]
        # self.conv1d_1 = Conv1d(in_channels=embed_dim,
        #                        out_channels=dim_feedforward,
        #                        kernel_size=1,
        #                        stride=1,
        #                        bias=False)
        # self.conv1d_2 = Conv1d(in_channels=dim_feedforward,
        #                        out_channels=embed_dim,
        #                        kernel_size=1,
        #                        stride=1,
        #                        bias=False)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, src, *args, **kwargs):
        """ input shape/output shape: [batch size, seq length, embed_dim] """
        x = self.self_attn(src)
        x = src + self.dropout1(x)
        x = self.norm1(x)
        x_2 = self.linear2(self.dropout(F.relu(self.linear1(x))))
        # x = torch.transpose(x, 1, 2)
        # x_2 = self.conv1d_2(self.dropout(F.relu(self.conv1d_1(x))))
        x = x + self.dropout2(x_2)
        # x = torch.transpose(x, 1, 2)
        x = self.norm2(x)
        return x