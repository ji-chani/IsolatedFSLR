import torch
import torch.nn as nn
import copy

class LearnedPositionalEncoding(nn.Module):
    """
    d_model-dimensional learnable vector per frame position, added element-wise to the pose vector at that frame position
    """
    def __init__(self, d_model, max_len=256):  # max_len: pre-allocated table length (must be > than longest clip/video)
        super().__init__()
        # a lookup table: row i = the encoding added to frame i
        self.pe = nn.Parameter(torch.rand(max_len, d_model))

    def forward(self, x):
        # x shape: (seq_len, batch, d_model)
        seq_len = x.size(0)
        return x + self.pe[:seq_len].unsqueeze(1)  # broadcast over batch | (seq_len x d_model) -> (seq_len x 1 x d_model)



class MultiHeadProjection(nn.Module):
    """
    Special case of multi-head attention when the query sequence has length 1.
    Softmax attention would trivially return 1, so we skip straight to
    projecting the single query vector through per-head linear layers,
    concatenating, and projecting back to d_model. (Paper, Sec 4.4.)
    """

    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.head_dim = d_model // nhead

        # One value-projection per head, done as a single linear layer
        # covering all heads at once (standard trick, e.g. nn.MultiheadAttention).
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (1, batch, d_model)  -- one query token
        seq_len, batch, d_model = x.shape

        v = self.value_proj(x)                                      # (1, batch, d_model)
        v = v.view(seq_len, batch, self.nhead, self.head_dim)       # separating by heads
        v = v.permute(1,2,0,3).contiguous()                         # reorde dimensions: (batch, nhead, 1, head_dim)
        v = v.view(batch, seq_len, self.nhead*self.head_dim)         # concat heads
        v = v.transpose(0,1)                                        # back to (1, batch, d_model)

        out = self.out_proj(v)
        return self.dropout(out)


_ = """
Why go through all this splitting/concatenating if there's no attention weighting happening?

Since with seq_len=1 you could arguably just do one Linear(108, 108) and get a similar-capacity result. 
The reason to keep the head-split structure is architectural consistency with the rest of the transformer — 
it preserves the same parameter count and per-head "mixing" structure that the original multi-head attention had, 
so that if you ever want to inspect or compare per-head behavior (or if seq_len weren't always 1), the module still
behaves like a proper multi-head component rather than a totally different mechanism. It's also just matching the 
paper's explicit description of this as a "Multi-Head Projection" module with heads that get "concatenated and 
processed by the final linear layer" (Section 4.4) rather than a generic single linear layer.

"""


# complete Transformer Decoder Layer
class SPOTERDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)
        # del self.self_attn  # not used, remove for clarity
        self.multi_head_projection = MultiHeadProjection(d_model, nhead, dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                **kwargs):
        # tgt shape: (1, batch, d_model)  -- class query
        # memory shape: (seq_len, batch, d_model)  -- encoder output

        
        # ---- Multi-head projection instead of self-attention -----
        tgt2 = self.multi_head_projection(tgt)                       # projected version of class query
        tgt = tgt + self.dropout1(tgt2)                              # residual connection
        tgt = self.norm1(tgt)                                        # normalization step (nn.Layer)
        # tgt has now encoded value-projected & normalized class query


        # ---- Cross-attention: class query attends into encoded frame sequence ----
            # self.multihead_attn(query, key, value, ...)
            # query = tgt -- the class query (1, batch, d_model)
            # key, value = memory  -- encoded frame sequence (seq_len, batch, d_model)
        tgt2 = self.multihead_attn(tgt, memory, memory,
                                   attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]  # outputs (attn_output, attn_output_weights)
        # tgt2 shape: (1, batch, d_model)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # tgt has now encoded weighted summary of pose sequences (cross-attended information)

        # Feed-forward network: transform information within a single position
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))  # 108 -> 2048 (linear1) -> 108 (linear2) 
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt
    

class SPOTER(nn.Module):
    """
    Sign Pose-based Transformer
    Input: a sequence of normalized pose vectors, one per frame
    Output: class logits for the sign gloss.
    """

    def __init__(self, num_classes, d_model=225, nhead=6,
                 num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1, max_len=256):
        super().__init__()
        self.d_model = d_model

        self.pos_encoding = LearnedPositionalEncoding(d_model, max_len)
        self.class_query = nn.Parameter(torch.rand(1,1,d_model))  # (1, 1, d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # replacing decoder block with custom Decoder Layer
        decoder_layer = SPOTERDecoderLayer(d_model,
                                           nhead,
                                           dim_feedforward,
                                           dropout,
                                           'relu')
        self.transformer.decoder.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(num_decoder_layers)]  # deepcopy to have independent weights
        )


        self.linear_class = nn.Linear(d_model, num_classes)     # (d_model -> num_classes)

    def forward(self, x, src_key_padding_mask=None):
        """
        x: (seq_len, batch, d_model)  - already normalized/augmented pose vectors.
        src_key_padding_mask: (batch, seq_len), True at PADDING position
        Returns: (batch, num_classes) logits
        """
        batch_size = x.size(1)

        x = self.pos_encoding(x)                                # add positional encoding
        query = self.class_query.expand(-1, batch_size, -1)     # (1, 1, d_model) -> (1, batch, d_model)

        decoded = self.transformer(x, 
                                   query,                                           #  (1, batch, d_model)
                                   src_key_padding_mask=src_key_padding_mask,
                                   memory_key_padding_mask=src_key_padding_mask)    # memory == encoded src, same mask applies                  
        # src_key_padding_mask stops encoder's self-attention from letting frames attend to padding
        # memory_key_padding_mask stops the decoder's cross-attention from letting class query pull information from padded positions

        decoded = decoded.squeeze(0)                            # (batch, d_model)

        logits = self.linear_class(decoded)
        return logits