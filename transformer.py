import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads, drop_prob):
        super().__init__()
    
        self.dim = dim # "$d_{model}$"
        self.n_heads = n_heads # "$h$"

        self.head_dim = dim // n_heads # "$d_{k}$, $d_{v}$"

        self.q_proj = nn.Linear(dim, dim, bias=False) # "$W^{Q}_{i}$"
        self.k_proj = nn.Linear(dim, dim, bias=False) # "$W^{K}_{i}$"
        self.v_proj = nn.Linear(dim, dim, bias=False) # "$W^{V}_{i}$"

        self.attn_drop = nn.Dropout(drop_prob) # Not in the paper
        self.out_proj = nn.Linear(dim, dim, bias=False) # "$W^{O}$"

    @staticmethod
    def _get_attention_score(q, k):
        attn_score = torch.einsum("bnid,bnjd->bnij", q, k)
        return attn_score

    def forward(self, q, k, v, mask=None):
        b, i, _ = q.shape
        _, j, _ = k.shape

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = q.view(b, self.n_heads, i, self.head_dim)
        k = k.view(b, self.n_heads, j, self.head_dim)
        v = v.view(b, self.n_heads, j, self.head_dim)

        attn_score = self._get_attention_score(q=q, k=k)
        if mask is not None:
            mask = einops.repeat(
                mask, pattern="b i j -> b n i j", n=self.n_heads,
            )
            attn_score.masked_fill_(mask=mask, value=-1e9) # "Mask (opt.)"
        attn_score /= (self.head_dim ** 0.5) # "Scale"
        attn_weight = F.softmax(attn_score, dim=3) # "Softmax"

        attn_weight_drop = self.attn_drop(attn_weight) # Not in the paper
        x = torch.einsum("bnij,bnjd->bnid", attn_weight_drop, v) # "MatMul"
        x = einops.rearrange(x, pattern="b n i d -> b i (n d)")

        x = self.out_proj(x)
        return x, attn_weight


class PositionwiseFeedForward(nn.Module):
    def __init__(self, dim, mlp_dim, drop_prob, activ="relu"):
        super().__init__()

        assert activ in ["relu", "gelu"], (
            """The argument `activ` must be one of (`"relu"`, `"gelu"`)"""
        )

        self.activ = activ

        self.proj1 = nn.Linear(dim, mlp_dim) # "$W_{1}$"
        if activ == "relu":
            self.relu = nn.ReLU()
        else:
            self.gelu = nn.GELU()
        self.proj2 = nn.Linear(mlp_dim, dim) # "$W_{2}$"
        self.mlp_drop = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.proj1(x)
        if self.activ == "relu":
            x = self.relu(x)
        else:
            x = self.gelu(x)
        x = self.proj2(x)
        x = self.mlp_drop(x) # Not in the paper
        return x


class ResidualConnection(nn.Module):
    def __init__(self, dim, drop_prob):
        super().__init__()

        self.resid_drop = nn.Dropout(drop_prob) # "Residual dropout"
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, sublayer):
        skip = x.clone()
        x = sublayer(x)
        x = self.resid_drop(x)
        x += skip # "Add"
        x = self.norm(x) # "& Norm"
        return x


class DecoderLayer(nn.Module):
    def __init__(
        self,
        n_heads,
        dim,
        mlp_dim,
        attn_drop_prob=0.1,
        ff_drop_prob=0.1,
        resid_drop_prob=0.1,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.dim = dim
        self.mlp_dim = mlp_dim

        self.self_attn = MultiHeadAttention(dim=dim, n_heads=n_heads, drop_prob=attn_drop_prob)
        self.self_attn_resid_conn = ResidualConnection(dim=dim, drop_prob=resid_drop_prob)
        self.enc_dec_attn = MultiHeadAttention(dim=dim, n_heads=n_heads, drop_prob=attn_drop_prob)
        self.enc_dec_attn_resid_conn = ResidualConnection(dim=dim, drop_prob=resid_drop_prob)
        self.feed_forward = PositionwiseFeedForward(
            dim=dim, mlp_dim=mlp_dim, drop_prob=ff_drop_prob, activ="relu",
        )
        self.ff_resid_conn = ResidualConnection(dim=dim, drop_prob=resid_drop_prob)

    def forward(self, x, enc_out):
        x = self.self_attn_resid_conn(
            x=x,
            sublayer=lambda x: self.self_attn(q=x, k=x, v=x)[0],
        )
        x = self.enc_dec_attn_resid_conn(
            x=x,
            sublayer=lambda x: self.enc_dec_attn(q=x, k=enc_out, v=enc_out)[0]
        )
        x = self.ff_resid_conn(x=x, sublayer=self.feed_forward)
        return x


class TransformerDecoder(nn.Module):
    """
    "Transformer module uses the standard Transformer decoder [41] to compute from image features F and N learnable positional embeddings (i.e., queries) its output, i.e., N per-segment embeddings Q 2 RCQ N of dimension CQ that encode global information about each segment MaskFormer predicts. Similarly to [4], the decoder yields all predictions in parallel."
    """
    def __init__(
        self,
        n_heads,
        dim,
        mlp_dim,
        n_layers,
        attn_drop_prob=0.1,
        ff_drop_prob=0.1,
        resid_drop_prob=0.1,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.dim = dim
        self.n_layers = n_layers

        self.dec_stack = nn.ModuleList(
            [
                DecoderLayer(
                    n_heads=n_heads,
                    dim=dim,
                    mlp_dim=mlp_dim,
                    attn_drop_prob=attn_drop_prob,
                    ff_drop_prob=ff_drop_prob,
                    resid_drop_prob=resid_drop_prob,
                )
                for _ in range(self.n_layers)
            ]
        )

    def forward(self, x, enc_out):
        for dec_layer in self.dec_stack:
            x = dec_layer(x, enc_out=enc_out)
        return x
