import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


class MultiHeadAttention(nn.Module):
    def __init__(self, width, num_heads, drop_prob):
        super().__init__()
    
        self.num_heads = num_heads

        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.to_multi_heads = Rearrange("b i (n h) -> b i n h", n=num_heads)
        self.scale = width ** (-0.5)
        self.attn_drop = nn.Dropout(drop_prob)
        self.to_one_head = Rearrange("b i n h -> b i (n h)")
        self.out_proj = nn.Linear(width, width, bias=False)

    def forward(self, q, k, v):
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        attn_score = torch.einsum(
            "binh,bjnh->bnij", self.to_multi_heads(q), self.to_multi_heads(k),
        ) * self.scale
        attn_weight = F.softmax(attn_score, dim=-1)
        x = self.to_one_head(
            torch.einsum(
                "bnij,bjnh->binh",
                self.attn_drop(attn_weight),
                self.to_multi_heads(v),
            )
        )
        x = self.out_proj(x)
        return x, attn_weight


class FFN(nn.Module):
    def __init__(self, width, mlp_width, drop_prob):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(width, mlp_width),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(mlp_width, width),
        )

    def forward(self, x):
        return self.layers(x)


class ResidualConnection(nn.Module):
    def __init__(self, fn, width, drop_prob):
        super().__init__()

        self.fn = fn
        self.res_drop = nn.Dropout(drop_prob)
        self.norm = nn.LayerNorm(width)

    def forward(self, skip, **kwargs):
        x = self.fn(**kwargs)
        x = self.res_drop(x)
        x += skip
        x = self.norm(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(
        self,
        num_heads,
        width,
        mlp_width,
        drop_prob,
    ):
        super().__init__()

        self.self_attn = MultiHeadAttention(
            width=width, num_heads=num_heads, drop_prob=drop_prob,
        )
        self.enc_dec_attn = MultiHeadAttention(
            width=width, num_heads=num_heads, drop_prob=drop_prob,
        )
        self.ffn = FFN(
            width=width,
            mlp_width=mlp_width,
            drop_prob=drop_prob,
        )

        self.self_attn_res_conn = ResidualConnection(
            fn=lambda x, out_pos_enc: self.self_attn(
                q=x + out_pos_enc, k=x + out_pos_enc, v=x)[0],
            width=width,
            drop_prob=drop_prob,
        )
        self.enc_dec_attn_res_conn = ResidualConnection(
            fn=lambda x, enc_mem, out_pos_enc: self.enc_dec_attn(
                q=x + out_pos_enc, k=self.spatial_pos_enc(enc_mem), v=enc_mem,
            )[0],
            width=width,
            drop_prob=drop_prob,
        )
        self.ffn_res_conn = ResidualConnection(
            fn=self.ffn, width=width, drop_prob=drop_prob,
        )

    def forward(self, query, enc_mem, out_pos_enc):
        x = self.self_attn_res_conn(
            skip=query, x=query, out_pos_enc=out_pos_enc,
        )
        x = self.enc_dec_attn_res_conn(
            skip=x, x=x, enc_mem=enc_mem, out_pos_enc=out_pos_enc,
        )
        x = self.ffn_res_conn(skip=x, x=x)
        return x


class TransformerDecoder(nn.Module):
    """
    "Transformer module uses the standard Transformer decoder [41] to compute from image features F and N learnable positional embeddings (i.e., queries) its output, i.e., N per-segment embeddings Q 2 RCQ N of dimension CQ that encode global information about each segment MaskFormer predicts. Similarly to [4], the decoder yields all predictions in parallel."
    """
    def __init__(
        self,
        num_heads,
        width,
        num_layers,
        drop_prob,
        mlp_width=None,
    ):
        super().__init__()

        self.mlp_width = mlp_width if mlp_width is not None else width * 4

        self.dec_stack = nn.ModuleList(
            [
                DecoderLayer(
                    num_heads=num_heads,
                    width=width,
                    mlp_width=self.mlp_width,
                    drop_prob=drop_prob,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, query, enc_mem, out_pos_enc):
        for dec_layer in self.dec_stack:
            query = dec_layer(
                query=query, enc_mem=enc_mem, out_pos_enc=out_pos_enc,
            )
        return query
