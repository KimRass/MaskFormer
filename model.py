# References:
    # https://github.com/facebookresearch/detr/blob/main/models/matcher.py

import torch
import torch.nn as nn
import einops

from transformer import DecoderLayer


img_size = 640
class Backbone(nn.Module):
    """
    "A backbone generates a (typically) low-resolution image feature map F $\mathcal{F} \in \mathbb{R}^{C_{\mathcal{F}} \times \frac{H}{S} \times \frac{W}{S}}$, where $C_{\mathcal{F}}$ is the number of channels and $S$ is the stride of the feature map ($C_{\mathcal{F}}$ depends on the specific backbone and we use $S = 32$ in this work)."
    """
    def __init__(self, feat_dim=128, stride=32):
        super().__init__()

        self.feat_dim = feat_dim
        self.stride = stride

    def forward(self, x):
        return torch.randn(
            (
                x.size(0),
                self.feat_dim,
                img_size // self.stride,
                img_size // self.stride,
            ),
        )


class PixelDecoder(nn.Module):
    """
    "Pixel-level module takes an image of size H   W as input.
    Then, a pixel decoder gradually upsamples the features to generate per-pixel embeddings Epixel 2 RCE H W, where CE is the embedding dimension. Note, that any per-pixel classificationbased segmentation model fits the pixel-level module design including recent Transformer-based models [37, 53, 29]."
    """
    def __init__(self, embed_dim=128):
        super().__init__()

        self.embed_dim = embed_dim

    def forward(self, x):
        return torch.randn((x.size(0), self.embed_dim, img_size, img_size))


class MLP(nn.Module):
    """
    "Segmentation module applies a linear classifier, followed by a softmax
    activation, on top of the per-segment embeddings $Q$ to yield class
    probability predictions $\{p_{i} \in \Delta^{K + 1}\}^{N}_{i = 1}$ for
    each segment."
    "The classifier predicts an additional 'no object' category ($\phi$) in case the
    embedding does not correspond to any region. For mask prediction, a Multi-Layer
    Perceptron (MLP) with 2 hidden layers converts the per-segment embeddings
    $\mathcal{Q}$ to $N$ mask embeddings
    $\epsilon_{\text{mask}} \in \mathbb{R}^{C_{\epsilon} \times N}$ of dimension
    $C_{\epsilon}$."
    """
    def __init__(self, q_dim, embed_dim, mlp_dim, num_classes):
        super().__init__()

        self.prob_proj = nn.Linear(q_dim, num_classes + 1)

        self.bbox_proj1 = nn.Linear(q_dim, mlp_dim)
        self.bbox_proj2 = nn.Linear(mlp_dim, embed_dim)

    def forward(self, x):
        prob = self.prob_proj(x)
        bbox = self.bbox_proj2(self.bbox_proj1(x))
        return prob, bbox


class MaskFormer(nn.Module):
    """
    "We obtain each binary mask prediction mi 2 [0; 1]H W via a dot product between the ith mask embedding and per-pixel embeddings Epixel computed by the pixel-level module. The dot product is followed by a sigmoid activation, i.e., mi[h;w] = sigmoid(Emask[:; i]T   Epixel[:; h;w]).
    "It is beneficial to not enforce mask predictions to be mutually exclusive to each other by using a softmax activation."
    "For simplicity we use the same Lmask as DETR [4], i.e., a linear combination of a focal loss [27] and a dice loss [33] multiplied by hyper-parameters  focal and  dice respectively."
    """
    def get_loss(self):
        pass
    """
    "Unlike DETR that uses bounding boxes to compute the assignment costs between prediction zi and ground truth zgt j for the matching problem, we directly use class and mask predictions, i.e., 􀀀pi(cgt j ) + Lmask(mi;mgt j ), where Lmask is a binary mask loss."
    "The main mask classification loss Lmask-cls is composed of a cross-entropy classification loss and a binary mask loss Lmask for each predicted segment: Lmask-cls(z; zgt) = XN j=1 h 􀀀log p (j)(cgt j ) + 1 cgt j 6=?Lmask(m (j);mgt j ) i :
    """


if __name__ == "__main__":
    num_classes = 80
    num_gts = 13
    gt_label = torch.randint(0, num_classes + 1, size=(num_gts,), dtype=torch.long)
    num_qs = 40
    q_dim = 384
    embed_dim = 128
    backbone = Backbone(feat_dim=q_dim)
    pixel_decoder = PixelDecoder()
    batch_size = 4
    x = torch.randn(size=(batch_size, 3, img_size, img_size))
    backbone_out = backbone(x)
    backbone_out = einops.rearrange(
        backbone_out, pattern="b c h w -> b (h w) c",
    )

    n_heads = 1
    dim = q_dim
    mlp_dim = 4 * dim
    dec_layer = DecoderLayer(
        n_heads=n_heads, dim=dim, mlp_dim=mlp_dim,
    )
    q = torch.randn((batch_size, num_qs, q_dim))
    x = dec_layer(x=q, enc_out=backbone_out)
    mlp = MLP(
        q_dim=q_dim,
        embed_dim=embed_dim,
        mlp_dim = embed_dim * 4,
        num_classes=num_classes,
    )
    cls_prob, pred_bbox = mlp(x)
    cls_prob.shape, pred_bbox.shape
    
    x = pixel_decoder(backbone_out)
    x = torch.einsum("bchw,bnc->bnhw", x, pred_bbox)
    x.shape
