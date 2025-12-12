# References:
# capi: https://github.com/facebookresearch/capi/blob/main/eval_classification.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# backbone classification wrappers adapted from capi with minor changes


class ClassificationWrapper(nn.Module):
    """
    Wrap a backbone embedding model together with a grid of classifier heads.

    backbone: backbone model that returns (cls_embeds, reg_embeds, patch_embeds)
    classifiers: map of (representation, hparams) -> classifier
    """

    def __init__(
        self,
        backbone: nn.Module,
        classifiers: dict[tuple[str, tuple[float, float]], nn.Module],
    ):
        super().__init__()
        self.representations = {key[0] for key in classifiers}
        self.backbone = backbone

        # can't use ModuleDict bc of restrictions of keys (must be strings, no dots).
        self.classifier_keys = list(classifiers)
        self.classifiers = nn.ModuleList(list(classifiers.values()))

    def forward(self, *args, **kwargs) -> Tensor:
        cls_embeds, reg_embeds, patch_embeds = self.backbone(*args, **kwargs)
        backbone_out = pool_representations(
            cls_embeds, reg_embeds, patch_embeds, self.representations
        )

        all_logit = []
        for ii, (feature_source, _) in enumerate(self.classifier_keys):
            clf = self.classifiers[ii]
            all_logit.append(clf(backbone_out[feature_source]))

        # [B, num_classes, num_classifiers]
        all_logit = torch.stack(all_logit, dim=-1)
        return all_logit


class LinearClassifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.init_weights()

    def init_weights(self):
        nn.init.trunc_normal_(self.linear.weight, std=0.02)
        nn.init.zeros_(self.linear.bias)

    def forward(self, cls_token):
        return self.linear(cls_token)


class AttnPoolClassifier(nn.Module):
    def __init__(self, in_dim, out_dim, embed_dim=None):
        super().__init__()
        embed_dim = embed_dim or in_dim
        assert embed_dim % 64 == 0
        self.query_token = nn.Parameter(torch.empty(embed_dim))
        self.embed_dim = embed_dim
        self.num_heads = embed_dim // 64
        self.kv = nn.Linear(in_dim, embed_dim * 2)
        self.linear = nn.Linear(embed_dim, out_dim)
        self.init_weights()

    def init_weights(self):
        nn.init.trunc_normal_(self.query_token, std=0.02)
        nn.init.trunc_normal_(self.kv.weight, std=0.02)
        nn.init.zeros_(self.kv.bias)
        nn.init.trunc_normal_(self.linear.weight, std=0.02)
        nn.init.zeros_(self.linear.bias)

    def forward(self, feat_tokens):
        B, N, _ = feat_tokens.shape
        D = self.embed_dim

        q = self.query_token.expand(B, 1, -1)
        q = q.reshape(B, 1, self.num_heads, D // self.num_heads)  # [B, 1, head, D_head]
        q = q.permute(0, 2, 1, 3)  # [B, head, 1, D_head]

        kv = self.kv(feat_tokens).reshape(
            B, N, 2, self.num_heads, D // self.num_heads
        )  # [B, N, 2, head, D_head]
        kv = kv.permute(2, 0, 3, 1, 4)  # [2, B, head, N, D_head]
        k, v = torch.unbind(kv, dim=0)  # 2 * [B, head, N, D_head]

        x = F.scaled_dot_product_attention(q, k, v)  # [B, head, 1, D_head]
        x = x.reshape(B, D)  # [B, D]
        return self.linear(x)


def pool_representations(
    cls_embeds: Tensor | None,
    reg_embeds: Tensor | None,
    patch_embeds: Tensor | None,
    representations: list[str],
):
    # Global features for the linear classifiers
    out: dict[str, Tensor] = {}
    if "cls" in representations:
        out["cls"] = cls_embeds.squeeze(1)  # [B, D]
    if "avg_patch" in representations:
        out["avg_patch"] = patch_embeds.mean(1)  # [B, D]
    if "cls_avg_patch" in representations:
        out["cls_avg_patch"] = torch.cat(
            [cls_embeds.squeeze(1), patch_embeds.mean(1)], dim=-1
        )  # [B, 2 * D]
    if "avg_reg" in representations:
        out["avg_reg"] = reg_embeds.mean(1)  # [B, D]
    if "concat_reg" in representations:
        out["concat_reg"] = reg_embeds.flatten(1, 2)  # [B, R * D]
    # Object features (registers) for the attention pooling classifiers
    if "reg" in representations:
        assert reg_embeds is not None
        out["reg"] = reg_embeds
    # Patch features for the attention pooling classifiers
    if "patch" in representations:
        assert patch_embeds is not None
        out["patch"] = patch_embeds  # [B, h * w, D]
    return out
