# References:
# capi: https://github.com/facebookresearch/capi/blob/main/eval_classification.py

import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# backbone classification wrappers adapted from capi


class ClassifierGrid(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        representation: str,
        classifiers: dict[tuple[float, float], nn.Module],
    ):
        super().__init__()
        self.representation = representation
        self.backbone = backbone

        # can't use ModuleDict bc of restrictions of keys (must be strings, no dots).
        self.hparams = list(classifiers)
        self.hparam_id_map = {hparam: ii for ii, hparam in enumerate(self.hparams)}
        self.classifiers = nn.ModuleList(list(classifiers.values()))

    def forward(self, *args, **kwargs) -> Tensor:
        cls_embeds, reg_embeds, patch_embeds = self.backbone(*args, **kwargs)
        all_embeds = {"cls": cls_embeds, "reg": reg_embeds, "patch": patch_embeds}
        embeds = all_embeds[self.representation]

        # [B, num_classes, num_classifiers]
        all_logit = torch.stack([clf(embeds) for clf in self.classifiers], dim=-1)
        return all_logit


class LinearClassifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.init_weights()

    def init_weights(self):
        nn.init.trunc_normal_(self.linear.weight, std=0.02)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor):
        assert x.ndim in {2, 3}, "linear classifier only accepts 2D or 3D inputs"
        if x.ndim == 3:
            x = x.mean(dim=1)
        x = self.linear(x)
        return x


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

    def forward(self, x: Tensor):
        assert x.ndim == 3, "attn classifier only accepts 3D inputs"
        B, N, _ = x.shape
        D = self.embed_dim

        q = self.query_token.expand(B, 1, -1)
        q = q.reshape(B, 1, self.num_heads, D // self.num_heads)  # [B, 1, head, D_head]
        q = q.permute(0, 2, 1, 3)  # [B, head, 1, D_head]

        # [B, N, 2, head, D_head]
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, D // self.num_heads)
        kv = kv.permute(2, 0, 3, 1, 4)  # [2, B, head, N, D_head]
        k, v = torch.unbind(kv, dim=0)  # 2 * [B, head, N, D_head]

        x = F.scaled_dot_product_attention(q, k, v)  # [B, head, 1, D_head]
        x = x.reshape(B, D)  # [B, D]
        x = self.linear(x)
        return x


class MLPClassifier(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        embed_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        embed_dim = embed_dim or in_dim

        self.fc1 = nn.Linear(in_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(embed_dim, out_dim)
        self.init_weights()

    def init_weights(self):
        nn.init.trunc_normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.trunc_normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim in {2, 3}, "mlp classifier only accepts 2D or 3D inputs"
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.norm(x)
        x = self.fc2(x)
        if x.ndim == 3:
            x = x.mean(dim=1)
        return x


CLASSIFIERS = {
    "linear": LinearClassifier,
    "attn": AttnPoolClassifier,
    "mlp": MLPClassifier,
}


def filter_kwargs(func, kwargs):
    sigature = inspect.signature(func)
    kwargs = {k: v for k, v in kwargs.items() if k in sigature.parameters}
    return kwargs


def create_classifier(name: str, in_dim: int, out_dim: int, **kwargs):
    clf_cls = CLASSIFIERS[name]
    kwargs = filter_kwargs(clf_cls, kwargs)
    clf = clf_cls(in_dim=in_dim, out_dim=out_dim, **kwargs)
    return clf


def list_classififiers():
    return list(CLASSIFIERS)
