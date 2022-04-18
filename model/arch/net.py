import time

import torch
import torch.nn as nn

from model.backbone import build_backbone
from model.fpn import build_fpn
from model.head import build_head
import pdb


class net(nn.Module):
    def __init__(
        self,
        backbone_cfg,
        fpn_cfg=None,
        head_cfg=None,
    ):
        super(net, self).__init__()
        self.backbone = build_backbone(backbone_cfg)
        if fpn_cfg is not None:
            self.fpn = build_fpn(fpn_cfg)
        if head_cfg is not None:
            self.head = build_head(head_cfg)

    def forward(self, x):
        pdb.set_trace()
        x = self.backbone(x)
        if hasattr(self, "fpn"):
            x = self.fpn(x)
        if hasattr(self, "head"):
            x = self.head(x)
        return x
