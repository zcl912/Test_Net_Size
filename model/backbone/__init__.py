import copy

from .shufflenetv2 import ShuffleNetV2
from .esnet import EsNet


def build_backbone(cfg):
    backbone_cfg = copy.deepcopy(cfg)
    name = backbone_cfg.pop("name")
    if name == "ShuffleNetV2":
        return ShuffleNetV2(**backbone_cfg)
    elif name == "EsNet":
        return EsNet(**backbone_cfg)
