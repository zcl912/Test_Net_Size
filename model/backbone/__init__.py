import copy

from .shufflenetv2 import ShuffleNetV2


def build_backbone(cfg):
    backbone_cfg = copy.deepcopy(cfg)
    name = backbone_cfg.pop("name")
    if name == "ShuffleNetV2":
        return ShuffleNetV2(**backbone_cfg)
