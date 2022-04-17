import copy
import warnings

from .nanodet_plus import nanodet_plus
from .net import net


def build_model(model_cfg):
    model_cfg = copy.deepcopy(model_cfg)
    name = model_cfg.arch.pop("name")
    if name == "nanodet-plus":
        #model = net(model_cfg.arch.backbone)
        model = nanodet_plus(model_cfg.arch.backbone, model_cfg.arch.fpn, model_cfg.arch.head)
    elif name == "net":
        model = net(model_cfg.arch.backbone, model_cfg.arch.fpn, model_cfg.arch.head)
    else:
        raise NotImplementedError
    return model
