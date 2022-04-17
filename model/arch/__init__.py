import copy
import warnings

from .net import net

def build_model(model_cfg):
    model_cfg = copy.deepcopy(model_cfg)
    name = model_cfg.arch.pop("name")
    if name == "net":
        #model = net(model_cfg.arch.backbone)
        model = net(model_cfg.arch.backbone, model_cfg.arch.fpn, model_cfg.arch.head)
    else:
        raise NotImplementedError
    return model
