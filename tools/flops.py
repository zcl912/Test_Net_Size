import argparse

import torch

from model.arch import build_model
from utils import cfg, load_config


def main(config, input_shape=(320, 320)):
    model = build_model(config.model)
    try:
        import mobile_cv.lut.lib.pt.flops_utils as flops_utils
    except ImportError:
        print("mobilde-cv is not installed. Skip flops calculation.")
        return
    first_batch = torch.rand(1, 3, input_shape[0], input_shape[1])
    input_args = (first_batch,)
    flops_utils.print_model_flops(model, input_args)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
        description="Convert .pth model to onnx.",
    )
    parser.add_argument("cfg", type=str, help="Path to .yml config file.")
    parser.add_argument("--input_shape", type=str, default=None, help="Model input shape.")
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()

    cfg_path = args.cfg
    load_config(cfg, cfg_path)

    input_shape = args.input_shape
    if input_shape is None:
        input_shape = cfg.data.train.input_size
    else:
        input_shape = tuple(map(int, input_shape.split(",")))
        assert len(input_shape) == 2
    main(config=cfg, input_shape=input_shape)
