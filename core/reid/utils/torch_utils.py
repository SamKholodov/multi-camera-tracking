import os

import torch

from core.reid.utils import logger as LOGGER


def parse_device(device):
    return (
        str(device)
        .lower()
        .replace("cuda:", "")
        .replace("none", "")
        .replace("(", "")
        .replace(")", "")
        .replace("[", "")
        .replace("]", "")
        .replace("'", "")
        .replace(" ", "")
    )


def select_device(device="", batch=0):
    device = parse_device(device)
    mps = device == "mps"
    cpu = device == "cpu" or (device == "" and not torch.cuda.is_available())

    if cpu or mps:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif device:
        os.environ["CUDA_VISIBLE_DEVICES"] = device

    if not cpu and not mps and torch.cuda.is_available():
        devices = device.split(",") if device else ["0"]
        arg = "cuda:" + devices[0]
    elif mps:
        arg = "mps"
    else:
        arg = "cpu"

    LOGGER.info(f"ReID device: {arg}")
    return torch.device(arg)
