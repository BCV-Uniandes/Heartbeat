import argparse

import torch


def boolean(text):
    value = text.strip().lower()
    if value in ("true", "1", "yes", "y", "on"):
        return True
    if value in ("false", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(
        f"{text!r} is not a boolean; use true or false")


def device_name(explicit=None):
    if explicit is not None:
        return explicit
    return "cuda" if torch.cuda.is_available() else "cpu"
