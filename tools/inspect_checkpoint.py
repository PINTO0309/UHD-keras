import argparse
import os
import sys

import numpy as np
import tensorflow as tf
import tf_keras as keras

from uhd_keras.model import UltraTinyOD, UltraTinyODConfig


def load_model_any(path: str):
    """Load a Keras/SavedModel checkpoint with custom objects wired."""
    custom_objects = {
        "UltraTinyOD": UltraTinyOD,
        "UltraTinyODConfig": UltraTinyODConfig,
    }
    if path.endswith(".keras"):
        return keras.models.load_model(path, compile=False, custom_objects=custom_objects)
    # SavedModel directory
    if os.path.isdir(path):
        return keras.models.load_model(path, compile=False, custom_objects=custom_objects)
    raise FileNotFoundError(f"Checkpoint not found: {path}")


def main():
    parser = argparse.ArgumentParser(description="Inspect anchors and wh_scale from a checkpoint")
    parser.add_argument("ckpt", help="Path to .keras or SavedModel directory")
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        sys.exit(f"Checkpoint path not found: {args.ckpt}")

    model = load_model_any(args.ckpt)
    anchors = getattr(model, "anchors", None)
    print("=== Anchors ===")
    if anchors is None:
        print("No anchors attribute found on model.")
    else:
        arr = np.array(anchors)
        print(arr)

    print("\n=== wh_scale ===")
    head = getattr(model, "head", None)
    if head is None:
        print("No head found on model.")
    else:
        wh_scale = getattr(head, "wh_scale", None)
        if wh_scale is None:
            print("wh_scale not present.")
        else:
            print(np.array(wh_scale))


if __name__ == "__main__":
    main()
