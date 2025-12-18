from __future__ import annotations

import os
import random
import warnings
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf

from .augment import build_augmentation_pipeline


def _read_labels(label_path: str):
    boxes = []
    labels = []
    if os.path.exists(label_path):
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(float(parts[0]))
                cx, cy, w, h = map(float, parts[1:5])
                boxes.append([cx, cy, w, h])
                labels.append(cls)
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    return np.asarray(boxes, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def _resize_image(img: np.ndarray, size: Tuple[int, int], mode: str) -> np.ndarray:
    import cv2

    h, w = size
    if mode == "opencv_inter_nearest":
        interp = cv2.INTER_NEAREST
    elif mode == "keras_nearest":
        interp = cv2.INTER_NEAREST
    else:
        interp = cv2.INTER_LINEAR
    return cv2.resize(img, (int(w), int(h)), interpolation=interp)


class SamplePool:
    """Lightweight sampler to support mixup/mosaic/copy-paste inside py_function."""

    def __init__(self, paths: Sequence[str], img_size: Tuple[int, int], resize_mode: str, max_boxes: int, seed: int = 42):
        self.paths = list(paths)
        self.img_h, self.img_w = int(img_size[0]), int(img_size[1])
        self.resize_mode = resize_mode
        self.max_boxes = max_boxes
        self.rng = random.Random(seed)

    def _load_single(self, path_str: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        import cv2

        label_path = os.path.splitext(path_str)[0] + ".txt"
        boxes, labels = _read_labels(label_path)
        img = cv2.imread(path_str)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {path_str}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = _resize_image(img, (self.img_h, self.img_w), self.resize_mode)
        img = img.astype("float32") / 255.0
        padded_boxes = np.zeros((self.max_boxes, 4), dtype=np.float32)
        padded_labels = -np.ones((self.max_boxes,), dtype=np.int32)
        num = min(len(boxes), self.max_boxes)
        if num > 0:
            padded_boxes[:num] = boxes[:num]
            padded_labels[:num] = labels[:num]
        return img, padded_boxes, padded_labels, num

    def sample_random(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # retry a few times to avoid missing files
        for _ in range(10):
            path = self.rng.choice(self.paths)
            try:
                img, boxes, labels, num = self._load_single(path)
                return img, boxes[:num], labels[:num]
            except FileNotFoundError:
                continue
        # fallback blank sample
        img = np.zeros((self.img_h, self.img_w, 3), dtype=np.float32)
        boxes = np.zeros((0, 4), dtype=np.float32)
        labels = np.zeros((0,), dtype=np.int32)
        return img, boxes, labels

    def load_and_aug(self, path: tf.Tensor, pipeline=None):
        path_str = path.numpy().decode("utf-8")
        try:
            img, boxes, labels, num = self._load_single(path_str)
        except FileNotFoundError as e:
            warnings.warn(str(e))
            img = np.zeros((self.img_h, self.img_w, 3), dtype=np.float32)
            boxes = np.zeros((self.max_boxes, 4), dtype=np.float32)
            labels = -np.ones((self.max_boxes,), dtype=np.int32)
            num = 0
        if pipeline is not None:
            boxes_valid = boxes[:num]
            labels_valid = labels[:num]
            img, boxes_valid, labels_valid = pipeline(img, boxes_valid, labels_valid)
            num = min(len(boxes_valid), self.max_boxes)
            padded_boxes = np.zeros((self.max_boxes, 4), dtype=np.float32)
            padded_labels = -np.ones((self.max_boxes,), dtype=np.int32)
            if num > 0:
                padded_boxes[:num] = boxes_valid[:num]
                padded_labels[:num] = labels_valid[:num]
            boxes, labels = padded_boxes, padded_labels
        return img.astype("float32"), boxes, labels, np.array(num, dtype=np.int32)


def build_dataset(
    list_path: Union[str, Sequence[str]],
    img_size: Tuple[int, int] = (64, 64),
    batch_size: int = 32,
    shuffle: bool = True,
    augment: bool = True,
    resize_mode: str = "opencv_inter_nearest",
    max_boxes: int = 50,
    seed: int = 42,
    aug_cfg: Optional[Dict] = None,
    num_workers: Optional[int] = None,
    cache: bool = False,
):
    """Create a tf.data pipeline from a YOLO-style train.txt or list of paths with UHD-style augmentation."""
    if isinstance(list_path, str) and os.path.isdir(list_path):
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        paths = [
            os.path.join(list_path, p)
            for p in os.listdir(list_path)
            if os.path.splitext(p.lower())[1] in exts
        ]
    elif isinstance(list_path, str):
        with open(list_path, "r", encoding="utf-8") as f:
            paths = [line.strip() for line in f if line.strip()]
    else:
        paths = list(list_path)

    # filter out missing files upfront
    existing_paths = []
    missing = []
    for p in paths:
        if os.path.exists(p):
            existing_paths.append(p)
        else:
            missing.append(p)
    if missing:
        warnings.warn(f"{len(missing)} entries in train list not found on disk; they will be skipped.")
    paths = existing_paths

    pool = SamplePool(paths, img_size=img_size, resize_mode=resize_mode, max_boxes=max_boxes, seed=seed)
    class_swap = {}
    if aug_cfg and isinstance(aug_cfg, dict):
        hf = aug_cfg.get("HorizontalFlip")
        if isinstance(hf, dict):
            class_swap = hf.get("class_swap_map", {}) or {}
    pipeline = (
        build_augmentation_pipeline(aug_cfg or {}, img_w=img_size[1], img_h=img_size[0], class_swap_map=class_swap, dataset=pool)
        if augment
        else None
    )

    def _loader(path: tf.Tensor):
        img, boxes, labels, num = tf.py_function(
            func=lambda p: pool.load_and_aug(p, pipeline),
            inp=[path],
            Tout=[tf.float32, tf.float32, tf.int32, tf.int32],
        )
        img.set_shape((img_size[0], img_size[1], 3))
        boxes.set_shape((max_boxes, 4))
        labels.set_shape((max_boxes,))
        num.set_shape(())
        return img, boxes, labels, num

    ds = tf.data.Dataset.from_tensor_slices(paths)
    if shuffle:
        ds = ds.shuffle(buffer_size=max(1024, len(paths)), seed=seed, reshuffle_each_iteration=True)
    npc = tf.data.AUTOTUNE if num_workers in (None, 0, -1) else int(num_workers)
    ds = ds.map(_loader, num_parallel_calls=npc)
    ds = ds.map(lambda i, b, l, n: {"image": i, "boxes": b, "labels": l, "num_boxes": n}, num_parallel_calls=npc)
    if cache:
        ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds
