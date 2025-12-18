from __future__ import annotations

import argparse
import json
import os
import random
import logging
import warnings
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

# Reduce TensorFlow log noise (suppress info/warnings) before TF import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_VLOG_LEVEL", "0")
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=Warning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

import numpy as np
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
import tf_keras as keras
import yaml
from tqdm import tqdm
from absl import logging as absl_logging

logger = tf.get_logger()
logger.setLevel(logging.FATAL)
tf.autograph.set_verbosity(0)
absl_logging.set_verbosity(absl_logging.FATAL)
try:
    logger.handlers.clear()
    logger.propagate = False
except Exception:
    pass
try:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
except Exception:
    pass


class _FilteredStderr:
    """Suppress specific TensorFlow C++ noise."""

    def __init__(self, wrapped):
        self._wrapped = wrapped
        self._block_substrings = [
            "local_rendezvous.cc:407",
            "OUT_OF_RANGE: End of sequence",
            "computation placer already registered",
            "Unable to register cuDNN factory",
            "Unable to register cuBLAS factory",
            "Unable to register cuFFT factory",
        ]

    def write(self, data):
        for sub in self._block_substrings:
            if sub in data:
                return
        return self._wrapped.write(data)

    def flush(self):
        try:
            return self._wrapped.flush()
        except Exception:
            return None


# Redirect stderr to filter noisy TF C++ logs
sys.stderr = _FilteredStderr(sys.stderr)

from .data import build_dataset
from .distill import distillation_loss
from .ema import ExponentialMovingAverage
from .losses import anchor_loss
from .model import UltraTinyOD, UltraTinyODConfig
from .ops import cxcywh_to_xyxy


def _parse_anchors(anchor_str: Optional[str]) -> Optional[List[tuple]]:
    """Parse anchors from a string like '0.1,0.2;0.2,0.3;0.3,0.4' or JSON."""
    if anchor_str is None:
        return None
    anchor_str = anchor_str.strip()
    if not anchor_str:
        return None
    if anchor_str.startswith("["):
        return [tuple(map(float, p)) for p in json.loads(anchor_str)]
    parts = anchor_str.split(";")
    anchors = []
    for p in parts:
        xy = p.strip().split(",")
        if len(xy) != 2:
            continue
        anchors.append((float(xy[0]), float(xy[1])))
    return anchors if anchors else None


def _resolve_num_classes(num_classes: Optional[int], names_path: Optional[str]) -> int:
    if num_classes is not None:
        return int(num_classes)
    if names_path and os.path.exists(names_path):
        with open(names_path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f if _.strip())
    raise ValueError("num_classes must be provided via --num-classes or --names")


def _parse_classes(arg: Optional[str]) -> Optional[List[int]]:
    if arg is None:
        return None
    s = str(arg).replace(" ", "")
    if s == "":
        return None
    return [int(p) for p in s.split(",") if p != ""]


def _parse_img_size(val) -> Tuple[int, int]:
    if isinstance(val, (list, tuple)):
        if len(val) != 2:
            raise ValueError("img-size tuple must have length 2")
        return int(val[0]), int(val[1])
    s = str(val).lower().replace(" ", "")
    if "x" in s:
        parts = s.split("x")
        if len(parts) != 2:
            raise ValueError("img-size must be HxW, e.g., 64x64")
        return int(parts[0]), int(parts[1])
    v = int(float(s))
    return v, v


def _find_resume_log(resume_path: Optional[str]) -> Optional[str]:
    """Find train.log associated with a resume checkpoint by walking up directories."""
    if not resume_path:
        return None
    cur = os.path.abspath(os.path.dirname(resume_path))
    for _ in range(4):
        cand = os.path.join(cur, "train.log")
        if os.path.exists(cand):
            return cand
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return None


def _load_resume_config(resume_path: Optional[str]) -> tuple[Optional[dict], Optional[str]]:
    """Load JSON config stored on the first line of train.log for the given checkpoint."""
    log_path = _find_resume_log(resume_path)
    if not log_path:
        return None, None
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            first = f.readline()
            cfg = json.loads(first)
        return cfg, log_path
    except Exception:
        return None, log_path


DEFAULT_AUG_CFG = {
    "RandomPhotometricDistort": {"prob": 0.5},
    "RandomHSV": {"prob": 0.75, "hue_gain": 0.015, "saturation_gain": 0.7, "value_gain": 0.4},
    "HorizontalFlip": {"prob": 0.5, "class_swap_map": {}},
    "CLAHE": {"prob": 0.01, "clip_limit": 4.0, "tile_grid_size": [8, 8]},
    "RemoveOutliers": 0.002197266,
}


def _collect_paths(input_path: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(input_path, (list, tuple)):
        return [p for p in input_path]
    if os.path.isdir(input_path):
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        return [
            os.path.join(input_path, p)
            for p in os.listdir(input_path)
            if os.path.splitext(p.lower())[1] in exts
        ]
    with open(input_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def _targets_from_batch(batch, class_map: Optional[dict] = None) -> List[dict]:
    targets = []
    num = batch["num_boxes"]
    boxes = batch["boxes"]
    labels = batch["labels"]
    bsz = int(boxes.shape[0])
    max_boxes = boxes.shape[1]
    for i in range(bsz):
        n = int(num[i].numpy()) if hasattr(num[i], "numpy") else int(num[i])
        n = max(0, min(n, max_boxes))
        box_np = boxes[i].numpy()[:n]
        label_np = labels[i].numpy()[:n].astype(np.int32)
        if class_map:
            keep = [idx for idx, l in enumerate(label_np) if int(l) in class_map]
            if len(keep) == 0:
                targets.append({"boxes": tf.zeros((0, 4), dtype=boxes.dtype), "labels": tf.zeros((0,), dtype=tf.int32)})
                continue
            box_np = box_np[keep]
            label_np = np.array([class_map[int(label_np[k])] for k in keep], dtype=np.int32)
        targets.append({"boxes": tf.convert_to_tensor(box_np, dtype=boxes.dtype), "labels": tf.convert_to_tensor(label_np, dtype=tf.int32)})
    return targets


def _bbox_iou_np(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Compute IoU between a single box and an array of boxes (xyxy)."""
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter + 1e-9
    return inter / union


def _voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


def _compute_map(all_preds, gt_store, num_classes: int, iou_thresh: float = 0.5) -> float:
    """Compute mAP@IoU over collected predictions/ground truths."""
    if not gt_store:
        return 0.0
    gt_count = [0 for _ in range(num_classes)]
    gt_by_img_cls = {}
    for img_id, g in gt_store.items():
        boxes = g["boxes"]
        labels = g["labels"]
        for cls_id in np.unique(labels):
            cls_id_int = int(cls_id)
            mask = labels == cls_id_int
            cls_boxes = boxes[mask]
            if cls_boxes.size == 0:
                continue
            gt_by_img_cls.setdefault(img_id, {})[cls_id_int] = {
                "boxes": cls_boxes,
                "matched": np.zeros((cls_boxes.shape[0],), dtype=bool),
            }
            gt_count[cls_id_int] += cls_boxes.shape[0]

    preds_by_cls = {c: [] for c in range(num_classes)}
    for img_id, cls_id, score, box in all_preds:
        if cls_id < 0 or cls_id >= num_classes:
            continue
        preds_by_cls[cls_id].append((img_id, float(score), box))

    ap_list = []
    for cls_id in range(num_classes):
        total_gt = gt_count[cls_id]
        if total_gt == 0:
            continue
        preds = sorted(preds_by_cls.get(cls_id, []), key=lambda x: x[1], reverse=True)
        if not preds:
            ap_list.append(0.0)
            continue
        tp = np.zeros((len(preds),), dtype=np.float32)
        fp = np.zeros((len(preds),), dtype=np.float32)
        for i, (img_id, score, box) in enumerate(preds):
            gts = gt_by_img_cls.get(img_id, {}).get(cls_id)
            if gts is None or gts["boxes"].size == 0:
                fp[i] = 1.0
                continue
            ious = _bbox_iou_np(box, gts["boxes"])
            best_idx = int(np.argmax(ious)) if ious.size else -1
            best_iou = float(ious[best_idx]) if ious.size else 0.0
            if best_iou >= iou_thresh and not gts["matched"][best_idx]:
                tp[i] = 1.0
                gts["matched"][best_idx] = True
            else:
                fp[i] = 1.0
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recall = cum_tp / max(float(total_gt), 1e-9)
        precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
        ap_list.append(_voc_ap(recall, precision))
    if not ap_list:
        return 0.0
    return float(np.mean(ap_list))


@dataclass
class TrainConfig:
    image_dir: str
    names_path: Optional[str] = None
    num_classes: Optional[int] = None
    classes: Optional[str] = None
    aug_config: Optional[str] = None
    use_augment: bool = True
    train_split: float = 0.8
    val_split: float = 0.2
    seed: int = 42
    exp_name: Optional[str] = None
    epochs: int = 100
    batch_size: int = 32
    img_size: Tuple[int, int] = (64, 64)
    lr: float = 1e-3
    weight_decay: float = 0.0
    log_interval: int = 10
    eval_interval: int = 1
    ckpt_out: Optional[str] = None
    log_dir: str = "runs/tensorboard"
    resume: Optional[str] = None
    teacher_ckpt: Optional[str] = None
    distill_weight: float = 0.0
    feature_distill_weight: float = 0.0
    use_ema: bool = False
    ema_decay: float = 0.999
    use_amp: bool = False
    use_xla: bool = False
    resize_mode: str = "opencv_inter_nearest"
    max_boxes: int = 50
    use_residual: bool = False
    use_improved_head: bool = False
    use_head_ese: bool = False
    use_iou_aware_head: bool = False
    quality_power: float = 1.0
    cls_bottleneck_ratio: float = 0.5
    anchors: Optional[str] = None
    c_stem: int = 64
    cnn_width: Optional[int] = None
    activation: str = "swish"
    assigner: str = "legacy"
    iou_loss: str = "giou"
    cls_loss_type: str = "bce"
    simota_topk: int = 10
    use_batchnorm: bool = False
    grad_clip_norm: Optional[float] = None
    num_workers: Optional[int] = None
    utod_head_ese: bool = False
    utod_large_obj_branch: bool = False
    utod_large_obj_depth: int = 1
    utod_large_obj_ch_scale: float = 1.0
    conf_thresh: float = 0.25
    auto_anchors: bool = False
    num_anchors: int = 3


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        tf.random.set_seed(cfg.seed)
        self.use_xla = cfg.use_xla
        if cfg.use_amp:
            from tf_keras import mixed_precision
            mixed_precision.set_global_policy("mixed_float16")
        if self.use_xla:
            try:
                tf.config.optimizer.set_jit(True)
            except Exception:
                pass
        self.aug_cfg = self._load_aug_cfg(cfg.aug_config) if cfg.use_augment else {}
        base = os.path.join("runs", cfg.exp_name or "ultratinyod")
        cfg.ckpt_out = base
        if cfg.log_dir == "runs/tensorboard":
            cfg.log_dir = base
        self.class_ids = _parse_classes(cfg.classes)
        if self.class_ids:
            self.num_classes = len(self.class_ids)
        else:
            self.num_classes = _resolve_num_classes(cfg.num_classes, cfg.names_path)
        self.class_map = {cid: idx for idx, cid in enumerate(self.class_ids)} if self.class_ids else None
        anchors = _parse_anchors(cfg.anchors)
        if cfg.auto_anchors:
            anchors = self._auto_compute_anchors(cfg.image_dir, cfg.num_anchors)
            print(f"Auto-computed anchors: {anchors}")
        ut_cfg = UltraTinyODConfig(
            num_classes=self.num_classes,
            anchors=anchors,
            cls_bottleneck_ratio=cfg.cls_bottleneck_ratio,
            use_improved_head=cfg.use_improved_head,
            use_head_ese=cfg.use_head_ese or cfg.utod_head_ese,
            use_iou_aware_head=cfg.use_iou_aware_head,
            quality_power=cfg.quality_power,
            use_batchnorm=cfg.use_batchnorm,
            activation=cfg.activation,
            use_large_obj_branch=cfg.utod_large_obj_branch,
            large_obj_branch_depth=cfg.utod_large_obj_depth,
            large_obj_branch_expansion=cfg.utod_large_obj_ch_scale,
        )
        stem = cfg.cnn_width if cfg.cnn_width is not None else cfg.c_stem
        self.model = UltraTinyOD(
            num_classes=self.num_classes,
            config=ut_cfg,
            c_stem=stem,
            use_residual=cfg.use_residual,
        )
        # build model
        h, w = cfg.img_size
        self.model(tf.zeros((1, h, w, 3)))
        if cfg.resume:
            self.model.load_weights(cfg.resume)
            print(f"Resumed weights from {cfg.resume}")
        self.teacher = None
        if cfg.teacher_ckpt:
            self.teacher = UltraTinyOD(
                num_classes=self.num_classes,
                config=ut_cfg,
                c_stem=cfg.c_stem,
                use_residual=cfg.use_residual,
            )
            self.teacher(tf.zeros((1, cfg.img_size[0], cfg.img_size[1], 3)))
            self.teacher.load_weights(cfg.teacher_ckpt)
            self.teacher.trainable = False
            print(f"Loaded teacher weights from {cfg.teacher_ckpt}")
        if cfg.use_amp:
            from tf_keras import mixed_precision
            base_opt = keras.optimizers.AdamW(learning_rate=cfg.lr, weight_decay=cfg.weight_decay)
            self.optimizer = mixed_precision.LossScaleOptimizer(base_opt)
        else:
            self.optimizer = keras.optimizers.AdamW(learning_rate=cfg.lr, weight_decay=cfg.weight_decay)
        self.ema = ExponentialMovingAverage(self.model, decay=cfg.ema_decay) if cfg.use_ema else None
        self.best_loss = float("inf")
        self.best_map = float("-inf")
        self.use_amp = cfg.use_amp
        self.fwd_fn = self._build_fwd_fn()
        self.train_step_fn = self._build_train_step(jit=cfg.use_xla)
        self.writer = tf.summary.create_file_writer(cfg.log_dir)
        self.text_log_path = os.path.join(cfg.log_dir, "train.log")
        os.makedirs(cfg.log_dir, exist_ok=True)
        with open(self.text_log_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(cfg.__dict__, indent=2) + "\n")
            f.write("epoch,loss,box,obj,cls,quality,val_loss,val_map\n")
        self.ckpt_base = cfg.ckpt_out or os.path.join("runs", cfg.exp_name or "ultratinyod")
        self.best_dir = self.ckpt_base
        self.last_dir = self.ckpt_base
        os.makedirs(self.ckpt_base, exist_ok=True)
        self._setup_checkpoint()
        self.start_epoch, self.global_step = self._restore_training_state(cfg.resume)

    def _setup_checkpoint(self):
        self.ckpt_global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")
        self.ckpt_epoch = tf.Variable(0, dtype=tf.int64, trainable=False, name="epoch")
        self.ckpt_best_map = tf.Variable(self.best_map, dtype=tf.float32, trainable=False, name="best_map")
        self.ckpt_best_loss = tf.Variable(self.best_loss, dtype=tf.float32, trainable=False, name="best_loss")
        ema_shadow = self.ema.shadow if self.ema else []
        self.ckpt = tf.train.Checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            ema_shadow=ema_shadow,
            global_step=self.ckpt_global_step,
            epoch=self.ckpt_epoch,
            best_map=self.ckpt_best_map,
            best_loss=self.ckpt_best_loss,
        )
        self.state_dir = os.path.join(self.ckpt_base, "train_state")
        os.makedirs(self.state_dir, exist_ok=True)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.state_dir, max_to_keep=3, checkpoint_name="train_state")

    def _restore_training_state(self, resume_path: Optional[str]) -> Tuple[int, int]:
        start_epoch = 1
        global_step = 0
        if not resume_path:
            return start_epoch, global_step
        latest = self.ckpt_manager.latest_checkpoint if self.ckpt_manager else None
        if latest:
            status = self.ckpt.restore(latest)
            try:
                status.expect_partial()
            except Exception:
                pass
            start_epoch = int(self.ckpt_epoch.numpy()) + 1
            global_step = int(self.ckpt_global_step.numpy())
            self.best_map = float(self.ckpt_best_map.numpy())
            self.best_loss = float(self.ckpt_best_loss.numpy())
            print(f"Restored training state from {latest} (epoch {start_epoch-1}, global_step {global_step}).")
            return start_epoch, global_step
        self.model.load_weights(resume_path)
        print(f"Resumed weights from {resume_path} (no optimizer state found).")
        return start_epoch, global_step

    def _save_state(self, epoch: int, global_step: int):
        if not self.ckpt_manager:
            return
        self.ckpt_epoch.assign(epoch)
        self.ckpt_global_step.assign(global_step)
        self.ckpt_best_map.assign(self.best_map)
        self.ckpt_best_loss.assign(self.best_loss)
        path = self.ckpt_manager.save(checkpoint_number=global_step)
        print(f"Saved training state to {path}")

    def train(self):
        train_paths, val_paths = self._split_paths(self.cfg.image_dir, self.cfg.train_split, self.cfg.val_split)
        print(f"Train/val split: {len(train_paths)} train, {len(val_paths)} val (seed={self.cfg.seed})")
        ds = build_dataset(
            train_paths,
            img_size=self.cfg.img_size,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            augment=True,
            resize_mode=self.cfg.resize_mode,
            max_boxes=self.cfg.max_boxes,
            seed=self.cfg.seed,
            num_workers=self.cfg.num_workers,
            aug_cfg=self.aug_cfg,
            cache=True,
        )
        val_ds = None
        if val_paths:
            val_ds = build_dataset(
                val_paths,
                img_size=self.cfg.img_size,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                augment=False,
                resize_mode=self.cfg.resize_mode,
                max_boxes=self.cfg.max_boxes,
                seed=self.cfg.seed,
                num_workers=self.cfg.num_workers,
                cache=True,
            )
        ckpt_dir = os.path.dirname(self.cfg.ckpt_out)
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(self.cfg.log_dir, exist_ok=True)
        global_step = self.global_step
        for epoch in range(self.start_epoch, self.cfg.epochs + 1):
            steps_per_epoch = tf.data.experimental.cardinality(ds).numpy()
            total_steps = steps_per_epoch if steps_per_epoch and steps_per_epoch > 0 else None
            pbar = tqdm(ds, total=total_steps, desc=f"Epoch {epoch}", leave=False, file=sys.stdout, dynamic_ncols=True)
            val_map_val = 0.0
            epoch_totals = {}
            epoch_count = 0
            train_vals = {"loss": 0.0, "box": 0.0, "obj": 0.0, "cls": 0.0, "quality": 0.0}
            for batch in pbar:
                global_step += 1
                targets = _targets_from_batch(batch, class_map=self.class_map)
                loss, loss_dict = self.train_step_fn(batch, targets)
                if self.ema:
                    self.ema.update()
                loss_value = float(loss.numpy())
                for k, v in loss_dict.items():
                    epoch_totals[k] = epoch_totals.get(k, 0.0) + float(v.numpy())
                epoch_count += 1

                if global_step % self.cfg.log_interval == 0:
                    loss_vals = {k: float(v.numpy()) for k, v in loss_dict.items()}
                    pbar.set_postfix(
                        {
                            "loss": f"{loss_vals.get('loss', 0):.5f}",
                            "box": f"{loss_vals.get('box', 0):.5f}",
                            "obj": f"{loss_vals.get('obj', 0):.5f}",
                            "cls": f"{loss_vals.get('cls', 0):.5f}",
                            "qual": f"{loss_vals.get('quality', 0):.5f}",
                        }
                    )
                    # skip per-step TensorBoard logging; epoch-level only

            if epoch_count > 0:
                avg_losses = {k: v / epoch_count for k, v in epoch_totals.items()}
                with self.writer.as_default():
                    order = ["loss", "box", "cls", "obj", "quality"]
                    for idx, k in enumerate(order, start=1):
                        if k in avg_losses:
                            tf.summary.scalar(f"train/{idx:02d}_{k}", avg_losses[k], step=epoch)
                    for k, v in avg_losses.items():
                        if k not in order:
                            tf.summary.scalar(f"train/{len(order)+1:02d}_{k}", v, step=epoch)
                for k in train_vals.keys():
                    train_vals[k] = float(avg_losses.get(k, 0.0))

            val_loss_val = None
            val_map_val = 0.0
            if val_ds is not None and (epoch % max(1, self.cfg.eval_interval) == 0):
                val_loss = self._run_eval(val_ds, epoch=epoch)
                val_loss_val = float(val_loss.get("loss", 0.0))
                val_map_val = float(val_loss.get("map", 0.0))
                with self.writer.as_default():
                    order = ["map", "loss", "box", "cls", "obj", "quality"]
                    for idx, k in enumerate(order, start=1):
                        if k in val_loss:
                            tf.summary.scalar(f"val/{idx:02d}_{k}", val_loss[k], step=epoch)
                    for k, v in val_loss.items():
                        if k not in order:
                            tf.summary.scalar(f"val/{len(order)+1:02d}_{k}", v, step=epoch)
                with open(self.text_log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"{epoch},"
                        f"{train_vals['loss']:.6f},{train_vals['box']:.6f},{train_vals['obj']:.6f},{train_vals['cls']:.6f},{train_vals['quality']:.6f},"
                        f"{val_loss_val:.6f},{val_map_val:.6f}\n"
                    )
                print(f"Epoch {epoch} val: loss={val_loss_val:.5f} AP@0.5={val_map_val:.5f}")
                if val_map_val > self.best_map:
                    self.best_map = val_map_val
                    self._save_best(val_map_val, epoch, global_step, metric_name="map")
                elif val_loss_val < self.best_loss:
                    # fallback to loss-based best if mAP does not improve (no file save)
                    self.best_loss = val_loss_val
            # fallback: if no val, use train loss at epoch end
            if val_loss_val is None:
                if loss_value < self.best_loss:
                    self.best_loss = loss_value
                    # no file save for loss-only best
                with open(self.text_log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"{epoch},"
                        f"{train_vals['loss']:.6f},{train_vals['box']:.6f},{train_vals['obj']:.6f},{train_vals['cls']:.6f},{train_vals['quality']:.6f},"
                        f"{0.0:.6f},{0.0:.6f}\n"
                    )
            self._save_last(epoch, map_value=val_map_val, global_step=global_step)
            print(f"Finished epoch {epoch}")

    def _run_eval(self, val_ds, epoch: int = 0) -> dict:
        totals = {}
        count = 0
        all_preds = []
        gt_store = {}
        img_offset = 0
        for batch in tqdm(val_ds, desc=f"Val {epoch}", leave=False, file=sys.stdout, dynamic_ncols=True):
            targets = _targets_from_batch(batch, class_map=self.class_map)
            raw, decoded = self.model(batch["image"], training=False, decode=True)
            loss_dict = anchor_loss(
                raw,
                targets,
                anchors=self.model.anchors,
                num_classes=self.num_classes,
                iou_loss=self.cfg.iou_loss,
                assigner=self.cfg.assigner,
                cls_loss_type=self.cfg.cls_loss_type,
                simota_topk=self.cfg.simota_topk,
                use_quality=self.model.has_quality_head,
                wh_scale=self.model.head.wh_scale if self.model.head.use_improved_head else None,
            )
            for k, v in loss_dict.items():
                totals[k] = totals.get(k, 0.0) + float(v.numpy())
            count += 1
            # collect GT and predictions for mAP
            bsz = batch["image"].shape[0]
            for bi in range(bsz):
                img_id = img_offset + bi
                gt_boxes = targets[bi]["boxes"]
                gt_labels = targets[bi]["labels"]
                if isinstance(gt_boxes, tf.Tensor):
                    gt_boxes = gt_boxes.numpy()
                if isinstance(gt_labels, tf.Tensor):
                    gt_labels = gt_labels.numpy()
                gt_xyxy = cxcywh_to_xyxy(tf.convert_to_tensor(gt_boxes)).numpy() if gt_boxes.size else np.zeros((0, 4), dtype=np.float32)
                gt_store[img_id] = {"boxes": gt_xyxy, "labels": gt_labels}
                for sc, cls_id, box in decoded[bi]:
                    box_xyxy = cxcywh_to_xyxy(tf.expand_dims(box, 0))[0].numpy()
                    all_preds.append((img_id, int(cls_id), float(sc), box_xyxy))
            img_offset += bsz
        if count == 0:
            return {k: 0.0 for k in totals}
        avg = {k: v / count for k, v in totals.items()}
        avg["map"] = _compute_map(all_preds, gt_store, self.num_classes, iou_thresh=0.5)
        return avg

    def _save_weights(self, base_path: str):
        """Save checkpoint (Keras v3 single-file only)."""
        if self.ema:
            self.ema.apply_shadow()
        keras_path = f"{base_path}.keras"
        os.makedirs(os.path.dirname(keras_path) or ".", exist_ok=True)
        import io
        import contextlib

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.model.save(keras_path)
        if self.ema:
            self.ema.restore()

    def _save_saved_model(self, export_dir: str):
        if self.ema:
            self.ema.apply_shadow()
        if os.path.exists(export_dir):
            if os.path.isdir(export_dir):
                for root, dirs, files in os.walk(export_dir, topdown=False):
                    for f in files:
                        os.remove(os.path.join(root, f))
                    for d in dirs:
                        os.rmdir(os.path.join(root, d))
                os.rmdir(export_dir)
            else:
                os.remove(export_dir)
        import io
        import contextlib

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            if hasattr(self.model, "export"):
                self.model.export(export_dir)
            else:
                tf.saved_model.save(self.model, export_dir)
        if self.ema:
            self.ema.restore()

    def _save_best(self, metric_value: float, epoch: int, step: int, metric_name: str = "loss"):
        if metric_name != "map":
            print(f"Best {metric_name} updated to {metric_value:.5f} (no best file saved for non-mAP metrics).")
            return
        base = f"best_utod_{epoch:04d}_map_{metric_value:.5f}"
        prefix = "best_utod_"
        path = os.path.join(self.best_dir, base)
        self._save_weights(path)
        sm_path = os.path.join(self.best_dir, base)
        self._save_saved_model(sm_path)
        self._prune_dir(self.best_dir, prefix=prefix, keep=10)
        print(f"Saved best checkpoint to {path}.keras ({metric_name}={metric_value:.5f})")

    def _save_last(self, epoch: int, map_value: float = 0.0, global_step: int = 0):
        base = f"last_utod_{epoch:04d}_map_{map_value:.5f}"
        path = os.path.join(self.last_dir, base)
        self._save_weights(path)
        self._save_saved_model(path)
        self._save_state(epoch, global_step)
        self._prune_dir(self.last_dir, prefix="last_", keep=10)

    def _prune_dir(self, directory: str, prefix: str, keep: int = 10):
        # group by base name (strip .keras or _saved_model)
        grouped = {}
        for name in os.listdir(directory):
            if not name.startswith(prefix):
                continue
            base = name
            if base.endswith(".keras"):
                base = base[:-6]
            full = os.path.join(directory, name)
            try:
                mtime = os.path.getmtime(full)
            except OSError:
                continue
            grouped[base] = max(grouped.get(base, mtime), mtime)
        entries = sorted(grouped.items(), key=lambda x: x[1], reverse=True)
        for base, _ in entries[keep:]:
            keras_path = os.path.join(directory, f"{base}.keras")
            sm_path = os.path.join(directory, base)
            for path in (keras_path, sm_path):
                if not os.path.exists(path):
                    continue
                try:
                    if os.path.isdir(path):
                        for root, dirs, files in os.walk(path, topdown=False):
                            for f in files:
                                os.remove(os.path.join(root, f))
                            for d in dirs:
                                os.rmdir(os.path.join(root, d))
                        os.rmdir(path)
                    else:
                        os.remove(path)
                except OSError:
                    pass

    def _load_aug_cfg(self, path: Optional[str]) -> dict:
        if path is None:
            return DEFAULT_AUG_CFG
        if not os.path.exists(path):
            print(f"Aug config {path} not found; falling back to defaults.")
            return DEFAULT_AUG_CFG
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if isinstance(cfg, dict) and "data_augment" in cfg:
            return cfg["data_augment"] or {}
        return cfg or {}

    def _auto_compute_anchors(self, data_path: str, k: int) -> Optional[List[tuple]]:
        """Compute anchors from YOLO label files using IoU-based k-means on w,h."""
        wh = []
        for img_path in _collect_paths(data_path):
            label_path = os.path.splitext(img_path)[0] + ".txt"
            if not os.path.exists(label_path):
                continue
            with open(label_path, "r", encoding="utf-8") as lf:
                for row in lf:
                    parts = row.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        w = float(parts[3])
                        h = float(parts[4])
                        cls = int(float(parts[0]))
                    except ValueError:
                        continue
                    if self.class_map and cls not in self.class_map:
                        continue
                    if w <= 0 or h <= 0:
                        continue
                    wh.append((w, h))
        boxes = np.asarray(wh, dtype=np.float32)
        if boxes.size == 0:
            print("Auto-anchor: no labels found; falling back to defaults.")
            return None
        anchors = self._kmeans_wh(boxes, k=k, iters=30)
        return [tuple(map(float, a)) for a in anchors.tolist()]

    @staticmethod
    def _kmeans_wh(boxes: np.ndarray, k: int = 3, iters: int = 20) -> np.ndarray:
        """K-means with IoU distance on widths/heights, sorted by area."""
        if boxes.shape[0] < k:
            boxes = np.concatenate([boxes, boxes[np.random.choice(boxes.shape[0], k - boxes.shape[0])]], axis=0)

        rng = np.random.default_rng(0)
        centers = boxes[rng.choice(boxes.shape[0], k, replace=False)]

        def wh_iou(b: np.ndarray, c: np.ndarray) -> np.ndarray:
            inter = np.minimum(b[:, None, :], c[None, :, :]).prod(axis=2)
            union = (b[:, 0] * b[:, 1])[:, None] + (c[:, 0] * c[:, 1])[None, :] - inter + 1e-9
            return inter / union

        for _ in range(iters):
            ious = wh_iou(boxes, centers)
            assign = ious.argmax(axis=1)
            new_centers = []
            for ki in range(k):
                mask = assign == ki
                if mask.sum() == 0:
                    new_centers.append(centers[ki])
                else:
                    new_centers.append(boxes[mask].mean(axis=0))
            new_centers = np.stack(new_centers, axis=0)
            if np.allclose(new_centers, centers):
                break
            centers = new_centers
        centers = centers[np.argsort(centers.prod(axis=1))]
        return centers

    def _split_paths(self, list_path: str, train_split: float, val_split: float) -> Tuple[List[str], List[str]]:
        paths = _collect_paths(list_path)
        rng = np.random.default_rng(self.cfg.seed)
        rng.shuffle(paths)
        total = max(train_split + val_split, 1e-9)
        train_ratio = train_split / total
        n = len(paths)
        n_train = int(round(n * train_ratio))
        if val_split > 0 and n_train == n and n > 1:
            n_train = n - 1
        if train_ratio > 0 and n_train == 0 and n > 0:
            n_train = 1
        train_paths = paths[:n_train]
        val_paths = paths[n_train:]
        return train_paths, val_paths

    def _build_train_step(self, jit: bool = False):
        optimizer = self.optimizer
        model = self.model
        teacher = self.teacher
        cfg = self.cfg
        num_classes = self.num_classes
        class_map = self.class_map
        anchors = model.anchors
        has_quality = model.has_quality_head
        fwd_fn = self.fwd_fn

        if jit:
            print("Warning: XLA/graph mode is not fully supported; running eager train step instead.")

        def step(batch, targets):
            with tf.GradientTape() as tape:
                pred, feat = fwd_fn(batch["image"])
                loss_dict = anchor_loss(
                    pred,
                    targets,
                    anchors=anchors,
                    num_classes=num_classes,
                    iou_loss=cfg.iou_loss,
                    assigner=cfg.assigner,
                    cls_loss_type=cfg.cls_loss_type,
                    simota_topk=cfg.simota_topk,
                    use_quality=has_quality,
                    wh_scale=model.head.wh_scale if model.head.use_improved_head else None,
                )
                loss = loss_dict["loss"]
                if teacher is not None and (cfg.distill_weight > 0.0 or cfg.feature_distill_weight > 0.0):
                    t_pred, t_feat = teacher(batch["image"], training=False, return_feat=True)
                    distill = distillation_loss(
                        student_pred=pred,
                        teacher_pred=t_pred,
                        student_feat=feat,
                        teacher_feat=t_feat,
                        logit_weight=cfg.distill_weight,
                        feature_weight=cfg.feature_distill_weight,
                    )
                    loss = loss + distill
            if self.use_amp:
                scaled_loss = optimizer.get_scaled_loss(loss)
                scaled_grads = tape.gradient(scaled_loss, model.trainable_variables)
                grads = optimizer.get_unscaled_gradients(scaled_grads)
            else:
                grads = tape.gradient(loss, model.trainable_variables)
            grad_var = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]
            if cfg.grad_clip_norm is not None and grad_var:
                g_list, v_list = zip(*grad_var)
                g_list, _ = tf.clip_by_global_norm(g_list, cfg.grad_clip_norm)
                grad_var = list(zip(g_list, v_list))
            if grad_var:
                optimizer.apply_gradients(grad_var)
            return loss, loss_dict

        return step

    def _build_fwd_fn(self):
        @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None, 3], dtype=tf.float32)])
        def forward_fn(images):
            x = images
            if self.use_amp:
                x = tf.cast(x, tf.float16)
            return self.model(x, training=True, return_feat=True)

        return forward_fn


def build_argparser():
    parser = argparse.ArgumentParser(description="Train UltraTinyOD (Keras)")
    parser.add_argument("--image-dir", default="data/wholebody34/obj_train_data", help="Path to image directory (labels alongside with .txt)")
    parser.add_argument("--names", dest="names_path", default=None, help="obj.names file to set class count")
    parser.add_argument("--num-classes", type=int, default=None, help="Override class count")
    parser.add_argument("--classes", type=str, default=None, help="Comma-separated subset of class ids to use")
    parser.add_argument("--aug-config", type=str, default="uhd_keras/aug.yaml", help="YAML file with data_augment config (UHD format)")
    parser.add_argument("--no-aug", action="store_true", help="Disable augmentation")
    parser.add_argument("--exp-name", type=str, default=None, help="Experiment name for outputs (ckpt/log dirs)")
    parser.add_argument("--train-split", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--val-split", type=float, default=0.2, help="Val split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split/augment")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=str, default="64", help="Input size (e.g., 64 or 64x64)")
    parser.add_argument("--num-workers", type=int, default=None, help="tf.data num_parallel_calls")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=1, help="Epoch interval for running val loss if split is available")
    parser.add_argument("--ckpt-out", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--teacher-ckpt", type=str, default=None, help="Teacher checkpoint for distillation")
    parser.add_argument("--distill-weight", type=float, default=0.0)
    parser.add_argument("--feature-distill-weight", type=float, default=0.0)
    parser.add_argument("--use-ema", action="store_true", help="Enable EMA tracking")
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--use-xla", action="store_true", help="Enable XLA compilation for train step")
    parser.add_argument("--use-amp", action="store_true", help="Enable mixed precision (float16) training")
    parser.add_argument(
        "--resize-mode",
        type=str,
        default="opencv_inter_nearest",
        choices=["opencv_inter_nearest", "keras_bilinear", "keras_nearest"],
    )
    parser.add_argument("--max-boxes", type=int, default=50)
    parser.add_argument("--use-residual", action="store_true")
    parser.add_argument("--use-improved-head", action="store_true")
    parser.add_argument("--use-head-ese", action="store_true")
    parser.add_argument("--utod-head-ese", action="store_true", help="Alias for --use-head-ese")
    parser.add_argument("--use-iou-aware-head", action="store_true")
    parser.add_argument("--quality-power", type=float, default=1.0)
    parser.add_argument("--cls-bottleneck-ratio", type=float, default=0.5)
    parser.add_argument("--anchors", type=str, default=None, help="Custom anchors 'w1,h1;w2,h2;w3,h3'")
    parser.add_argument("--utod-residual", action="store_true", help="Enable residual skips inside UltraTinyOD backbone")
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "silu", "swish"],
        help="Activation function for UltraTinyOD",
    )
    parser.add_argument("--auto-anchors", action="store_true", help="Compute anchors from training labels")
    parser.add_argument("--num-anchors", type=int, default=3, help="Number of anchors when auto-computing")
    parser.add_argument("--cnn-width", type=int, default=64, help="UltraTinyOD width (replaces c-stem)")
    parser.add_argument("--utod-large-obj-branch", action="store_true", help="Enable UTOD large object branch")
    parser.add_argument("--utod-large-obj-depth", type=int, default=1, help="Depth of large object branch")
    parser.add_argument("--utod-large-obj-ch-scale", type=float, default=1.0, help="Channel scale for large object branch")
    parser.add_argument("--assigner", type=str, default="legacy", choices=["legacy", "simota"])
    parser.add_argument("--iou-loss", type=str, default="ciou", choices=["iou", "giou", "ciou"])
    parser.add_argument("--cls-loss-type", type=str, default="bce", choices=["bce", "vfl"])
    parser.add_argument("--simota-topk", type=int, default=10)
    parser.add_argument("--use-batchnorm", action="store_true", help="Enable BatchNorm layers (off by default)")
    parser.add_argument("--grad-clip-norm", type=float, default=None, help="Global norm gradient clipping")
    parser.add_argument("--log-dir", type=str, default="runs/tensorboard", help="TensorBoard log directory")
    parser.add_argument("--conf-thresh", type=float, default=0.25, help="Confidence threshold (compat)")
    parser.add_argument("--opencv_inter_nearest", action="store_true", help="Shortcut to set resize-mode opencv_inter_nearest")
    parser.add_argument("--keras_bilinear", action="store_true", help="Shortcut to set resize-mode keras_bilinear")
    parser.add_argument("--keras_nearest", action="store_true", help="Shortcut to set resize-mode keras_nearest")
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()
    if args.opencv_inter_nearest:
        args.resize_mode = "opencv_inter_nearest"
    if args.keras_bilinear:
        args.resize_mode = "keras_bilinear"
    if args.keras_nearest:
        args.resize_mode = "keras_nearest"
    defaults = parser.parse_args([])
    resume_cfg, resume_log_path = _load_resume_config(args.resume)
    if resume_cfg:
        print(f"Loaded config from {resume_log_path}. CLI overrides will be applied.")
    elif args.resume:
        msg = f"No train.log found for resume path {args.resume}." if resume_log_path is None else f"Failed to parse {resume_log_path}; using CLI defaults."
        print(msg)

    def set_field(cfg_dict, name, arg_val, default_val, transform=lambda x: x):
        if name not in cfg_dict or arg_val != default_val:
            cfg_dict[name] = transform(arg_val)

    cfg_dict = dict(resume_cfg) if resume_cfg else {}
    set_field(cfg_dict, "image_dir", args.image_dir, defaults.image_dir)
    set_field(cfg_dict, "names_path", args.names_path, defaults.names_path)
    set_field(cfg_dict, "num_classes", args.num_classes, defaults.num_classes)
    set_field(cfg_dict, "classes", args.classes, defaults.classes)
    set_field(cfg_dict, "aug_config", args.aug_config, defaults.aug_config)
    set_field(cfg_dict, "use_augment", not args.no_aug, not defaults.no_aug)
    set_field(cfg_dict, "exp_name", args.exp_name, defaults.exp_name)
    set_field(cfg_dict, "train_split", args.train_split, defaults.train_split)
    set_field(cfg_dict, "val_split", args.val_split, defaults.val_split)
    set_field(cfg_dict, "seed", args.seed, defaults.seed)
    set_field(cfg_dict, "epochs", args.epochs, defaults.epochs)
    set_field(cfg_dict, "batch_size", args.batch_size, defaults.batch_size)
    set_field(cfg_dict, "img_size", args.img_size, defaults.img_size, _parse_img_size)
    set_field(cfg_dict, "lr", args.lr, defaults.lr)
    set_field(cfg_dict, "weight_decay", args.weight_decay, defaults.weight_decay)
    set_field(cfg_dict, "log_interval", args.log_interval, defaults.log_interval)
    set_field(cfg_dict, "eval_interval", args.eval_interval, defaults.eval_interval)
    set_field(cfg_dict, "ckpt_out", args.ckpt_out, defaults.ckpt_out)
    cfg_dict["resume"] = args.resume
    set_field(cfg_dict, "teacher_ckpt", args.teacher_ckpt, defaults.teacher_ckpt)
    set_field(cfg_dict, "distill_weight", args.distill_weight, defaults.distill_weight)
    set_field(cfg_dict, "feature_distill_weight", args.feature_distill_weight, defaults.feature_distill_weight)
    set_field(cfg_dict, "use_ema", args.use_ema, defaults.use_ema)
    set_field(cfg_dict, "ema_decay", args.ema_decay, defaults.ema_decay)
    set_field(cfg_dict, "use_amp", args.use_amp, defaults.use_amp)
    set_field(cfg_dict, "use_xla", args.use_xla, defaults.use_xla)
    set_field(cfg_dict, "resize_mode", args.resize_mode, defaults.resize_mode)
    set_field(cfg_dict, "max_boxes", args.max_boxes, defaults.max_boxes)
    set_field(cfg_dict, "use_residual", args.use_residual or args.utod_residual, defaults.use_residual or defaults.utod_residual)
    set_field(cfg_dict, "use_improved_head", args.use_improved_head, defaults.use_improved_head)
    set_field(cfg_dict, "use_head_ese", args.use_head_ese or args.utod_head_ese, defaults.use_head_ese or defaults.utod_head_ese)
    set_field(cfg_dict, "use_iou_aware_head", args.use_iou_aware_head, defaults.use_iou_aware_head)
    set_field(cfg_dict, "quality_power", args.quality_power, defaults.quality_power)
    set_field(cfg_dict, "cls_bottleneck_ratio", args.cls_bottleneck_ratio, defaults.cls_bottleneck_ratio)
    set_field(cfg_dict, "anchors", args.anchors, defaults.anchors)
    set_field(cfg_dict, "auto_anchors", args.auto_anchors, defaults.auto_anchors)
    set_field(cfg_dict, "num_anchors", args.num_anchors, defaults.num_anchors)
    set_field(cfg_dict, "c_stem", args.cnn_width, defaults.cnn_width)
    set_field(cfg_dict, "cnn_width", args.cnn_width, defaults.cnn_width)
    set_field(cfg_dict, "activation", args.activation, defaults.activation)
    set_field(cfg_dict, "assigner", args.assigner, defaults.assigner)
    set_field(cfg_dict, "iou_loss", args.iou_loss, defaults.iou_loss)
    set_field(cfg_dict, "cls_loss_type", args.cls_loss_type, defaults.cls_loss_type)
    set_field(cfg_dict, "simota_topk", args.simota_topk, defaults.simota_topk)
    set_field(cfg_dict, "use_batchnorm", args.use_batchnorm, defaults.use_batchnorm)
    set_field(cfg_dict, "grad_clip_norm", args.grad_clip_norm, defaults.grad_clip_norm)
    set_field(cfg_dict, "log_dir", args.log_dir, defaults.log_dir)
    set_field(cfg_dict, "num_workers", args.num_workers, defaults.num_workers)
    set_field(cfg_dict, "utod_head_ese", args.utod_head_ese or args.use_head_ese, defaults.utod_head_ese or defaults.use_head_ese)
    set_field(cfg_dict, "utod_large_obj_branch", args.utod_large_obj_branch, defaults.utod_large_obj_branch)
    set_field(cfg_dict, "utod_large_obj_depth", args.utod_large_obj_depth, defaults.utod_large_obj_depth)
    set_field(cfg_dict, "utod_large_obj_ch_scale", args.utod_large_obj_ch_scale, defaults.utod_large_obj_ch_scale)
    set_field(cfg_dict, "conf_thresh", args.conf_thresh, defaults.conf_thresh)
    cfg = TrainConfig(**cfg_dict)
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
