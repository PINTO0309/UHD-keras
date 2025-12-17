# UHD-keras

Keras reimplementation of the UHD `ultratinyod` training pipeline. The goal is to keep the original training flow (anchors, SimOTA/legacy assigner, IoU-aware heads, improved head, and quality-aware scoring) while restricting the scope to the `ultratinyod` architecture. Distillation supports only UltraTinyOD teacher→student; transformer teachers are intentionally unsupported. EMA tracking and multiple resize backends are available.

## Features
- UltraTinyOD backbone/head rebuilt in `tf.keras` with the same blocks (DWConv, SPPFmin, lightweight SE/RFB, optional improved head and IoU-aware quality branch).
- Anchor-based loss ported from UHD (`legacy` and `simota` assigners, IoU/GIoU/CIoU, optional Varifocal loss).
- Optional auto-anchor computation from training labels (`--auto-anchors`, `--num-anchors`).
- Distillation between UltraTinyOD models (logit + optional feature MSE), no transformer distill path.
- EMA shadow weights (`--use-ema`) and configurable decay.
- Optional BatchNorm (`--use-batchnorm`) matching UHD flag; off by default.
- Optional residual connections in the UTOD backbone (`--utod-residual`).
- Activation choice (`--activation relu|silu|swish`, default swish).
- Class subset selection via `--classes "0,1,3"` (num_classes inferred from the list), or use `--num-classes`/`--names`.
- Deterministic split helpers: `--train-split` (default 0.8), `--val-split` (default 0.2), `--seed` (default 42). Val split drives a simple loss-only evaluation loop.
- Logging: `--log-interval` (default 10 steps) and `--eval-interval` (default 1 epoch) with TensorBoard summaries written to `--log-dir` (default `runs/tensorboard`).
- UHD-style augmentation pipeline (HorizontalFlip, PhotometricDistort, HSV jitter, CLAHE, RemoveOutliers, plus MixUp/Mosaic/CopyPaste and more) configurable via `--aug-config` (UHD `aug.yaml` format) or defaults matching `uhd/aug.yaml`. Disable with `--no-aug`.
- Resize backends: `opencv_inter_nearest`, `keras_bilinear`, `keras_nearest`.

## Setup
```bash
# Recommended: uv environment (installs from pyproject.toml)
uv sync
```

Notes:
- The stack assumes `tf-keras`/`tensorflow` (see requirements). Pillow is not used; image I/O/resize relies on OpenCV and TensorFlow ops.

## Data expectation
- Point `--image-dir` to a directory (e.g., `data/wholebody34/obj_train_data`); images inside (jpg/png/bmp/webp) are discovered automatically. Label files live next to images with the same stem and `.txt` extension.
- Each label line: `class cx cy w h` (normalized).
- Class count is resolved from `--names obj.names` (one class per line) or `--num-classes`.

## Training (student only)
```bash
uv run python -m uhd_keras.train \
--image-dir data/wholebody34/obj_train_data \
--names data/wholebody34/obj.names \
--epochs 50 \
--batch-size 64 \
--img-size 64 \
--resize-mode opencv_inter_nearest \
--ckpt-out runs/ultratinyod_keras_best
```

Common knobs:
- `--auto-anchors --num-anchors K` to k-means anchors from label widths/heights.
- `--cnn-width` (default 64) to set the UTOD stem width.
- `--use-batchnorm` to enable BN layers (disabled by default to mirror UHD flag).
- `--utod-residual` to enable backbone residuals; `--activation relu|silu|swish` to set activations.
- `--use-improved-head`, `--use-head-ese`, `--use-iou-aware-head`, `--quality-power` to mirror UHD head options.
- `--assigner legacy|simota`, `--iou-loss iou|giou|ciou`, `--cls-loss-type bce|vfl`.
- `--anchors "0.1,0.2;0.2,0.3;0.3,0.4"` to override defaults (normalized).
- `--resize-mode` chooses resize backend: OpenCV nearest (`cv2` required), TF bilinear, or TF nearest.
- `--grad-clip-norm` to apply global-norm clipping.
- `--train-split` / `--val-split` / `--seed` to control deterministic path split (val split is used for loss-only eval every `--eval-interval` epochs).
- `--aug-config path/to/aug.yaml` to load UHD-style augmentation, `--no-aug` to disable, `--log-dir` to set TensorBoard output.

## Distillation (UltraTinyOD → UltraTinyOD only)
```bash
uv run python -m uhd_keras.train \
--image-dir data/wholebody34/obj_train_data \
--names data/wholebody34/obj.names \
--teacher-ckpt path/to/teacher_weights \
--distill-weight 1.0 \
--feature-distill-weight 0.5 \
--ckpt-out runs/ultratinyod_student_distill
```
Teacher and student share the same UltraTinyOD config; transformer teachers are not wired.

## EMA tracking
Add `--use-ema --ema-decay 0.9999` to maintain and save EMA weights (EMA weights are swapped in only for checkpoint writing).

## Notes
- Training loop runs in eager mode to keep the anchor assigner logic simple.
- Checkpoints are written whenever the running best loss improves; `--resume` can restore a student.
- Only `ultratinyod` is implemented; other UHD architectures are intentionally left out.
