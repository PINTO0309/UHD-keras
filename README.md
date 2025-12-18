# WIP
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
SIZE=64x64
ANCHOR=8
CNNWIDTH=64
LR=0.0003
RESIZEMODE=opencv_inter_nearest
uv run python -m uhd_keras.train \
--image-dir data/wholebody34/obj_train_data \
--img-size ${SIZE} \
--exp-name ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_loese_${SIZE}_lr${LR}_${RESIZEMODE} \
--batch-size 64 \
--epochs 300 \
--lr ${LR} \
--weight-decay 0.0001 \
--classes 0 \
--cnn-width ${CNNWIDTH} \
--auto-anchors \
--num-anchors ${ANCHOR} \
--iou-loss ciou \
--conf-thresh 0.15 \
--use-ema \
--ema-decay 0.9999 \
--grad-clip-norm 10.0 \
--use-batchnorm \
--utod-residual \
--use-improved-head \
--use-iou-aware-head \
--utod-head-ese \
--activation relu \
--utod-large-obj-branch \
--utod-large-obj-depth 2 \
--utod-large-obj-ch-scale 1.25 \
--${RESIZEMODE}
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

## Inspect Anchors/wh_scale

```bash
uv run python tools/inspect_checkpoint.py \
runs/ultratinyod_res_anc8_w64_loese_64x64_lr0.0003_opencv_inter_nearest/last_utod_0001_map_0.00000.keras
```
```
=== Anchors ===
[[0.02343096 0.04865872]
 [0.04480992 0.10889184]
 [0.08770896 0.16278133]
 [0.09427162 0.3447917 ]
 [0.21822597 0.32518694]
 [0.17143537 0.61753505]
 [0.34108981 0.6963622 ]
 [0.6419175  0.81884986]]

=== wh_scale ===
[[1.0001173  1.0001669 ]
 [1.0002174  1.0002654 ]
 [1.0002551  1.000126  ]
 [1.0000645  1.0001314 ]
 [0.99998957 1.0000978 ]
 [0.9999818  1.0000243 ]
 [0.9999917  0.99985534]
 [0.9999727  0.9999857 ]]
```

## ONNX Export

```bash
SIZE=64x64
ANCHOR=8
CNNWIDTH=64
LR=0.0003
RESIZEMODE=opencv_inter_nearest
uv run python -m tf2onnx.convert \
--saved-model runs/ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_loese_${SIZE}_lr${LR}_${RESIZEMODE}/best_utod_0001_map_0.00000 \
--output model.onnx \
--inputs-as-nchw input_1 \
--outputs-as-nchw output_0 \
--opset 17
uv run onnxsim model.onnx model.onnx \
--overwrite-input-shape "input_1:1,3,64,64"
```
