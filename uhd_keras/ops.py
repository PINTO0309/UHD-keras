from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import tensorflow as tf


def cxcywh_to_xyxy(boxes: tf.Tensor) -> tf.Tensor:
    """Convert center-format boxes to corner format."""
    cx, cy, w, h = tf.unstack(boxes, axis=-1)
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return tf.stack([x1, y1, x2, y2], axis=-1)


def pairwise_iou(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    """Compute pairwise IoU for two sets of boxes (cx,cy,w,h)."""
    if tf.size(a) == 0 or tf.size(b) == 0:
        return tf.zeros((tf.shape(a)[0], tf.shape(b)[0]), dtype=a.dtype)
    a_xyxy = cxcywh_to_xyxy(a)
    b_xyxy = cxcywh_to_xyxy(b)
    a_xyxy = tf.expand_dims(a_xyxy, axis=1)  # N,1,4
    b_xyxy = tf.expand_dims(b_xyxy, axis=0)  # 1,M,4
    tl = tf.maximum(a_xyxy[..., :2], b_xyxy[..., :2])
    br = tf.minimum(a_xyxy[..., 2:], b_xyxy[..., 2:])
    wh = tf.maximum(br - tl, 0.0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a_xyxy[..., 2] - a_xyxy[..., 0]) * (a_xyxy[..., 3] - a_xyxy[..., 1])
    area_b = (b_xyxy[..., 2] - b_xyxy[..., 0]) * (b_xyxy[..., 3] - b_xyxy[..., 1])
    union = area_a + area_b - inter + 1e-6
    return inter / union


def bbox_iou_aligned(boxes1: tf.Tensor, boxes2: tf.Tensor, iou_type: str = "iou", eps: float = 1e-7) -> tf.Tensor:
    """Compute IoU/GIoU/CIoU for aligned box pairs."""
    x1, y1, x2, y2 = tf.unstack(cxcywh_to_xyxy(boxes1), axis=-1)
    x1g, y1g, x2g, y2g = tf.unstack(cxcywh_to_xyxy(boxes2), axis=-1)

    inter_x1 = tf.maximum(x1, x1g)
    inter_y1 = tf.maximum(y1, y1g)
    inter_x2 = tf.minimum(x2, x2g)
    inter_y2 = tf.minimum(y2, y2g)

    inter_w = tf.maximum(inter_x2 - inter_x1, 0.0)
    inter_h = tf.maximum(inter_y2 - inter_y1, 0.0)
    inter_area = inter_w * inter_h

    area1 = tf.maximum(x2 - x1, 0.0) * tf.maximum(y2 - y1, 0.0)
    area2 = tf.maximum(x2g - x1g, 0.0) * tf.maximum(y2g - y1g, 0.0)
    union = area1 + area2 - inter_area + eps
    iou = inter_area / union

    if iou_type == "iou":
        return iou

    cw = tf.maximum(tf.maximum(x2, x2g) - tf.minimum(x1, x1g), 0.0)
    ch = tf.maximum(tf.maximum(y2, y2g) - tf.minimum(y1, y1g), 0.0)
    c_area = cw * ch + eps
    giou = iou - (c_area - union) / c_area
    if iou_type == "giou":
        return giou

    rho2 = tf.square(boxes1[..., 0] - boxes2[..., 0]) + tf.square(boxes1[..., 1] - boxes2[..., 1])
    c2 = tf.maximum(tf.square(cw) + tf.square(ch), eps)
    pi_const = tf.constant(3.141592653589793, dtype=boxes1.dtype)
    v = (4.0 / (pi_const**2)) * tf.square(tf.atan(boxes1[..., 2] / (boxes1[..., 3] + eps)) - tf.atan(boxes2[..., 2] / (boxes2[..., 3] + eps)))
    alpha = v / tf.maximum(1.0 - iou + v, eps)
    ciou = iou - rho2 / c2 - alpha * v
    return ciou


def _activate_wh(tw: tf.Tensor, th: tf.Tensor, max_scale: Optional[float] = None) -> Tuple[tf.Tensor, tf.Tensor]:
    w = tf.nn.softplus(tw)
    h = tf.nn.softplus(th)
    if max_scale is not None:
        w = tf.minimum(w, max_scale)
        h = tf.minimum(h, max_scale)
    return w, h


def decode_anchor(
    pred: tf.Tensor,
    anchors: tf.Tensor,
    num_classes: int,
    conf_thresh: float = 0.3,
    nms_thresh: float = 0.5,
    has_quality: bool = False,
    wh_scale: Optional[tf.Tensor] = None,
    score_mode: str = "obj_quality_cls",
    quality_power: float = 1.0,
) -> List[List[Tuple[float, int, tf.Tensor]]]:
    """
    YOLO-style decode: pred shape [B, H, W, A*(5+Q+C)], anchors [A,2].
    Returns list per batch of (score, cls, box[cx,cy,w,h]) tuples.
    """
    bsz = tf.shape(pred)[0]
    h = tf.shape(pred)[1]
    w = tf.shape(pred)[2]
    anchors = tf.cast(tf.convert_to_tensor(anchors), pred.dtype)
    if wh_scale is not None:
        anchors = anchors * tf.cast(wh_scale, pred.dtype)
    na = tf.shape(anchors)[0]
    extra = 1 if has_quality else 0
    pred = tf.reshape(pred, (bsz, h, w, na, 5 + extra + num_classes))
    tx = pred[..., 0]
    ty = pred[..., 1]
    tw = pred[..., 2]
    th = pred[..., 3]
    obj = tf.nn.sigmoid(pred[..., 4])
    quality = tf.nn.sigmoid(pred[..., 5]) if has_quality else None
    cls = tf.nn.sigmoid(pred[..., (5 + extra):])

    gx, gy = tf.meshgrid(tf.range(w, dtype=pred.dtype), tf.range(h, dtype=pred.dtype))
    gx = tf.reshape(gx, (1, h, w, 1))
    gy = tf.reshape(gy, (1, h, w, 1))

    pred_cx = (tf.nn.sigmoid(tx) + gx) / tf.cast(w, pred.dtype)
    pred_cy = (tf.nn.sigmoid(ty) + gy) / tf.cast(h, pred.dtype)
    pw = anchors[:, 0][tf.newaxis, tf.newaxis, tf.newaxis, :]
    ph = anchors[:, 1][tf.newaxis, tf.newaxis, tf.newaxis, :]
    act_w, act_h = _activate_wh(tw, th)
    pred_w = pw * act_w
    pred_h = ph * act_h

    score_mode = (score_mode or "obj_quality_cls").lower()
    qp = float(quality_power)
    if quality is not None and qp != 1.0:
        quality = tf.pow(tf.clip_by_value(quality, 0.0, 1.0), qp)

    if score_mode == "quality_cls" and quality is not None:
        score_base = quality
    elif score_mode == "obj_cls":
        score_base = obj
    else:
        score_base = obj
        if quality is not None:
            score_base = score_base * quality
    scores = score_base[..., tf.newaxis] * cls  # B,H,W,A,C

    outputs: List[List[Tuple[float, int, tf.Tensor]]] = []
    for bi in range(bsz):
        scores_b = scores[bi]  # H,W,A,C
        pred_cx_b = pred_cx[bi]
        pred_cy_b = pred_cy[bi]
        pred_w_b = pred_w[bi]
        pred_h_b = pred_h[bi]

        scores_flat = tf.reshape(scores_b, (-1, num_classes))
        max_scores = tf.reduce_max(scores_flat, axis=1)
        max_cls = tf.argmax(scores_flat, axis=1, output_type=tf.int32)
        mask = max_scores >= conf_thresh
        if not tf.reduce_any(mask):
            outputs.append([])
            continue
        sel_scores = tf.boolean_mask(max_scores, mask)
        sel_cls = tf.boolean_mask(max_cls, mask)
        idxs = tf.where(mask)[:, 0]
        a_idx = idxs // (h * w)
        rem = idxs % (h * w)
        gy_idx = rem // w
        gx_idx = rem % w
        cx_sel = tf.gather_nd(pred_cx_b, tf.stack([gy_idx, gx_idx, a_idx], axis=1))
        cy_sel = tf.gather_nd(pred_cy_b, tf.stack([gy_idx, gx_idx, a_idx], axis=1))
        bw_sel = tf.gather_nd(pred_w_b, tf.stack([gy_idx, gx_idx, a_idx], axis=1))
        bh_sel = tf.gather_nd(pred_h_b, tf.stack([gy_idx, gx_idx, a_idx], axis=1))

        boxes_raw = []
        for sc, cls_id, cxv, cyv, bwv, bhv in zip(sel_scores.numpy(), sel_cls.numpy(), cx_sel.numpy(), cy_sel.numpy(), bw_sel.numpy(), bh_sel.numpy()):
            boxes_raw.append((float(sc), int(cls_id), tf.convert_to_tensor([cxv, cyv, bwv, bhv], dtype=pred.dtype)))
        outputs.append(nms_per_class(boxes_raw, iou_thresh=nms_thresh))
    return outputs


def nms_per_class(boxes: Sequence[Tuple[float, int, tf.Tensor]], iou_thresh: float = 0.5) -> List[Tuple[float, int, tf.Tensor]]:
    """Simple per-class NMS operating on CPU tensors."""
    if not boxes:
        return []
    grouped = {}
    for sc, cls_id, box in boxes:
        grouped.setdefault(cls_id, []).append((sc, box))
    kept: List[Tuple[float, int, tf.Tensor]] = []
    for cls_id, items in grouped.items():
        items = sorted(items, key=lambda x: x[0], reverse=True)
        while items:
            sc, box = items.pop(0)
            keep = True
            survivors = []
            for sc2, box2 in items:
                iou = pairwise_iou(tf.expand_dims(box, 0), tf.expand_dims(box2, 0))[0, 0].numpy()
                if iou >= iou_thresh:
                    continue
                survivors.append((sc2, box2))
            kept.append((sc, cls_id, box))
            items = survivors
    kept.sort(key=lambda x: x[0], reverse=True)
    return kept
