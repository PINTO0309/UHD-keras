from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Optional

import tensorflow as tf

from .ops import bbox_iou_aligned, pairwise_iou, _activate_wh


def anchor_loss(
    pred: tf.Tensor,
    targets: Sequence[Dict[str, tf.Tensor]],
    anchors: tf.Tensor,
    num_classes: int,
    iou_loss: str = "giou",
    assigner: str = "legacy",
    cls_loss_type: str = "bce",
    simota_topk: int = 10,
    use_quality: bool = False,
    wh_scale: Optional[tf.Tensor] = None,
) -> Dict[str, tf.Tensor]:
    """
    YOLO-style anchor loss with optional IoU/GIoU/CIoU regression.
    pred: [B, H, W, A*(5+extra+C)]
    anchors: [A, 2] normalized w,h
    """
    pred = tf.convert_to_tensor(pred)
    anchors = tf.convert_to_tensor(anchors, dtype=tf.float32)
    if wh_scale is not None:
        anchors = tf.cast(anchors, pred.dtype) * tf.cast(wh_scale, pred.dtype)
    else:
        anchors = tf.cast(anchors, pred.dtype)

    b = tf.shape(pred)[0]
    h = tf.shape(pred)[1]
    w = tf.shape(pred)[2]
    h_int = int(pred.shape[1]) if pred.shape[1] is not None else None
    w_int = int(pred.shape[2]) if pred.shape[2] is not None else None
    na = tf.shape(anchors)[0]
    extra = 1 if use_quality else 0
    pred = tf.reshape(pred, (b, h, w, na, 5 + extra + num_classes))
    pred = tf.transpose(pred, (0, 3, 1, 2, 4))  # B,A,H,W,*

    tx = pred[..., 0]
    ty = pred[..., 1]
    tw = pred[..., 2]
    th = pred[..., 3]
    obj_logit = pred[..., 4]
    qual_logit = pred[..., 5] if use_quality else None
    cls_logit = pred[..., (5 + extra):]

    target_obj = tf.zeros_like(obj_logit)
    target_cls = tf.zeros((b, na, h, w, num_classes), dtype=pred.dtype)
    target_box = tf.zeros((b, na, h, w, 4), dtype=pred.dtype)
    target_quality = tf.zeros_like(obj_logit) if use_quality else None

    mesh_dtype = tf.float32 if pred.dtype == tf.float16 else pred.dtype
    gx, gy = tf.meshgrid(tf.range(w, dtype=mesh_dtype), tf.range(h, dtype=mesh_dtype))
    gx = tf.reshape(gx, (1, 1, h, w))
    gy = tf.reshape(gy, (1, 1, h, w))
    gx = tf.cast(gx, pred.dtype)
    gy = tf.cast(gy, pred.dtype)
    pred_cx = (tf.nn.sigmoid(tx) + gx) / tf.cast(w, pred.dtype)
    pred_cy = (tf.nn.sigmoid(ty) + gy) / tf.cast(h, pred.dtype)
    act_w, act_h = _activate_wh(tw, th)
    pred_w = anchors[:, 0][tf.newaxis, :, tf.newaxis, tf.newaxis] * act_w
    pred_h = anchors[:, 1][tf.newaxis, :, tf.newaxis, tf.newaxis] * act_h
    pred_box = tf.stack([pred_cx, pred_cy, pred_w, pred_h], axis=-1)

    if assigner not in ("legacy", "simota"):
        raise ValueError(f"Unknown assigner: {assigner}")

    obj_indices = []
    obj_values = []
    cls_indices = []
    cls_values = []
    box_indices = []
    box_values = []
    qual_indices = []

    for bi, tgt in enumerate(targets):
        boxes = tf.cast(tf.convert_to_tensor(tgt.get("boxes", [])), pred.dtype)
        labels = tf.cast(tf.convert_to_tensor(tgt.get("labels", [])), tf.int32)
        if tf.size(boxes) == 0:
            continue
        gxs = tf.clip_by_value(boxes[:, 0] * tf.cast(w, pred.dtype), 0.0, tf.cast(w, pred.dtype) - 1e-3)
        gys = tf.clip_by_value(boxes[:, 1] * tf.cast(h, pred.dtype), 0.0, tf.cast(h, pred.dtype) - 1e-3)
        gis = tf.cast(gxs, tf.int32)
        gjs = tf.cast(gys, tf.int32)

        if assigner == "legacy":
            awh = tf.expand_dims(anchors, axis=1)  # A,1,2
            wh = boxes[:, 2:4]
            inter = tf.reduce_prod(tf.minimum(awh, wh[tf.newaxis, :, :]), axis=2)
            union = (awh[:, :, 0] * awh[:, :, 1]) + (wh[tf.newaxis, :, 0] * wh[tf.newaxis, :, 1]) - inter + 1e-7
            anchor_iou = inter / union  # A,G
            best_anchor = tf.cast(tf.argmax(anchor_iou, axis=0), tf.int32)
            boxes_len = int(tf.shape(boxes)[0])
            for idx in range(boxes_len):
                gi = int(gis[idx])
                gj = int(gjs[idx])
                if w_int is not None and (gi < 0 or gi >= w_int):
                    continue
                if h_int is not None and (gj < 0 or gj >= h_int):
                    continue
                a = int(best_anchor[idx])
                cls_id = int(labels[idx])
                obj_indices.append([bi, a, gj, gi])
                obj_values.append(1.0)
                cls_indices.append([bi, a, gj, gi])
                cls_vec = tf.one_hot(cls_id, num_classes, dtype=pred.dtype)
                cls_values.append(cls_vec)
                box_indices.append([bi, a, gj, gi])
                box_values.append(boxes[idx])
                if use_quality:
                    qual_indices.append([bi, a, gj, gi])
        else:  # simota
            pb = tf.reshape(pred_box[bi], (-1, 4))
            pb = tf.clip_by_value(tf.where(tf.math.is_finite(pb), pb, 0.0), 0.0, 10.0)
            boxes_clean = tf.clip_by_value(tf.where(tf.math.is_finite(boxes), boxes, 0.0), 0.0, 1.0)
            ious = pairwise_iou(pb, boxes_clean)  # N,G
            grid_size = h_int * w_int
            for gt_idx in range(tf.shape(boxes_clean)[0]):
                cls_id = int(labels[gt_idx])
                iou_g = ious[:, gt_idx]
                topk = tf.minimum(simota_topk, tf.shape(iou_g)[0])
                topk_vals, topk_idx = tf.math.top_k(iou_g, k=topk, sorted=True)
                dynamic_k = int(tf.minimum(topk, tf.maximum(tf.reduce_sum(topk_vals), 1.0)).numpy())
                selected = topk_idx[:dynamic_k]
                for idx in selected.numpy().tolist():
                    a = idx // grid_size
                    rem = idx % grid_size
                    gj = rem // w_int
                    gi = rem % w_int
                    obj_indices.append([bi, a, gj, gi])
                    obj_values.append(1.0)
                    cls_indices.append([bi, a, gj, gi])
                    cls_vec = tf.one_hot(cls_id, num_classes, dtype=pred.dtype)
                    cls_values.append(cls_vec)
                    box_indices.append([bi, a, gj, gi])
                    box_values.append(boxes_clean[gt_idx])
                    if use_quality:
                        qual_indices.append([bi, a, gj, gi])

    if obj_indices:
        obj_indices_tf = tf.constant(obj_indices, dtype=tf.int32)
        obj_values_tf = tf.constant(obj_values, dtype=pred.dtype)
        target_obj = tf.tensor_scatter_nd_update(target_obj, obj_indices_tf, obj_values_tf)

        cls_indices_tf = tf.constant(cls_indices, dtype=tf.int32)
        cls_values_tf = tf.stack(cls_values, axis=0)
        target_cls = tf.tensor_scatter_nd_update(target_cls, cls_indices_tf, cls_values_tf)

        box_indices_tf = tf.constant(box_indices, dtype=tf.int32)
        box_values_tf = tf.stack(box_values, axis=0)
        target_box = tf.tensor_scatter_nd_update(target_box, box_indices_tf, box_values_tf)

        if use_quality and qual_indices:
            qual_indices_tf = tf.constant(qual_indices, dtype=tf.int32)
            qual_values_tf = tf.ones((len(qual_indices),), dtype=pred.dtype)
            target_quality = tf.tensor_scatter_nd_update(target_quality, qual_indices_tf, qual_values_tf)

    bce = tf.nn.sigmoid_cross_entropy_with_logits
    obj_loss = tf.reduce_mean(bce(labels=target_obj, logits=obj_logit))
    quality_loss = tf.constant(0.0, dtype=pred.dtype)

    pos_mask = target_obj > 0.5
    num_pos = tf.reduce_sum(tf.cast(pos_mask, pred.dtype))

    def _non_empty():
        t_box = tf.boolean_mask(target_box, pos_mask)
        p_box = tf.boolean_mask(pred_box, pos_mask)
        iou_val = bbox_iou_aligned(p_box, t_box, iou_type=iou_loss)
        box_loss = tf.reduce_mean(1.0 - iou_val)
        obj_l = obj_loss
        q_loss = quality_loss
        if use_quality and qual_logit is not None and target_quality is not None:
            tq = tf.tensor_scatter_nd_update(tf.zeros_like(obj_logit), tf.where(pos_mask), tf.clip_by_value(iou_val, 0.0, 1.0))
            q_loss = tf.reduce_mean(bce(labels=tq, logits=qual_logit))
            obj_l = tf.reduce_mean(bce(labels=tq, logits=obj_logit))
        cls_targets = tf.boolean_mask(target_cls, pos_mask)
        cls_logits_pos = tf.boolean_mask(cls_logit, pos_mask)
        if cls_loss_type == "vfl":
            pred_sigmoid = tf.nn.sigmoid(cls_logits_pos)
            weight = cls_targets * cls_targets + 0.75 * (1.0 - cls_targets) * tf.pow(pred_sigmoid, 2.0)
            cls_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=cls_targets, logits=cls_logits_pos) * weight)
        else:
            cls_loss = tf.reduce_sum(bce(labels=cls_targets, logits=cls_logits_pos)) / tf.maximum(num_pos, 1.0)
        return box_loss, cls_loss, q_loss, obj_l

    def _empty():
        zero = tf.constant(0.0, dtype=pred.dtype)
        return zero, zero, zero, obj_loss

    box_loss, cls_loss, quality_loss, obj_loss = tf.cond(num_pos > 0.0, _non_empty, _empty)

    total = box_loss + obj_loss + cls_loss + quality_loss
    return {"loss": total, "box": box_loss, "obj": obj_loss, "cls": cls_loss, "quality": quality_loss}
