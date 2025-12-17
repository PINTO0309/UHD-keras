from __future__ import annotations

from typing import Optional

import tensorflow as tf


def _safe_resize_teacher(tensor: tf.Tensor, target_shape) -> tf.Tensor:
    """Resize teacher outputs if spatial dims do not match student."""
    if tensor.shape == target_shape:
        return tensor
    tgt_h, tgt_w = target_shape[1], target_shape[2]
    return tf.image.resize(tensor, size=(tgt_h, tgt_w), method="bilinear")


def distillation_loss(
    student_pred: tf.Tensor,
    teacher_pred: tf.Tensor,
    student_feat: Optional[tf.Tensor] = None,
    teacher_feat: Optional[tf.Tensor] = None,
    logit_weight: float = 1.0,
    feature_weight: float = 0.0,
) -> tf.Tensor:
    """Simple L2 distillation on logits and optional feature maps."""
    loss = tf.constant(0.0, dtype=student_pred.dtype)
    if logit_weight > 0.0:
        t_pred = tf.stop_gradient(_safe_resize_teacher(teacher_pred, tf.shape(student_pred)))
        loss += logit_weight * tf.reduce_mean(tf.square(student_pred - t_pred))
    if feature_weight > 0.0 and student_feat is not None and teacher_feat is not None:
        t_feat = tf.stop_gradient(_safe_resize_teacher(teacher_feat, tf.shape(student_feat)))
        loss += feature_weight * tf.reduce_mean(tf.square(student_feat - t_feat))
    return loss
