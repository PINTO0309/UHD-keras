from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import tensorflow as tf
import tf_keras as keras

from .ops import decode_anchor


def _make_activation(name: str):
    act = (name or "silu").lower()
    if act in ("silu", "swish"):
        return tf.nn.silu
    if act == "relu":
        return tf.nn.relu
    raise ValueError(f"Unsupported activation: {name}")


class ConvBNAct(keras.layers.Layer):
    """Conv -> BatchNorm -> Activation block."""

    def __init__(
        self,
        c_out: int,
        kernel_size: int = 3,
        strides: int = 1,
        groups: int = 1,
        use_bias: bool = False,
        act: bool = True,
        act_name: str = "silu",
        use_bn: bool = True,
    ):
        super().__init__()
        self.conv = keras.layers.Conv2D(
            filters=c_out,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            use_bias=use_bias,
            groups=groups,
            kernel_initializer="he_normal",
        )
        self.bn = keras.layers.BatchNormalization(momentum=0.03, epsilon=1e-3) if use_bn else None
        self.act_fn = _make_activation(act_name) if act else None

    def call(self, x, training: bool = False):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x, training=training)
        return self.act_fn(x) if self.act_fn is not None else x


class DWConv(keras.layers.Layer):
    """Depthwise separable conv block."""

    def __init__(
        self,
        c_out: int,
        kernel_size: int = 3,
        strides: int = 1,
        act: bool = True,
        act_name: str = "silu",
        use_bn: bool = True,
    ):
        super().__init__()
        self.dw = keras.layers.DepthwiseConv2D(
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            use_bias=False,
            depthwise_initializer="he_normal",
        )
        self.dw_bn = keras.layers.BatchNormalization(momentum=0.03, epsilon=1e-3) if use_bn else None
        self.dw_act = _make_activation(act_name) if act else None
        self.pw = ConvBNAct(c_out, kernel_size=1, strides=1, act=act, act_name=act_name, use_bn=use_bn)

    def call(self, x, training: bool = False):
        x = self.dw(x)
        if self.dw_bn is not None:
            x = self.dw_bn(x, training=training)
        x = self.dw_act(x) if self.dw_act is not None else x
        x = self.pw(x, training=training)
        return x


class EfficientSE(keras.layers.Layer):
    """Lightweight squeeze-and-excite using 1x1 conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.fc = keras.layers.Conv2D(channels, kernel_size=1, padding="same", use_bias=True)

    def call(self, x):
        w = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        w = tf.nn.sigmoid(self.fc(w))
        return x * w


class ReceptiveFieldEnhancer(keras.layers.Layer):
    """Dilated + wide depthwise fusion for cheap context."""

    def __init__(self, channels: int, dilation: int = 2, act_name: str = "silu", use_bn: bool = True):
        super().__init__()
        self.dw_dil = keras.layers.DepthwiseConv2D(
            kernel_size=3,
            strides=1,
            padding="same",
            dilation_rate=dilation,
            use_bias=False,
            depthwise_initializer="he_normal",
        )
        self.dw_dil_bn = keras.layers.BatchNormalization(momentum=0.03, epsilon=1e-3) if use_bn else None
        self.dw_dil_act = _make_activation(act_name)

        self.dw_wide = keras.layers.DepthwiseConv2D(
            kernel_size=5,
            strides=1,
            padding="same",
            use_bias=False,
            depthwise_initializer="he_normal",
        )
        self.dw_wide_bn = keras.layers.BatchNormalization(momentum=0.03, epsilon=1e-3) if use_bn else None
        self.dw_wide_act = _make_activation(act_name)

        self.fuse = ConvBNAct(channels, kernel_size=1, strides=1, act_name=act_name, use_bn=use_bn)

    def call(self, x, training: bool = False):
        b1 = self.dw_dil(x)
        if self.dw_dil_bn is not None:
            b1 = self.dw_dil_bn(b1, training=training)
        b1 = self.dw_dil_act(b1)

        b2 = self.dw_wide(x)
        if self.dw_wide_bn is not None:
            b2 = self.dw_wide_bn(b2, training=training)
        b2 = self.dw_wide_act(b2)
        return x + self.fuse(tf.concat([b1, b2], axis=-1), training=training)


class SPPFmin(keras.layers.Layer):
    """Lightweight SPPF variant."""

    def __init__(self, c_in: int, c_out: int, pool_k: int = 5, act_name: str = "silu", use_bn: bool = True):
        super().__init__()
        c_hidden = max(1, c_in // 2)
        self.cv1 = ConvBNAct(c_hidden, kernel_size=1, strides=1, act_name=act_name, use_bn=use_bn)
        self.pool = keras.layers.MaxPool2D(pool_size=pool_k, strides=1, padding="same")
        self.cv2 = ConvBNAct(c_out, kernel_size=1, strides=1, act_name=act_name, use_bn=use_bn)

    def call(self, x, training: bool = False):
        x = self.cv1(x, training=training)
        y = self.pool(x)
        x = tf.concat([x, y], axis=-1)
        x = self.cv2(x, training=training)
        return x


class UltraTinyODBackbone(keras.layers.Layer):
    """UltraTinyOD backbone for 64x64-ish inputs."""

    def __init__(
        self,
        c_stem: int = 16,
        use_residual: bool = False,
        out_stride: int = 8,
        activation: str = "silu",
        use_batchnorm: bool = False,
    ):
        super().__init__()
        if out_stride not in (4, 8, 16):
            raise ValueError("out_stride must be 4/8/16")
        act_name = activation
        self.use_residual = bool(use_residual)
        self.out_stride = int(out_stride)

        self.stem = ConvBNAct(c_stem, kernel_size=3, strides=2, act_name=act_name, use_bn=use_batchnorm)
        stride_block2 = 2 if self.out_stride >= 8 else 1
        stride_block3 = 2 if self.out_stride == 16 else 1
        self.block1 = DWConv(c_stem * 2, kernel_size=3, strides=2, act_name=act_name, use_bn=use_batchnorm)
        self.block2 = DWConv(c_stem * 4, kernel_size=3, strides=stride_block2, act_name=act_name, use_bn=use_batchnorm)
        self.block3 = DWConv(c_stem * 8, kernel_size=3, strides=stride_block3, act_name=act_name, use_bn=use_batchnorm)
        self.block4 = DWConv(c_stem * 8, kernel_size=3, strides=1, act_name=act_name, use_bn=use_batchnorm)
        if self.use_residual:
            self.block3_skip = ConvBNAct(
                c_stem * 8,
                kernel_size=1,
                strides=stride_block3,
                act=False,
                act_name=act_name,
                use_bn=use_batchnorm,
            )
            self.block4_skip = keras.layers.Layer()  # identity
        self.sppf = SPPFmin(c_stem * 8, c_stem * 4, act_name=act_name, use_bn=use_batchnorm)
        self.out_channels = c_stem * 4

    def call(self, x, training: bool = False):
        x = self.stem(x, training=training)
        x = self.block1(x, training=training)
        x2 = self.block2(x, training=training)
        x3 = self.block3(x2, training=training)
        if self.use_residual:
            x3 = x3 + self.block3_skip(x2, training=training)
        x4 = self.block4(x3, training=training)
        if self.use_residual:
            x4 = x4 + x3
        x = self.sppf(x4, training=training)
        return x


@dataclass
class UltraTinyODConfig:
    num_classes: int = 1
    stride: int = 8
    anchors: Optional[Sequence[Tuple[float, float]]] = None
    cls_bottleneck_ratio: float = 0.5
    use_improved_head: bool = False
    use_head_ese: bool = False
    use_iou_aware_head: bool = False
    quality_power: float = 1.0
    activation: str = "silu"
    use_context_rfb: bool = False
    context_dilation: int = 2
    use_large_obj_branch: bool = False
    large_obj_branch_depth: int = 1
    large_obj_branch_expansion: float = 1.0
    use_batchnorm: bool = False

    def __post_init__(self):
        if self.anchors is None:
            self.anchors = [
                (8 / 64.0, 16 / 64.0),
                (12 / 64.0, 28 / 64.0),
                (20 / 64.0, 40 / 64.0),
            ]
        self.anchors = [(float(w), float(h)) for w, h in self.anchors]
        self.cls_bottleneck_ratio = float(max(0.05, min(1.0, self.cls_bottleneck_ratio)))
        self.activation = "silu" if str(self.activation).lower() == "swish" else str(self.activation).lower()
        self.use_improved_head = bool(self.use_improved_head)
        self.use_head_ese = bool(self.use_head_ese)
        self.use_iou_aware_head = bool(self.use_iou_aware_head)
        self.use_context_rfb = bool(self.use_context_rfb)
        self.context_dilation = max(1, int(self.context_dilation))
        self.use_large_obj_branch = bool(self.use_large_obj_branch)
        self.large_obj_branch_depth = max(1, int(self.large_obj_branch_depth))
        self.large_obj_branch_expansion = float(max(0.25, self.large_obj_branch_expansion))
        self.use_batchnorm = bool(self.use_batchnorm)

    def to_dict(self) -> dict:
        return {
            "num_classes": self.num_classes,
            "stride": self.stride,
            "anchors": [list(a) for a in self.anchors],
            "cls_bottleneck_ratio": self.cls_bottleneck_ratio,
            "use_improved_head": self.use_improved_head,
            "use_head_ese": self.use_head_ese,
            "use_iou_aware_head": self.use_iou_aware_head,
            "quality_power": self.quality_power,
            "activation": self.activation,
            "use_context_rfb": self.use_context_rfb,
            "context_dilation": self.context_dilation,
            "use_large_obj_branch": self.use_large_obj_branch,
            "large_obj_branch_depth": self.large_obj_branch_depth,
            "large_obj_branch_expansion": self.large_obj_branch_expansion,
            "use_batchnorm": self.use_batchnorm,
        }

    @classmethod
    def from_dict(cls, cfg: dict) -> "UltraTinyODConfig":
        anchors = cfg.get("anchors")
        if anchors is not None:
            anchors = [tuple(a) for a in anchors]
        return cls(
            num_classes=cfg.get("num_classes", 1),
            stride=cfg.get("stride", 8),
            anchors=anchors,
            cls_bottleneck_ratio=cfg.get("cls_bottleneck_ratio", 0.5),
            use_improved_head=cfg.get("use_improved_head", False),
            use_head_ese=cfg.get("use_head_ese", False),
            use_iou_aware_head=cfg.get("use_iou_aware_head", False),
            quality_power=cfg.get("quality_power", 1.0),
            activation=cfg.get("activation", "silu"),
            use_context_rfb=cfg.get("use_context_rfb", False),
            context_dilation=cfg.get("context_dilation", 2),
            use_large_obj_branch=cfg.get("use_large_obj_branch", False),
            large_obj_branch_depth=cfg.get("large_obj_branch_depth", 1),
            large_obj_branch_expansion=cfg.get("large_obj_branch_expansion", 1.0),
            use_batchnorm=cfg.get("use_batchnorm", False),
        )


class UltraTinyODHead(keras.layers.Layer):
    """Single-scale detection head for UltraTinyOD."""

    def __init__(self, in_channels: int, cfg: UltraTinyODConfig):
        super().__init__()
        self.nc = int(cfg.num_classes)
        self.in_channels = in_channels
        self.cls_ratio = float(cfg.cls_bottleneck_ratio)
        self.cls_mid = max(8, min(in_channels, int(round(in_channels * self.cls_ratio))))
        self.use_improved_head = bool(cfg.use_improved_head)
        self.use_head_ese = bool(cfg.use_head_ese)
        self.use_iou_aware_head = bool(cfg.use_iou_aware_head)
        self.use_context_rfb = bool(cfg.use_context_rfb)
        self.context_dilation = int(cfg.context_dilation)
        self.use_large_obj_branch = bool(cfg.use_large_obj_branch)
        self.large_obj_branch_depth = int(cfg.large_obj_branch_depth)
        self.large_obj_branch_expansion = float(cfg.large_obj_branch_expansion)
        self.has_quality = self.use_improved_head or self.use_iou_aware_head
        self.quality_power = float(cfg.quality_power)
        anchor_tensor = tf.convert_to_tensor(cfg.anchors, dtype=tf.float32)
        self.anchors = anchor_tensor
        self.num_anchors = int(anchor_tensor.shape[0])
        self.no = self.nc + 5 + (1 if self.has_quality else 0)
        self.wh_scale = (
            self.add_weight(
                "wh_scale",
                shape=(self.num_anchors, 2),
                initializer="ones",
                trainable=True,
                dtype=tf.float32,
            )
            if self.use_improved_head
            else tf.constant(1.0, shape=(self.num_anchors, 2), dtype=tf.float32)
        )
        self.score_mode = "quality_cls" if self.use_iou_aware_head else "obj_quality_cls"
        act_name = cfg.activation
        use_bn = cfg.use_batchnorm

        self.context = DWConv(in_channels, kernel_size=3, strides=1, act=True, act_name=act_name, use_bn=use_bn)
        self.context_res = (
            keras.Sequential(
                [
                    DWConv(in_channels, kernel_size=3, strides=1, act=True, act_name=act_name, use_bn=use_bn),
                    ConvBNAct(in_channels, kernel_size=1, strides=1, act_name=act_name, use_bn=use_bn),
                    DWConv(in_channels, kernel_size=3, strides=1, act=True, act_name=act_name, use_bn=use_bn),
                ]
            )
            if self.use_improved_head
            else None
        )
        self.head_se = EfficientSE(in_channels) if self.use_head_ese else None
        self.head_rfb = (
            ReceptiveFieldEnhancer(in_channels, dilation=self.context_dilation, act_name=act_name, use_bn=use_bn)
            if self.use_context_rfb
            else None
        )
        if self.use_large_obj_branch:
            lob_ch = int(round(in_channels * self.large_obj_branch_expansion))
            self.lod_dw = keras.layers.DepthwiseConv2D(
                kernel_size=3,
                strides=2,
                padding="same",
                use_bias=False,
                depthwise_initializer="he_normal",
            )
            self.lod_bn = keras.layers.BatchNormalization(momentum=0.03, epsilon=1e-3) if use_bn else None
            self.lod_act = _make_activation(act_name)
            self.lod_pw = ConvBNAct(lob_ch, kernel_size=1, strides=1, act_name=act_name, use_bn=use_bn)
            self.large_obj_blocks = keras.Sequential(
                [
                    DWConv(lob_ch, kernel_size=3, strides=1, act=True, act_name=act_name, use_bn=use_bn)
                    for _ in range(self.large_obj_branch_depth)
                ]
            )
            self.large_obj_fuse = ConvBNAct(in_channels, kernel_size=1, strides=1, act_name=act_name, use_bn=use_bn)
        else:
            self.lod_dw = None

        if self.use_iou_aware_head:
            self.box_tower = keras.Sequential(
                [
                    DWConv(in_channels, kernel_size=3, strides=1, act=True, act_name=act_name, use_bn=use_bn),
                    DWConv(in_channels, kernel_size=3, strides=1, act=True, act_name=act_name, use_bn=use_bn),
                ]
            )
        else:
            self.box_conv = DWConv(in_channels, kernel_size=3, strides=1, act=True, act_name=act_name, use_bn=use_bn)
        self.box_out = keras.layers.Conv2D(
            filters=self.num_anchors * 4,
            kernel_size=1,
            padding="same",
            use_bias=True,
            kernel_initializer="he_normal",
        )

        if self.has_quality:
            if self.use_iou_aware_head:
                self.quality_tower = keras.Sequential(
                    [
                        DWConv(in_channels, kernel_size=3, strides=1, act=True, act_name=act_name, use_bn=use_bn),
                        DWConv(in_channels, kernel_size=3, strides=1, act=True, act_name=act_name, use_bn=use_bn),
                    ]
                )
            else:
                self.quality_conv = DWConv(in_channels, kernel_size=3, strides=1, act=True, act_name=act_name, use_bn=use_bn)
            self.quality_out = keras.layers.Conv2D(
                filters=self.num_anchors,
                kernel_size=1,
                padding="same",
                use_bias=True,
                kernel_initializer="he_normal",
            )

        obj_bias = float(tf.math.log(0.01 / (1.0 - 0.01)))
        cls_bias = float(tf.math.log(0.01 / (1.0 - 0.01)))
        self.obj_conv = DWConv(in_channels, kernel_size=3, strides=1, act=True, act_name=act_name, use_bn=use_bn)
        self.obj_out = keras.layers.Conv2D(
            filters=self.num_anchors,
            kernel_size=1,
            padding="same",
            use_bias=True,
            kernel_initializer="he_normal",
            bias_initializer=keras.initializers.Constant(obj_bias),
        )

        if self.use_iou_aware_head:
            self.cls_tower = keras.Sequential(
                [
                    ConvBNAct(self.cls_mid, kernel_size=1, strides=1, act_name=act_name, use_bn=use_bn),
                    DWConv(self.cls_mid, kernel_size=3, strides=1, act=True, act_name=act_name, use_bn=use_bn),
                    DWConv(self.cls_mid, kernel_size=3, strides=1, act=True, act_name=act_name, use_bn=use_bn),
                ]
            )
        else:
            self.cls_reduce = ConvBNAct(self.cls_mid, kernel_size=1, strides=1, act_name=act_name, use_bn=use_bn)
            self.cls_conv = DWConv(self.cls_mid, kernel_size=3, strides=1, act=True, act_name=act_name, use_bn=use_bn)
        self.cls_out = keras.layers.Conv2D(
            filters=self.num_anchors * self.nc,
            kernel_size=1,
            padding="same",
            use_bias=True,
            kernel_initializer="he_normal",
            bias_initializer=keras.initializers.Constant(cls_bias),
        )

    def call(self, x, training: bool = False):
        b = tf.shape(x)[0]
        h = tf.shape(x)[1]
        w = tf.shape(x)[2]
        x = self.context(x, training=training)
        if self.use_improved_head and self.context_res is not None:
            x = x + self.context_res(x, training=training)
        if self.head_rfb is not None:
            x = self.head_rfb(x, training=training)
        if self.lod_dw is not None:
            lob = self.lod_dw(x)
            if self.lod_bn is not None:
                lob = self.lod_bn(lob, training=training)
            lob = self.lod_act(lob)
            lob = self.lod_pw(lob, training=training)
            lob = self.large_obj_blocks(lob, training=training)
            lob = tf.image.resize(lob, size=(h, w), method="nearest")
            x = x + self.large_obj_fuse(lob, training=training)
        if self.use_head_ese and self.head_se is not None:
            x = self.head_se(x)

        if self.use_iou_aware_head:
            box_feat = self.box_tower(x, training=training)
        else:
            box_feat = self.box_conv(x, training=training)
        box = self.box_out(box_feat)
        box = tf.reshape(box, (b, h, w, self.num_anchors, 4))

        obj = self.obj_conv(x, training=training)
        obj = self.obj_out(obj)
        obj = tf.reshape(obj, (b, h, w, self.num_anchors, 1))

        quality = None
        if self.has_quality:
            if self.use_iou_aware_head:
                quality_feat = self.quality_tower(x, training=training)
            else:
                quality_feat = self.quality_conv(x, training=training)
            quality = self.quality_out(quality_feat)
            quality = tf.reshape(quality, (b, h, w, self.num_anchors, 1))

        if self.use_iou_aware_head:
            cls_feat = self.cls_tower(x, training=training)
        else:
            cls_feat = self.cls_reduce(x, training=training)
            cls_feat = self.cls_conv(cls_feat, training=training)
        cls = self.cls_out(cls_feat)
        cls = tf.reshape(cls, (b, h, w, self.num_anchors, self.nc))

        parts = [box, obj]
        if self.has_quality and quality is not None:
            parts.append(quality)
        parts.append(cls)
        raw = tf.concat(parts, axis=-1)
        raw = tf.reshape(raw, (b, h, w, self.num_anchors * self.no))
        return raw


class UltraTinyOD(keras.Model):
    """Keras implementation of UltraTinyOD."""

    def __init__(
        self,
        num_classes: int = 1,
        config: Optional[UltraTinyODConfig] = None,
        c_stem: int = 16,
        use_residual: bool = False,
    ):
        if config is None:
            config = UltraTinyODConfig(num_classes=num_classes)
        else:
            config.num_classes = num_classes
        super().__init__(name="UltraTinyOD")
        self.config = config
        self.num_classes = num_classes
        self.c_stem = c_stem
        self.use_residual_flag = use_residual
        self.backbone = UltraTinyODBackbone(
            c_stem=c_stem,
            use_residual=use_residual,
            out_stride=int(config.stride),
            activation=config.activation,
            use_batchnorm=config.use_batchnorm,
        )
        self.head = UltraTinyODHead(self.backbone.out_channels, config)
        self.anchors = self.head.anchors
        self.out_stride = int(config.stride)
        self.use_improved_head = self.head.use_improved_head
        self.use_iou_aware_head = self.head.use_iou_aware_head
        self.has_quality_head = self.head.has_quality
        self.score_mode = self.head.score_mode
        self.quality_power = self.head.quality_power

    def call(self, inputs, training: bool = False, decode: bool = False, return_feat: bool = False):
        feat = self.backbone(inputs, training=training)
        raw = self.head(feat, training=training)
        if decode:
            decoded = decode_anchor(
                raw,
                anchors=self.anchors,
                num_classes=self.num_classes,
                has_quality=self.head.has_quality,
                wh_scale=self.head.wh_scale if self.head.use_improved_head else None,
                score_mode=self.head.score_mode,
                quality_power=self.head.quality_power,
            )
            return (raw, decoded, feat) if return_feat else (raw, decoded)
        if return_feat:
            return raw, feat
        return raw

    def get_config(self) -> dict:
        return {
            "num_classes": self.num_classes,
            "config": self.config.to_dict() if hasattr(self.config, "to_dict") else {},
            "c_stem": self.c_stem,
            "use_residual": self.use_residual_flag,
        }

    @classmethod
    def from_config(cls, cfg: dict):
        cfg_dict = cfg.get("config", {})
        ut_cfg = UltraTinyODConfig.from_dict(cfg_dict) if isinstance(cfg_dict, dict) else UltraTinyODConfig()
        return cls(
            num_classes=cfg.get("num_classes", ut_cfg.num_classes),
            config=ut_cfg,
            c_stem=cfg.get("c_stem", 16),
            use_residual=cfg.get("use_residual", False),
        )
