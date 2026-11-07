"""看：一帧画面 → 几个"什么东西、在哪、多远"。

用的是 NanoDet-Plus（3.6MB），80 类 COCO。选它的理由：
  * 小到能塞进 8MB PSRAM 的 ESP32-S3，也小到能在 Mac 上跑几百帧
  * 单阶段、无 anchor 聚类，解码逻辑只有三十行，出问题能查
  * 官方有 OpenCV DNN 的参考实现，不容易抄错

真正的难点不在检测，在**"检测到之后说什么"** —— 那在 narrate.py。
"""
from __future__ import annotations

import os
from typing import List, Optional

import cv2
import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(HERE, "models", "nanodet_320.onnx")

# COCO 80 类里日常生活会遇到的，给中文。没列到的用英文名。
ZH = {
    "person": "人", "bicycle": "自行车", "car": "汽车", "motorcycle": "摩托车",
    "bus": "公交车", "train": "火车", "truck": "卡车",
    "traffic light": "红绿灯", "stop sign": "停车牌", "bench": "长椅",
    "cat": "猫", "dog": "狗", "bird": "鸟", "horse": "马",
    "backpack": "背包", "umbrella": "雨伞", "handbag": "手提包", "suitcase": "行李箱",
    "bottle": "瓶子", "cup": "杯子", "bowl": "碗", "wine glass": "酒杯",
    "fork": "叉子", "knife": "刀", "spoon": "勺子", "banana": "香蕉",
    "apple": "苹果", "orange": "橙子", "sandwich": "三明治",
    "chair": "椅子", "couch": "沙发", "bed": "床", "dining table": "桌子",
    "toilet": "马桶", "tv": "电视", "laptop": "笔记本电脑", "mouse": "鼠标",
    "remote": "遥控器", "keyboard": "键盘", "cell phone": "手机",
    "microwave": "微波炉", "oven": "烤箱", "toaster": "烤面包机",
    "refrigerator": "冰箱", "sink": "水槽",
    "book": "书", "clock": "钟", "vase": "花瓶", "scissors": "剪刀",
    "teddy bear": "玩偶", "toothbrush": "牙刷",
}

# 真实世界里的典型尺寸（米），用来从"画面里多大"倒推"离多远"
# 只列会对着估距离的；没有先验的类别不报距离。
REAL_HEIGHT = {
    "person": 1.70, "book": 0.24, "cup": 0.10, "bottle": 0.25, "bowl": 0.08,
    "cell phone": 0.15, "laptop": 0.23, "keyboard": 0.02, "mouse": 0.04,
    "chair": 0.90, "dining table": 0.75, "tv": 0.60, "bench": 0.85,
    "backpack": 0.45, "handbag": 0.25, "umbrella": 0.85, "cat": 0.25,
    "dog": 0.50, "vase": 0.30, "clock": 0.25, "remote": 0.05,
    "wine glass": 0.20, "banana": 0.20, "apple": 0.08, "orange": 0.08,
}


class Detector:
    def __init__(self, model_path: str = MODEL, size: int = 0,
                 prob: float = 0.35, iou: float = 0.55):
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.sess = ort.InferenceSession(model_path, sess_options=opts,
                                         providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        # 输入尺寸从模型里读，不猜 —— 文件名写 320、实际 416 这种事很常见
        self.input_name = inp.name
        self.size = size or int(inp.shape[2])
        self.prob = prob
        self.iou = iou
        self.strides = (8, 16, 32)  # 三个层级：52x52 / 26x26 / 13x13（416 输入）
        self.reg_max = 7
        self.project = np.arange(self.reg_max + 1)
        self.mean = np.array([103.53, 116.28, 123.675], np.float32).reshape(1, 1, 3)
        self.std = np.array([57.375, 57.12, 58.395], np.float32).reshape(1, 1, 3)
        self.names = self._load_names()
        self.anchors = self._make_anchors(self.size)

    @staticmethod
    def _load_names() -> List[str]:
        # COCO 的类别名有时在模型里读不到，就用标准顺序
        return [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
            "truck", "boat", "traffic light", "fire hydrant", "stop sign",
            "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
            "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
            "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
            "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "couch", "potted plant", "bed", "dining table", "toilet", "tv",
            "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
            "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
            "scissors", "teddy bear", "hair drier", "toothbrush",
        ]

    def _make_anchors(self, size: int) -> List[np.ndarray]:
        out = []
        for s in self.strides:
            h = w = size // s
            sx = np.arange(w) * s
            sy = np.arange(h) * s
            xv, yv = np.meshgrid(sx, sy)
            cx = xv.flatten() + 0.5 * (s - 1)
            cy = yv.flatten() + 0.5 * (s - 1)
            out.append(np.column_stack((cx, cy)))
        return out

    def detect(self, frame: np.ndarray) -> List[dict]:
        # 保持长宽比缩放，再补边 —— 直接拉伸会让框变形
        h0, w0 = frame.shape[:2]
        r = min(self.size / w0, self.size / h0)
        img = cv2.resize(frame, (int(round(w0 * r)), int(round(h0 * r))))
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        dh, dw = (self.size - img.shape[0]) // 2, (self.size - img.shape[1]) // 2
        canvas[dh : dh + img.shape[0], dw : dw + img.shape[1]] = img

        blob = (canvas.astype(np.float32) - self.mean) / self.std
        blob = blob.transpose(2, 0, 1)[None, ...]  # HWC → NCHW
        outs = self.sess.run(None, {self.input_name: blob})
        return self._decode(outs, r, dw, dh, w0, h0)

    def _decode(self, outs, r, dw, dh, w0, h0) -> List[dict]:
        # 输出顺序各家不一样，不靠猜：分类头是 80 通道，框头是 4*(reg_max+1)=32 通道
        cls_scores = [o for o in outs if o.shape[-1] == 80]
        bbox_preds = [o for o in outs if o.shape[-1] == 32]
        # 每层候选框数量从多到少，正好对应 stride 从小到大
        cls_scores.sort(key=lambda o: -o.shape[1])
        bbox_preds.sort(key=lambda o: -o.shape[1])
        boxes, scores = [], []
        for stride, cs, bp, anchors in zip(self.strides, cls_scores, bbox_preds, self.anchors):
            cs = cs.squeeze(0) if cs.ndim == 3 else cs
            bp = bp.squeeze(0) if bp.ndim == 3 else bp
            # 减最大值再 exp：直接 exp 会溢出成 inf，inf/inf = nan，框就废了
            z = bp.reshape(-1, self.reg_max + 1)
            z = np.nan_to_num(z, nan=0.0, posinf=30.0, neginf=-30.0)
            e = np.exp(z - z.max(axis=1, keepdims=True))
            dist = (e / np.maximum(e.sum(1, keepdims=True), 1e-9)) @ self.project
            dist = dist.reshape(-1, 4) * stride
            x1 = anchors[:, 0] - dist[:, 0]
            y1 = anchors[:, 1] - dist[:, 1]
            x2 = anchors[:, 0] + dist[:, 2]
            y2 = anchors[:, 1] + dist[:, 3]
            boxes.append(np.column_stack([x1, y1, x2 - x1, y2 - y1]))
            scores.append(cs)

        boxes = np.nan_to_num(np.concatenate(boxes), nan=0.0)
        scores = np.nan_to_num(np.concatenate(scores), nan=0.0)
        cls_ids = np.argmax(scores, axis=1)
        conf = np.max(scores, axis=1)

        keep = cv2.dnn.NMSBoxes(
            boxes.tolist(), conf.tolist(), self.prob, self.iou
        )
        out = []
        for i in np.array(keep).flatten():
            x, y, w, h = boxes[i]
            # 从 320 画布映射回原图
            x = (x - dw) / r
            y = (y - dh) / r
            w /= r
            h /= r
            if w < 8 or h < 8:
                continue
            name = self.names[cls_ids[i]] if cls_ids[i] < len(self.names) else "?"
            out.append(
                {
                    "name": name,
                    "zh": ZH.get(name, name),
                    "conf": float(conf[i]),
                    "box": (float(x), float(y), float(w), float(h)),
                    "cx": float(x + w / 2) / max(w0, 1),
                    "cy": float(y + h / 2) / max(h0, 1),
                    "h_px": float(h),
                }
            )
        return out


def draw(frame: np.ndarray, dets: List[dict], focal_px: Optional[float] = None) -> np.ndarray:
    """把框画出来。中文画不上（cv2 不带字体），所以框上标英文名。"""
    img = frame.copy()
    for d in dets:
        x, y, w, h = (int(v) for v in d["box"])
        cv2.rectangle(img, (x, y), (x + w, y + h), (90, 220, 90), 2)
        label = "%s %.0f%%" % (d["name"], d["conf"] * 100)
        if focal_px and d["name"] in REAL_HEIGHT:
            m = focal_px * REAL_HEIGHT[d["name"]] / max(d["h_px"], 1)
            label += "  %.1fm" % m
        cv2.putText(img, label, (x, max(14, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 220, 90), 1, cv2.LINE_AA)
    return img
