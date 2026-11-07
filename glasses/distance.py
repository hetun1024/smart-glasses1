"""距离：从"画面里多大"倒推"大概多远"。

原理是针孔相机：d = f × H / h
    f  焦距（像素），由水平视场角和画面宽度算出来
    H  这个类别在现实里的典型高度（米）
    h  它在画面里占多少像素

误差很大，原因写在下面，用的时候别装准：
  * H 是**假设**的 —— 一个杯子可能 8cm 也可能是 12cm，这一项就差 50%
  * 人眼/鞋/帽子会让"人的高度"浮动
  * 斜着看的时候，框高不等于真实高度

所以对外只说"大概一米二"，不说"1.23 米"。
"""
from __future__ import annotations

import math
from typing import Optional

from .detect import REAL_HEIGHT

# 默认按 60° 水平视场角。手机/眼镜摄像头的实际值差别很大，可以 --fov 改。
DEFAULT_FOV = 60.0


def focal_px(width_px: int, fov_deg: float = DEFAULT_FOV) -> float:
    """水平视场角 → 焦距（像素）。"""
    return (width_px / 2.0) / math.tan(math.radians(fov_deg) / 2.0)


def estimate(name: str, h_px: float, f: float) -> Optional[float]:
    """估计距离（米）。没有尺寸先验的类别返回 None。"""
    real_h = REAL_HEIGHT.get(name)
    if not real_h or h_px < 4:
        return None
    return f * real_h / h_px


def phrase(meters: Optional[float]) -> str:
    """把数字变成一句人话。故意给区间，不给小数点。"""
    if meters is None:
        return ""
    if meters < 0.4:
        return "就在手边"
    if meters > 8:
        return ""
    if meters < 1.0:
        return "不到一米"
    if meters < 1.5:
        return "大概一米"
    if meters < 2.5:
        return "两米左右"
    if meters < 4:
        return "三四米"
    return "比较远"
