"""说：把"检测到什么"翻译成一句人能听的话。

这一层才是这个项目真正难的地方。检测模型给你的是 5 个框，人需要的是**一句话**。

三条规矩：

1. **宁可不报，不许报错。** 假阳性对一个看不见的人来说是灾难 —— 他可能因为
   "前面有个台阶"里其实没有台阶而摔倒。所以有多帧确认 + 面积门槛 + 类别白名单。
2. **只说最近的三件事。** 报十个框等于没报。
3. **先说方位，再说距离。** "左边有人" 比 "一米二的地方有人" 有用得多。
"""
from __future__ import annotations

from typing import Dict, List, Optional

from . import distance as dist

# 只报这些类别。椅子是"有用的东西"，徽章/领带/钟是"噪音"。
USEFUL = {
    "person", "chair", "dining table", "couch", "bed", "toilet", "tv", "laptop",
    "cup", "bottle", "bowl", "book", "cell phone", "keyboard", "mouse",
    "backpack", "handbag", "suitcase", "umbrella", "bench",
    "car", "bicycle", "motorcycle", "bus", "truck", "traffic light", "stop sign",
    "dog", "cat", "potted plant", "sink", "refrigerator", "door",
}

MIN_CONF = 0.45  # 置信度门槛
MIN_AREA = 0.006  # 至少占画面 0.6%，太小的忽略
NEED_HITS = 2  # 连续几帧看到才报


def direction(cx: float) -> str:
    if cx < 0.33:
        return "左边"
    if cx > 0.67:
        return "右边"
    return "前面"


def merge(frames: List[List[dict]], need: int = NEED_HITS) -> List[dict]:
    """多帧确认：同一个类别、位置接近、连续出现够多次，才算数。

    frames 是最新的在前。返回的是"确认过"的目标，按距离/大小排好。
    """
    if not frames:
        return []
    latest = frames[0]
    out = []
    for d in latest:
        if d["name"] not in USEFUL:
            continue
        if d["conf"] < MIN_CONF:
            continue
        box = d["box"]
        area = (box[2] * box[3]) / max(1.0, d.get("_frame_area", 1.0))
        if area < MIN_AREA:
            continue

        hits = 1
        for prev in frames[1:need]:
            for q in prev:
                if q["name"] != d["name"]:
                    continue
                # 位置接近（中心差 < 画面 12%）
                if abs(q["cx"] - d["cx"]) < 0.12 and abs(q["cy"] - d["cy"]) < 0.15:
                    hits += 1
                    break
        if hits >= need:
            out.append(d)
    return out


def describe(dets: List[dict], focal: float, limit: int = 3) -> str:
    """把确认过的目标变成一句中文。"""
    if not dets:
        return "前面没看到明显的东西。"

    scored = []
    for d in dets:
        m = dist.estimate(d["name"], d["h_px"], focal)
        # 有距离的按距离排，没距离的按"画面里多大"排
        key = m if m is not None else (10.0 - d["h_px"] / 100.0)
        scored.append((key, d, m))
    scored.sort(key=lambda t: t[0])

    parts = []
    seen = set()
    for _, d, m in scored[:limit]:
        if d["name"] in seen:
            continue
        seen.add(d["name"])
        where = direction(d["cx"])
        how_far = dist.phrase(m)
        noun = d.get("zh", d["name"])
        if how_far:
            parts.append("%s有%s，%s" % (where, noun, how_far))
        else:
            parts.append("%s有%s" % (where, noun))

    if not parts:
        return "前面没看到明显的东西。"
    return "；".join(parts) + "。"


def urgency(dets: List[dict], focal: float) -> Optional[str]:
    """特别近的人/车，单独喊一声 —— 这才是"辅助"该有的优先级。"""
    for d in dets:
        if d["name"] not in ("person", "car", "bicycle", "motorcycle", "bus", "truck"):
            continue
        m = dist.estimate(d["name"], d["h_px"], focal)
        if m is not None and m < 0.8:
            return "小心，%s离得很近" % d.get("zh", d["name"])
    return None


def announce(dets: List[dict], focal: float, last: str = "") -> Optional[str]:
    """决定这一刻要不要开口。返回要说的话，或 None 表示不用说。"""
    alarm = urgency(dets, focal)
    if alarm and alarm != last:
        return alarm
    text = describe(dets, focal)
    if text == last:
        return None  # 一模一样就别重复了，很烦
    return text
