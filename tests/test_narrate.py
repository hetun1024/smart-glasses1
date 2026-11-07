#!/usr/bin/env python3
"""播报层的测试：不加载模型，纯逻辑。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glasses import distance, narrate  # noqa: E402


def det(name, cx=0.5, cy=0.5, conf=0.8, h=200, zh=None):
    return {"name": name, "zh": zh or name, "conf": conf,
            "box": (cx * 1000 - 50, cy * 1000 - 50, 100, h),
            "cx": cx, "cy": cy, "h_px": h, "_frame_area": 1000 * 1000}


def test_方位():
    assert narrate.direction(0.1) == "左边"
    assert narrate.direction(0.5) == "前面"
    assert narrate.direction(0.9) == "右边"


def test_距离用区间不用小数():
    assert distance.phrase(0.3) == "就在手边"
    assert distance.phrase(0.8) == "不到一米"
    assert distance.phrase(1.2) == "大概一米"
    assert distance.phrase(2.0) == "两米左右"
    assert distance.phrase(9.0) == ""


def test_没有尺寸先验就不报距离():
    assert distance.estimate("giraffe", 200, 500) is None
    assert distance.estimate("book", 200, 500) is not None


def test_多帧确认_只出现一帧不算():
    one = [det("chair")]
    assert narrate.merge([one]) == []


def test_多帧确认_连续出现才算():
    a = [det("chair", cx=0.3)]
    b = [det("chair", cx=0.31)]
    assert len(narrate.merge([a, b])) == 1


def test_位置跳太远算不同东西():
    a = [det("chair", cx=0.1)]
    b = [det("chair", cx=0.9)]
    assert narrate.merge([a, b]) == []


def test_低置信度不报():
    assert narrate.merge([[det("chair", conf=0.30)]]) == []


def test_不相干类别不报():
    assert narrate.merge([[det("tie", conf=0.9)] * 2], need=1) == []


def test_生成一句话():
    text = narrate.describe([det("person", cx=0.5, h=300, zh="人")], focal=500)
    assert "前面" in text and "人" in text


def test_没东西时说没东西():
    assert "没看到" in narrate.describe([], focal=500)


def test_太近会单独喊一声():
    close = det("person", h=1500, zh="人")
    assert narrate.urgency([close], focal=500) is not None


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print("ok    %s" % fn.__name__)
        except AssertionError as e:
            failed += 1
            print("FAIL  %s: %s" % (fn.__name__, e))
        except Exception as e:  # noqa: BLE001
            failed += 1
            print("ERROR %s: %s: %s" % (fn.__name__, type(e).__name__, e))
    print("\n%d/%d passed" % (len(tests) - failed, len(tests)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
