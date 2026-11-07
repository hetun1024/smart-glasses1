#!/usr/bin/env python3
"""下模型。仓库里不放大文件，克隆下来跑这一句就行。

    python3 setup_models.py
"""
from __future__ import annotations

import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, "models", "nanodet_320.onnx")
URL = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
       "models/object_detection_nanodet/object_detection_nanodet_2022nov.onnx")


def main() -> int:
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    if os.path.exists(DEST) and os.path.getsize(DEST) > 1_000_000:
        print("已经有了：%s（%.1f MB）" % (DEST, os.path.getsize(DEST) / 1e6))
        return 0
    print("下载中（约 3.6MB）…")
    try:
        urllib.request.urlretrieve(URL, DEST)
    except Exception as e:  # noqa: BLE001
        print("下载失败：%s" % e)
        print("手动下载这个地址，放到 models/nanodet_320.onnx：")
        print(URL)
        return 1
    print("好了：%s（%.1f MB）" % (DEST, os.path.getsize(DEST) / 1e6))
    return 0


if __name__ == "__main__":
    sys.exit(main())
