#!/usr/bin/env python3
"""智能眼镜的软件部分：看见 → 看懂 → 说出来。

    python3 glasses.py --image 照片.jpg          # 分析一张照片
    python3 glasses.py --camera 0                # 用摄像头实时看
    python3 glasses.py --camera 0 --speak        # 实时看 + 语音播报
    python3 glasses.py --camera 0 --speak --record out.mp4

设计上，**眼镜只负责采画面和播声音**，识别和说话都在外面（Mac 上跑，
以后搬到 RDK）。ESP32-S3 只做采集和 I2S 输出，跟"汤圆"是同一套结构。
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
import time

import cv2

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)  # 模型内部偶尔出 nan，不影响框

from glasses import narrate
from glasses.detect import Detector, draw
from glasses.distance import DEFAULT_FOV, focal_px
from glasses.speak import speak


def analyze_frame(det, frame, history, focal, args):
    dets = det.detect(frame)
    h, w = frame.shape[:2]
    for d in dets:
        d["_frame_area"] = float(w * h)
    history.append(dets)

    confirmed = narrate.merge(list(history), need=args.need)
    text = narrate.announce(confirmed, focal, last=getattr(analyze_frame, "last", ""))
    if text and text != getattr(analyze_frame, "last", ""):
        analyze_frame.last = text
        print("  🔊 %s" % text)
        if args.speak:
            speak(text)
    return dets, confirmed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None, help="分析一张照片")
    ap.add_argument("--camera", type=int, default=None, help="摄像头编号，一般是 0")
    ap.add_argument("--video", default=None, help="视频文件")
    ap.add_argument("--model", default="models/nanodet_320.onnx")
    ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--fov", type=float, default=DEFAULT_FOV, help="水平视场角，默认 60°")
    ap.add_argument("--need", type=int, default=2, help="连续几帧确认才播报")
    ap.add_argument("--speak", action="store_true", help="语音播报")
    ap.add_argument("--record", default=None, help="录一段带框的视频")
    ap.add_argument("--seconds", type=float, default=0, help="跑多久（0=一直）")
    args = ap.parse_args(argv)

    if not os.path.exists(args.model):
        print("找不到模型：%s\n先跑 python3 setup_models.py" % args.model)
        return 2

    det = Detector(args.model, prob=args.conf)
    history = collections.deque(maxlen=6)

    # ---------- 单张图片 ----------
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print("读不到图片：%s" % args.image)
            return 2
        focal = focal_px(frame.shape[1], args.fov)
        args.need = 1  # 单张图没法"连续看到"，一帧就算数
        dets, confirmed = analyze_frame(det, frame, history, focal, args)
        print()
        print("  画面里：%d 个目标，确认 %d 个" % (len(dets), len(confirmed)))
        for d in dets:
            m = narrate.dist.estimate(d["name"], d["h_px"], focal) if hasattr(narrate, "dist") else None
            print("    %-12s %3.0f%%  %s" % (
                d["name"], d["conf"] * 100,
                ("%.1f 米" % m) if m else ""))
        out = "/tmp/glasses_out.jpg"
        cv2.imwrite(out, draw(frame, dets, focal))
        print("  标注图：%s" % out)
        return 0

    # ---------- 摄像头 / 视频 ----------
    src = args.camera if args.camera is not None else args.video
    if src is None:
        print("要么 --image，要么 --camera 0")
        return 2
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("打不开 %r。如果是摄像头，去「系统设置 → 隐私与安全性 → 摄像头」授权。" % (src,))
        return 2

    ok, frame = cap.read()
    if not ok:
        print("读不到画面。")
        return 2
    focal = focal_px(frame.shape[1], args.fov)

    writer = None
    t0 = time.time()
    n = 0
    fps_t = time.time()
    print("开始。按 q 退出。")
    try:
        while True:
            if n:
                ok, frame = cap.read()
                if not ok:
                    break
            n += 1
            dets, confirmed = analyze_frame(det, frame, history, focal, args)

            view = draw(frame, dets, focal)
            if time.time() - fps_t > 1.0:
                print("  %.0f fps" % (n / max(time.time() - t0, 1e-6)))
                fps_t = time.time()

            if args.record:
                if writer is None:
                    h, w = view.shape[:2]
                    writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"),
                                             15.0, (w, h))
                writer.write(view)

            if args.camera is not None:
                cv2.imshow("glasses", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.seconds and time.time() - t0 > args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    print("\n跑了 %d 帧。" % n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
