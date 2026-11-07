"""嘴：把话说出来。

两级：先试在线音色（更好听），失败退回本机 `say`。
眼镜上以后换成骨传导 / I2S 喇叭，只改这个文件。
"""
from __future__ import annotations

import shutil
import subprocess
import sys

ONLINE_VOICE = "zh-CN-XiaoxiaoNeural"
_down = False


def _local(text: str, voice: str = "Meijia", rate: int = 190) -> None:
    if not shutil.which("say"):
        print(text)
        return
    subprocess.run(["say", "-v", voice, "-r", str(rate), text], timeout=60)


def speak(text: str, rate: int = 190, online: bool = True) -> None:
    global _down
    if not text:
        return
    if online and not _down:
        try:
            import asyncio

            import edge_tts

            path = "/tmp/_glasses_tts.mp3"
            communicate = edge_tts.Communicate(text, ONLINE_VOICE, rate="+0%")
            asyncio.run(communicate.save(path))
            subprocess.run(["afplay", path], timeout=60)
            return
        except Exception as e:  # noqa: BLE001
            _down = True
            print("（在线音色不可用，改用本机：%s）" % str(e)[:60], file=sys.stderr)
    _local(text)
