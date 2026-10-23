"""
devices
=======

设备驱动 + 它们共享的抽象基类。

公开 API
--------
BaseCamera      —— 相机 / 视频设备的抽象接口
BaseGlove       —— 数据手套的抽象接口
Frame           —— 具名元组: (index, image, time_sec, timestamp)
GoProCamera     —— GoPro MP4 随机读取器 (别名 GoProVideo)
WujiGlove       —— wuji 数据手套驱动

新增设备
--------
相机::

    from devices.base import BaseCamera, Frame
    class MyCamera(BaseCamera): ...

手套::

    from devices.base import BaseGlove
    class MyGlove(BaseGlove): ...
"""

from .base import BaseCamera, BaseGlove, Frame
from .gopro_camera import GoProCamera, GoProVideo
from .wuji_glove import WujiGlove
from .manus_glove import ManusGlove


def __getattr__(name):
    """按需加载需要可选 OpenXR 依赖的 tracker 驱动。"""
    if name == "ViveTracker":
        from .vive_tracker import ViveTracker
        return ViveTracker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BaseCamera",
    "BaseGlove",
    "Frame",
    "GoProCamera",
    "GoProVideo",
    "WujiGlove",
    "ViveTracker",
    "ManusGlove",
]
