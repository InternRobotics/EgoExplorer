# 硬件

[English](README.md) | **中文**

> [!NOTE]
> 清单仍在补充中：型号、购买链接和打印参数会陆续补上。

穿戴方式和走线见 [EgoExplorer Guidelines](https://internrobotics.github.io/EgoExplorer-Guidelines/)。

## 采购清单

| 部件 | 型号 | 数量 | 购买链接 | 备注 |
| :--- | :--- | :---: | :--- | :--- |
| VIVE Ultimate Tracker（含接收器） | | 5 | | 头部、双手腕、双脚踝 |
| GoPro HERO13 Black | | 1 | | 头部相机 |
| GoPro Media Mod | | 1 | | |
| USB HDMI 采集卡 | | 1 | | 把相机画面接入采集端 |
| micro-HDMI → HDMI 线 | | 1 | | |
| MANUS Metagloves Pro **或** Wuji 手套 | | 1 副 | | Wuji 带触觉 |
| USB 以太网卡 | | 2 | | 仅 Wuji 方案，每只手套一个 |
| 头环、绑带 | | | | |

## 3D 打印件

STL 文件在 [`3d_print/`](3d_print/)。

| 打印件 | 用途 | 文件 | 数量 | 打印参数 |
| :--- | :--- | :--- | :---: | :--- |
| 头部转接件 | GoPro 与头部 tracker 刚性连接 | [`gopro-tracker.stl`](3d_print/gopro-tracker.stl) | 1 | |
| 手背底座 | 把 tracker 固定在手套手背 | | 2 | |
| 踝部底座 | 把 tracker 固定在脚踝外侧 | | 2 | |
| Wuji 接触件 | | [`wuji-contact.stl`](3d_print/wuji-contact.stl) | | |
