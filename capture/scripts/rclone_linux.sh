#!/usr/bin/env bash
# 直接运行本脚本即可上传; 上传前把下面两个路径改成自己的。
set -e

LOCAL_PATH="./data"                              # 本地数据目录
REMOTE_PATH="myremote:bucket/ego_data"         # 远端路径（rclone remote:path）

exec rclone copy --progress "$LOCAL_PATH" "$REMOTE_PATH"
