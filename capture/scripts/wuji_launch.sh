#!/usr/bin/env bash
# One-command launch of GoPro + Wuji gloves + Windows tracker capture (Linux side).
# Edit PYTHON and the options below for this machine; set EGO_TRACKER_TOKEN to the same secret as the Windows sender.
set -e

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"

exec "$PYTHON" record_ego_wuji.py \
    --task-setting "task_setting/example.json" \
    --tracker-host "0.0.0.0" \
    --tracker-port "8766" \
    --host "127.0.0.1" \
    --preview-width "640" \
    --preview-height "360" \
    --storage-limit-gb "50" \
    --storage-check-interval "30"
