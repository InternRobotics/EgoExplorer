@echo off
rem Run this script directly; edit the two paths below first.
set "LOCAL_PATH=E:\EgoDataCollection\data"
set "REMOTE_PATH=myremote:bucket/ego_data"
call rclone copy --progress "%LOCAL_PATH%" "%REMOTE_PATH%"
