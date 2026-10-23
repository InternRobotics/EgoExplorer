"""任务流程定义文件的读取与校验。"""

import json
import os
import re
import unicodedata


def load_task_setting(path):
    """读取 JSON 任务列表，返回去除首尾空白后的任务名称。"""
    task_path = os.path.abspath(os.path.expanduser(os.fspath(path)))
    try:
        with open(task_path, "r", encoding="utf-8-sig") as file_obj:
            data = json.load(file_obj)
    except OSError as exc:
        raise ValueError(f"无法读取任务定义文件 {task_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"任务定义文件不是有效 JSON: {task_path}（第 {exc.lineno} 行第 {exc.colno} 列）"
        ) from exc

    if not isinstance(data, list) or not data:
        raise ValueError("任务定义文件的顶层必须是非空字符串数组")

    tasks = []
    for index, value in enumerate(data, start=1):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"第 {index} 个任务名称必须是非空字符串")
        tasks.append(value.strip())
    return task_path, tasks


def task_filename_component(task_name, max_length=80):
    """把任务名称转换为安全、可读的单个文件名组成部分。"""
    normalized = unicodedata.normalize("NFKC", str(task_name or "").strip())
    safe_name = "".join(
        char if char.isalnum() or char in "-_" else "_"
        for char in normalized
    )
    safe_name = re.sub(r"_+", "_", safe_name).strip("_-")
    return safe_name[:max_length] or "task"
