"""通过 rclone 在后台上传已完成的 Ego 会话。"""

import os
import shutil
import subprocess
import threading
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class RcloneUploader:
    """跨平台 rclone 上传器。

    手动模式使用 ``copy --ignore-existing``，允许安全地重复点击；容量超限
    模式使用 ``move --checksum``，只清理已传输或校验成功的本地文件。
    """

    def __init__(self, save_dir, remote, executable="rclone", config_path=None,
                 on_finished=None):
        self.save_dir = os.path.abspath(save_dir)
        self.remote = (remote or "").strip()
        self.executable = executable or "rclone"
        self.config_path = config_path
        self.on_finished = on_finished
        self._lock = threading.Lock()
        self._running = False
        self._state = "idle"
        self._message = "等待上传" if self.remote else "未配置 rclone 远端"
        self._started = None
        self._finished = None
        self._exit_code = None
        self._output = []
        self._cleanup = False
        self._automatic = False

    def status(self):
        with self._lock:
            return {
                "enabled": bool(self.remote),
                "running": self._running,
                "state": self._state,
                "message": self._message,
                "remote": self.remote,
                "source": self.save_dir,
                "started": self._started,
                "finished": self._finished,
                "exit_code": self._exit_code,
                "mode": "move" if self._cleanup else "copy",
                "automatic": self._automatic,
                "output": "\n".join(self._output[-8:]),
            }

    def start(self, cleanup=False, automatic=False):
        """启动后台任务。

        ``cleanup=False`` 用于手动上传，保留本地文件；``cleanup=True`` 用于
        超限自动清理，执行 rclone move，仅删除成功复制到远端的源文件。
        """
        with self._lock:
            if self._running:
                return False, "已有上传任务正在运行"
            if not self.remote:
                return False, "未配置上传目标，请使用 --rclone-remote"
            if not os.path.isdir(self.save_dir):
                return False, f"数据目录不存在：{self.save_dir}"
            if cleanup:
                safety_error = self._cleanup_safety_error()
                if safety_error:
                    return False, safety_error
            self._running = True
            self._state = "running"
            self._cleanup = bool(cleanup)
            self._automatic = bool(automatic)
            self._message = (
                "存储超限：正在上传并清理已确认的本地文件…"
                if self._cleanup else "正在检查远端并上传新文件…")
            self._started = datetime.now().isoformat(timespec="seconds")
            self._finished = None
            self._exit_code = None
            self._output = []

        thread = threading.Thread(target=self._run, name="rclone-upload", daemon=True)
        thread.start()
        if self._cleanup:
            return True, "自动清理已开始；只删除 rclone 确认已传到远端的文件"
        return True, "上传任务已开始；远端已有文件会自动跳过"

    def _cleanup_safety_error(self):
        """拒绝把根目录、家目录或项目根目录作为自动删除源。"""
        source = os.path.realpath(self.save_dir)
        root = os.path.realpath(os.path.abspath(os.sep))
        forbidden = {
            root,
            os.path.realpath(os.path.expanduser("~")),
            os.path.realpath(os.getcwd()),
            os.path.realpath(PROJECT_ROOT),
        }
        if source in forbidden or os.path.dirname(source) == source:
            return f"拒绝自动清理过于宽泛的数据目录：{self.save_dir}"
        if os.name != "nt" and os.path.dirname(source) == root:
            return f"拒绝自动清理一级系统目录：{self.save_dir}"
        return None

    def _resolve_executable(self):
        resolved = shutil.which(self.executable)
        if resolved:
            return resolved
        # shutil.which 也支持完整路径；保留明确检查可给出更直观的错误。
        if os.path.isfile(self.executable):
            return os.path.abspath(self.executable)
        return None

    def _command(self, executable):
        command = [
            executable,
            "move" if self._cleanup else "copy",
            self.save_dir,
            self.remote,
            "--stats",
            "1s",
            "--stats-one-line",
        ]
        if self._cleanup:
            command.extend([
                "--delete-empty-src-dirs",
                "--checksum",
                "--exclude",
                "/recording_log.csv",
            ])
        else:
            command.extend(["--ignore-existing", "--create-empty-src-dirs"])
        if self.config_path:
            command.extend(["--config", os.path.abspath(self.config_path)])
        return command

    def _append_output(self, line):
        line = line.strip()
        if not line:
            return
        with self._lock:
            self._output.append(line)
            if len(self._output) > 80:
                del self._output[:-80]
            self._message = line[-300:]

    def _finish(self, state, message, exit_code=None):
        with self._lock:
            self._running = False
            self._state = state
            self._message = message
            self._finished = datetime.now().isoformat(timespec="seconds")
            self._exit_code = exit_code
        if self.on_finished is not None:
            try:
                self.on_finished()
            except Exception:
                pass

    def _run(self):
        executable = self._resolve_executable()
        if executable is None:
            self._finish("error", "找不到 rclone；请安装并加入 PATH，或设置 --rclone-bin")
            return

        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            process = subprocess.Popen(
                self._command(executable),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=creationflags,
            )
            if process.stdout is not None:
                for line in process.stdout:
                    self._append_output(line)
            exit_code = process.wait()
        except OSError as exc:
            self._finish("error", f"无法启动 rclone：{exc}")
            return
        except Exception as exc:
            self._finish("error", f"上传异常：{type(exc).__name__}: {exc}")
            return

        if exit_code == 0 and self._cleanup:
            self._finish("success", "自动清理完成；已上传文件已从本地删除", exit_code)
        elif exit_code == 0:
            self._finish("success", "上传完成；远端已有文件已跳过", exit_code)
        else:
            action = "自动上传清理" if self._cleanup else "上传"
            self._finish("error", f"rclone {action}失败（退出码 {exit_code}）", exit_code)


def directory_usage(path):
    """返回目录内普通文件的 ``(总字节数, 文件数)``，忽略符号链接。"""
    total_bytes = 0
    total_files = 0
    pending = [os.path.abspath(path)]
    while pending:
        current = pending.pop()
        try:
            entries = os.scandir(current)
        except OSError:
            continue
        try:
            with entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            pending.append(entry.path)
                        elif entry.is_file(follow_symlinks=False):
                            total_bytes += entry.stat(follow_symlinks=False).st_size
                            total_files += 1
                    except OSError:
                        continue
        except OSError:
            continue
    return total_bytes, total_files
