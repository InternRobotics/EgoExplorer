"""长度分帧的 MessagePack 网络协议。"""

from __future__ import annotations

import socket
import struct
from typing import Any

import msgpack
import numpy as np


PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 128 * 1024 * 1024
_HEADER = struct.Struct("!I")
_NDARRAY_MARKER = "__ego_ndarray__"


class ProtocolError(RuntimeError):
    """收到不兼容或损坏的数据包。"""


def _pack_default(value: Any):
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject:
            raise TypeError("协议不支持 object dtype 的 ndarray")
        return {
            _NDARRAY_MARKER: True,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "data": array.tobytes(),
        }
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"不支持的协议对象: {type(value).__name__}")


def _unpack_object(value):
    if not value.get(_NDARRAY_MARKER):
        return value
    try:
        dtype = np.dtype(value["dtype"])
        shape = tuple(int(item) for item in value["shape"])
        return np.frombuffer(value["data"], dtype=dtype).reshape(shape).copy()
    except Exception as exc:
        raise ProtocolError(f"无法解码 ndarray: {exc}") from exc


def pack_message(message):
    payload = msgpack.packb(
        message, default=_pack_default, use_bin_type=True, strict_types=False)
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ProtocolError(f"消息过大: {len(payload)} bytes")
    return _HEADER.pack(len(payload)) + payload


def unpack_message(payload):
    try:
        message = msgpack.unpackb(
            payload, raw=False, object_hook=_unpack_object, strict_map_key=False)
    except Exception as exc:
        if isinstance(exc, ProtocolError):
            raise
        raise ProtocolError(f"无法解码消息: {exc}") from exc
    if not isinstance(message, dict):
        raise ProtocolError("消息顶层必须是字典")
    return message


def _validate_size(size):
    if size <= 0 or size > MAX_MESSAGE_BYTES:
        raise ProtocolError(f"非法消息长度: {size}")
    return size


def recv_message(sock):
    header = _recv_exact(sock, _HEADER.size)
    size = _validate_size(_HEADER.unpack(header)[0])
    return unpack_message(_recv_exact(sock, size))


def send_message(sock, message):
    sock.sendall(pack_message(message))


def _recv_exact(sock: socket.socket, size):
    chunks = bytearray()
    while len(chunks) < size:
        block = sock.recv(size - len(chunks))
        if not block:
            raise ConnectionError("网络对端已关闭连接")
        chunks.extend(block)
    return bytes(chunks)
