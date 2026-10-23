"""Wuji SDK 本地扫描、标定、连接和订阅流程。"""

from .wuji_glove import STREAMS, WujiGlove


def connect_wuji_local(args):
    """导入用户标定、扫描并连接全部 Wuji 手套。"""
    try:
        from wuji_sdk import SdkManager
    except Exception as ex:
        print(f"[wuji] ✖ 无法导入 wuji_sdk: {ex}")
        print("[wuji]   请执行: python -m pip install wuji-sdk")
        return []

    manager = SdkManager.instance()
    if not args.no_wuji_calibration:
        try:
            from wuji_calibration import import_and_select_bundle
            report = import_and_select_bundle(args.wuji_calibration, manager=manager)
            print(f"[wuji] ✔ 已加载标定用户 {report['owner_display_name']} "
                  f"({report['owner_user_id']})")
        except Exception as ex:
            print(f"[wuji] ⚠ 未加载标定 bundle: {ex}")
            try:
                user = manager.current_user()
                print(f"[wuji]   继续使用当前 SDK 用户: "
                      f"{user['display_name']} ({user['user_id'] or 'default'})")
            except Exception:
                print("[wuji]   继续扫描手套；请升级 SDK 后再启用 bundle。")
    else:
        try:
            user = manager.current_user()
            print(f"[wuji] 跳过 bundle，使用当前 SDK 用户: "
                  f"{user['display_name']} ({user['user_id'] or 'default'})")
        except Exception:
            pass

    try:
        gloves = WujiGlove.discover(manager)
    except Exception as ex:
        print(f"[wuji] ✖ 扫描手套失败: {type(ex).__name__}: {ex}")
        return []
    if not gloves:
        print("[wuji] ✖ 未发现手套；请检查手套网络、SDK 和设备状态。")
        return []

    connected = []
    for glove in gloves:
        try:
            glove.connect()
            glove.subscribe(STREAMS)
            connected.append(glove)
            info = glove.glove_info()
            print(f"[wuji] ✔ 已连接 {info.get('slot') or '?'}手 "
                  f"sn={info.get('sn')}，订阅 {len(glove._streams)} 路数据")
        except Exception as ex:
            print(f"[wuji] ✖ 连接 {glove.glove_info()} 失败: "
                  f"{type(ex).__name__}: {ex}")
            try:
                glove.disconnect()
            except Exception:
                pass
    return connected
