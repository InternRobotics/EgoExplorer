#!/usr/bin/env bash
# 自动发现直连 Wuji Glove 的 Linux 以太网口，配置持久网络并验证 SDK 连接。
#
# 官方默认地址：左手 192.168.1.100，右手 192.168.1.101；主机使用
# 192.168.1.x/24。脚本不会清空已有地址，也不会修改默认路由网卡。

set -Eeuo pipefail

LEFT_GLOVE_IP="${WUJI_LEFT_GLOVE_IP:-192.168.1.100}"
RIGHT_GLOVE_IP="${WUJI_RIGHT_GLOVE_IP:-192.168.1.101}"
HOST_PREFIX="${WUJI_HOST_PREFIX:-192.168.1}"
HOST_START="${WUJI_HOST_START:-10}"
SDK_PYTHON="${WUJI_PYTHON:-python}"
PING_TIMEOUT=1
EXPECTED_GLOVES=0
CHECK_ONLY=0
SDK_CHECK=1
SDK_CONNECT_CHECK=1
ALLOW_UFW=0
VERBOSE=0
declare -a EXPLICIT_INTERFACES=()
declare -a CANDIDATES=()
declare -a TARGET_IPS=()
declare -A HOST_IP_FOR=()
declare -A TEMP_CIDR_FOR=()
declare -A MATCH_IFACES_FOR=()
declare -A GLOVE_IFACE_FOR=()
declare -A TARGETS_FOR_IFACE=()

usage() {
    cat <<'EOF'
自动发现并配置 Wuji Glove 网络（原生 Linux + NetworkManager）

用法：
  ./scripts/setup_wuji_gloves.sh
  ./scripts/setup_wuji_gloves.sh --check
  ./scripts/setup_wuji_gloves.sh --interface enxAAA --interface enxBBB

默认行为：
  1. 查找已插线的物理以太网口；排除 Wi-Fi、虚拟网卡、默认路由网卡和业务网卡。
  2. 临时分配 192.168.1.10 起的主机地址，逐网卡探测左/右手 .100/.101。
  3. 为每个命中的网卡建立 wuji-glove-<网卡名> NetworkManager 持久配置。
  4. 建立每只手套的 /32 路由，避免双网卡同网段时把数据发错网口。
  5. 调用 wuji_sdk.SdkManager.scan()，再逐 SN 连接并断开，验证真实 SDK 链路。

参数：
  --check                  只检查当前配置，不修改地址、路由或 NetworkManager
  --interface IFACE        只检查指定网卡；可重复（显式指定时允许已有业务 IPv4）
  --left-ip IP             左手地址，默认 192.168.1.100
  --right-ip IP            右手地址，默认 192.168.1.101
  --host-prefix A.B.C      主机地址前三段，默认 192.168.1
  --host-start N           第一张网卡主机地址末段，默认 10
  --expected-gloves N      要求 SDK 至少发现 N 只；0 表示按网络探测数量判断
  --ping-timeout SEC       单次 ping 超时，默认 1
  --allow-ufw              UFW 启用时放行来源网段的 UDP 50000/50001
  --python COMMAND          运行 wuji_sdk 的 Python，默认 python
  --no-sdk-check           只配置/检查网络，不加载 Wuji SDK
  --scan-only              SDK 只 scan，不执行 connect/disconnect 验证
  -v, --verbose            输出探测细节
  -h, --help               显示帮助

环境变量等价项：
  WUJI_LEFT_GLOVE_IP, WUJI_RIGHT_GLOVE_IP, WUJI_HOST_PREFIX, WUJI_HOST_START,
  WUJI_PYTHON

示例：
  # 自动配置当前插入的左右手
  ./scripts/setup_wuji_gloves.sh

  # 机器网卡很多时明确指定两个 USB 网卡
  ./scripts/setup_wuji_gloves.sh --interface enx001 --interface enx002

  # 手套地址曾被修改
  ./scripts/setup_wuji_gloves.sh --left-ip 10.10.1.100 --right-ip 10.10.1.101 \
    --host-prefix 10.10.1
EOF
}

log() { printf '%s\n' "$*"; }
warn() { printf '[警告] %s\n' "$*" >&2; }
fail() { printf '[错误] %s\n' "$*" >&2; exit 1; }
debug() { ((VERBOSE == 0)) || printf '[调试] %s\n' "$*"; }

while (($#)); do
    case "$1" in
        --check) CHECK_ONLY=1 ;;
        --interface)
            (($# >= 2)) || fail "--interface 缺少网卡名"
            EXPLICIT_INTERFACES+=("$2")
            shift
            ;;
        --left-ip)
            (($# >= 2)) || fail "--left-ip 缺少地址"
            LEFT_GLOVE_IP="$2"
            shift
            ;;
        --right-ip)
            (($# >= 2)) || fail "--right-ip 缺少地址"
            RIGHT_GLOVE_IP="$2"
            shift
            ;;
        --host-prefix)
            (($# >= 2)) || fail "--host-prefix 缺少 A.B.C"
            HOST_PREFIX="$2"
            shift
            ;;
        --host-start)
            (($# >= 2)) || fail "--host-start 缺少数字"
            HOST_START="$2"
            shift
            ;;
        --expected-gloves)
            (($# >= 2)) || fail "--expected-gloves 缺少数字"
            EXPECTED_GLOVES="$2"
            shift
            ;;
        --ping-timeout)
            (($# >= 2)) || fail "--ping-timeout 缺少秒数"
            PING_TIMEOUT="$2"
            shift
            ;;
        --allow-ufw) ALLOW_UFW=1 ;;
        --python)
            (($# >= 2)) || fail "--python 缺少命令或路径"
            SDK_PYTHON="$2"
            shift
            ;;
        --no-sdk-check) SDK_CHECK=0 ;;
        --scan-only) SDK_CONNECT_CHECK=0 ;;
        -v|--verbose) VERBOSE=1 ;;
        -h|--help) usage; exit 0 ;;
        *) fail "未知参数：$1（使用 --help 查看用法）" ;;
    esac
    shift
done

is_ipv4() {
    local ip="$1" part
    local -a parts=()
    [[ "$ip" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]] || return 1
    IFS=. read -r -a parts <<<"$ip"
    for part in "${parts[@]}"; do
        ((10#$part <= 255)) || return 1
    done
}

[[ "$HOST_PREFIX" =~ ^([0-9]{1,3}\.){2}[0-9]{1,3}$ ]] \
    || fail "--host-prefix 必须是 A.B.C：$HOST_PREFIX"
is_ipv4 "$HOST_PREFIX.1" || fail "--host-prefix 包含越界数字：$HOST_PREFIX"
is_ipv4 "$LEFT_GLOVE_IP" || fail "左手 IP 非法：$LEFT_GLOVE_IP"
is_ipv4 "$RIGHT_GLOVE_IP" || fail "右手 IP 非法：$RIGHT_GLOVE_IP"
[[ "$LEFT_GLOVE_IP" != "$RIGHT_GLOVE_IP" ]] || fail "左右手 IP 不能相同"
[[ "${LEFT_GLOVE_IP%.*}" == "$HOST_PREFIX" ]] \
    || fail "左手 $LEFT_GLOVE_IP 与主机网段 $HOST_PREFIX.x 不一致"
[[ "${RIGHT_GLOVE_IP%.*}" == "$HOST_PREFIX" ]] \
    || fail "右手 $RIGHT_GLOVE_IP 与主机网段 $HOST_PREFIX.x 不一致"
[[ "$HOST_START" =~ ^[0-9]+$ ]] && ((HOST_START >= 1 && HOST_START <= 252)) \
    || fail "--host-start 必须在 1..252"
[[ "$EXPECTED_GLOVES" =~ ^[0-9]+$ ]] && ((EXPECTED_GLOVES <= 2)) \
    || fail "--expected-gloves 必须是 0、1 或 2"
[[ "$PING_TIMEOUT" =~ ^[0-9]+$ ]] && ((PING_TIMEOUT >= 1)) \
    || fail "--ping-timeout 必须是正整数"

for command_name in ip ping; do
    command -v "$command_name" >/dev/null 2>&1 \
        || fail "未安装 $command_name 命令"
done
if ((SDK_CHECK)); then
    command -v "$SDK_PYTHON" >/dev/null 2>&1 \
        || fail "找不到 SDK Python：$SDK_PYTHON"
fi
if ((CHECK_ONLY == 0)); then
    command -v nmcli >/dev/null 2>&1 \
        || fail "自动持久配置需要 NetworkManager/nmcli"
    if ((EUID != 0)); then
        command -v sudo >/dev/null 2>&1 || fail "需要 root 权限，但未找到 sudo"
        log "[权限] 网络配置需要 sudo，正在请求授权……"
        sudo -v
    fi
fi

run_root() {
    if ((EUID == 0)); then
        "$@"
    else
        sudo "$@"
    fi
}

interface_exists() { [[ -d "/sys/class/net/$1" ]]; }

is_default_route_iface() {
    local iface="$1"
    ip -4 route show default 2>/dev/null \
        | awk '{for (i=1; i<=NF; i++) if ($i=="dev") print $(i+1)}' \
        | grep -Fxq -- "$iface"
}

has_foreign_global_ipv4() {
    local iface="$1" cidr ip_addr
    while IFS= read -r cidr; do
        [[ -n "$cidr" ]] || continue
        ip_addr="${cidr%/*}"
        [[ "$ip_addr" == "$HOST_PREFIX".* ]] || return 0
    done < <(ip -4 -o addr show dev "$iface" scope global 2>/dev/null \
        | awk '{print $4}')
    return 1
}

discover_candidates() {
    local iface path carrier type
    if ((${#EXPLICIT_INTERFACES[@]})); then
        for iface in "${EXPLICIT_INTERFACES[@]}"; do
            interface_exists "$iface" || fail "找不到指定网卡：$iface"
            [[ "$iface" != "lo" ]] || fail "不能使用 loopback 网卡"
            [[ ! -d "/sys/class/net/$iface/wireless" ]] || fail "$iface 是 Wi-Fi，不适合直连手套"
            CANDIDATES+=("$iface")
        done
        return
    fi

    for path in /sys/class/net/*; do
        [[ -e "$path" ]] || continue
        iface="${path##*/}"
        [[ "$iface" != "lo" ]] || continue
        [[ ! -d "$path/wireless" ]] || continue
        [[ -e "$path/device" ]] || { debug "跳过虚拟网卡 $iface"; continue; }
        type="$(<"$path/type")"
        [[ "$type" == "1" ]] || continue
        is_default_route_iface "$iface" \
            && { debug "跳过默认路由网卡 $iface"; continue; }
        has_foreign_global_ipv4 "$iface" \
            && { debug "跳过已有业务 IPv4 的网卡 $iface"; continue; }
        if ((CHECK_ONLY == 0)); then
            run_root ip link set dev "$iface" up
        fi
        carrier=0
        # 部分 USB 以太网驱动在接口尚未 up 时读取 carrier 会返回 EINVAL。
        # 在 set -e 下，直接使用 $(<file) 会让整个发现流程提前退出。
        [[ ! -r "$path/carrier" ]] \
            || carrier="$(cat "$path/carrier" 2>/dev/null || printf '0')"
        if [[ "$carrier" != "1" && "$CHECK_ONLY" == "0" ]]; then
            sleep 0.3
            [[ ! -r "$path/carrier" ]] \
                || carrier="$(cat "$path/carrier" 2>/dev/null || printf '0')"
        fi
        [[ "$carrier" == "1" ]] \
            || { debug "跳过未插线网卡 $iface"; continue; }
        CANDIDATES+=("$iface")
    done
}

deduplicate_candidates() {
    local iface
    declare -A seen=()
    declare -a unique=()
    for iface in "${CANDIDATES[@]}"; do
        [[ -z "${seen[$iface]:-}" ]] || continue
        seen[$iface]=1
        unique+=("$iface")
    done
    CANDIDATES=("${unique[@]}")
}

subnet_address_on_iface() {
    local iface="$1"
    ip -4 -o addr show dev "$iface" scope global 2>/dev/null \
        | awk -v prefix="$HOST_PREFIX." '$4 ~ ("^" prefix) {sub(/\/.*/, "", $4); print $4; exit}'
}

cleanup_temporary_addresses() {
    local iface cidr
    for iface in "${!TEMP_CIDR_FOR[@]}"; do
        cidr="${TEMP_CIDR_FOR[$iface]}"
        [[ -n "$cidr" ]] || continue
        run_root ip addr del "$cidr" dev "$iface" >/dev/null 2>&1 || true
        unset 'TEMP_CIDR_FOR[$iface]'
    done
}
trap cleanup_temporary_addresses EXIT

prepare_probe_addresses() {
    local index=0 iface current host_ip host_last
    for iface in "${CANDIDATES[@]}"; do
        current="$(subnet_address_on_iface "$iface" || true)"
        if [[ -n "$current" ]]; then
            HOST_IP_FOR[$iface]="$current"
            log "[网卡] $iface 已有同网段地址 $current"
            continue
        fi
        if ((CHECK_ONLY)); then
            warn "$iface 当前没有 $HOST_PREFIX.x 地址；--check 不会临时配置，无法探测该网卡"
            HOST_IP_FOR[$iface]=""
            continue
        fi
        host_last=$((HOST_START + index))
        ((host_last <= 254)) || fail "可分配的主机地址超出 IPv4 范围"
        host_ip="$HOST_PREFIX.$host_last"
        [[ "$host_ip" != "$LEFT_GLOVE_IP" && "$host_ip" != "$RIGHT_GLOVE_IP" ]] \
            || fail "主机候选地址与手套地址冲突：$host_ip"
        HOST_IP_FOR[$iface]="$host_ip"
        TEMP_CIDR_FOR[$iface]="$host_ip/24"
        run_root ip addr add "$host_ip/24" dev "$iface"
        log "[网卡] $iface 临时配置 $host_ip/24，用于硬件探测"
        index=$((index + 1))
    done
}

probe_target() {
    local iface="$1" target="$2"
    [[ -n "${HOST_IP_FOR[$iface]:-}" ]] || return 1
    ping -n -q -I "$iface" -c 1 -W "$PING_TIMEOUT" "$target" >/dev/null 2>&1
}

probe_all() {
    local iface target matches
    for target in "${TARGET_IPS[@]}"; do
        matches=""
        for iface in "${CANDIDATES[@]}"; do
            debug "从 $iface (${HOST_IP_FOR[$iface]:-无地址}) 探测 $target"
            if probe_target "$iface" "$target"; then
                log "[发现] $target 可通过 $iface 到达"
                matches+="${matches:+ }$iface"
            fi
        done
        MATCH_IFACES_FOR[$target]="$matches"
    done
}

resolve_mappings() {
    local target matches count iface
    for target in "${TARGET_IPS[@]}"; do
        matches="${MATCH_IFACES_FOR[$target]:-}"
        if [[ -z "$matches" ]]; then
            warn "未发现 $target"
            continue
        fi
        count="$(wc -w <<<"$matches")"
        if ((count > 1)); then
            fail "$target 同时从多个网卡响应（$matches）。这通常表示 IP 冲突或交换机重复接入，请只保留真实手套链路。"
        fi
        iface="$matches"
        GLOVE_IFACE_FOR[$target]="$iface"
        TARGETS_FOR_IFACE[$iface]="${TARGETS_FOR_IFACE[$iface]:-}${TARGETS_FOR_IFACE[$iface]:+ }$target"
    done
}

remove_temporary_for_iface() {
    local iface="$1" cidr
    cidr="${TEMP_CIDR_FOR[$iface]:-}"
    [[ -n "$cidr" ]] || return 0
    run_root ip addr del "$cidr" dev "$iface" >/dev/null 2>&1 || true
    unset 'TEMP_CIDR_FOR[$iface]'
}

configure_profile() {
    local iface="$1" host_ip="$2" targets="$3"
    local profile="wuji-glove-$iface" legacy legacy_iface mac routes="" target
    mac="$(<"/sys/class/net/$iface/address")"
    for target in $targets; do
        routes+="${routes:+,}$target/32"
    done

    # 兼容本项目旧脚本创建的固定名称，避免同一网卡留下两个自动连接 profile。
    if ! nmcli -g connection.id connection show "$profile" >/dev/null 2>&1; then
        for legacy in wuji-left-glove wuji-right-glove; do
            legacy_iface="$(nmcli -g connection.interface-name connection show "$legacy" \
                2>/dev/null || true)"
            if [[ "$legacy_iface" == "$iface" ]]; then
                profile="$legacy"
                break
            fi
        done
    fi

    remove_temporary_for_iface "$iface"
    if ! nmcli -g connection.id connection show "$profile" >/dev/null 2>&1; then
        log "[配置] 创建 NetworkManager profile：$profile"
        run_root nmcli connection add type ethernet ifname "$iface" con-name "$profile" \
            >/dev/null
    else
        log "[配置] 更新 NetworkManager profile：$profile"
    fi
    run_root nmcli connection modify "$profile" \
        connection.interface-name "$iface" \
        connection.autoconnect yes \
        connection.autoconnect-priority 100 \
        connection.autoconnect-retries 0 \
        802-3-ethernet.mac-address "$mac" \
        ipv4.method manual \
        ipv4.addresses "$host_ip/24" \
        ipv4.gateway "" \
        ipv4.dns "" \
        ipv4.ignore-auto-dns yes \
        ipv4.never-default yes \
        ipv4.routes "$routes" \
        ipv6.method disabled
    run_root nmcli connection up "$profile" ifname "$iface" >/dev/null
    log "[配置] $iface：$host_ip/24，手套路由：${targets// /, }"
}

configure_network() {
    local iface
    for iface in "${!TARGETS_FOR_IFACE[@]}"; do
        configure_profile "$iface" "${HOST_IP_FOR[$iface]}" "${TARGETS_FOR_IFACE[$iface]}"
    done
}

verify_routes_and_ping() {
    local target iface route failed=0
    for target in "${!GLOVE_IFACE_FOR[@]}"; do
        iface="${GLOVE_IFACE_FOR[$target]}"
        route="$(ip route get "$target" 2>&1)" || {
            warn "$target 没有可用路由"
            failed=1
            continue
        }
        if [[ "$route" != *"dev $iface"* ]]; then
            warn "$target 路由未走探测到的 $iface：$route"
            failed=1
        elif ! probe_target "$iface" "$target"; then
            warn "$target 配置后无法从 $iface ping 通"
            failed=1
        else
            log "[正常] $target -> $iface；$route"
        fi
    done
    return "$failed"
}

configure_ufw_if_requested() {
    command -v ufw >/dev/null 2>&1 || return 0
    local status
    status="$(run_root ufw status 2>/dev/null | head -n 1 || true)"
    [[ "$status" == *active* || "$status" == *活动* ]] || return 0
    if ((ALLOW_UFW)); then
        run_root ufw allow from "$HOST_PREFIX.0/24" to any port 50000 proto udp >/dev/null
        run_root ufw allow from "$HOST_PREFIX.0/24" to any port 50001 proto udp >/dev/null
        log "[防火墙] 已允许 $HOST_PREFIX.0/24 访问 UDP 50000/50001"
    else
        warn "UFW 已启用。若 SDK scan 失败，请重跑并加 --allow-ufw 放行 UDP 50000/50001。"
    fi
}

sdk_verify() {
    local required="$1"
    WUJI_EXPECTED_GLOVES="$required" WUJI_CONNECT_CHECK="$SDK_CONNECT_CHECK" \
        "$SDK_PYTHON" - <<'PY'
import os
import sys

try:
    from wuji_sdk import DeviceType, SdkManager
except Exception as exc:
    print(f"[SDK 错误] 无法导入 wuji_sdk: {exc}", file=sys.stderr)
    print("           请先执行: python -m pip install wuji-sdk", file=sys.stderr)
    raise SystemExit(1)

manager = SdkManager.instance()
try:
    discovered = list(manager.scan())
except Exception as exc:
    print(f"[SDK 错误] SdkManager.scan() 失败: {type(exc).__name__}: {exc}",
          file=sys.stderr)
    raise SystemExit(1)

glove_type = getattr(DeviceType, "WujiGlove", None)
gloves = []
for device in discovered:
    device_type = getattr(device, "device_type", None)
    if glove_type is not None and device_type == glove_type:
        gloves.append(device)
    elif "glove" in str(device_type).lower():
        gloves.append(device)

print(f"[SDK] 共扫描到 {len(discovered)} 个 Wuji 设备，其中手套 {len(gloves)} 只")
for device in gloves:
    sn = str(getattr(device, "sn", "?"))
    side = "左手" if len(sn) > 3 and sn[3].upper() == "J" else (
        "右手" if len(sn) > 3 and sn[3].upper() == "K" else "未知手性")
    print(f"[SDK]   {side}: sn={sn} address={getattr(device, 'address', '?')}")

required = int(os.environ.get("WUJI_EXPECTED_GLOVES", "1"))
if len(gloves) < required:
    print(f"[SDK 错误] 期望至少 {required} 只手套，实际发现 {len(gloves)} 只。",
          file=sys.stderr)
    raise SystemExit(1)

if os.environ.get("WUJI_CONNECT_CHECK") != "1":
    raise SystemExit(0)

connected = []
try:
    for index, device in enumerate(gloves):
        sn = str(getattr(device, "sn", ""))
        name = f"wuji_network_check_{index}"
        handle = manager.connect(sn=sn, device_name=name)
        connected.append((name, handle))
        serial = getattr(handle, "serial_number", sn)
        print(f"[SDK]   连接成功: {serial}")
finally:
    for name, handle in reversed(connected):
        try:
            handle.disconnect()
        except Exception:
            try:
                manager.disconnect(device_name=name)
            except Exception:
                pass

print("[SDK] 所有已发现手套均完成连接/断开验证")
PY
}

TARGET_IPS=("$LEFT_GLOVE_IP" "$RIGHT_GLOVE_IP")

log "[信息] Wuji 默认网络探测：左手=$LEFT_GLOVE_IP，右手=$RIGHT_GLOVE_IP"
discover_candidates
deduplicate_candidates
((${#CANDIDATES[@]} > 0)) || fail "没有找到可用于手套的已插线物理以太网口。请检查供电、USB-C 正反面和网线，或用 --interface 指定。"
log "[信息] 候选网卡：${CANDIDATES[*]}"

prepare_probe_addresses
probe_all
resolve_mappings

FOUND_COUNT=${#GLOVE_IFACE_FOR[@]}
((FOUND_COUNT > 0)) || fail "未在候选网卡上探测到 $LEFT_GLOVE_IP 或 $RIGHT_GLOVE_IP。请检查线缆方向、供电及手套是否使用过自定义 IP。"
if ((EXPECTED_GLOVES > 0 && FOUND_COUNT < EXPECTED_GLOVES)); then
    fail "要求发现 $EXPECTED_GLOVES 只手套，但网络层只发现 $FOUND_COUNT 只"
fi

if ((CHECK_ONLY == 0)); then
    configure_network
    configure_ufw_if_requested
fi

verify_routes_and_ping || fail "网络验证失败；没有继续执行 SDK 连接测试"

if ((SDK_CHECK)); then
    required="$FOUND_COUNT"
    ((EXPECTED_GLOVES == 0)) || required="$EXPECTED_GLOVES"
    sdk_verify "$required"
fi

log "[结果] 成功配置并验证 $FOUND_COUNT 只 Wuji 手套。"
for target in "${TARGET_IPS[@]}"; do
    iface="${GLOVE_IFACE_FOR[$target]:-}"
    [[ -n "$iface" ]] || continue
    if [[ "$target" == "$LEFT_GLOVE_IP" ]]; then side="左手"; else side="右手"; fi
    log "       $side $target -> $iface (${HOST_IP_FOR[$iface]})"
done
