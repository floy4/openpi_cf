#!/usr/bin/env bash

set -euo pipefail

# ==========================================
# 基础配置
# ==========================================
SESSION="pi05-libero"
SERVER_WIN="server"
CLIENT_WIN="client"

# Python 环境（使用 openvla 环境）
PYTHON="/home/zhy/miniconda3/envs/openvla/bin/python"

# 硬件与网络配置
SERVER_GPU=1
CLIENT_GPU=1
HOST="127.0.0.1"
PORT=8888

SERVER_DIR="$PWD"
CLIENT_DIR="$PWD"

# 评估模式与任务套件
MODES=("E" "B" "C" "D")
TASK_SUITES=(
    libero_spatial
    libero_object
    libero_goal
    libero_10
)

SERVER_TARGET="${SESSION}:${SERVER_WIN}"
CLIENT_TARGET="${SESSION}:${CLIENT_WIN}"

# ==========================================
# 辅助函数定义
# ==========================================

check_tmux_target() {
    local target=$1
    tmux list-panes -t "$target" >/dev/null 2>&1
}

create_tmux_window() {
    local session=$1
    local window=$2
    echo "[INFO] 创建 tmux 窗口: ${session}:${window}"
    tmux new-window -t "${session}" -n "${window}"
    sleep 1
}

ensure_tmux_session() {
    local session=$1
    if ! tmux has-session -t "$session" 2>/dev/null; then
        echo "[INFO] tmux 会话 '$session' 不存在，自动创建..."
        tmux new-session -d -s "$session" -n "$SERVER_WIN"
        sleep 1
        # 创建 client 窗口
        create_tmux_window "$session" "$CLIENT_WIN"
        echo "[INFO] ✅ tmux 会话 '$session' 已创建，包含 '$SERVER_WIN' 和 '$CLIENT_WIN' 窗口"
    fi
}

ensure_tmux_windows() {
    local session=$1
    local server_win=$2
    local client_win=$3

    # 检查 server 窗口
    if ! check_tmux_target "${session}:${server_win}"; then
        echo "[INFO] server 窗口不存在，自动创建..."
        create_tmux_window "$session" "$server_win"
    fi

    # 检查 client 窗口
    if ! check_tmux_target "${session}:${client_win}"; then
        echo "[INFO] client 窗口不存在，自动创建..."
        create_tmux_window "$session" "$client_win"
    fi
}

clear_window_cmdline() {
    local target=$1
    tmux send-keys -t "$target" C-c
    sleep 1
    tmux send-keys -t "$target" C-m
    sleep 1
}

stop_server() {
    local mode=${1:-""}
    echo "[INFO] 停止服务端进程 (Mode: ${mode})..."
    tmux send-keys -t "$SERVER_TARGET" C-c
    sleep 3

    # 先按端口回收残留服务，避免复用旧代码进程
    local port_pids
    port_pids=$(ss -ltnp 2>/dev/null | awk -v port=":${PORT}" '$4 ~ port {print $NF}' | grep -o 'pid=[0-9]\+' | cut -d= -f2 | sort -u || true)
    if [[ -n "${port_pids}" ]]; then
        for pid in ${port_pids}; do
            kill "$pid" 2>/dev/null || true
            echo "[INFO] 已按端口终止 Server PID: $pid"
        done
    fi

    # 精准匹配当前 mode 的 Server 进程，避免误杀其他实验
    if [[ -n "$mode" ]] && [[ -f "/tmp/server_${mode}.pid" ]]; then
        local pid
        pid=$(cat "/tmp/server_${mode}.pid")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            echo "[INFO] 已终止 Server PID: $pid"
        fi
        rm -f "/tmp/server_${mode}.pid"
    else
        # 兜底：按命令行模式匹配清理
        pkill -f "serve_policy.py --env LIBERO --cf_mode ${mode}" >/dev/null 2>&1 || true
    fi

    sleep 2
}

wait_for_client_finish() {
    local timeout_sec=${1:-86400}
    local out_dir=${2:-""}  # 输出目录参数，用于精确匹配当前客户端
    local elapsed=0

    echo "[INFO] 等待客户端任务执行完毕（最长 ${timeout_sec} 秒）..."
    echo "[INFO] 监控输出目录: ${out_dir}"

    while (( elapsed < timeout_sec )); do
        # 使用输出目录精确匹配当前客户端进程
        if [[ -n "$out_dir" ]]; then
            if ! pgrep -f "python -u examples/libero/main.py.*${out_dir}" >/dev/null 2>&1; then
                echo "[INFO] ✅ 当前客户端任务已完成。"
                return 0
            fi
        else
            # 兜底：无输出目录时使用端口匹配
            if ! pgrep -f "python -u examples/libero/main.py.*--args.port ${PORT}" >/dev/null 2>&1; then
                echo "[INFO] ✅ 当前客户端任务已完成。"
                return 0
            fi
        fi
        sleep 10
        elapsed=$(( elapsed + 10 ))

        if (( elapsed % 60 == 0 )); then
            echo "[INFO] ... 客户端仍在运行 (${elapsed}s / ${timeout_sec}s) ..."
        fi
    done

    echo "[ERROR] ❌ 客户端任务超时（超过 ${timeout_sec} 秒）。"
    return 1
}

wait_for_server_port() {
    local host=$1
    local port=$2
    local timeout_sec=$3
    local elapsed=0

    echo "[INFO] 等待服务端加载模型并绑定端口 $host:$port（最长 $timeout_sec 秒）..."

    while (( elapsed < timeout_sec )); do
        # 用 ss 检查端口是否处于 LISTEN 状态，不建立实际连接
        if ss -tlnp | grep -q ":${port} "; then
            echo "[INFO] ✅ 检测到服务端端口已就绪！(耗时: ${elapsed} 秒)"
            sleep 3
            return 0
        fi

        sleep 2
        elapsed=$(( elapsed + 2 ))

        if (( elapsed % 10 == 0 )); then
            echo "[INFO] ... 仍在加载模型 (${elapsed}s / ${timeout_sec}s) ..."
        fi
    done

    echo "[ERROR] ❌ 等待服务端端口超时（超过 ${timeout_sec} 秒）。"
    return 1
}


# ==========================================
# 前置运行检查（自动创建会话和窗口）
# ==========================================

# 确保 tmux 会话存在
ensure_tmux_session "$SESSION"

# 确保 server 和 client 窗口存在
ensure_tmux_windows "$SESSION" "$SERVER_WIN" "$CLIENT_WIN"

echo "[INFO] Python 环境: ${PYTHON}"
echo "[INFO] tmux 会话: ${SESSION}"
echo "[INFO] Server 窗口: ${SERVER_TARGET}"
echo "[INFO] Client 窗口: ${CLIENT_TARGET}"

# ==========================================
# 主干调度流程
# ==========================================

for MODE in "${MODES[@]}"; do
    echo "================================================="
    echo "[INFO] 开始评估模式: Mode ${MODE}"
    echo "================================================="

    # 启动前先清理上一轮可能残留的进程和窗口输入
    stop_server "$MODE"
    clear_window_cmdline "$SERVER_TARGET"
    clear_window_cmdline "$CLIENT_TARGET"

    # 构建并发送 Server 启动命令，后台记录 PID 供精准清理
    SERVER_CMD="cd ${SERVER_DIR} && CUDA_VISIBLE_DEVICES=${SERVER_GPU} ${PYTHON} scripts/serve_policy.py \
--env LIBERO --use_cf_sampling --cf_mode ${MODE} \
--cf_guidance_scale 0.1 --vlm_effect_upper_threshold 0.5 --reweight_action_with_cf \
--return_cf_metrics \
& echo \$! > /tmp/server_${MODE}.pid; wait"

    echo "[INFO] 发送服务端启动命令..."
    tmux send-keys -t "$SERVER_TARGET" "$SERVER_CMD" C-m

    # 阻塞等待 Server 端口就绪
    if ! wait_for_server_port "$HOST" "$PORT" 600; then
        echo "[WARN] ⚠️ 放弃当前 Mode ${MODE}，跳过执行下一个模式。"
        stop_server "$MODE"
        continue
    fi

    # 遍历所有任务套件
    BASE_DIR="results/pi05_cf_libero_${MODE}"

    for suite in "${TASK_SUITES[@]}"; do
        out_dir="${BASE_DIR}/${suite}"

        if [ -d "$out_dir" ]; then
            echo "[SKIP] ${suite} -> 输出目录 ${out_dir} 已存在，跳过该套件。"
            continue
        fi

        mkdir -p "$out_dir"
        echo "==== 运行任务套件: ${suite} (Mode ${MODE}) ===="

        LOG_FILE="${BASE_DIR}/${suite}/run.log"

        # 构建 Client 命令
        CLIENT_CMD="export MUJOCO_GL=egl; \
export PYOPENGL_PLATFORM=egl; \
export CUDA_VISIBLE_DEVICES=${CLIENT_GPU}; \
cd ${CLIENT_DIR} && ${PYTHON} -u examples/libero/main.py \
--args.task_suite_name ${suite} \
--args.video_out_path ${out_dir} \
2>&1 | tee ${LOG_FILE}"

        echo "[INFO] 发送客户端执行命令..."
        tmux send-keys -t "$CLIENT_TARGET" "$CLIENT_CMD" C-m

        # 等待当前 suite 跑完再发下一个
        if ! wait_for_client_finish 86400 "$out_dir"; then
            echo "[WARN] ⚠️ 套件 ${suite} 执行超时，继续下一个套件。"
        fi

        echo "[INFO] 套件 ${suite} 运行结束。日志已保存至 ${LOG_FILE}"
    done

    # 当前 Mode 所有 suite 执行完毕，关闭 Server
    echo "[INFO] 模式 ${MODE} 的所有套件评估完毕，准备关闭 Server..."
    stop_server "$MODE"
    sleep 3

done

echo "================================================="
echo "[INFO] 🎉 所有模式与任务套件执行完毕！"
echo "================================================="
