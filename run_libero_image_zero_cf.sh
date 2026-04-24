#!/usr/bin/env bash

set -euo pipefail

# ==========================================
# 仅图像置零（像素级）CF 评估脚本
# ==========================================
# 逻辑参考 run_libero.sh，底层对应 Pi0.sample_actions_with_cf:
#   obs_img0 = _make_cf_observation_image_zero(...)
#   actions_img0 = sample_actions(...)
#
# 本脚本只做图像置零，不做语言置零：
#   --clear_prompt_for_image_cf False
#
# 说明：
# - 使用 input-level CF（--use_cf_sampling）
# - 默认使用 Mode A（只使用图像分支 delta）
# ==========================================

# 基础配置
SESSION="pi05-libero-image-zero-cf"
SERVER_WIN="server"
CLIENT_WIN="client"

# Python 环境（使用 openvla 环境）
PYTHON="/home/zhy/miniconda3/envs/openvla/bin/python"

# 硬件与网络配置
SERVER_GPU=3
CLIENT_GPU=3
HOST="127.0.0.1"
PORT=8890

SERVER_DIR="$PWD"
CLIENT_DIR="$PWD"

# 评估模式与任务套件
# 仅图像分支建议使用 Mode A
MODES=("E")
TASK_SUITES=(
    libero_spatial
    libero_object
    libero_goal
    libero_10
)

# CF 参数
CF_GUIDANCE_SCALE=0.1
VLM_EFFECT_THRESHOLD=0.5

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
        create_tmux_window "$session" "$CLIENT_WIN"
        echo "[INFO] ✅ tmux 会话 '$session' 已创建，包含 '$SERVER_WIN' 和 '$CLIENT_WIN' 窗口"
    fi
}

ensure_tmux_windows() {
    local session=$1
    local server_win=$2
    local client_win=$3

    if ! check_tmux_target "${session}:${server_win}"; then
        echo "[INFO] server 窗口不存在，自动创建..."
        create_tmux_window "$session" "$server_win"
    fi

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
    echo "[INFO] 停止服务端进程 (Mode: ${mode}, image-zero-only)..."
    tmux send-keys -t "$SERVER_TARGET" C-c
    sleep 3

    local port_pids
    port_pids=$(ss -ltnp 2>/dev/null | awk -v port=":${PORT}" '$4 ~ port {print $NF}' | grep -o 'pid=[0-9]\+' | cut -d= -f2 | sort -u || true)
    if [[ -n "${port_pids}" ]]; then
        for pid in ${port_pids}; do
            kill "$pid" 2>/dev/null || true
            echo "[INFO] 已按端口终止 Server PID: $pid"
        done
    fi

    if [[ -n "$mode" ]] && [[ -f "/tmp/server_image_zero_${mode}.pid" ]]; then
        local pid
        pid=$(cat "/tmp/server_image_zero_${mode}.pid")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            echo "[INFO] 已终止 Server PID: $pid"
        fi
        rm -f "/tmp/server_image_zero_${mode}.pid"
    else
        pkill -f "serve_policy.py --env LIBERO --use_cf_sampling --cf_mode ${mode}" >/dev/null 2>&1 || true
    fi

    sleep 2
}

wait_for_client_finish() {
    local timeout_sec=${1:-86400}
    local elapsed=0

    echo "[INFO] 等待客户端任务执行完毕（最长 ${timeout_sec} 秒）..."

    while (( elapsed < timeout_sec )); do
        if ! pgrep -f "python -u examples/libero/main.py" >/dev/null 2>&1; then
            echo "[INFO] ✅ 当前客户端任务已完成。"
            return 0
        fi
        sleep 10
        elapsed=$(( elapsed + 10 ))
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
# 前置运行检查
# ==========================================

ensure_tmux_session "$SESSION"
ensure_tmux_windows "$SESSION" "$SERVER_WIN" "$CLIENT_WIN"

echo "[INFO] Python 环境: ${PYTHON}"
echo "[INFO] tmux 会话: ${SESSION}"
echo "[INFO] Server 窗口: ${SERVER_TARGET}"
echo "[INFO] Client 窗口: ${CLIENT_TARGET}"

echo "================================================="
echo "[INFO] 仅图像置零（像素级）CF 评估"
echo "[INFO] clear_prompt_for_image_cf=False (不置零语言)"
echo "[INFO] 模式: ${MODES[*]}"
echo "[INFO] 端口: ${PORT}"
echo "================================================="

for MODE in "${MODES[@]}"; do
    echo "================================================="
    echo "[INFO] 开始评估模式: Mode ${MODE} (image-zero-only CF)"
    echo "================================================="

    stop_server "$MODE"
    clear_window_cmdline "$SERVER_TARGET"
    clear_window_cmdline "$CLIENT_TARGET"

    SERVER_CMD="cd ${SERVER_DIR} && CUDA_VISIBLE_DEVICES=${SERVER_GPU} ${PYTHON} scripts/serve_policy.py \\
--env LIBERO \\
--use_cf_sampling \\
--cf_mode ${MODE} \\
--cf_guidance_scale ${CF_GUIDANCE_SCALE} \\
--vlm_effect_upper_threshold ${VLM_EFFECT_THRESHOLD} \\
--reweight_action_with_cf \\
--clear_prompt_for_image_cf False \\
--return_cf_metrics \\
--port ${PORT} \\
& echo \$! > /tmp/server_image_zero_${MODE}.pid; wait"

    echo "[INFO] 发送服务端启动命令（仅图像置零）..."
    tmux send-keys -t "$SERVER_TARGET" "$SERVER_CMD" C-m

    if ! wait_for_server_port "$HOST" "$PORT" 600; then
        echo "[WARN] ⚠️ 放弃当前 Mode ${MODE}，跳过执行下一个模式。"
        stop_server "$MODE"
        continue
    fi

    BASE_DIR="results/pi05_cf_image_zero_libero_${MODE}"

    for suite in "${TASK_SUITES[@]}"; do
        out_dir="${BASE_DIR}/${suite}"

        if [ -d "$out_dir" ]; then
            echo "[SKIP] ${suite} -> 输出目录 ${out_dir} 已存在，跳过该套件。"
            continue
        fi

        mkdir -p "$out_dir"
        echo "==== 运行任务套件: ${suite} (Mode ${MODE}, image-zero-only CF) ===="

        LOG_FILE="${BASE_DIR}/${suite}/run.log"

        CLIENT_CMD="export MUJOCO_GL=egl; \\
export PYOPENGL_PLATFORM=egl; \\
export CUDA_VISIBLE_DEVICES=${CLIENT_GPU}; \\
cd ${CLIENT_DIR} && ${PYTHON} -u examples/libero/main.py \\
--args.host ${HOST} \\
--args.port ${PORT} \\
--args.task_suite_name ${suite} \\
--args.video_out_path ${out_dir} \\
2>&1 | tee ${LOG_FILE}"

        echo "[INFO] 发送客户端执行命令..."
        tmux send-keys -t "$CLIENT_TARGET" "$CLIENT_CMD" C-m

        if ! wait_for_client_finish 86400; then
            echo "[WARN] ⚠️ 套件 ${suite} 执行超时，继续下一个套件。"
        fi

        echo "[INFO] 套件 ${suite} 运行结束。日志已保存至 ${LOG_FILE}"
    done

    echo "[INFO] 模式 ${MODE} 的所有套件评估完毕，准备关闭 Server..."
    stop_server "$MODE"
    sleep 3
done

echo "================================================="
echo "[INFO] 🎉 所有 image-zero-only CF 任务执行完毕！"
echo "================================================="
echo "[INFO] 结果保存路径: results/pi05_cf_image_zero_libero_{MODE}/"
