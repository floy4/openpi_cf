#!/usr/bin/env bash

set -euo pipefail

# ==========================================
# Attention-Guided CF 评估脚本
# ==========================================
# 支持 Token-level CF（注意力屏蔽）和 Pixel-level CF（像素置零）
#
# CF 模式：
#   - token: 在 cf_attn_mask 中屏蔽高注意力的图像 token
#   - pixel: 将高注意力像素块置零后重新采样
#
# ==========================================
# 使用示例
# ==========================================
# 基本用法:
#   ./run_libero_image_attention_cf.sh
#
# 指定干预方式和加权模式:
#   CF_MODE_TYPE=token CF_MODE=E ./run_libero_image_attention_cf.sh
#   CF_MODE_TYPE=pixel CF_MODE=A ./run_libero_image_attention_cf.sh
#
# 选择特定层attention:
#   CF_ATTN_LAYER_INDEX=8  ./run_libero_image_attention_cf.sh   # 较浅层
#   CF_ATTN_LAYER_INDEX=16 ./run_libero_image_attention_cf.sh   # 中间层(默认)
#   CF_ATTN_LAYER_INDEX=24 ./run_libero_image_attention_cf.sh   # 较深层
#
# 开启attention可视化(测试阶段):
#   VISUALIZE_ATTENTION=true ./run_libero_image_attention_cf.sh
#
# 调整干预比例和加权强度:
#   CF_ATTN_TOPK_RATIO=0.3 CF_GUIDANCE_SCALE=0.2 ./run_libero_image_attention_cf.sh
#
# 完整测试命令示例:
#   VISUALIZE_ATTENTION=true CF_ATTN_LAYER_INDEX=16 CF_MODE_TYPE=token CF_MODE=E \
#     ./run_libero_image_attention_cf.sh
#
# 参数说明：
#   --cf_attn_layer_index: Transformer 层索引（默认 16）
#   --cf_attn_time: 提取 attention 的扩散时间点（默认 1.0）
#   --cf_attn_topk_ratio: 屏蔽 top-k 比例的高注意力 token/像素（默认 0.2）
# ==========================================

# 基础配置
SESSION="pi05-libero-attention-cf"
SERVER_WIN="server"
CLIENT_WIN="client"

# Python 环境（使用 openvla 环境）
PYTHON="/home/zhy/miniconda3/envs/openvla/bin/python"

# 硬件与网络配置
SERVER_GPU=0
CLIENT_GPU=0
HOST="127.0.0.1"
PORT=8891

SERVER_DIR="$PWD"
CLIENT_DIR="$PWD"

# 评估任务套件
TASK_SUITES=(
    libero_spatial
    libero_object
    libero_goal
    libero_10
)

# ==========================================
# Attention-Guided CF 参数（可通过命令行修改）
# ==========================================
CF_MODE_TYPE="pixel"  # 可选: "token" 或 "pixel"
CF_ATTN_LAYER_INDEX=16
CF_ATTN_TIME=1.0
CF_ATTN_TOPK_RATIO=0.2

# ==========================================
# CF 加权参数（复用 input-level CF 的参数）
# ==========================================
CF_MODE="E"  # 加权模式: BASE/A/B/C/D/E/F
CF_GUIDANCE_SCALE=0.1
VLM_EFFECT_THRESHOLD=0.5
REWEIGHT_ACTION_WITH_CF=true

# ==========================================
# Attention 可视化参数
# ==========================================
VISUALIZE_ATTENTION=false          # 是否可视化attention (测试时开启)
VISUALIZATION_DIR="results/attention_vis"  # 可视化输出目录
VISUALIZATION_FREQUENCY=10         # 每N步保存一次可视化

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
    local cf_type=${1:-""}
    local cf_weight_mode=${2:-""}
    echo "[INFO] 停止服务端进程 (Attention-CF: ${cf_type}, Mode: ${cf_weight_mode})..."
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

    local pid_file="/tmp/server_attn_cf_${cf_type}_${cf_weight_mode}.pid"
    if [[ -n "$cf_type" ]] && [[ -n "$cf_weight_mode" ]] && [[ -f "$pid_file" ]]; then
        local pid
        pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            echo "[INFO] 已终止 Server PID: $pid"
        fi
        rm -f "$pid_file"
    else
        pkill -f "serve_policy.py --env LIBERO --use_cf_${cf_type}_sampling" >/dev/null 2>&1 || true
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
echo "[INFO] Attention-Guided CF 评估"
echo "[INFO] CF 模式类型: ${CF_MODE_TYPE}"
echo "[INFO] 层索引: ${CF_ATTN_LAYER_INDEX}"
echo "[INFO] Attention 时间点: ${CF_ATTN_TIME}"
echo "[INFO] Top-k 比例: ${CF_ATTN_TOPK_RATIO}"
echo "[INFO] 加权模式: ${CF_MODE}"
echo "[INFO] 加权强度: ${CF_GUIDANCE_SCALE}"
echo "[INFO] Effect 阈值: ${VLM_EFFECT_THRESHOLD}"
echo "[INFO] 可视化: ${VISUALIZE_ATTENTION}"
if [ "${VISUALIZE_ATTENTION}" = "true" ]; then
    echo "[INFO] 可视化输出目录: ${VISUALIZATION_DIR}"
fi
echo "[INFO] 端口: ${PORT}"
echo "================================================="

stop_server "$CF_MODE_TYPE" "$CF_MODE"
clear_window_cmdline "$SERVER_TARGET"
clear_window_cmdline "$CLIENT_TARGET"

SERVER_CMD="cd ${SERVER_DIR} && CUDA_VISIBLE_DEVICES=${SERVER_GPU} ${PYTHON} scripts/serve_policy.py \\
--env LIBERO \\
--use_cf_${CF_MODE_TYPE}_sampling \\
--cf_attn_layer_index ${CF_ATTN_LAYER_INDEX} \\
--cf_attn_time ${CF_ATTN_TIME} \\
--cf_attn_topk_ratio ${CF_ATTN_TOPK_RATIO} \\
--cf_mode ${CF_MODE} \\
--cf_guidance_scale ${CF_GUIDANCE_SCALE} \\
--vlm_effect_upper_threshold ${VLM_EFFECT_THRESHOLD} \\
--return_cf_metrics \\
--port ${PORT} \\
"

# Add boolean flags only when true
# Note: reweight_action_with_cf default is True in serve_policy.py
# When false, use --no-reweight_action_with_cf
if [ "${REWEIGHT_ACTION_WITH_CF}" = "false" ]; then
    SERVER_CMD="${SERVER_CMD}--no-reweight_action_with_cf \\
"
fi

# Add visualization flags only if enabled
if [ "${VISUALIZE_ATTENTION}" = "true" ]; then
    SERVER_CMD="${SERVER_CMD}--visualize_attention \\
--visualization_dir ${VISUALIZATION_DIR} \\
--visualization_frequency ${VISUALIZATION_FREQUENCY} \\
"
fi

SERVER_CMD="${SERVER_CMD}& echo \$! > /tmp/server_attn_cf_${CF_MODE_TYPE}_${CF_MODE}.pid; wait"

echo "[INFO] 发送服务端启动命令（Attention-CF: ${CF_MODE_TYPE}）..."
tmux send-keys -t "$SERVER_TARGET" "$SERVER_CMD" C-m

if ! wait_for_server_port "$HOST" "$PORT" 600; then
    echo "[WARN] ⚠️ 服务端启动失败，退出。"
    stop_server "$CF_MODE_TYPE" "$CF_MODE"
    exit 1
fi

BASE_DIR="results/pi05_cf_attention_${CF_MODE_TYPE}_libero_${CF_MODE}"

for suite in "${TASK_SUITES[@]}"; do
    out_dir="${BASE_DIR}/${suite}"

    if [ -d "$out_dir" ]; then
        echo "[SKIP] ${suite} -> 输出目录 ${out_dir} 已存在，跳过该套件。"
        continue
    fi

    mkdir -p "$out_dir"
    echo "==== 运行任务套件: ${suite} (Attention-CF: ${CF_MODE_TYPE}) ===="

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

    if ! wait_for_client_finish 86400 "$out_dir"; then
        echo "[WARN] ⚠️ 套件 ${suite} 执行超时，继续下一个套件。"
    fi

    echo "[INFO] 套件 ${suite} 运行结束。日志已保存至 ${LOG_FILE}"
done

echo "================================================="
echo "[INFO] 🎉 所有 Attention-CF (${CF_MODE_TYPE}, Mode ${CF_MODE}) 任务执行完毕！"
echo "================================================="
echo "[INFO] 结果保存路径: results/pi05_cf_attention_${CF_MODE_TYPE}_libero_${CF_MODE}/"

# 停止服务端
stop_server "$CF_MODE_TYPE" "$CF_MODE"