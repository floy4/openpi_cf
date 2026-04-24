#!/usr/bin/env bash

set -euo pipefail

# ==========================================
# 特征级反事实推理评估脚本
# ==========================================
#
# 本脚本使用特征级 CF（Feature-level CF）进行评估：
# - 干预位置：VLM 输出特征（prefix_tokens）
# - 零化内容：VLM 特征部分（image + language tokens）
# - 与输入级 CF 的区别：直接修改模型内部特征而非输入观测
#
# 对比脚本：
#   - run_libero.sh: 输入级 CF (--use_cf_sampling)
#   - run_libero_feature_cf.sh: 特征级 CF (--use_cf_feature_sampling)
#
# ==========================================

# 基础配置
SESSION="pi05-libero-feature-cf"
SERVER_WIN="server"
CLIENT_WIN="client"

# Python 环境（使用 openvla 环境）
PYTHON="/home/zhy/miniconda3/envs/openvla/bin/python"

# 硬件与网络配置
SERVER_GPU=2
CLIENT_GPU=2
HOST="127.0.0.1"
PORT=8889  # 使用不同端口避免冲突

SERVER_DIR="$PWD"
CLIENT_DIR="$PWD"

# 评估模式与任务套件
# 特征级 CF 使用与输入级 CF 相同的重加权模式
MODES=("A" "B" "C" "D")
TASK_SUITES=(
    libero_spatial
    libero_object
    libero_goal
    libero_10
)

You are an educational dialogue trajectory constructor. Given a question, a student's reasoning, a student's answer, and a teacher's feedback, your sole task is to rewrite the teacher's feedback as the student's own first-person self-reflection. You must output only a JSON object with no explanation or commentary.

**Input:​**
- Question: 如图所示，在△ABC中，D点在AB边，从‖BC在AC上到E点，AE=\\ frac{1}{3}AC，
如果分段BC=30，则分段的长度为（）\nA. 15\nB. 20\nC. 5\nD. 10\nAnswer with the option's letter from the given choices directly.
- Student's Thinking: Since AE is one-third of AC, triangle ADE is similar to triangle ABC with a ratio of 1:3. Therefore, DE should be one-third of BC.
- Student's Answer: C
- Teacher's Feedback: You are on the right track by identifying the similarity between triangles ADE and ABC. However, you made an error in calculating the length of DE. If the ratio is 1:3, DE should be one-third of BC, but your calculation seems off. Please recheck your calculation.

**Task:​**
Rewrite the teacher's feedback as the student's internal self-correction monologue in first person. The reflection must directly continue from the student's own thinking process — identify exactly where the reasoning in Student's Thinking went wrong, explain why that step was incorrect, and redirect toward the right approach. The reflection should read as a natural continuation of the student's thought process, not as an external correction imposed from outside.

**Rules:​**
1. Write strictly in first person using "I" as the subject, as if the student is thinking aloud.
2. Explicitly reference the specific reasoning step in Student's Thinking that led to the wrong answer, then correct it. Do not write a generic reflection that could apply to any student.
3. Do not use the teacher's voice or any second-person language (e.g., "you should...") or anything about teacher or student.
4. Remove all scores or evaluative language (e.g., "Score: 3", "Well done", "Perfect").
5. The transition from Student's Thinking to Reflection must feel seamless — the reflection picks up where the thinking left off and steers it in the right direction.
6. Output only a single JSON object.

**Output:​**
{
  "reflection": "..."
}


# 特征级 CF 特有参数
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
    echo "[INFO] 停止服务端进程 (Mode: ${mode}, Feature-level CF)..."
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

    # 精准匹配当前 mode 和 feature-cf 的 Server 进程
    if [[ -n "$mode" ]] && [[ -f "/tmp/server_feature_cf_${mode}.pid" ]]; then
        local pid
        pid=$(cat "/tmp/server_feature_cf_${mode}.pid")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            echo "[INFO] 已终止 Server PID: $pid"
        fi
        rm -f "/tmp/server_feature_cf_${mode}.pid"
    else
        # 兜底：按命令行模式匹配清理
        pkill -f "serve_policy.py --env LIBERO --use_cf_feature_sampling --cf_mode ${mode}" >/dev/null 2>&1 || true
    fi

    sleep 2
}

wait_for_client_finish() {
    local timeout_sec=${1:-86400}
    local out_dir=${2:-""}  # 输出目录参数，用于精确匹配当前客户端
    local elapsed=0
    local grace_period=30  # 等待客户端启动的宽限期
    local client_started=false

    echo "[INFO] 等待客户端任务执行完毕（最长 ${timeout_sec} 秒）..."
    echo "[INFO] 监控输出目录: ${out_dir}"

    # 先等待客户端启动（宽限期）
    while (( elapsed < grace_period )); do
        # 使用输出目录精确匹配
        if [[ -n "$out_dir" ]]; then
            if pgrep -f "python -u examples/libero/main.py.*${out_dir}" >/dev/null 2>&1; then
                client_started=true
                echo "[INFO] ✅ 检测到客户端进程已启动。"
                break
            fi
        else
            if pgrep -f "python -u examples/libero/main.py.*--args.port ${PORT}" >/dev/null 2>&1; then
                client_started=true
                echo "[INFO] ✅ 检测到客户端进程已启动。"
                break
            fi
        fi
        sleep 2
        elapsed=$(( elapsed + 2 ))
    done

    if [[ "$client_started" == "false" ]]; then
        echo "[ERROR] ❌ 客户端进程未在 ${grace_period} 秒内启动，可能启动失败。"
        return 1
    fi

    # 等待客户端完成
    while (( elapsed < timeout_sec )); do
        if [[ -n "$out_dir" ]]; then
            if ! pgrep -f "python -u examples/libero/main.py.*${out_dir}" >/dev/null 2>&1; then
                echo "[INFO] ✅ 当前客户端任务已完成（总耗时: ${elapsed} 秒）。"
                return 0
            fi
        else
            if ! pgrep -f "python -u examples/libero/main.py.*--args.port ${PORT}" >/dev/null 2>&1; then
                echo "[INFO] ✅ 当前客户端任务已完成（总耗时: ${elapsed} 秒）。"
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
# 前置运行检查（自动创建会话和窗口）
# ==========================================

# 确保 tmux 会话存在（添加错误处理防止脚本退出）
ensure_tmux_session "$SESSION" || {
    echo "[ERROR] tmux 会话创建失败，退出脚本。"
    exit 1
}

# 确保 server 和 client 窗口存在
ensure_tmux_windows "$SESSION" "$SERVER_WIN" "$CLIENT_WIN" || {
    echo "[ERROR] tmux 窗口创建失败，退出脚本。"
    exit 1
}

echo "[INFO] Python 环境: ${PYTHON}"
echo "[INFO] tmux 会话: ${SESSION}"
echo "[INFO] Server 窗口: ${SERVER_TARGET}"
echo "[INFO] Client 窗口: ${CLIENT_TARGET}"

# ==========================================
# 主干调度流程
# ==========================================

echo "================================================="
echo "[INFO] 特征级反事实推理评估 (Feature-level CF)"
echo "[INFO] 干预位置: VLM 输出特征 (prefix_tokens)"
echo "[INFO] CF 模式: ${MODES[*]}"
echo "[INFO] 服务端口: ${PORT}"
echo "================================================="

for MODE in "${MODES[@]}"; do
    echo "================================================="
    echo "[INFO] 开始评估模式: Mode ${MODE} (Feature-level CF)"
    echo "================================================="

    # 启动前先清理上一轮可能残留的进程和窗口输入
    stop_server "$MODE"
    clear_window_cmdline "$SERVER_TARGET"
    clear_window_cmdline "$CLIENT_TARGET"

    # 构建并发送 Server 启动命令（使用特征级 CF）
    # 关键参数: --use_cf_feature_sampling (而非 --use_cf_sampling)
    SERVER_CMD="cd ${SERVER_DIR} && CUDA_VISIBLE_DEVICES=${SERVER_GPU} ${PYTHON} scripts/serve_policy.py \
--env LIBERO \
--use_cf_feature_sampling \
--cf_mode ${MODE} \
--cf_guidance_scale ${CF_GUIDANCE_SCALE} \
--vlm_effect_upper_threshold ${VLM_EFFECT_THRESHOLD} \
--reweight_action_with_cf \
--return_cf_metrics \
--port ${PORT} \
& echo \$! > /tmp/server_feature_cf_${MODE}.pid; wait"

    echo "[INFO] 发送服务端启动命令（特征级 CF）..."
    echo "[INFO] 命令: --use_cf_feature_sampling --cf_mode ${MODE}"
    tmux send-keys -t "$SERVER_TARGET" "$SERVER_CMD" C-m

    # 阻塞等待 Server 端口就绪
    if ! wait_for_server_port "$HOST" "$PORT" 600; then
        echo "[WARN] ⚠️ 放弃当前 Mode ${MODE}，跳过执行下一个模式。"
        stop_server "$MODE"
        continue
    fi

    # 遍历所有任务套件
    BASE_DIR="results/pi05_feature_cf_libero_${MODE}"

    for suite in "${TASK_SUITES[@]}"; do
        out_dir="${BASE_DIR}/${suite}"

        if [ -d "$out_dir" ]; then
            # 检查是否有有效的结果文件（避免跳过已失败的运行）
            if [ -f "${out_dir}/results.txt" ]; then
                echo "[SKIP] ${suite} -> 输出目录 ${out_dir} 已存在且有结果，跳过该套件。"
                continue
            else
                echo "[INFO] ${suite} -> 输出目录已存在但无结果文件，重新运行。"
                rm -rf "${out_dir}"
            fi
        fi

        mkdir -p "$out_dir"
        echo "==== 运行任务套件: ${suite} (Mode ${MODE}, Feature-level CF) ===="
        echo "[INFO] 输出目录: ${out_dir}"

        LOG_FILE="${BASE_DIR}/${suite}/run.log"

        # 构建 Client 命令（连接到特征级 CF 服务端）
        CLIENT_CMD="export MUJOCO_GL=egl; \
export PYOPENGL_PLATFORM=egl; \
export CUDA_VISIBLE_DEVICES=${CLIENT_GPU}; \
cd ${CLIENT_DIR} && ${PYTHON} -u examples/libero/main.py \
--args.host ${HOST} \
--args.port ${PORT} \
--args.task_suite_name ${suite} \
--args.video_out_path ${out_dir} \
2>&1 | tee ${LOG_FILE}"

        echo "[INFO] 发送客户端执行命令..."
        tmux send-keys -t "$CLIENT_TARGET" "$CLIENT_CMD" C-m

        # 等待当前 suite 跑完再发下一个
        if ! wait_for_client_finish 86400 "$out_dir"; then
            echo "[WARN] ⚠️ 套件 ${suite} 执行失败或超时，继续下一个套件。"
            # 记录失败状态
            echo "FAILED: ${suite} - timeout or client startup failure" >> "${BASE_DIR}/failed_tasks.txt"
        else
            echo "[INFO] 套件 ${suite} 运行结束。日志已保存至 ${LOG_FILE}"
            # 检查是否有结果文件
            if [ -f "${out_dir}/results.txt" ]; then
                echo "[INFO] ✅ 结果文件已生成: ${out_dir}/results.txt"
            else
                echo "[WARN] ⚠️ 未找到结果文件，任务可能未正常完成。"
                echo "NO_RESULTS: ${suite}" >> "${BASE_DIR}/failed_tasks.txt"
            fi
        fi
    done

    # 当前 Mode 所有 suite 执行完毕，关闭 Server
    echo "[INFO] 模式 ${MODE} 的所有套件评估完毕，准备关闭 Server..."
    stop_server "$MODE"
    sleep 3

done

echo "================================================="
echo "[INFO] 🎉 所有特征级 CF 模式与任务套件执行完毕！"
echo "================================================="
echo "[INFO] 结果保存路径: results/pi05_feature_cf_libero_{MODE}/"
echo ""

# 汇总失败任务
if [ -f "${BASE_DIR}/../failed_tasks_summary.txt" ]; then
    echo "[INFO] 失败任务汇总:"
    cat "${BASE_DIR}/../failed_tasks_summary.txt"
else
    echo "[INFO] 所有任务均已完成。"
fi

echo ""
echo "[INFO] 对比两种 CF 方法的结果："
echo "  输入级 CF: results/pi05_cf_libero_{MODE}/"
echo "  特征级 CF: results/pi05_feature_cf_libero_{MODE}/"