# 反事实推理方法对比：输入级 CF vs 特征级 CF

本文档详细对比两种反事实推理（Counterfactual Reasoning）方法的实现原理、推理流程和具体运行代码。

---

## 一、概述

| 方法 | 参数 | 干预位置 | 干预对象 | 计算开销 |
|---|---|---|---|---|
| **输入级 CF** | `--use_cf_sampling` | Embed 之前 | 输入观测数据 | 较高（3× 全量推理） |
| **特征级 CF** | `--use_cf_feature_sampling` | VLM 输出之后 | prefix_tokens 特征 | 较高（3× 全量推理） |

两种方法都通过三分支对比计算效应，并按相同模式（B/C/D/E/F）进行重加权。

---

## 二、输入级 CF（Input-level CF）

### 2.1 核心原理

输入级 CF 通过**零化输入观测**来创建反事实场景：

```
原始观测 (observation)
    │
    ├─── 分支 1 (BASE): sample_actions(observation) → actions_base
    │
    ├─── 分支 2 (VLM_ZERO): 零化 images + prompt → sample_actions → actions_vlm0
    │
    └─── 分支 3 (STATE_ZERO): 零化 state → sample_actions → actions_state0
    │
    ↓
计算效应: effect_vlm = L2(actions_base - actions_vlm0)
         effect_state = L2(actions_base - actions_state0)
    │
    ↓
重加权: actions_final = actions_base + α·δ_vlm + w·δ_state
```

### 2.2 推理流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        输入级 CF 推理流程                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  输入: observation                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ images: {base_0_rgb: [H,W,3], wrist_0_rgb: [H,W,3]}                │    │
│  │ state: [s]  (机器人状态向量)                                        │    │
│  │ tokenized_prompt: [L]  (语言指令 tokens)                            │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌─────────────────────────────── 分支构造 ───────────────────────────┐    │
│  │                                                                     │    │
│  │  BASE 分支:                                                         │    │
│  │    observation → preprocess → embed_prefix → LLM → decode → actions │    │
│  │                                                                     │    │
│  │  VLM_ZERO 分支 (_make_cf_observation_image_zero):                   │    │
│  │    images = zeros_like(images)                                      │    │
│  │    prompt = None (可选)                                             │    │
│  │    state = 保持不变                                                  │    │
│  │    → preprocess → embed_prefix → LLM → decode → actions_vlm0       │    │
│  │                                                                     │    │
│  │  STATE_ZERO 分支 (_make_cf_observation_state_zero):                │    │
│  │    images = 保持不变                                                 │    │
│  │    prompt = 保持不变                                                 │    │
│  │    state = zeros_like(state)                                        │    │
│  │    → preprocess → embed_prefix → LLM → decode → actions_state0    │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌─────────────────────────────── 效应计算 ───────────────────────────┐    │
│  │                                                                     │    │
│  │  effect_vlm = mean(L2_norm(actions_base - actions_vlm0))            │    │
│  │  effect_state = mean(L2_norm(actions_base - actions_state0))        │    │
│  │                                                                     │    │
│  │  delta_vlm = actions_base - actions_vlm0                            │    │
│  │  delta_state = actions_base - actions_state0                        │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌─────────────────────────────── 重加权 ─────────────────────────────┐    │
│  │                                                                     │    │
│  │  if effect_vlm > threshold:                                        │    │
│  │      actions_final = actions_base  (回退到基线)                     │    │
│  │  else:                                                             │    │
│  │      actions_final = actions_base + α·delta_vlm + w·delta_state    │    │
│  │      (w 和 δ_state 的处理方式由 cf_mode 决定)                        │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 关键代码位置

**模型层** ([pi0.py](../src/openpi/models/pi0.py)):

```python
# 第 387-430 行: CF 观测构造方法
def _make_cf_observation_state_zero(self, observation):
    """零化本体感知 state"""
    return Observation(
        images=observation.images,          # 保持不变
        state=jnp.zeros_like(observation.state),  # 零化
        ...
    )

def _make_cf_observation_image_zero(self, observation, clear_prompt=True):
    """零化图像和语言"""
    zero_images = {name: jnp.zeros_like(image) for name, image in observation.images.items()}
    return Observation(
        images=zero_images,                 # 零化
        state=observation.state,            # 保持不变
        tokenized_prompt=None,              # 零化（可选）
        ...
    )

# 第 439-553 行: 主 CF 方法
def sample_actions_with_cf(self, rng, observation, cf_mode="E", ...):
    # 1. 三分支采样
    actions_base = self.sample_actions(rng, observation, ...)
    actions_prop0 = self.sample_actions(rng, obs_state_zero, ...)
    actions_img0 = self.sample_actions(rng, obs_image_zero, ...)
    
    # 2. 计算效应
    effect_prop = self._compute_cf_action_diff(actions_base, actions_prop0)
    effect_vlm = self._compute_cf_action_diff(actions_base, actions_img0)
    
    # 3. 按模式重加权
    delta_img = actions_base - actions_img0
    delta_prop = actions_base - actions_prop0
    actions_final = actions_base + cf_guidance_scale * delta_img + ...
```

---

## 三、特征级 CF（Feature-level CF）

### 3.1 核心原理

特征级 CF 通过**零化 VLM 输出特征**来创建反事实场景：

```
原始观测 (observation)
    │
    ↓ embed_prefix()
    │
prefix_tokens (VLM 输出特征)
    │
    ├─── 分支 1 (BASE): 使用原 prefix_tokens → decode → actions_base
    │
    ├─── 分支 2 (VLM_ZERO): prefix_tokens_vlm0 (零化 VLM 部分) → decode → actions_vlm0
    │
    └─── 分支 3 (STATE_ZERO): 
    │       pi05: prefix_tokens_state0 (零化 state tokens 部分)
    │       pi0:  suffix_tokens_state0 (零化 suffix 中 state token)
    │       → decode → actions_state0
    │
    ↓
计算效应 + 重加权 (同输入级 CF)
```

### 3.2 推理流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        特征级 CF 推理流程                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  输入: observation                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ images, state, tokenized_prompt                                    │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌───────────────────────── 第一步: Embed Prefix ─────────────────────┐    │
│  │                                                                     │    │
│  │  observation → preprocess                                          │    │
│  │                                                                     │    │
│  │  for each image:                                                   │    │
│  │      image_tokens = PaliGemma.img(image)  # SigLIP 编码            │    │
│  │      tokens.append(image_tokens)  # 每个 256 tokens                │    │
│  │                                                                     │    │
│  │  if tokenized_prompt:                                              │    │
│  │      language_tokens = PaliGemma.llm(prompt, method="embed")       │    │
│  │      tokens.append(language_tokens)                                │    │
│  │                                                                     │    │
│  │  prefix_tokens_base = concat(tokens)  # [B, P, E]                  │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌───────────────────────── 第二步: 计算边界 ─────────────────────────┐    │
│  │                                                                     │    │
│  │  bounds = _compute_prefix_token_bounds(observation, prefix_tokens) │    │
│  │                                                                     │    │
│  │  对于 Pi05:                                                         │    │
│  │    vlm_start = 0                                                    │    │
│  │    vlm_end = image_len + language_len  # VLM 特征边界              │    │
│  │    total_len = prefix_tokens.shape[1]                               │    │
│  │                                                                     │    │
│  │  对于 Pi0:                                                          │    │
│  │    vlm_start = 0                                                    │    │
│  │    vlm_end = total_len  # 整个 prefix 都是 VLM                      │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌───────────────────────── 第三步: 构造 CF 分支 ─────────────────────┐    │
│  │                                                                     │    │
│  │  BASE: prefix_tokens_base → kv_cache → decode → actions_base      │    │
│  │                                                                     │    │
│  │  VLM_ZERO:                                                          │    │
│  │    mask = zeros(vlm_start:vlm_end)                                  │    │
│  │    prefix_tokens_vlm0 = prefix_tokens_base * mask                  │    │
│  │    → kv_cache → decode → actions_vlm0                              │    │
│  │                                                                     │    │
│  │  STATE_ZERO:                                                        │    │
│  │    对于 Pi05:                                                       │    │
│  │      mask = zeros(state_start:state_end)                            │    │
│  │      prefix_tokens_state0 = prefix_tokens_base * mask              │    │
│  │      → decode → actions_state0                                     │    │
│  │    对于 Pi0:                                                        │    │
│  │      suffix_tokens_state0 = zero_first_token(suffix_tokens)        │    │
│  │      → decode → actions_state0                                     │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌───────────────────────── 第四步: 效应计算与重加权 ─────────────────┐    │
│  │                                                                     │    │
│  │  effect_vlm = L2(actions_base - actions_vlm0)                      │    │
│  │  effect_state = L2(actions_base - actions_state0)                  │    │
│  │                                                                     │    │
│  │  actions_final = actions_base + α·delta_vlm + w·delta_state        │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 关键代码位置

**模型层** ([pi0.py](../src/openpi/models/pi0.py)):

```python
# 第 555-698 行: 特征级 CF 实现

def _compute_prefix_token_bounds(self, observation, prefix_tokens):
    """计算 VLM 特征在 prefix_tokens 中的边界"""
    # 对于 Pi05: state 在 prefix_tokens 中（离散化）
    # 对于 Pi0: state 在 suffix_tokens 中（投影）
    ...

def _make_cf_prefix_tokens_vlm_zero(self, prefix_tokens, bounds):
    """零化 prefix_tokens 中的 VLM 部分"""
    mask = zeros(bounds["vlm_start"]:bounds["vlm_end"])
    return prefix_tokens * mask

def _sample_actions_with_modified_prefix(self, rng, observation, 
                                          prefix_tokens, prefix_mask, 
                                          zero_suffix_state=False):
    """使用修改后的 prefix_tokens 进行解码"""
    # 1. 用修改后的 prefix 构建 kv_cache
    _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], ...)
    
    # 2. Flow Matching 解码
    while time > 0:
        suffix_tokens = self.embed_suffix(observation, x_t, time)
        if zero_suffix_state:
            suffix_tokens = self._make_cf_suffix_tokens_state_zero(suffix_tokens)
        v_t = self.PaliGemma.llm([None, suffix_tokens], kv_cache=kv_cache)
        x_t = x_t + dt * v_t
    ...

def sample_actions_with_feature_cf(self, rng, observation, cf_mode="E", ...):
    # 1. 预处理并获取原始 prefix_tokens
    observation = preprocess(observation)
    prefix_tokens_base, prefix_mask, _ = self.embed_prefix(observation)
    bounds = self._compute_prefix_token_bounds(observation, prefix_tokens_base)
    
    # 2. 三分支采样
    actions_base = self._sample_actions_with_modified_prefix(..., prefix_tokens_base)
    prefix_tokens_vlm0 = self._make_cf_prefix_tokens_vlm_zero(prefix_tokens_base, bounds)
    actions_vlm0 = self._sample_actions_with_modified_prefix(..., prefix_tokens_vlm0)
    
    # State zero 分支 (根据 pi05/pi0 不同处理)
    if self.pi05:
        prefix_tokens_state0 = ...  # 零化 prefix 中 state tokens
        actions_state0 = self._sample_actions_with_modified_prefix(..., prefix_tokens_state0)
    else:
        actions_state0 = self._sample_actions_with_modified_prefix(..., zero_suffix_state=True)
    
    # 3. 效应计算与重加权
    ...
```

---

## 四、服务端与客户端代码

### 4.1 服务端代码

**入口脚本** ([serve_policy.py](../scripts/serve_policy.py)):

```python
@dataclasses.dataclass
class Args:
    # CF 配置参数
    use_cf_sampling: bool = False           # 输入级 CF
    use_cf_feature_sampling: bool = False   # 特征级 CF
    
    cf_mode: CfMode = CfMode.E              # 重加权模式
    cf_guidance_scale: float = 0.1          # VLM 引导强度
    vlm_effect_upper_threshold: float = 0.5 # VLM 效应阈值
    clear_prompt_for_image_cf: bool = True  # 输入级 CF 参数


def create_policy(args: Args) -> Policy:
    sample_kwargs = None
    if args.use_cf_sampling or args.use_cf_feature_sampling:
        sample_kwargs = {
            "cf_mode": args.cf_mode.value,
            "cf_guidance_scale": args.cf_guidance_scale,
            "vlm_effect_upper_threshold": args.vlm_effect_upper_threshold,
            ...
        }
    
    return create_trained_policy(
        train_config,
        checkpoint_dir,
        use_cf_sampling=args.use_cf_sampling,
        use_cf_feature_sampling=args.use_cf_feature_sampling,
        sample_kwargs=sample_kwargs,
    )


def main(args: Args):
    policy = create_policy(args)
    server = WebsocketPolicyServer(policy, host="0.0.0.0", port=args.port)
    server.serve_forever()
```

**启动命令**:

```bash
# 输入级 CF
python scripts/serve_policy.py \
    --env LIBERO \
    --use_cf_sampling \
    --cf_mode E \
    --cf_guidance_scale 0.1 \
    --port 8888

# 特征级 CF
python scripts/serve_policy.py \
    --env LIBERO \
    --use_cf_feature_sampling \
    --cf_mode E \
    --cf_guidance_scale 0.1 \
    --port 8888
```

### 4.2 Policy 路由代码

**Policy 类** ([policy.py](../src/openpi/policies/policy.py)):

```python
class Policy(BasePolicy):
    def __init__(self, model, use_cf_sampling=False, use_cf_feature_sampling=False, ...):
        # 选择采样函数
        if use_cf_feature_sampling and hasattr(model, "sample_actions_with_feature_cf"):
            sample_fn = model.sample_actions_with_feature_cf
        elif use_cf_sampling and hasattr(model, "sample_actions_with_cf"):
            sample_fn = model.sample_actions_with_cf
        else:
            sample_fn = model.sample_actions
        
        # JIT 编译
        self._sample_actions = nnx_utils.module_jit(sample_fn)
    
    def infer(self, obs: dict) -> dict:
        inputs = self._input_transform(obs)
        observation = Observation.from_dict(inputs)
        actions = self._sample_actions(rng, observation, **self._sample_kwargs)
        outputs = self._output_transform({"actions": actions})
        return outputs
```

### 4.3 客户端代码

**LIBERO 客户端** ([examples/libero/main.py](../examples/libero/main.py)):

```python
from openpi_client import WebsocketClientPolicy

def eval_libero(args):
    # 创建 WebSocket 客户端
    client = WebsocketClientPolicy(host=args.host, port=args.port)
    
    for task_id in range(num_tasks):
        env, task_description = get_libero_env(task)
        
        for episode_idx in range(num_trials):
            obs = env.reset()
            action_plan = deque()
            
            while t < max_steps:
                # 图像预处理
                img = obs["agentview_image"][::-1, ::-1]
                wrist_img = obs["robot0_eye_in_hand_image"][::-1, ::-1]
                img = resize_with_pad(img, 224, 224)
                wrist_img = resize_with_pad(wrist_img, 224, 224)
                
                if not action_plan:
                    # 构建观测字典
                    element = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": np.concatenate([
                            obs["robot0_eef_pos"],
                            quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        ]),
                        "prompt": task_description,
                    }
                    
                    # 发送请求并接收动作
                    result = client.infer(element)
                    action_chunk = result["actions"]
                    action_plan.extend(action_chunk[:replan_steps])
                
                action = action_plan.popleft()
                obs, reward, done, info = env.step(action)
                
                if done:
                    break
```

**启动命令**:

```bash
python examples/libero/main.py \
    --args.host 127.0.0.1 \
    --args.port 8888 \
    --args.task_suite_name libero_spatial \
    --args.video_out_path data/cf_libero_results
```

### 4.4 运行脚本

**run_libero.sh**:

```bash
#!/bin/bash

SESSION="pi05-libero"
SERVER_GPU=4
CLIENT_GPU=4
MODES=("B" "C" "D" "F")
TASK_SUITES=(libero_spatial libero_object libero_goal libero_10)

for MODE in "${MODES[@]}"; do
    # 启动服务端
    SERVER_CMD="python scripts/serve_policy.py \
        --env LIBERO --use_cf_sampling --cf_mode ${MODE}"
    tmux send-keys -t "$SERVER_TARGET" "$SERVER_CMD" C-m
    
    # 等待端口就绪
    wait_for_server_port 8888
    
    # 运行客户端任务
    for suite in "${TASK_SUITES[@]}"; do
        CLIENT_CMD="python examples/libero/main.py \
            --args.task_suite_name ${suite} \
            --args.video_out_path data/pi05_cf_libero_${MODE}/${suite}"
        tmux send-keys -t "$CLIENT_TARGET" "$CLIENT_CMD" C-m
        wait_for_client_finish
    done
    
    # 关闭服务端
    stop_server "$MODE"
done
```

---

## 五、两种方法的差异对比

### 5.1 理论差异

| 维度 | 输入级 CF | 特征级 CF |
|---|---|---|
| **干预层级** | 输入层（观测数据） | 特征层（VLM 输出） |
| **干预时机** | Embed 之前 | Embed 之后 |
| **VLM 编码次数** | 3 次（每个分支独立） | 3 次（但可共享 prefix cache） |
| **特征空间** | 原始图像/语言空间 | 高维特征空间 |
| **理论意义** | "如果看不到图像会怎样" | "如果 VLM 特征无效会怎样" |

### 5.2 实现差异

```python
# 输入级 CF: 零化输入观测
def _make_cf_observation_image_zero(observation):
    zero_images = {name: zeros_like(image) for name, image in observation.images}
    return Observation(images=zero_images, state=observation.state, ...)

# 特征级 CF: 零化 VLM 输出特征
def _make_cf_prefix_tokens_vlm_zero(prefix_tokens, bounds):
    mask = zeros(bounds["vlm_start"]:bounds["vlm_end"])
    return prefix_tokens * mask  # 直接修改特征
```

### 5.3 适用场景

| 场景 | 推荐方法 |
|---|---|
| 分析输入模态重要性 | 输入级 CF |
| 分析模型内部特征依赖 | 特征级 CF |
| 研究 VLM 编码质量 | 特征级 CF |
| 部署到生产环境 | 输入级 CF（更稳定） |

---

## 六、CF 重加权模式详解

两种 CF 方法共用相同的重加权模式（`CfMode`）：

| 模式 | 公式 | 说明 |
|---|---|---|
| **BASE** | `actions_base` | 无 CF 干扰 |
| **A** | `base + α·δ_vlm` | 仅 VLM 引导 |
| **B** | `base + α·δ_vlm + 0.05·δ_state` | VLM + 固定权重 state |
| **C** | `base + α·(w_vlm·δ_vlm + w_state·0.1·δ_state_norm)` | 效应比例加权 |
| **D** | `base + α·δ_vlm + 0.05·clip(δ_state)` | VLM + 硬截断 state |
| **E** | `base + α·δ_vlm + prop_weight·clip(δ_state)` | 自适应 state 权重 |
| **F** | `base + α·δ_vlm + 0.05·tanh(δ_state)` | VLM + 柔截断 state |

**通用回退机制**: 若 `effect_vlm > vlm_effect_upper_threshold`，直接返回 `actions_base`。

---

## 七、CF 权重记录

### 7.1 返回的 metrics 字段

当启用 CF sampling 时，服务端会返回完整的权重信息：

| 字段 | 说明 |
|---|---|
| `effect_vlm` | VLM 效应 L2 范数 |
| `effect_prop` / `effect_state` | 本体感知效应 L2 范数 |
| `use_base` | 是否回退到基线动作 |
| `cf_mode` | CF 重加权模式 |
| `cf_level` | CF 类型（"input" 或 "feature"） |
| `effect_ratio_vlm` | VLM 效应占比 |
| `effect_ratio_prop` / `effect_ratio_state` | 本体感知效应占比 |
| `actual_vlm_weight` | 实际应用的 VLM 权重 |
| `actual_prop_weight` / `actual_state_weight` | 实际应用的本体感知权重 |
| `delta_vlm_norm` | VLM delta 的 L2 范数 |
| `delta_prop_norm` / `delta_state_norm` | 本体感知 delta 的 L2 范数 |
| `cf_guidance_scale` | CF 引导强度参数 |
| `vlm_effect_threshold` | VLM 效应阈值参数 |

### 7.2 客户端保存格式

CF metrics 会保存在每个 episode 的 JSON 文件中：

```json
{
  "task_description": "pick up the black mug...",
  "episode_idx": 0,
  "success": true,
  "cf_metrics_history": [
    {
      "step": 10,
      "effect_vlm": 0.35,
      "effect_state": 0.15,
      "use_base": false,
      "cf_mode": "E",
      "cf_level": "input",
      "effect_ratio_vlm": 0.70,
      "effect_ratio_state": 0.30,
      "actual_vlm_weight": 0.10,
      "actual_state_weight": 0.05,
      "delta_vlm_norm": 0.42,
      "delta_state_norm": 0.18,
      "cf_guidance_scale": 0.1,
      "vlm_effect_threshold": 0.5
    },
    // ... 每个 replan 步骤的 metrics
  ]
}
```

### 7.3 日志输出示例

运行时会实时打印 CF 权重信息：

```
[INFO] CF metrics at step 10: effect_vlm=0.3500, effect_state=0.1500, vlm_weight=0.1000, state_weight=0.0500
```

---

## 八、参考文件

| 文件 | 作用 |
|---|---|
| [pi0.py](../src/openpi/models/pi0.py) | CF 方法实现 |
| [policy.py](../src/openpi/policies/policy.py) | Policy 路由逻辑 |
| [policy_config.py](../src/openpi/policies/policy_config.py) | Policy 创建函数 |
| [serve_policy.py](../scripts/serve_policy.py) | 服务端 CLI |
| [main.py](../examples/libero/main.py) | LIBERO 客户端 |
| [run_libero.sh](../run_libero.sh) | 运行调度脚本 |