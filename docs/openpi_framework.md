# OpenPI 框架架构详解

OpenPI 是一个基于 **VLA (Vision-Language-Action)** 的机器人学习框架，实现了 Pi0 系列模型。本文档详细介绍框架的各个模块、数据流和关键变量。

---

## 一、整体架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                        推理/训练流程                              │
│                                                                 │
│  环境观测 → 数据变换 → 模型输入 → 模型推理 → 动作输出 → 逆变换    │
│            (transforms)   (Observation)  (Actions)  (transforms) │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、核心模块详解

### 1. 模型层 (Models)

**位置:** `src/openpi/models/`

#### 1.1 基础类 `BaseModel`

**文件:** `model.py`

所有模型的基类，定义了统一的接口：

```python
class BaseModel(nnx.Module):
    action_dim: int        # 动作空间维度 (如32维)
    action_horizon: int    # 动作序列长度 (如50帧)
    max_token_len: int     # 文本prompt最大长度
```

**核心方法：**

| 方法 | 输入 | 输出 | 用途 |
|------|------|------|------|
| `compute_loss(rng, observation, actions)` | 随机数、观测、动作 | `Float[Array, "*b ah"]` | 训练时计算损失 |
| `sample_actions(rng, observation, **kwargs)` | 随机数、观测 | `Actions` | 推理时采样动作 |

#### 1.2 `Observation` 数据结构

**文件:** `model.py`

模型的标准化输入数据结构：

```python
@struct.dataclass
class Observation(Generic[ArrayT]):
    # 必需字段
    images: dict[str, Float[ArrayT, "*b h w c"]]     # 多相机图像, 范围[-1,1]
    image_masks: dict[str, Bool[ArrayT, "*b"]]       # 图像有效性掩码
    state: Float[ArrayT, "*b s"]                     # 低维机器人状态 (关节角度等)
    
    # 可选字段 - 语言prompt
    tokenized_prompt: Int[ArrayT, "*b l"] | None     # token化的语言指令
    tokenized_prompt_mask: Bool[ArrayT, "*b l"] | None
    
    # FAST模型专用字段
    token_ar_mask: Int[ArrayT, "*b l"] | None        # 自回归掩码
    token_loss_mask: Bool[ArrayT, "*b l"] | None     # 损失掩码
```

**预期图像键名:**

```python
IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
IMAGE_RESOLUTION = (224, 224)  # 统一 resize 到这个尺寸
```

**从字典创建 Observation:**

```python
# 原始数据格式
data = {
    "image": {
        "base_0_rgb": np.array([h, w, 3]),      # uint8 或 float32[-1,1]
        "left_wrist_0_rgb": np.array([h, w, 3]),
        ...
    },
    "image_mask": {
        "base_0_rgb": bool,                     # True表示图像有效
        ...
    },
    "state": np.array([s]),                     # 机器人状态
    "tokenized_prompt": np.array([l]),          # 可选
    "tokenized_prompt_mask": np.array([l]),     # 可选
}

# 转换为 Observation 对象
observation = Observation.from_dict(data)
```

#### 1.3 `Actions` 数据类型

```python
Actions = Float[ArrayT, "*b ah ad"]
# 形状: batch_size × action_horizon × action_dim
```

#### 1.4 Pi0 模型

**文件:** `pi0.py`

**核心架构:** PaliGemma (视觉语言) + Action Expert (动作预测)

```python
class Pi0(BaseModel):
    PaliGemma:
        ├── llm: nnx_bridge.ToNNX    # Gemma LLM (双配置：paligemma + action_expert)
        └── img: nnx_bridge.ToNNX    # SigLIP 视觉编码器
    
    # 投影层
    action_in_proj: nnx.Linear       # action_dim → action_expert_width
    state_proj: nnx.Linear           # state → action_expert_width (非pi05)
    time_mlp_in: nnx.Linear          # 时间步MLP (pi05用)
    time_mlp_out: nnx.Linear         # 时间步MLP (pi05用)
    action_time_mlp_in: nnx.Linear   # 非pi05用
    action_time_mlp_out: nnx.Linear  # 非pi05用
    action_out_proj: nnx.Linear      # action_expert_width → action_dim
```

**Pi0 的两种模式对比:**

| 特性 | Pi0 | Pi05 |
|------|-----|------|
| state输入位置 | 连续的suffix部分 | 离散的language tokens |
| 时间步注入方式 | action_time_mlp | adaRMSNorm |
| 默认max_token_len | 48 | 200 |
| discrete_state_input | False | True |

#### 1.5 Pi0Config

**文件:** `pi0_config.py`

模型配置类：

```python
@dataclass
class Pi0Config(BaseModelConfig):
    dtype: str = "bfloat16"                           # 计算精度
    paligemma_variant: str = "gemma_2b"               # LLM变体
    action_expert_variant: str = "gemma_300m"         # action expert变体
    action_dim: int = 32                              # 动作维度
    action_horizon: int = 50                          # 动作序列长度
    max_token_len: int = 48                           # 最大token数
    pi05: bool = False                                # 是否用pi05架构
    discrete_state_input: bool = False                # state是否离散化输入
```

**LoRA 配置变体:**

```python
# LoRA 微调配置
Pi0Config(
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora"
)
```

---

### 2. 策略层 (Policies)

**位置:** `src/openpi/policies/`

#### 2.1 Policy 类

**文件:** `policy.py`

封装模型推理的完整流程：

```python
class Policy(BasePolicy):
    _model: BaseModel                      # 底层模型
    _input_transform: DataTransformFn      # 输入变换链
    _output_transform: DataTransformFn     # 输出变换链
    _sample_kwargs: dict                   # 采样参数 (如num_steps)
    _is_pytorch_model: bool                # 是否PyTorch模型
    _use_cf_sampling: bool                 # 是否用反事实采样
```

**核心方法 `infer(obs) → dict`:**

```
原始观测 → input_transform → Observation → sample_actions → Actions → output_transform → 最终动作
```

**输入输出规范:**

| 方向 | 字段 | 类型 | 说明 |
|------|------|------|------|
| 输入 | `image` | dict[str, array] | 多相机图像 |
| 输入 | `image_mask` | dict[str, bool] | 图像有效性 |
| 输入 | `state` | array | 机器人状态 |
| 输入 | `prompt` | str | 语言指令 (可选) |
| 输出 | `actions` | array[ah, ad] | 动作序列 |
| 输出 | `state` | array | 状态副本 |
| 输出 | `policy_timing` | dict | 推理时间统计 |

#### 2.2 Policy 创建

**文件:** `policy_config.py`

```python
def create_trained_policy(
    train_config: TrainConfig,           # 训练配置
    checkpoint_dir: str,                 # 检查点目录
    repack_transforms: Group | None,     # 数据重打包变换
    sample_kwargs: dict | None,          # 采样参数 {num_steps: 10}
    default_prompt: str | None,          # 默认prompt
    norm_stats: dict | None,             # 归一化统计
    pytorch_device: str | None,          # PyTorch设备
    use_cf_sampling: bool,               # 是否使用反事实采样
) -> Policy
```

**自动检测模型类型:**

- 存在 `model.safetensors` → PyTorch 模型
- 否则 → JAX 模型

---

### 3. 数据变换层 (Transforms)

**位置:** `src/openpi/transforms.py`

#### 3.1 核心概念

```python
@dataclass
class Group:
    inputs: Sequence[DataTransformFn]   # 推理和训练都用
    outputs: Sequence[DataTransformFn]  # 仅推理时用 (反变换)
```

**变换链顺序:**

```
inputs: repack → data_transforms → normalize → model_transforms
outputs: model_transforms → unnormalize → data_transforms → repack
```

#### 3.2 主要变换类型

| 变换类 | 功能 | 输入→输出 |
|--------|------|-----------|
| `RepackTransform` | 重命名字段 | 数据集格式→统一格式 |
| `InjectDefaultPrompt` | 注入默认prompt | 无prompt时添加 |
| `ResizeImages` | 调整图像大小 | 任意尺寸→224×224 |
| `Normalize` | 数据归一化 | 原始值→归一化值 |
| `Unnormalize` | 反归一化 | 归一化值→原始值 |
| `TokenizePrompt` | 文本token化 | 字符串→token IDs |
| `DeltaActions` | 绝对→相对动作 | 绝对坐标→相对坐标 |
| `AbsoluteActions` | 相对→绝对动作 | 相对坐标→绝对坐标 |
| `PadStatesAndActions` | 填充维度 | 任意维度→action_dim |
| `SubsampleActions` | 动作子采样 | 原序列→稀疏序列 |

#### 3.3 归一化类型

**Z-score 归一化:**

```python
def _normalize(x, stats):
    mean, std = stats.mean, stats.std
    return (x - mean) / (std + 1e-6)
```

**分位数归一化 (Quantile):**

```python
def _normalize_quantile(x, stats):
    q01, q99 = stats.q01, stats.q99
    return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
    # 输出范围: [-1, 1]
```

#### 3.4 数据流示例

```python
# 推理时的完整变换链
input_transforms = [
    RepackTransform({
        "images": {"cam_high": "observation.images.top"},
        "state": "observation.state",
        "actions": "action",
    }),
    InjectDefaultPrompt("fold towel"),
    ResizeImages(224, 224),
    Normalize(norm_stats, use_quantiles=True),
    TokenizePrompt(tokenizer),
    PadStatesAndActions(action_dim=32),
]

output_transforms = [
    Unnormalize(norm_stats, use_quantiles=True),
    AbsoluteActions(mask=make_bool_mask(6, -1, 6, -1)),
]
```

---

### 4. 训练层 (Training)

**位置:** `src/openpi/training/`

#### 4.1 TrainConfig

**文件:** `config.py`

训练的核心配置：

```python
@dataclass
class TrainConfig:
    name: str                              # 配置名称 (如 "pi0_libero")
    project_name: str = "openpi"           # 项目名
    exp_name: str                          # 实验名称
    
    # 模型配置
    model: BaseModelConfig                 # Pi0Config 或 Pi0FASTConfig
    
    # 权重加载
    weight_loader: WeightLoader            # 预训练权重加载器
    pytorch_weight_path: str | None        # PyTorch权重路径
    
    # 训练参数
    lr_schedule: LRScheduleConfig          # 学习率调度
    optimizer: OptimizerConfig             # AdamW 等
    ema_decay: float | None = 0.99         # EMA衰减率
    freeze_filter: Filter                  # LoRA冻结过滤
    
    # 数据配置
    data: DataConfigFactory                # 数据配置工厂
    
    # 训练设置
    batch_size: int = 32                   # 全局batch size
    num_train_steps: int = 30_000          # 训练步数
    num_workers: int = 2                   # DataLoader工作数
    seed: int = 42                         # 随机种子
    
    # 保存设置
    log_interval: int = 100                # 日志间隔
    save_interval: int = 1000              # 保存间隔
    keep_period: int | None = 5000         # 保留周期
    
    # 分布式
    fsdp_devices: int = 1                  # FSDP设备数
```

#### 4.2 DataConfig

```python
@dataclass
class DataConfig:
    repo_id: str | None                    # LeRobot数据集ID (如 "physical-intelligence/libero")
    asset_id: str | None                   # 资产ID (如 "trossen")
    norm_stats: dict[NormStats] | None     # 归一化统计
    
    # 变换组
    repack_transforms: Group               # 数据重打包
    data_transforms: Group                 # 数据变换 (机器人特定)
    model_transforms: Group                # 模型变换 (token化等)
    
    # 配置
    use_quantile_norm: bool = False        # 是否用分位数归一化
    action_sequence_keys: Sequence[str]    # 动作序列键
    prompt_from_task: bool                 # 从任务提取prompt
    
    # RLDS专用
    rlds_data_dir: str | None              # RLDS数据目录
    action_space: DroidActionSpace | None  # 动作空间类型
    datasets: Sequence[RLDSDataset]        # 数据集列表
```

#### 4.3 DataLoader

**文件:** `data_loader.py`

**两种数据源:**

| 数据源 | DataLoader类 | 适用场景 |
|--------|--------------|----------|
| LeRobot格式 | `TorchDataLoader` | 小数据集 (<10小时) |
| RLDS格式 | `RLDSDataLoader` | 大数据集 (如DROID) |

**数据加载流程:**

```
原始数据集 
    → repack_transforms (重映射键名)
    → data_transforms (机器人特定变换)
    → Normalize (归一化)
    → model_transforms (token化等)
    → (Observation, Actions)
```

#### 4.4 NormStats 归一化统计

```python
@dataclass
class NormStats:
    mean: np.ndarray          # 均值
    std: np.ndarray           # 标准差
    q01: np.ndarray | None    # 1%分位数 (分位数归一化用)
    q99: np.ndarray | None    # 99%分位数
```

---

### 5. Flow Matching 动作采样

**文件:** `pi0.py`

Pi0 使用 **Flow Matching** 方法生成动作序列。

#### 5.1 训练损失计算

```python
def compute_loss(self, rng, observation, actions):
    # 1. 采样时间 t ~ Beta(1.5, 1) * 0.999 + 0.001
    time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
    
    # 2. 采样噪声 noise ~ N(0, 1)
    noise = jax.random.normal(noise_rng, actions.shape)
    
    # 3. 计算噪声动作: x_t = t * noise + (1-t) * actions
    x_t = time * noise + (1 - time) * actions
    
    # 4. 计算真实速度: u_t = noise - actions
    u_t = noise - actions
    
    # 5. 模型预测速度: v_t = model(x_t, t, observation)
    #    通过 embed_prefix + embed_suffix + LLM forward
    v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
    
    # 6. 返回 MSE 损失: mean((v_t - u_t)^2)
    return jnp.mean(jnp.square(v_t - u_t), axis=-1)
```

#### 5.2 推理动作采样

```python
def sample_actions(self, rng, observation, num_steps=10, noise=None):
    # dt = -1/num_steps (从噪声到数据的方向)
    dt = -1.0 / num_steps
    
    # 初始化噪声: x_1 ~ N(0, 1)
    noise = jax.random.normal(rng, (batch_size, action_horizon, action_dim))
    
    # 首先缓存 prefix (图像+文本) 的 KV cache
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], ...)
    
    # 迭代采样
    def step(x_t, time):
        # 嵌入 suffix (噪声动作 + 时间步)
        suffix_tokens, ... = self.embed_suffix(observation, x_t, time)
        
        # 通过 LLM 预测速度
        _, suffix_out = self.PaliGemma.llm([None, suffix_tokens], kv_cache=kv_cache)
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
        
        # 更新: x_{t+dt} = x_t + dt * v_t
        return x_t + dt * v_t, time + dt
    
    # 从 t=1 到 t=0 迭代
    x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
    return x_0  # 生成的动作序列
```

**关键时间约定:**

- `t=1` → 纯噪声
- `t=0` → 目标数据 (真实动作)
- 这与 Pi0 论文的约定相反

---

### 6. 反事实动作分析 (Counterfactual)

**文件:** `pi0.py`

可选的高级功能，用于分析不同模态对动作的影响。

#### 6.1 方法签名

```python
def sample_actions_with_cf(
    self, rng, observation,
    num_steps: int = 10,
    reweight_action_with_cf: bool = True,
    cf_guidance_scale: float = 0.1,
    vlm_effect_upper_threshold: float = 0.5,
    cf_mode: str = "E",
    clear_prompt_for_image_cf: bool = True,
    return_cf_metrics: bool = False,
) -> Actions | tuple[Actions, dict]
```

#### 6.2 反事实分支

| 分支 | 操作 | 目的 |
|------|------|------|
| baseline | 原始观测采样 | 参考基准 |
| prop0 | state归零后采样 | 评估语言影响 |
| img0 | 图像归零后采样 (可选清除prompt) | 评估视觉影响 |

#### 6.3 CfMode 模式

| 模式 | 修正策略 |
|------|----------|
| BASE | 不使用修正，直接返回baseline |
| A | `actions_cf = actions_base + guidance * (base - img0)` |
| B | A + 小比例 proprio 修正 |
| C | 自适应权重 + 归一化 proprio |
| D | A + 裁剪 proprio |
| E | 自适应比例 + 裁剪 proprio (默认) |
| F | A + 软裁剪 proprio |

#### 6.4 使用示例

```python
# 在 Policy 中启用
policy = create_trained_policy(
    train_config,
    checkpoint_dir,
    use_cf_sampling=True,
)

# 推理时自动使用 CF 采样
result = policy.infer(obs)
# 或获取 CF 分析指标
actions, cf_metrics = model.sample_actions_with_cf(
    rng, observation,
    cf_mode="E",
    return_cf_metrics=True,
)
# cf_metrics = {"effect_vlm": float, "effect_prop": float, "use_base": bool, "cf_mode": str}
```

---

## 三、关键变量意义总结

### 维度符号约定

| 符号 | 含义 | 典型值 |
|------|------|--------|
| `*b` | batch维度 (可变) | 32 |
| `h` | 图像高度 | 224 |
| `w` | 图像宽度 | 224 |
| `c` | 通道数 | 3 |
| `s` | state维度 | 14 (ALOHA) / 7 (Libero) |
| `l` | token序列长度 | 48 / 200 |
| `ah` | action_horizon | 50 / 10 |
| `ad` | action_dim | 32 |
| `emb` | embedding维度 | Gemma width |

### 数值范围约定

| 数据类型 | 范围 | 说明 |
|----------|------|------|
| 图像输入 | [-1, 1] | float32, 已归一化 |
| 原始图像 | [0, 255] | uint8, 自动转换 |
| 状态/动作 (归一化后) | ~[-1, 1] | 分位数归一化 |
| 状态/动作 (原始) | 各不相同 | 需要norm_stats反归一化 |

---

## 四、数据流全景图

### 推理流程

```
┌──────────────────────────────────────────────────────────────────┐
│  环境                                                             │
│    │  obs = {                                                     │
│    │      image: {base_0_rgb: np.array[h,w,3], ...},              │
│    │      image_mask: {base_0_rgb: True, ...},                    │
│    │      state: np.array[s],                                     │
│    │      prompt: "fold the towel"                                │
│    │  }                                                            │
│    ▼                                                              │
│  Policy.infer(obs)                                                │
│    │                                                              │
│    ├─ input_transform(obs)                                        │
│    │    ├─ RepackTransform → 统一键名                              │
│    │    ├─ InjectDefaultPrompt → 添加默认prompt                    │
│    │    ├─ ResizeImages → 224×224                                 │
│    │    ├─ Normalize → 归一化 (z-score 或 quantile)               │
│    │    ├─ TokenizePrompt → token_ids + mask                      │
│    │    └─ PadStatesAndActions → 填充到 action_dim                 │
│    │                                                              │
│    ├─ Observation.from_dict() → Observation对象                   │
│    │                                                              │
│    ├─ model.sample_actions(rng, obs, num_steps=10)                │
│    │    │                                                         │
│    │    ├─ embed_prefix():                                        │
│    │    │    ├─ SigLIP 编码图像 → image_tokens                     │
│    │    │    ├─ Gemma embed 文本 → text_tokens                     │
│    │    │    └─ 返回 prefix_tokens, prefix_mask, ar_mask           │
│    │    │                                                         │
│    │    ├─ 缓存 prefix KV cache                                    │
│    │    │                                                         │
│    │    ├─ Flow Matching 迭代 (10步):                              │
│    │    │    ├─ embed_suffix(x_t, time)                           │
│    │    │    ├─ LLM forward with KV cache                         │
│    │    │    ├─ 预测速度 v_t                                       │
│    │    │    └─ x_{t+dt} = x_t + dt * v_t                         │
│    │    │                                                         │
│    │    └─ 返回 x_0 (生成的动作序列)                                │
│    │                                                              │
│    ├─ output_transform(results)                                   │
│    │    ├─ Unnormalize → 原始尺度                                  │
│    │    └─ AbsoluteActions → 相对转绝对动作                        │
│    │                                                              │
│    ▼                                                              │
│  返回 {                                                           │
│      actions: np.array[ah, ad],  # 动作序列                        │
│      state: np.array[s],         # 状态副本                        │
│      policy_timing: {infer_ms: float}                             │
│  }                                                                │
└──────────────────────────────────────────────────────────────────┘
```

### 训练流程

```
┌──────────────────────────────────────────────────────────────────┐
│  DataLoader                                                       │
│    │                                                              │
│    ├─ create_data_loader(config)                                  │
│    │    ├─ create_torch_dataset() → LeRobot Dataset               │
│    │    │    或 create_rlds_dataset() → DROID RLDS Dataset        │
│    │    └─ transform_dataset()                                    │
│    │        ├─ repack_transforms                                  │
│    │        ├─ data_transforms                                    │
│    │        ├─ Normalize(norm_stats)                              │
│    │        └─ model_transforms                                   │
│    │                                                              │
│    └─ 迭代返回 (Observation, Actions)                              │
│    ▼                                                              │
│  训练循环                                                         │
│    │                                                              │
│    ├─ model.compute_loss(rng, observation, actions)               │
│    │    ├─ Flow Matching 损失计算                                  │
│    │    └─ 返回 loss: Float[Array, "*b ah"]                       │
│    │                                                              │
│    ├─ optimizer.update(grads, params)                             │
│    ├─ EMA 更新 (可选)                                              │
│    ├─ 日志记录 (wandb)                                             │
│    └─ 定期保存 checkpoint                                          │
│    ▼                                                              │
│  训练完成 → 保存最终 checkpoint                                    │
└──────────────────────────────────────────────────────────────────┘
```

---

## 五、预训练配置列表

### 推理配置

| 配置名 | 模型 | 数据集 | 用途 |
|--------|------|--------|------|
| `pi0_aloha` | Pi0 | ALOHA | ALOHA机器人推理 |
| `pi05_aloha` | Pi05 | ALOHA | ALOHA机器人推理 (新架构) |
| `pi0_aloha_towel` | Pi0 | ALOHA | 折毛巾任务 |
| `pi0_aloha_tupperware` | Pi0 | ALOHA | 开盒子任务 |
| `pi0_droid` | Pi0 | DROID | DROID机器人推理 |
| `pi05_droid` | Pi05 | DROID | DROID机器人推理 |

### 微调配置

| 配置名 | 模型 | 数据集 | 特点 |
|--------|------|--------|------|
| `pi0_libero` | Pi0 | Libero | 全参数微调 |
| `pi0_libero_low_mem_finetune` | Pi0 (LoRA) | Libero | LoRA微调 |
| `pi0_fast_libero` | Pi0-FAST | Libero | FAST架构 |
| `pi05_libero` | Pi05 | Libero | 新架构 |
| `pi0_aloha_pen_uncap` | Pi0 | ALOHA | 笔帽任务 |
| `pi05_full_droid_finetune` | Pi05 | DROID (RLDS) | 大规模DROID |

---

## 六、扩展指南

### 创建自定义数据集配置

1. 创建新的 `DataConfigFactory` 类：

```python
@dataclasses.dataclass(frozen=True)
class MyCustomDataConfig(DataConfigFactory):
    repo_id: str = "my_dataset"
    
    @override
    def create(self, assets_dirs, model_config) -> DataConfig:
        # 定义 repack 变换 (数据集键名 → 统一格式)
        repack_transform = _transforms.Group(
            inputs=[_transforms.RepackTransform({
                "images": {"cam_high": "observation.images.my_camera"},
                "state": "observation.state",
                "actions": "action",
            })]
        )
        
        # 定义数据变换 (机器人特定)
        data_transforms = _transforms.Group(
            inputs=[MyCustomInputs()],
            outputs=[MyCustomOutputs()],
        )
        
        # 使用标准模型变换
        model_transforms = ModelTransformFactory()(model_config)
        
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
```

2. 创建自定义变换：

```python
@dataclasses.dataclass(frozen=True)
class MyCustomInputs(DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # 处理输入数据
        data["state"] = process_state(data["state"])
        data["actions"] = process_actions(data["actions"])
        return data

@dataclasses.dataclass(frozen=True)
class MyCustomOutputs(DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # 反变换输出数据
        data["actions"] = unprocess_actions(data["actions"])
        return data
```

3. 创建训练配置：

```python
TrainConfig(
    name="pi0_my_custom",
    model=pi0_config.Pi0Config(),
    data=MyCustomDataConfig(
        repo_id="my_hf_username/my_dataset",
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi0_base/params"
    ),
    num_train_steps=20_000,
)
```

---

## 七、常见问题

### Q1: 如何选择 Pi0 vs Pi05 vs Pi0-FAST?

| 模型 | 优点 | 适用场景 |
|------|------|----------|
| Pi0 | 稳定、广泛验证 | 通用机器人任务 |
| Pi05 | 更好性能、更长context | 复杂任务、双臂机器人 |
| Pi0-FAST | 离散token、快速推理 | 简单任务、低延迟需求 |

### Q2: delta_actions vs absolute_actions?

- **训练:** 如果数据集使用绝对位置动作，需要 `DeltaActions` 变换
- **推理:** 需要 `AbsoluteActions` 反变换，返回给机器人执行

```python
# 判断mask: 正数表示转为delta，负数表示保持absolute
delta_action_mask = make_bool_mask(6, -1)  # 前6维转delta，第7维保持
```

### Q3: 如何处理不同相机配置?

默认期望 3 个相机: `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`

如果只有部分相机:

```python
# 在变换中填充缺失的相机
data["image"]["left_wrist_0_rgb"] = np.zeros((224, 224, 3))
data["image_mask"]["left_wrist_0_rgb"] = False  # 标记为无效
```

---

## 八、参考文件索引

| 功能 | 主要文件 |
|------|----------|
| 模型定义 | `src/openpi/models/pi0.py`, `model.py`, `pi0_config.py` |
| 策略封装 | `src/openpi/policies/policy.py`, `policy_config.py` |
| 数据变换 | `src/openpi/transforms.py` |
| 数据加载 | `src/openpi/training/data_loader.py` |
| 训练配置 | `src/openpi/training/config.py` |
| 归一化 | `src/openpi/shared/normalize.py` |
| Tokenizer | `src/openpi/models/tokenizer.py` |
| 视觉编码 | `src/openpi/models/siglip.py`, `vit.py` |
| LLM | `src/openpi/models/gemma.py` |
| 推理服务 | `scripts/serve_policy.py` |
| 训练脚本 | `scripts/train.py`, `scripts/train_pytorch.py` |