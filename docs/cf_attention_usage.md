# Counterfactual Attention 模块使用指南

## 概述

`cf_attention` 模块提供了在不修改原始模型代码的情况下进行反事实分析的能力。通过操控 attention mask，可以分析不同模态（图像、语言、状态）对动作生成的贡献。

## 模块结构

```
src/openpi/models/cf_attention/
├── __init__.py           # 模块导出
├── modality_bounds.py    # 模态边界数据结构
├── tokenizer_ext.py      # 扩展tokenizer
├── attention_mask.py     # 反事实attention mask函数
├── cf_sampler.py         # 反事实采样器
└── policy_cf.py          # Policy封装
```

## 核心组件

### 1. ModalityBounds - 模态边界追踪

记录各模态在 token 序列中的位置：

```python
from openpi.models.cf_attention import ModalityBounds

bounds = ModalityBounds(
    image_bounds={"base_0_rgb": (0, 100), "wrist_0_rgb": (100, 200)},
    language_bounds=(200, 250),
    state_bounds=(250, 270),
    prefix_len=270,
    suffix_start=270,
)

# 查询属性
print(f"Image tokens: {bounds.total_image_tokens}")  # 200
print(f"Language tokens: {bounds.total_language_tokens}")  # 50
print(f"State tokens: {bounds.total_state_tokens}")  # 20

# 判断位置所属模态
print(bounds.get_modality_at_position(50))  # "image_base_0_rgb"
print(bounds.get_modality_at_position(220))  # "language"
```

### 2. CfAttnMode - 反事实模式

定义了多种 attention mask 模式：

| 模式 | 描述 |
|------|------|
| `BASE` | 正常 attention（所有模态可见） |
| `NO_IMAGE` | 阻断图像 → 动作 |
| `NO_LANGUAGE` | 阻断语言 → 动作 |
| `NO_STATE` | 阻断状态 → 动作 |
| `IMAGE_ONLY` | 只保留图像 → 动作 |
| `LANGUAGE_ONLY` | 只保留语言 → 动作 |
| `STATE_ONLY` | 只保留状态 → 动作 |
| `IMAGE_LANG` | 图像 + 语言（阻断状态） |
| `IMAGE_STATE` | 图像 + 状态（阻断语言） |
| `LANG_STATE` | 语言 + 状态（阻断图像） |

### 3. Attention Mask 创建

```python
from openpi.models.cf_attention import (
    create_prefix_suffix_attn_mask,
    make_cf_attn_mask,
    CfAttnMode,
)

# 创建基础 prefix-suffix mask
base_mask = create_prefix_suffix_attn_mask(
    seq_len=300,      # 总序列长度
    prefix_len=270,   # prefix 长度
    batch_size=1,
)

# 应用反事实模式
cf_mask = make_cf_attn_mask(
    base_mask,
    bounds,
    CfAttnMode.NO_IMAGE,  # 阻断图像
)

# 可视化 mask
from openpi.models.cf_attention import visualize_attn_mask
print(visualize_attn_mask(cf_mask, prefix_len=270))
```

### 4. ExtendedPaligemmaTokenizer - 边界追踪

扩展 tokenizer，返回模态边界信息：

```python
from openpi.models.cf_attention import ExtendedPaligemmaTokenizer
import numpy as np

tokenizer = ExtendedPaligemmaTokenizer(max_len=48)

# Pi05 格式（state 在 token 序列中）
state = np.array([0.1, -0.5, 0.3])  # 归一化状态
tokens, mask, bounds = tokenizer.tokenize_with_bounds(
    prompt="pick up the cup",
    state=state,
    image_token_counts={"base_0_rgb": 100},
)

print(f"Language bounds: {bounds.language_bounds}")
print(f"State bounds: {bounds.state_bounds}")

# Pi0 格式（state 为连续输入）
tokens, mask, bounds = tokenizer.tokenize_with_bounds(
    prompt="pick up the cup",
    state=None,  # Pi0 格式
    image_token_counts={"base_0_rgb": 100},
)
```

### 5. CfPolicy - Policy 封装

最常用的接口，封装现有 Policy：

```python
from openpi.policies.policy import create_trained_policy
from openpi.models.cf_attention import CfPolicy, CfAttnMode

# 创建原始 policy
original_policy = create_trained_policy(config, checkpoint_dir)

# 封装为 CF policy
cf_policy = CfPolicy(original_policy)

# 标准推理（与原始 policy 相同）
result = cf_policy.infer(obs)
actions = result["actions"]

# 带 CF 分析的推理
cf_result = cf_policy.infer_with_cf(
    obs,
    cf_modes=[
        CfAttnMode.BASE,
        CfAttnMode.NO_IMAGE,
        CfAttnMode.NO_LANGUAGE,
        CfAttnMode.NO_STATE,
    ],
    return_importance=True,
)

# 获取结果
baseline_actions = cf_result.get_actions()
effects = cf_result.get_cf_effects()

print(f"Image effect: {effects['image_effect']}")
print(f"Language effect: {effects['language_effect']}")
print(f"State effect: {effects['state_effect']}")

# 模态重要性统计分析
importance = cf_policy.analyze_modality_importance(
    obs,
    num_samples=10,  # 多次采样统计
)
print(importance)
# 输出:
# {
#     "image": {"mean": 0.5, "std": 0.1, "min": 0.3, "max": 0.8},
#     "language": {"mean": 0.2, "std": 0.05, ...},
#     "state": {"mean": 0.3, "std": 0.08, ...},
# }
```

## 完整使用示例

### 示例 1：简单 CF 分析

```python
import jax
from openpi.policies.policy import create_trained_policy
from openpi.models.cf_attention import wrap_policy_with_cf, CfAttnMode

# 创建和封装 policy
policy = create_trained_policy(config, checkpoint)
cf_policy = wrap_policy_with_cf(policy)

# 准备观测数据
obs = {
    "base_0_rgb": image_array,      # [H, W, 3]
    "state": robot_state,            # [state_dim]
    "prompt": "pick up the red cup",
}

# CF 分析
rng = jax.random.PRNGKey(42)
result = cf_policy.infer_with_cf(
    obs,
    cf_modes=[CfAttnMode.BASE, CfAttnMode.NO_IMAGE],
    rng=rng,
)

# 比较结果
baseline = result.actions_by_mode["base"]
no_image = result.actions_by_mode["no_image"]
print(f"Baseline vs No-Image diff: {result.get_cf_effects()['image_effect']}")
```

### 示例 2：统计分析模态重要性

```python
# 多次采样获得统计结果
importance = cf_policy.analyze_modality_importance(
    obs,
    num_samples=20,
    rng=jax.random.PRNGKey(0),
)

# 找出最重要的模态
most_important = max(
    importance.items(),
    key=lambda x: x[1]["mean"]
)
print(f"Most important modality: {most_important[0]}")
print(f"Mean effect: {most_important[1]['mean']}")
```

### 示例 3：单模态分析

```python
# 只测试图像的贡献
result = cf_policy.infer_with_cf(
    obs,
    cf_modes=[
        CfAttnMode.BASE,       # 全模态
        CfAttnMode.IMAGE_ONLY, # 只有图像
        CfAttnMode.NO_IMAGE,   # 无图像
    ],
)

# 比较三种情况
baseline = result.cf_analysis.actions_by_mode["base"]
image_only = result.cf_analysis.actions_by_mode["image_only"]
no_image = result.cf_analysis.actions_by_mode["no_image"]

# 计算差异
from openpi.models.cf_attention import compute_modality_effect

image_contrib = compute_modality_effect(baseline, no_image)
other_contrib = compute_modality_effect(baseline, image_only)

print(f"Image contribution: {image_contrib}")
print(f"Other modalities contribution: {other_contrib}")
```

## Attention Mask 工作原理

### Token 序列结构

```
┌─────────────────────────────────────────────────────────────────┐
│  Prefix (bidirectional attention)                               │
│  ┌────────────────┐ ┌──────────────┐ ┌──────────────┐          │
│  │ Image Tokens   │ │ Language     │ │ State Tokens │          │
│  │ (SigLIP)       │ │ Tokens       │ │ (Pi05 only)  │          │
│  │ ar_mask=False  │ │ ar_mask=False│ │ ar_mask=False│          │
│  └────────────────┘ └──────────────┘ └──────────────┘          │
│                                                                  │
│  Suffix (causal attention)                                       │
│  ┌────────────────────────────────────────────────────┐         │
│  │ Action Tokens                                      │         │
│  │ ar_mask=[True, False, False, ...] (causal)         │         │
│  └────────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

### Attention Mask 修改原理

当使用 `NO_IMAGE` 模式时：

```
原始 Mask (suffix → prefix):
┌─────────────────────────────────────────────────────┐
│  [1,1,1,1,1,1,1,1,1,1,1,1, ...]  ← 全部可见         │
│  Image │ Language │ State │                        │
└─────────────────────────────────────────────────────┘

修改后 Mask (NO_IMAGE):
┌─────────────────────────────────────────────────────┐
│  [0,0,0,0,0,1,1,1,1,1,1,1, ...]  ← Image被阻断      │
│  Image │ Language │ State │                        │
└─────────────────────────────────────────────────────┘
```

这样，action tokens 在生成时无法 "看到" image tokens 的信息，
从而可以分析图像对动作的贡献。

## 注意事项

### 当前实现限制

1. **完整 CF attention 控制需要模型级修改**：当前的封装方式只能
   进行部分 CF 分析。完全控制 attention mask 需要在模型内部实现。

2. **Token 边界估算**：如果没有精确的 tokenizer 信息，边界位置
   可能是估算值。

3. **Pi0 vs Pi05**：Pi0 格式中 state 不在 token 序列中（是连续输入），
   所以 `NO_STATE` 模式对 Pi0 效果有限。

### 实现完整 CF 控制的建议

如果需要完全控制 attention mask，需要：

1. 修改模型的 `embed_prefix` 方法返回边界信息
2. 修改 `sample_actions` 接受自定义 attention mask
3. 或在 forward pass 中注入 mask

这些修改需要触及原始模型代码，但通过本模块的设计，
可以最小化修改范围。

## 测试

运行测试：

```bash
pytest src/openpi/models/cf_attention_test.py -v
```

## API 参考

### ModalityBounds

```python
@dataclass
class ModalityBounds:
    image_bounds: dict[str, tuple[int, int]]
    language_bounds: tuple[int, int]
    state_bounds: tuple[int, int]
    prefix_len: int
    suffix_start: int
    
    # 属性
    total_image_tokens: int
    total_language_tokens: int
    total_state_tokens: int
    has_state_tokens: bool
    
    # 方法
    get_modality_at_position(pos: int) -> str
    to_dict() -> dict
    from_dict(data: dict) -> ModalityBounds
```

### CfAttnMode

```python
class CfAttnMode(str, Enum):
    BASE = "base"
    NO_IMAGE = "no_image"
    NO_LANGUAGE = "no_lang"
    NO_STATE = "no_state"
    IMAGE_ONLY = "image_only"
    LANGUAGE_ONLY = "lang_only"
    STATE_ONLY = "state_only"
    IMAGE_LANG = "image_lang"
    IMAGE_STATE = "image_state"
    LANG_STATE = "lang_state"
```

### 主要函数

```python
# 创建 attention mask
def create_prefix_suffix_attn_mask(
    seq_len: int,
    prefix_len: int,
    batch_size: int = 1,
) -> jnp.ndarray

# 应用 CF 模式
def make_cf_attn_mask(
    base_attn_mask: jnp.ndarray,
    modality_bounds: ModalityBounds,
    cf_mode: CfAttnMode,
) -> jnp.ndarray

# 计算模态效应
def compute_modality_effect(
    baseline_actions: jnp.ndarray,
    cf_actions: jnp.ndarray,
    metric: str = "l2",  # "l2", "l1", "cosine"
) -> jnp.ndarray

# 获取模态可见性配置
def get_modality_visibility(mode: CfAttnMode) -> dict[str, bool]
```

### CfPolicy

```python
class CfPolicy:
    def __init__(self, policy: Policy, **kwargs)
    
    def infer(self, obs: dict) -> dict
    def infer_with_cf(self, obs: dict, **kwargs) -> CfPolicyResult
    def analyze_modality_importance(self, obs: dict, **kwargs) -> dict
    def get_modality_bounds(self, obs: dict) -> ModalityBounds
```

## LIBERO 评估

### 评估脚本

`scripts/eval_cf_attn_libero.py` 提供了在 LIBERO 基准测试上评估 CF Attention 的完整流程：

```bash
# 基本用法
python scripts/eval_cf_attn_libero.py \
    --checkpoint_dir /data4/zhy/models/openpi-assets/checkpoints/pi05_libero \
    --cf_mode E \
    --task_suite_name libero_spatial

# 或使用 shell 脚本
./run_cf_attn_libero.sh
```

### CF 模式说明

| 模式 | 描述 |
|------|------|
| `BASE` | 无反事实加权（标准推理） |
| `A` | 仅重加权图像 tokens |
| `B` | 仅重加权文本 tokens |
| `C` | 仅重加权状态 tokens |
| `D` | 重加权图像 + 文本 tokens |
| `E` | 重加权图像 + 状态 tokens |
| `F` | 重加权文本 + 状态 tokens |

### 可用任务套件

| 套件 | 最大步数 | 描述 |
|------|----------|------|
| `libero_spatial` | 220 | 空间推理任务 |
| `libero_object` | 280 | 物体操作任务 |
| `libero_goal` | 300 | 目标导向任务 |
| `libero_10` | 520 | 10 任务基准 |
| `libero_90` | 400 | 90 任务基准 |

### 参数配置

```bash
# 环境变量方式
export CF_MODE=E
export TASK_SUITE=libero_spatial
export NUM_TRIALS=10
export GPU_ID=0
./run_cf_attn_libero.sh

# 命令行参数方式
python scripts/eval_cf_attn_libero.py \
    --checkpoint_dir /path/to/checkpoint \
    --cf_mode E \
    --task_suite_name libero_spatial \
    --num_trials_per_task 10 \
    --gpu_id 0 \
    --cf_guidance_scale 0.1 \
    --vlm_effect_upper_threshold 0.5
```

### 输出结果

评估结果保存在 `data/cf_attn_libero/{cf_mode}_{task_suite}/` 目录：

```
data/cf_attn_libero/
├── E_libero_spatial/
│   ├── videos/                    # 回放视频
│   │   ├── task0_ep0_success.mp4
│   │   ├── task0_ep1_failure.mp4
│   │   └── ...
│   ├── results.txt                # 成功率统计
│   └── cf_attn_effects.json       # 详细 CF 效果数据
```

### 批量对比测试

```bash
# 对比不同 CF 模式
for mode in BASE A B C D E F; do
    CF_MODE=$mode ./run_cf_attn_libero.sh
done

# 结果对比
cat data/cf_attn_libero/*/results.txt
```

## 相关文档

- [OpenPI Framework Overview](../docs/openpi_framework.md)
- [Pi0 Model Architecture](../src/openpi/models/pi0.py)
- [Policy Implementation](../src/openpi/policies/policy.py)
