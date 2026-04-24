# Pi0.5 模型结构分析 - sample_actions 与 embed_prefix

## Context

分析 pi0.5 模型的核心推理流程，重点关注 `sample_actions` 方法和 `embed_prefix` 部分的实现细节，理解 SigLIP 图像编码和 Gemma 文本嵌入如何协作生成 prefix tokens。

---

## 0. Attention-Guided 反事实推理详解

### 0.1 核心思想

Attention-guided CF 通过**分析模型内部注意力机制**来识别关键区域，然后针对性地进行反事实干预：

```
原始观测 → 提取Attention(指定层) → 选择干预方式 → 计算CF Action → A-E模式加权 → 最终Action
```

与传统 Input-level CF（直接将输入数据置零）不同，Attention-guided CF:
- **Token-level**: 通过 `cf_attn_mask` 阻断高注意力图像 token 的信息流
- **Pixel-level**: 将高注意力像素块置零后重新通过 SigLIP 编码

### 0.2 完整流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│            Attention-Guided CF: Token-level 和 Pixel-level 对比              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  阶段1: Prefix预填充（共享，一次性执行）                                │ │
│  │                                                                        │ │
│  │    Observation → embed_prefix → SigLIP编码 + Gemma嵌入                │ │
│  │                   ↓                                                    │ │
│  │              prefix_tokens [B, P, E]                                   │ │
│  │              prefix_mask   [B, P]                                      │ │
│  │              kv_cache (Prefix KV缓存)                                  │ │
│  │                                                                        │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                              ↓                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  阶段2: Baseline采样（无干预）                                         │ │
│  │                                                                        │ │
│  │    sample_actions_from_precomputed_prefix(rng, obs, kv_cache,         │ │
│  │                                            cf_attn_mask=None)         │ │
│  │                   ↓                                                    │ │
│  │              actions_baseline [B, AH, AD]                              │ │
│  │                                                                        │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                              ↓                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  阶段3: Attention提取                                                  │ │
│  │                                                                        │ │
│  │    参数:                                                               │ │
│  │      - cf_attn_layer_index: Transformer层索引 (默认16)                │ │
│  │      - cf_attn_time: 扩散时间点 (默认1.0 = 纯噪声)                    │ │
│  │                                                                        │ │
│  │    _extract_single_step_attention_map_from_precomputed_prefix(        │ │
│  │        observation, prefix_mask, kv_cache,                             │ │
│  │        layer_index=16, time=1.0, noise=初始噪声                       │ │
│  │    )                                                                   │ │
│  │                                                                        │ │
│  │    流程:                                                               │ │
│  │    ┌─────────────────────────────────────────────────────────────┐   │ │
│  │    │  1. embed_suffix: 编码噪声动作 + 时间步嵌入                  │   │ │
│  │    │     suffix_tokens = action_in_proj(noise)                   │   │ │
│  │    │     time_emb = posemb_sincos(time) + MLP → adaRMS参数       │   │ │
│  │    │                                                              │   │ │
│  │    │  2. 构建attention mask                                       │   │ │
│  │    │     full_attn_mask = concat(prefix_mask扩展, suffix因果mask)│   │ │
│  │    │                                                              │   │ │
│  │    │  3. LLM forward (decode_with_last_query_attention)          │   │ │
│  │    │     调用 Gemma 的特定层 (layer_index)                        │   │ │
│  │    │     返回 last_query_attn_by_head [B, heads, P+S]            │   │ │
│  │    │     ↓                                                        │   │ │
│  │    │     平均所有head: last_query_attn_head_avg [B, P+S]         │   │ │
│  │    │                                                              │   │ │
│  │    │  4. 按相机分割attention                                       │   │ │
│  │    │     image_token_bounds = {"base_0_rgb": (0,256),            │   │ │
│  │    │                            "wrist_0_rgb": (256,512)}        │   │ │
│  │    │     cam_attn = last_query_attn[:, start:end]                │   │ │
│  │    │     reshape为2D grid (16x16) → image_attn_maps              │   │ │
│  │    └─────────────────────────────────────────────────────────────┘   │ │
│  │                                                                        │ │
│  │    输出:                                                               │ │
│  │      - last_query_attn_head_avg: 最后query对各token的平均注意力      │ │
│  │      - image_token_bounds: 各相机图像token位置                        │ │
│  │      - image_attn_maps: 按相机分割的2D attention grid [B, 16, 16]    │ │
│  │                                                                        │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                              ↓                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  阶段4: 干预分支 (Token-level vs Pixel-level)                         │ │
│  │                                                                        │ │
│  │  ┌─────────────────────────────┐  ┌─────────────────────────────┐   │ │
│  │  │  Token-level CF             │  │  Pixel-level CF              │   │ │
│  │  │                             │  │                              │   │ │
│  │  │  参数: cf_attn_topk_ratio   │  │  参数: cf_attn_topk_ratio    │   │ │
│  │  │        (默认0.2)            │  │        (默认0.2)             │   │ │
│  │  │                             │  │                              │   │ │
│  │  │  流程:                      │  │  流程:                       │   │ │
│  │  │  ┌─────────────────────┐   │  │  ┌─────────────────────┐    │   │ │
│  │  │  │ build_token_level_  │   │  │  │ _zero_high_attention_│    │   │ │
│  │  │  │ cf_attn_mask_from_  │   │  │  │ pixel_patches        │    │   │ │
│  │  │  │ attention           │   │  │  │                      │    │   │ │
│  │  │  └─────────────────────┘   │  │  └─────────────────────┘    │   │ │
│  │  │  ↓                         │  │  ↓                          │   │ │
│  │  │  选择top-k高注意力token    │  │  将attention grid映射回像素 │   │ │
│  │  │  在cf_attn_mask中设为False │  │  高注意力像素块(14x14)置零 │   │ │
│  │  │  (阻断信息流)              │  │  创建modified_observation   │   │ │
│  │  │                             │  │                              │   │ │
│  │  │  ↓                         │  │  ↓                          │   │ │
│  │  │  sample_actions_from_      │  │  sample_actions(            │   │ │
│  │  │  precomputed_prefix(       │  │      modified_observation   │   │ │
│  │  │    cf_attn_mask=mask       │  │  )                          │   │ │
│  │  │  )                         │  │                              │   │ │
│  │  │                             │  │  ★ 需重新SigLIP编码         │   │ │
│  │  │                             │  │  (与token级不同!)           │   │ │
│  │  └─────────────────────────────┘  └─────────────────────────────┘   │ │
│  │                                                                        │ │
│  │              actions_cf [B, AH, AD]                                    │ │
│  │                                                                        │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                              ↓                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  阶段5: CF效果计算与动作重加权                                         │ │
│  │                                                                        │ │
│  │    effect_cf = L2(actions_baseline - actions_cf)                      │ │
│  │    delta_cf = actions_baseline - actions_cf                           │ │
│  │                                                                        │ │
│  │    CF加权模式 (cf_mode):                                               │ │
│  │    ┌─────────────────────────────────────────────────────────────┐   │ │
│  │    │  BASE: actions_final = actions_baseline (无干预)            │   │ │
│  │    │  A:    actions_final = baseline + guidance * delta_cf       │   │ │
│  │    │  D:    actions_final = baseline + guidance * clip(delta_cf) │   │ │
│  │    │  E:    actions_final = baseline + guidance * delta_cf       │   │ │
│  │    │        (adaptive state weight, 默认模式)                     │   │ │
│  │    └─────────────────────────────────────────────────────────────┘   │ │
│  │                                                                        │ │
│  │    安全fallback:                                                       │ │
│  │    if effect_cf > effect_threshold:                                   │ │
│  │        actions_final = actions_baseline  # 回退到baseline             │ │
│  │                                                                        │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 0.3 Token-level CF 详细流程

**位置**: [pi0.py:384-434](src/openpi/models/pi0.py#L384-L434)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    build_token_level_cf_attn_mask_from_attention            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  输入:                                                                      │
│    - last_query_attn_head_avg [B, P+S]: 最后query对各token的平均注意力     │
│    - prefix_mask [B, P]: 有效token指示                                     │
│    - image_token_bounds: {"cam_name": (start, end)}                        │
│    - suffix_len: suffix token数量                                          │
│    - topk_ratio: 干预比例 (默认0.2)                                        │
│                                                                             │
│  步骤:                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  1. 构建image_token_mask                                             │  │
│  │     image_token_mask[prefix_len] = False (全False)                  │  │
│  │     for (start, end) in image_token_bounds:                         │  │
│  │         image_token_mask[start:end] = True                          │  │
│  │                                                                      │  │
│  │  2. 计算候选token数量                                                │  │
│  │     candidate_mask = AND(prefix_mask, image_token_mask)             │  │
│  │     candidate_counts = sum(candidate_mask)                          │  │
│  │     k = ceil(candidate_counts * topk_ratio)                         │  │
│  │     k = max(k, 1) if candidate_counts > 0                          │  │
│  │                                                                      │  │
│  │  3. 选择top-k高注意力token                                           │  │
│  │     scores = last_query_attn[:, :prefix_len]                        │  │
│  │     masked_scores = where(candidate_mask, scores, -inf)             │  │
│  │     sorted_scores = sort(masked_scores, descending)                 │  │
│  │     threshold = sorted_scores[k-1]                                  │  │
│  │     block_prefix = AND(candidate_mask, scores >= threshold)         │  │
│  │                                                                      │  │
│  │  4. 构建cf_attn_mask                                                 │  │
│  │     cf_mask = ones([B, S, P+S])                                      │  │
│  │     cf_mask[:, :, :P] = NOT(block_prefix)[:, None, :]               │  │
│  │     # True = 可见, False = 阻断                                     │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  输出:                                                                      │
│    - cf_attn_mask [B, S, P+S]:                                             │
│      True = 该位置可见（可attention）                                      │
│      False = 该位置阻断（高注意力token被屏蔽）                             │
│                                                                             │
│  效果示意:                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  原始attention (所有prefix可见):                                     │  │
│  │  [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 ...]                               │  │
│  │   ↓                                                                  │  │
│  │  [图][图][图][图][文][文][状][状]...                                  │  │
│  │                                                                      │  │
│  │  CF attention mask (top-k图像token阻断):                             │  │
│  │  [░░░░1 1 1 1░░░░1 1 1 1 1 1 1 1 ...]                                │  │
│  │   ↓       ↓                                                          │  │
│  │  高注意力图像token被阻断(F)                                          │  │
│  │  其他token保持可见(T)                                                │  │
│  │                                                                      │  │
│  │  ░░░░ = False (阻断), 1 = True (允许)                                │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 0.4 Pixel-level CF 详细流程

**位置**: [pi0.py:577-670](src/openpi/models/pi0.py#L577-L670)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    _zero_high_attention_pixel_patches                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  输入:                                                                      │
│    - images: {"cam_name": [B, H, W, C]} 原始RGB图像                        │
│    - image_attn_maps: {"cam_name": [B, 16, 16]} 2D attention grid          │
│    - topk_ratio: 干预比例                                                  │
│                                                                             │
│  SigLIP So400m/14 参数:                                                    │
│    - 输入: 224x224 RGB图像                                                 │
│    - Patch size: 14x14                                                     │
│    - Grid: 16x16 = 256 tokens                                              │
│                                                                             │
│  步骤:                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  1. 计算patch大小                                                    │  │
│  │     patch_size = H / 16 = 224 / 16 = 14                             │  │
│  │                                                                      │  │
│  │  2. 展平attention并选择top-k                                         │  │
│  │     flat_attn = attn[b].flatten()  # [256]                          │  │
│  │     k = max(1, int(256 * topk_ratio))                               │  │
│  │     top_indices = argsort(flat_attn)[-k:]                           │  │
│  │                                                                      │  │
│  │  3. 将top-k像素块置零                                                │  │
│  │     for idx in top_indices:                                         │  │
│  │         row = idx // 16                                             │  │
│  │         col = idx % 16                                              │  │
│  │         y_start = row * 14                                          │  │
│  │         x_start = col * 14                                          │  │
│  │         modified[b, y:y+14, x:x+14] = 0                             │  │
│  │                                                                      │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  输出:                                                                      │
│    - modified_images: {"cam_name": [B, H, W, C]}                           │
│      高注意力像素块置零的图像                                               │
│                                                                             │
│  效果示意:                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  原始224x224图像:                                                    │  │
│  │  ┌────────────────────────────────────┐                            │  │
│  │  │  [RGB像素值]                        │                            │  │
│  │  │  全部有效                           │                            │  │
│  │  └────────────────────────────────────┘                            │  │
│  │                                                                      │  │
│  │  Attention grid (16x16):                                             │  │
│  │  ┌────────────────────────────────────┐                            │  │
│  │  │  [0.1 0.2 ████ 0.3 ...]            │                            │  │
│  │  │        ↑                           │                            │  │
│  │  │      高注意力区域                   │                            │  │
│  │  └────────────────────────────────────┘                            │  │
│  │                                                                      │  │
│  │  Modified图像:                                                        │  │
│  │  ┌────────────────────────────────────┐                            │  │
│  │  │  [RGB ░░░░░░░░ RGB ...]            │                            │  │
│  │  │       ↑                            │                            │  │
│  │  │     高注意力像素块置零              │                            │  │
│  │  │     (14x14区域)                    │                            │  │
│  │  └────────────────────────────────────┘                            │  │
│  │                                                                      │  │
│  │  ░░░░ = 像素值全为0                                                   │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  关键区别:                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  Token-level CF:                                                     │  │
│  │    - 在attention层阻断信息流                                         │  │
│  │    - 不改变输入数据                                                   │  │
│  │    - 共享kv_cache                                                    │  │
│  │                                                                      │  │
│  │  Pixel-level CF:                                                     │  │
│  │    - 在输入层修改像素                                                 │  │
│  │    - 需重新SigLIP编码 ★                                              │  │
│  │    - 不能共享kv_cache                                                │  │
│  │    - 更直接、效果更强                                                 │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 0.5 Attention可视化功能

当 `VISUALIZE_ATTENTION=true` 时，系统会保存attention热力图：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Attention 可视化流程                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  参数:                                                                      │
│    - VISUALIZE_ATTENTION: 是否开启可视化                                   │
│    - VISUALIZATION_DIR: 输出目录 (默认 "results/attention_vis")            │
│    - VISUALIZATION_FREQUENCY: 保存频率 (每N步)                             │
│                                                                             │
│  流程:                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  1. 检查频率条件                                                     │  │
│  │     if step_idx % visualization_frequency == 0:                     │  │
│  │         _visualize_attention(...)                                   │  │
│  │                                                                      │  │
│  │  2. 转换JAX数组为numpy                                               │  │
│  │     images_np = {cam: np.asarray(img) for cam, img in images}       │  │
│  │                                                                      │  │
│  │  3. 创建热力图叠加                                                   │  │
│  │     create_attention_heatmap(                                       │  │
│  │         attention_map,  # [16,16] 或展平                             │  │
│  │         original_image, # [H, W, 3]                                  │  │
│  │         alpha=0.4,        # 透明度                                   │  │
│  │         colormap="jet"    # 颜色映射                                 │  │
│  │     )                                                                │  │
│  │                                                                      │  │
│  │  4. 保存为PNG                                                        │  │
│  │     filename = "base_0_rgb_ep0_step0_layer16_cf_E.png"              │  │
│  │     save to VISUALIZATION_DIR                                        │  │
│  │                                                                      │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  输出文件命名:                                                              │
│    {camera_name}_ep{episode}_step{step}_layer{layer}_cf_{mode}.png         │
│                                                                             │
│  可视化内容:                                                                │
│    - 热力图: jet colormap显示attention强度                                │
│    - 叠加: 半透明叠加在原始RGB图像上                                        │
│    - 红色区域: 高注意力 (关键决策区域)                                     │
│    - 蓝色区域: 低注意力 (模型较少关注)                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 0.6 脚本使用示例

**文件**: `run_libero_image_attention_cf.sh`

```bash
# 基本用法 (Token-level CF, 模式E)
./run_libero_image_attention_cf.sh

# Pixel-level CF
CF_MODE_TYPE=pixel CF_MODE=A ./run_libero_image_attention_cf.sh

# 选择不同Transformer层
CF_ATTN_LAYER_INDEX=8  ./run_libero_image_attention_cf.sh   # 较浅层
CF_ATTN_LAYER_INDEX=24 ./run_libero_image_attention_cf.sh   # 较深层

# 开启可视化 (测试阶段)
VISUALIZE_ATTENTION=true ./run_libero_image_attention_cf.sh

# 调整干预比例
CF_ATTN_TOPK_RATIO=0.3 ./run_libero_image_attention_cf.sh   # 干预30%高注意力区域

# 调整加权强度
CF_GUIDANCE_SCALE=0.2 ./run_libero_image_attention_cf.sh    # 增强CF影响
```

### 0.7 关键参数说明

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `CF_MODE_TYPE` | "token" | 干预类型: "token"(attention阻断) 或 "pixel"(像素置零) |
| `CF_ATTN_LAYER_INDEX` | 16 | 从哪个Transformer层提取attention |
| `CF_ATTN_TIME` | 1.0 | 扩散时间点 (1.0=纯噪声, 0.0=干净信号) |
| `CF_ATTN_TOPK_RATIO` | 0.2 | 干预比例 (top-k高注意力区域) |
| `CF_MODE` | "E" | 加权模式: BASE/A/B/C/D/E/F |
| `CF_GUIDANCE_SCALE` | 0.1 | CF delta的加权强度 |
| `VLM_EFFECT_THRESHOLD` | 0.5 | 效果阈值 (超过则回退baseline) |

### 0.8 不同CF模式对比

```
┌─────────────────────────────────────────────────────────────────────────────┐
│               Token-level vs Pixel-level vs Input-level CF 对比              │
├───────────────┬───────────────────┬─────────────────────┬─────────────────┤
│    特性       │   Token-level CF │    Pixel-level CF   │  Input-level CF │
├───────────────┼───────────────────┼─────────────────────┼─────────────────┤
│               │                   │                     │                 │
│ 干扰位置      │ Attention层       │ 输入像素层          │ 输入观测层      │
│               │                   │                     │                 │
│ 干扰方式      │ cf_attn_mask      │ 像素块置零          │ 整张图像置零    │
│               │ 阻断高注意力token │                     │                 │
│               │                   │                     │                 │
│ 需重新编码    │ 否                │ 是 ★                │ 是              │
│               │                   │ (SigLIP)            │ (SigLIP)        │
│               │                   │                     │                 │
│ KV Cache复用  │ 可共享            │ 不能共享            │ 不能共享        │
│               │                   │                     │                 │
│ 理论纯净度    │ 最高              │ 中                  │ 较低            │
│               │ (不改数据)        │                     │ (改变分布)      │
│               │                   │                     │                 │
│ 计算效率      │ 最高              │ 中                  │ 最低            │
│               │                   │                     │                 │
│ 干扰精确度    │ Token级别         │ Patch级别           │ 整图级别        │
│               │ (细粒度)          │ (14x14像素)         │ (粗粒度)        │
│               │                   │                     │                 │
│ 适用场景      │ 精细分析          │ 物理干预验证        │ 快速实验        │
│               │ 学术研究          │                     │                 │
│               │                   │                     │                 │
└───────────────┴───────────────────┴─────────────────────┴─────────────────┘
```

---

## 1. 整体架构

Pi0.5 是基于 **PaliGemma + Action Expert** 的多模态 VLA 模型：

```
PaliGemma (冻结预训练)
├── SigLIP So400m/14 (视觉编码器)
└── Gemma 2B (语言模型)

Action Expert (可训练)
├── action_in_proj: 动作投影层
├── time_mlp: 时间步 MLP (用于 adaRMS)
└── action_out_proj: 动作输出投影层
```

**关键配置** ([pi0.py:116-124](src/openpi/models/pi0.py#L116-L124)):
- SigLIP variant: `So400m/14` (patch size 14x14)
- pool_type: `none` (保留所有 patch tokens，不池化)
- 输出维度匹配 Gemma hidden_size

---

## 2. sample_actions 流程详解

### 2.1 方法入口 ([pi0.py:252-270](src/openpi/models/pi0.py#L252-L270))

```python
def sample_actions(rng, observation, num_steps=10, noise=None, cf_attn_mask=None):
    # Step 1: 预处理 observation，计算 prefix KV cache (只执行一次)
    observation, prefix_mask, kv_cache = self.prepare_prefix_for_sampling(observation)
    
    # Step 2: 使用预计算的 KV cache 进行 flow matching 迭代解码
    return self.sample_actions_from_precomputed_prefix(
        rng, observation, prefix_mask, kv_cache,
        num_steps=num_steps, noise=noise, cf_attn_mask=cf_attn_mask,
    )
```

### 2.2 两阶段设计

| 阶段 | 方法 | 执行次数 | 目的 |
|------|------|----------|------|
| Prefix 预填充 | `prepare_prefix_for_sampling` | **1次** | 编码图像+文本，构建 KV cache |
| Action 解码 | `sample_actions_from_precomputed_prefix` | **N次** (num_steps) | Flow matching Euler 积分 |

这种设计避免了每次迭代都重新编码 prefix，大幅提升推理效率。

---

## 3. embed_prefix 方法 ([pi0.py:141-172](src/openpi/models/pi0.py#L141-L172))

### 3.1 核心逻辑

```python
def embed_prefix(obs):
    input_mask = []
    ar_mask = []
    tokens = []
    
    # 1. 编码所有图像
    for name in obs.images:
        # SigLIP 编码 → (b, 256, width)
        image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
        tokens.append(image_tokens)
        input_mask.append(obs.image_masks[name].repeat(256))
        ar_mask += [False] * 256  # 图像 token 双向注意
    
    # 2. 编码文本
    if obs.tokenized_prompt is not None:
        # Gemma embed → (b, text_len, width)
        text_tokens = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
        tokens.append(text_tokens)
        input_mask.append(obs.tokenized_prompt_mask)
        ar_mask += [False] * text_len  # 文本 token 双向注意
    
    # 3. 拼接所有 tokens
    tokens = jnp.concatenate(tokens, axis=1)      # (b, s, emb)
    input_mask = jnp.concatenate(input_mask, axis=1)  # (b, s)
    ar_mask = jnp.array(ar_mask)  # (s,) 全为 False
    
    return tokens, input_mask, ar_mask
```

### 3.2 输出三元组

| 输出 | Shape | 含义 |
|------|-------|------|
| `tokens` | `(b, s, emb)` | prefix tokens = 图像tokens + 文本tokens |
| `input_mask` | `(b, s)` | 每个 token 是否有效 |
| `ar_mask` | `(s,)` | **全为 False** → prefix 内双向注意力 |

### 3.3 Token 序列结构

```
[Image1 tokens (256)] [Image2 tokens (256)] [Image3 tokens (256)] [Text tokens (variable)]
```

- 每张 224x224 图像 → 16x16 = 256 个 patch tokens
- 标准配置有 3 个相机视角 (`base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`)
- 总 prefix 长度 ≈ 768 + text_len

---

## 4. SigLIP 图像编码 ([siglip.py](src/openpi/models/siglip.py))

### 4.1 编码流程

```
输入图像 (224, 224, 3)
    ↓ Patch Embedding (Conv 14x14, stride 14)
Patch tokens (16, 16, width) → reshape → (256, width)
    ↓ + Positional Embedding (sincos 2D)
    ↓ Transformer Encoder (27 layers)
输出 (b, 256, width)
```

### 4.2 关键参数

- **Patch size**: 14x14
- **Num patches**: 16×16 = 256
- **Output dim**: `paligemma_config.width` (Gemma hidden_size)
- **Pool type**: `none` (保留所有 patch tokens)

---

## 5. Gemma 文本嵌入 ([gemma.py](src/openpi/models/gemma.py))

### 5.1 Embedder 实现

```python
class Embedder(nn.Module):
    vocab_size: int
    embed_dim: int
    
    def encode(self, tokens):
        x = self.input_embedding_table[(tokens,)]  # lookup
        x *= jnp.sqrt(self.embed_dim)  # 缩放 (标准 Transformer 实践)
        return x
```

调用方式: `self.PaliGemma.llm(tokens, method="embed")`

---

## 6. Attention Mask 机制 ([pi0.py:54-79](src/openpi/models/pi0.py#L54-L79))

### 6.1 `make_attn_mask` 函数

```python
def make_attn_mask(input_mask, mask_ar):
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)
```

### 6.2 AR Mask 行为

| ar_mask 值 | cumsum 结果 | 注意力类型 |
|------------|-------------|-----------|
| 全 False (prefix) | 全 0 | **双向注意力** (prefix-LM) |
| [True, False, ...] (suffix) | [1, 1, 1, ...] | 因果注意力 |

**Prefix-LM 模式**: prefix 内所有 token 可以互相注意，这是 PaliGemma 的设计特点。

---

## 7. embed_suffix 方法 ([pi0.py:174-221](src/openpi/models/pi0.py#L174-L221))

### 7.1 Pi0.5 的 suffix 结构

```python
def embed_suffix(obs, noisy_actions, timestep):
    # Pi0.5 没有 state token!
    
    # 1. 动作投影
    action_tokens = self.action_in_proj(noisy_actions)  # (b, action_horizon, width)
    
    # 2. 时间步编码 (sine-cosine)
    time_emb = posemb_sincos(timestep, width, min_period=4e-3, max_period=4.0)
    
    # 3. Time MLP (用于 adaRMS normalization)
    time_emb = self.time_mlp_in(time_emb)
    time_emb = nnx.swish(time_emb)
    time_emb = self.time_mlp_out(time_emb)
    time_emb = nnx.swish(time_emb)
    
    ar_mask = [True] + [False] * (action_horizon - 1)
    # 第一个 action token 是 AR 边界，后续 token 因果注意
    
    return action_tokens, input_mask, ar_mask, time_emb
```

### 7.2 Pi0 vs Pi0.5 区别

| 特性 | Pi0 | Pi0.5 |
|------|-----|-------|
| Suffix 中的 state token | ✓ 有 | ✗ 没有 |
| 时间步处理方式 | 拼接到 action tokens | **adaRMS conditioning** |
| Time MLP | action_time_mlp (拼接) | time_mlp_in/out (独立) |

---

## 8. Flow Matching 解码循环 ([pi0.py:286-351](src/openpi/models/pi0.py#L286-L351))

### 8.1 Euler 积分

```python
def sample_actions_from_precomputed_prefix(...):
    dt = -1.0 / num_steps  # 从 t=1 到 t=0
    x_t = noise  # 初始噪声
    
    def step(carry):
        x_t, time = carry
        
        # 编码 suffix
        suffix_tokens, suffix_mask, suffix_ar, adarms_cond = self.embed_suffix(obs, x_t, time)
        
        # 构建 attention mask
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar)
        prefix_attn_mask = prefix_mask.repeat(suffix_len)
        full_attn_mask = concat([prefix_attn_mask, suffix_attn_mask])
        
        # LLM forward (只跑 suffix，prefix KV cache 复用)
        (_, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        
        # 预测速度场 v_t
        v_t = self.action_out_proj(suffix_out[:, -action_horizon:])
        
        # Euler 更新
        return x_t + dt * v_t, time + dt
    
    # while loop: t=1 → t=0
    x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
    return x_0
```

### 8.2 KV Cache 复用

- Prefix KV cache 在 `prepare_prefix_for_sampling` 中预计算一次
- 每次迭代只需:
  1. 编码 suffix (action tokens + time emb)
  2. 执行 suffix 的 LLM forward pass
  3. 复用 prefix 的 KV cache

---

## 9. 关键文件汇总

| 组件 | 文件路径 |
|------|----------|
| Pi0 模型主类 | [src/openpi/models/pi0.py](src/openpi/models/pi0.py) |
| SigLIP 视觉编码器 | [src/openpi/models/siglip.py](src/openpi/models/siglip.py) |
| Gemma 语言模型 | [src/openpi/models/gemma.py](src/openpi/models/gemma.py) |
| 模型基类/数据结构 | [src/openpi/models/model.py](src/openpi/models/model.py) |
| PyTorch 版本 | [src/openpi/models_pytorch/pi0_pytorch.py](src/openpi/models_pytorch/pi0_pytorch.py) |

---

## 10. 总结

Pi0.5 的 `sample_actions` 采用**两阶段设计**：

1. **Prefix 预填充**: 一次性编码所有图像(SigLIP)和文本(Gemma)，构建 KV cache
2. **Flow Matching 解码**: 迭代采样时只需编码 suffix (action tokens)，复用 prefix KV cache

**核心创新**:
- AdaRMS conditioning: 时间步信息通过 adaptive RMS normalization 注入，而非拼接
- 移除 suffix 中的 state token: 更简洁的架构
- Prefix-LM 模式: 图像和文本 token 之间双向注意力