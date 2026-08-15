# LLM-SNN — 200M-A25M MoE 从零预训练 + 工具调用微调

在 A100 40G 单卡上，用 ~2.8B token（代码 + 中文 + 英文 + 现有语料）从零预训练一个
**200M 总参数 / ~25M 激活** 的 MoE 模型，再 SFT 成中英文本工具调用助手。

## 模型规格（`model/config.py`）

| 项 | 值 |
|---|---|
| 总参数 | ~199M（含 tied embedding） |
| 每 token 激活 | ~37M（attention 全算）+ 纯 MoE FFN ~17.7M |
| 结构 | 12 层 / d=768 / 12 头 / GQA(4 KV) / RoPE |
| MoE | 16 专家 / Top-2 / 专家 SwiGLU hidden=320 / 共享 router |
| 词表 | 50K（sentencepiece BPE 重训） |
| 上下文 | 2048 |

## 数据配比（目标 ~2.8B token）

| 数据源 | 类型 | 目标 |
|---|---|---|
| `bigcode/starcoderdata` | 代码（py/js/html/css/java） | ~1.5B token |
| `Skywork/SkyPile-150B`（中文）+ `0xDing/wikipedia-cn` | 中文文本 | ~0.8B token |
| `HuggingFaceFW/fineweb-edu` | 英文文本 | ~0.5B token |
| 旧语料（`train_zh/en/en2/zh2/tool`） | 中英+工具 | 保留补充 |

> 说明：`0xDing/wikipedia-cn` 过滤版仅 191M 字符（~112M token），不足以支撑 0.8B token
> 中文目标，因此主中文源改为 `Skywork/SkyPile-150B`（gated，需 token；150B-token 超大规模）。

## 完整流程

### 0. 环境

```powershell
pip install -r requirements.txt
# 可选：国内加速
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

**在 Ascend 910 (NPU) 上训练**：

目标环境：**CANN 8.0 + torch 2.1 + torch_npu 2.1.0**（torch_npu 版本须与 torch 严格匹配）：

```bash
# 1. 安装 CANN 8.0，并 source 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 2. 安装与 torch 2.1 匹配的 torch_npu 2.1.0（华云镜像）
pip install torch_npu==2.1.0 -i https://repo.huaweicloud.com/repository/pypi/simple
```

代码已通过 `device.py` 自动检测设备（NPU > CUDA > CPU），脚本无需任何改动即可在 910 上运行：

```bash
python train.py --total-tokens 2800000000 --batch-size 32 --micro-batch 8 --ctx 2048 --out-dir runs/moe-200m
python sft_train.py --init-from runs/moe-200m/ckpt_best.pt --out-dir runs/moe-200m-sft
python chat.py --checkpoint runs/moe-200m-sft/sft_best.pt
```

> 说明：
> - 910 上默认走 `npu_fusion_attention`（CANN 的 FA 算子），通过显式 `atten_mask`
>   构造下三角因果掩码，保证与 CUDA flash-attn 因果语义一致；
> - AMP 精度默认 **bf16**（无需 loss scaling）。若在 CANN8 + torch_npu 2.1.0 下
>   bf16 autocast 报错或精度异常，可设环境变量 `LLM_SNN_AMP=fp16` 切到 fp16
>   （此时建议配合 GradScaler，脚本当前未内置）；
> - AdamW 的 `fused=True` 仅 CUDA 启用，NPU 自动降级为普通实现；
> - 显存有限时减小 `--micro-batch` 即可（如 `--micro-batch 4`）。

**在 OpenI 启智云脑上训练（C2NET 接入）**：

云脑训练任务通过 C2NET 模块注入数据/输出路径。入口脚本 `openi_train.py`
已封装好 `c2net.context.prepare()`（解析 `data_url`/`train_url`/预训练模型）
与 `upload_output()`（回传输出），并把路径自动映射到 `train.py`/`sft_train.py`：

**数据集布局约定**：平台把数据集挂载到 `dataset_path`，训练数据放在其
`LLM` 子目录，即 `dataset_path/LLM/`（内含 `train.bin`、`val.bin`、`meta.json`；
SFT 还需 `sft_data.bin`、`sft_mask.bin`、`sft_val_*`、`sft_meta.json`）。
入口脚本默认把 `--data-dir` 指向 `dataset_path/LLM`。

```bash
# 预训练（数据在 dataset_path/LLM/ 下，输出自动回传）
python openi_train.py --mode pretrain --total-tokens 2800000000 \
    --batch-size 32 --micro-batch 8 --ctx 2048

# 若训练数据在数据集根目录而非 LLM 子目录：
python openi_train.py --mode pretrain --c2net-data-subdir "" \
    --total-tokens 2800000000 --batch-size 32 --micro-batch 8 --ctx 2048

# SFT（把预训练 ckpt 作为"预训练模型"挂载，传相对文件名即可自动定位）
python openi_train.py --mode sft --init-from ckpt_best.pt --epochs 3
```

要点：
- `--c2net-data-subdir` 指定 `dataset_path` 下的数据子目录，默认 `LLM`；传空串 `""` 表示数据就在数据集根目录；
- `--init-from` 会依次在 `pretrain_model_path`、`dataset_path` 下查找，无需写完整容器路径；
- 其他参数（`--lr`、`--micro-batch` 等）原样透传给底层脚本；
- 本地无 c2net 时自动回退到 `--data-dir`/`--out-dir` 本地路径，便于离线调试。

**checkpoint 保存与回传**：
- 只保留两个 checkpoint，避免海量文件累积与回传过慢：
  - 预训练：`ckpt_last.pt`（最新训练状态）+ `ckpt_best.pt`（验证集 loss 最优）；
  - SFT：`sft_last.pt`（最新）+ `sft_best.pt`（验证集最优）；
  - 每次保存都是**覆盖写**，磁盘上始终只有 2 个文件；
- checkpoint 写入 `output_path`（即 `--out-dir`），训练结束后由 `upload_output()`
  回传云端；**即使训练中途失败/被中断，也会尝试回传已保存的 checkpoint**；
- 每个 ckpt 约 2.4GB（模型 0.8GB + AdamW 状态 1.6GB），云脑回传时总共约 5GB；
- 续训用 `--init-from <dir>/ckpt_last.pt`；SFT 默认从 `ckpt_best.pt` 起步，
  推理用 `sft_best.pt`。

### 1. 下载语料

```powershell
# 中英文本（约 3.5B 字符）
python download_text.py
```

**代码语料（两种选择）**

`bigcode/starcoderdata` 是 gated 数据集，需要认证：

1. 注册/登录 https://huggingface.co
2. 打开 https://huggingface.co/datasets/bigcode/starcoderdata，
   点 **"Agree and send request to access repo"**（通常自动批准）
3. 在 https://huggingface.co/settings/tokens 创建一个 **read** token
4. 带上 token 下载（约 6B 字符 → 1.5B token）：

```powershell
$env:HF_TOKEN = "hf_xxx"              # 或直接 --token hf_xxx
python download_code.py          # 全部 5 语言；--langs 可挑
```

> 注意：新版 starcoderdata 的**语言通过 `data_dir` 选择**（只有一个 `default` config），
> 脚本已用 `data_dir=<lang>` 加载，无需手动指定 config。

不想折腾认证的话，改用非 gated 的 Python 专用源 `codeparrot/codeparrot-clean`：

```powershell
python download_code.py --source codeparrot   # Python only，无需 token
```

### 2. 重训 50K tokenizer

```powershell
# 需要新的中英+代码混合语料；命令可直接复用（仅改 vocab-size）
python tokenizer/train_tokenizer.py --vocab-size 50000 --sample-chars 60000000
# 输出 tokenizer/spm.model（若 SP 实际产出 50000 - 5 个特殊符，以实际为准，meta.json 会自动记录）
```

> 注意：换 tokenizer 后**所有旧 bins 作废**，必须重新跑第 3 步。

### 3. 重新打包 bins（memmap 流式，内存友好）

```powershell
python build_bins.py --sp-model tokenizer/spm.model --train-tokens 3000000000 --val-tokens 2000000
```

### 4. 预训练

```powershell
python train.py --total-tokens 2800000000 --batch-size 32 --micro-batch 8 --ctx 2048 --out-dir runs/moe-200m
```

- `batch-size * ctx` = 每步 token 数（32×2048 = 65K，3B token 约 43K 步）
- `micro-batch` 为单卡显存内前向批大小，剩余部分由梯度累积补齐
- 默认 cosine LR，min_lr = 0.1×lr，warmup 60M token
- checkpoint 存 `runs/moe-200m/ckpt_*.pt`，可 `--init-from` 续训
- 想跑小规模快速验证：`--total-tokens 50000000 --ctx 512`

### 5. SFT 工具调用微调

```powershell
# 下载/打包指令数据（ultrachat + alpaca-zh + 本地合成工具对话）
python prepare_sft.py --download
python prepare_sft.py --pack

# 微调
python sft_train.py --init-from runs/moe-200m/ckpt_best.pt --out-dir runs/moe-200m-sft
```

### 6. 推理 / 工具调用

```powershell
python chat.py --checkpoint runs/moe-200m-sft/sft_best.pt
python chat.py --checkpoint runs/moe-200m-sft/sft_best.pt --prompt "帮我算一下 3+5*2"
```

模型按 ChatML 输出 `<|tool_call|>{"name": "...", "arguments": {...}}` 时，
`chat.py` 自动执行本地工具（calculator/get_datetime/get_weather/web_search）并把
结果回填为 `<|tool_result|>`，循环至模型给出最终答复。

## 关键实现细节

- **RoPE + RMSNorm + GQA + SwiGLU**：标准 Llama 结构，attention 部分每 token 全激活
- **共享 Router**：所有层共用一个 router（参数量可忽略），支持 top-k softmax 路由
- **平衡正则**：`router_z_loss`（抑制 logits 爆炸）+ `router_aux_loss`（Switch-Transformer 式负载均衡），系数见 config
- **GQA**：4 个 KV 头，减少激活内存
- **weight tying**：lm_head 复用 embedding 权重
- **数据管线**：`build_bins.py` 用 memmap 逐文档 tokenize，避免 3B token 一次进内存

## 训练速度参考

A100 40G 上 200M 参数 + 梯度累积，预计 15–30K tokens/s（取决于 flash-attn 是否可用）。
按 25K tok/s 计，2.8B token 约 **31 小时**；若 `--micro-batch` 调大或开 flash-attn 更快。
