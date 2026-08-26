# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目定位

`autoresearch` 是一个**自主机器学习研究项目**：让 AI agent 通过反复修改 `train.py` 来优化模型在固定时间预算下的验证指标 `val_bpb`。基于 Karpathy 的 nanochat 简化而来。

设计上的核心约束：每个实验**固定 5 分钟 wall-clock 训练时间**（不含启动和编译），使得跨平台/跨架构的结果仍可比较 —— 优化的是「在我的机器上 5 分钟内能训出的最佳模型」。

## 三文件架构

| 文件 | 角色 | 是否可改 |
|------|------|---------|
| `prepare.py` | 数据下载、BPE tokenizer 训练、BOS-aligned dataloader、`evaluate_bpb` 评估 | **禁止修改** |
| `train.py` | GPT 模型、Muon+AdamW 优化器、训练循环、所有超参数 | **agent 唯一可改的文件** |
| `program.md` | agent 的运行手册：实验流程、results.tsv 格式、循环规则 | 是流程文档 |

`prepare.py` 中的 `evaluate_bpb` 是**固定的真值指标** —— bits per byte（vocab-size-independent），排除 special tokens。要使用 `val_bpb` 之外的指标或修改评估逻辑都违反项目规则。

## 关键路径与缓存

- 数据 & tokenizer 缓存：`~/.cache/autoresearch/`
  - 训练分片：`~/.cache/autoresearch/data/shard_*.parquet`（默认 10 个分片 + 1 个固定验证分片 `shard_06542.parquet`）
  - Tokenizer：`~/.cache/autoresearch/tokenizer/{tokenizer.pkl, token_bytes.pt}`
- 数据集源：`https://hf-mirror.com/datasets/karpathy/climbmix-400b-shuffle/resolve/main`
- 不提交到 git：`results.tsv`、`CLAUDE.md`、`AGENTS.md`、`.venv/`、`results/`、`queue/`、`dev/`（见 `.gitignore`）

## 常用命令

```bash
uv sync                        # 安装依赖（PyTorch 2.9.1+cu128，从 PyTorch 官方源）
uv run prepare.py              # 下载数据 + 训练 tokenizer（一次性，约 2 分钟）
uv run prepare.py --num-shards 8  # 只下载 8 个分片用于测试
uv run train.py                # 跑一次 5 分钟训练（agent 用：uv run train.py > run.log 2>&1）
grep "^val_bpb:\|^peak_vram_mb:" run.log  # 抽取关键指标
```

Python 版本要求：`>=3.10`（`.python-version` 固定为 `3.10`）。需要在单 NVIDIA GPU 上运行（H100 上验证）。

## 训练架构要点（agent 修改时需理解）

`train.py` 中的关键设计（来自 nanochat 简化）：

- **模型**：GPT with RoPE、QK-norm、value-embedding（ResFormer）、`window_pattern="SSSL"` 交替滑动窗口注意力、logits softcap=15。attention 使用 FlashAttention-3（Hopper 上用 `varunneal/flash-attention-3`，其他用 `kernels-community/flash-attn3`），按 `torch.cuda.get_device_capability()` 自动切换。
- **优化器**：组合 `MuonAdamW` —— 2D 矩阵参数走 Muon（含 Polar Express 正交化 + NorMuon 方差缩减 + cautious weight decay），其他参数走 AdamW。LR 按 `1/√(dmodel/768)` 缩放。
- **超参数块**（在 `train.py` 底部直接编辑，无 CLI flag）。架构：`ASPECT_RATIO=64`（`model_dim = depth * ASPECT_RATIO`）、`HEAD_DIM=128`、`WINDOW_PATTERN="SSSL"`（L=full, S=half context；小机器可改 `"L"`）、`DEPTH=8`。训练：`DEVICE_BATCH_SIZE=128`、`TOTAL_BATCH_SIZE=2**19`（必须能被 `DEVICE_BATCH_SIZE * MAX_SEQ_LEN` 整除）。优化器：`MATRIX_LR=0.04`（Muon）、`EMBEDDING_LR=0.6`（AdamW）、`UNEMBEDDING_LR=0.004`、`SCALAR_LR=0.5`、`WEIGHT_DECAY=0.2`、`ADAM_BETAS=(0.8, 0.95)`。调度：`WARMUP_RATIO=0.0`、`WARMDOWN_RATIO=0.5`、`FINAL_LR_FRAC=0.0`。
- **时间预算守门**：循环在 `step > 10 and total_training_time >= TIME_BUDGET` 时退出；前 10 步用于 warmup（排除编译耗时）。
- **OOM/异常处理**：训练 loss > 100 或 NaN 会打印 "FAIL" 并 `exit(1)`；单次实验若超过 10 分钟应被外部 kill。
- **复现性**：seed 固定为 42（CPU+CUDA），`PYTORCH_ALLOC_CONF=expandable_segments:True` 和 `HF_HUB_DISABLE_PROGRESS_BARS=1` 已在 `train.py` 顶部设置。
- **MFU 口径**：`mfu_percent` 用 `H100_BF16_PEAK_FLOPS = 989.5e12` 计算；非 H100 平台上报值会偏低（仅反映相对 H100 的利用率，不是平台真实 MFU）。

## 实验循环约定（详见 `program.md`）

1. agent 在专用分支上工作：`git checkout -b autoresearch/<tag>` from master（tag 用日期，如 `mar5`）。
2. 修改 `train.py` → `git commit` → `uv run train.py > run.log 2>&1`（不要 `tee`，避免输出淹没上下文）。
3. 从 `run.log` 抽取 `val_bpb` 和 `peak_vram_mb`，追加到 `results.tsv`（**TSV 不要提交**）。
4. 若 `val_bpb` 降低 → 保留提交；否则 `git reset` 回退。
5. 一旦开始实验循环**不要停下来询问用户** —— agent 设计上无人值守运行到被手动中断。

`results.tsv` 列：`commit	val_bpb	memory_gb	status	description`（status: `keep` / `discard` / `crash`）。可用 `analysis.ipynb` 绘制 `val_bpb` 随 commit 的趋势图（需先 `uv sync` 装好 `matplotlib`/`pandas`）。

## 简化原则（program.md 强调）

当 `val_bpb` 改善幅度相近时，**更简单的实现胜出**。删除代码得到持平或更好的结果是有价值的「simplification win」。`~0.001 val_bpb` 的提升若伴随 20 行 hack 不值得；若来自删代码则必保留。