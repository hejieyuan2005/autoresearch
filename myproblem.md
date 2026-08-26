# 安装&启动

## A800 80G

pip install uv -i https://mirrors.aliyun.com/pypi/simple

pip download -r requirements.txt -d /root/autodl-tmp/packages/

pip install --no-index --find-links=/root/autodl-tmp/packages/ -r requirements.txt

export HF_ENDPOINT=https://hf-mirror.com

uv run prepare.py

uv run train.py

COMMAND: claude
/init
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.


# 我的问题

## Q1: 是如何实现自我研究的？

核心是一个**agent 驱动的「编辑-训练-评估-保留/回滚」循环**，所有「研究决策」都由 agent 自己做出：

**1. 唯一可变文件 = `train.py`**（CLAUDE.md:13）
- `prepare.py` 冻结（数据 + tokenizer + `evaluate_bpb`），所以「研究」被严格收束在模型架构、超参数、优化器这几个维度
- 真值指标固定为 `val_bpb`（bits per byte，vocab-size-independent），消除了「换个指标就涨」的取巧空间

**2. 5 分钟 wall-clock 预算**（CLAUDE.md:11）
- 所有实验等长训练时间，只比较「在我的机器上 5 分钟内能训出的最佳模型」
- 强制 agent 在**有限 FLOPs 下**做架构/算法权衡，而不是堆资源

**3. 实验循环**（CLAUDE.md:103-108）
```
git checkout -b autoresearch/<tag>  →  分支隔离
改 train.py → git commit → uv run train.py > run.log 2>&1
                ↓ grep val_bpb / peak_vram_mb → 追加 results.tsv
                ↓
       val_bpb 降低？  是 → 保留；否 → git reset 回滚
```
- `results.tsv` 不进 git，所以它只是 agent 的本地记忆
- 「保留/回滚」这个二值反馈让 agent 本身充当了 bandit 控制器

**4. 无人值守**（CLAUDE.md:110）
- 一旦进入循环**不被询问**，靠 `commit + val_bpb` 这一对信号自我驱动
- 简化偏好也是内置规则：「~0.001 val_bpb 提升若伴随 20 行 hack 不值得，删代码的胜出必保留」—— 这是把「奥卡姆剃刀」写进了 reward function

所以「自我研究」其实不是模型自己在研究，而是**一个 LLM agent 在一个被刻意收紧的沙盒（冻结真值、固定时间预算、二值 reward）里做自动 hyperparameter/architecture search**。`analysis.ipynb` 画出的 val_bpb 随 commit 的曲线，本质上是一次 random/greedy search 的轨迹。

## Q2: 真值指标固定为 val_bpb（bits per byte，vocab-size-independent）是什么意思？

**BPB = 把交叉熵按「原始文本字节数」归一化**，而不是按 token 数。

**计算方式**（粗略）：
$$\text{bpb} = \frac{\sum_t \text{CE}(\text{token}_t)}{\sum_t \text{bytes}(\text{token}_t)} \cdot \log_2 e$$

- 分子：验证集上每个 token 的交叉熵之和（nats，转成 bits）
- 分母：这些 token **原本对应的 UTF-8 字节数**（即 `token_bytes.pt` 里那张表）

**为什么 vocab-size-independent**：

常规 cross-entropy / perplexity 都是「per token」。vocab 一扩，每个 token 携带的信息变多，loss 自然下降 —— 这是**度量伪影**，不是模型变好了。例如：
- 模型 A：vocab=256（字符级），val_loss=1.2 nats/token
- 模型 B：vocab=32k（BPE），val_loss=0.8 nats/token

直观上 B 更好，但其中一部分是因为 BPE 把「经常共现的字符」打包成了单个 token，分母天然小。

BPB 的分母是**真实字节数**，所以「一个 token 覆盖多少字节」这件事被抵消了 —— vocab=256 和 vocab=64k 的模型，在同一个 val_bpb 下可直接比较，**不存在「换个 tokenizer 就能白嫖分数」这条路**。

这也是项目把它钉死为 ground truth 的原因：agent 不能靠改 tokenizer、改 vocab、改 chunk size 来制造虚假改进，能动的只有 `train.py` 里的架构和优化器。
