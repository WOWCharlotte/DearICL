# ARC-C 与 OpenBookQA 实验方案

## 1. 实验目的

在 ARC-Challenge（ARC-C）和 OpenBookQA 上，使用 Llama3-8B 与 Qwen2.5-7B-Instruct 分别评估 DearICL 4-shot 的表现，报告 Accuracy、Weighted-F1 与 run_time。共 4 组实验：2 个数据集 × 2 个模型。实验不更新 LLM 权重。

## 2. 训练方法

采用有限预算的 DearICL 子集选择流程：

1. **数据预处理**：将两个数据集统一为 `id`、`question`、`choices`、`answer`；ARC-C 的选项标签存在 `A–D` 与 `1–4` 两种格式，按选项位置统一映射为 `A–E`，并同步重映射 `answerKey`；OpenBookQA 保持 `A–D`。示例仅拼接题干、选项和正确选项字母。
2. **语义编码**：使用 `/home/model/roberta-large` 对题干和选项文本做 mean pooling，得到 1024 维 L2 归一化向量。
3. **候选构造**：在官方 train 集上执行 KMeans（4 簇），每个候选子集从 4 个簇各采 1 条示例，共生成 20 个 4-shot 候选子集。
4. **验证反馈**：每个数据集使用 seed 42 固定抽取 100 道 validation 题；对每个模型、每个候选子集生成答案，以答案完全正确为 1、错误或不可解析为 0，得到 `(query, candidate)` 奖励。
5. **排序器训练**：以查询向量与 4 个示例向量的均值为候选特征，使用 RankNet 学习候选项的相对优劣；每个模型和数据集独立训练一个排序器。
6. **测试选择**：对每道 test 题给 20 个候选子集打分，选择 top-1 子集生成最终答案。

候选臂首轮全部评估一次，不进行重复 bandit 拉取，以控制单卡实验成本。LLM 推理全程冻结权重；训练仅针对 DearICL 排序器。

## 3. 硬软件环境

| 项目 | 配置 |
|---|---|
| 操作系统 | Ubuntu 22.04.5 LTS |
| GPU | NVIDIA GeForce RTX 4090，24 GB |
| CPU / 内存 | 16 vCPU / 62 GB |
| NVIDIA 驱动 / CUDA | 595.80 / 13.2 |
| Python | 3.12.11 |
| PyTorch | 2.13.0+cu132 |
| Transformers | 5.22.2 |
| 精度与批大小 | bf16，batch size 1 |
| 执行方式 | 两个模型串行加载，每次仅驻留一个生成模型 |

## 4. 数据集介绍

| 数据集 | 设置 | Train | Validation | Test | 数据形式 |
|---|---|---:|---:|---:|---|
| [ARC-Challenge](https://huggingface.co/datasets/allenai/ai2_arc) | `ARC-Challenge` | 1119 | 299 | 1172 | 小学科学多选题，3–5 个选项，统一映射为 A–E |
| [OpenBookQA](https://huggingface.co/datasets/allenai/openbookqa) | `main`，闭卷 | 4957 | 500 | 500 | 四选一科学题，答案标号为 A–D |

- ARC-C 使用官方 `train/validation/test` 划分。
- OpenBookQA 仅使用题干和选项，不注入 `fact1`，不采用 `additional` 配置。
- test 集仅用于最终评测，不参与候选生成、奖励计算或模型选择。

## 5. 模型介绍

| 模型 | 本地权重 | 类型 | 上下文长度 |
|---|---|---|---:|
| Llama3-8B | `/home/model/Meta-Llama-3-8B` | Decoder-only base model | 8192 |
| Qwen2.5-7B-Instruct | `/home/model/Qwen2.5-7B-Instruct` | Decoder-only instruction-tuned model | 32768 |

环境设置：

- Llama3-8B 使用纯 completion prompt，按 few-shot 文本续写方式生成答案。
- Qwen2.5-7B-Instruct 使用 tokenizer 自带的 chat template。
- 两个模型均要求只输出一个选项字母，不生成解题理由。

## 6. 评价指标

### 6.1 Accuracy

\[
\text{Accuracy} = \frac{\text{预测正确的题目数}}{\text{测试题总数}}
\]

答案必须与 `answerKey` 完全一致；无法解析、空输出或输出多个冲突答案均判错。

### 6.2 Weighted-F1

使用 `sklearn.metrics.f1_score`，按真实标签支持度加权：

```python
f1_score(y_true, y_pred, labels=true_labels, average="weighted", zero_division=0)
```

无法解析的预测统一记为 `INVALID`；`true_labels` 为测试集真实标签集合。Weighted-F1 作为类别不均衡情况下的补充指标。

### 6.3 run_time

`run_time` 为离线训练与在线检索的总耗时：

```text
run_time_s = offline_train_time_s + online_retrieval_time_s
```

- `offline_train_time_s`：离线耗时，包含模型加载、候选构造、100 道 validation 题的 LLM 反馈生成及 RankNet 训练。
- `online_retrieval_time_s`：在线耗时，包含 test 查询编码、20 个候选子集打分与 top-1 检索、prompt 构造、LLM 生成及答案解析。
- `avg_latency_ms`、`p95_latency_ms`：在线阶段单题平均/P95 延迟。
- 不将数据集和模型权重的首次下载时间计入 `run_time`；两个模型使用相同的 batch size 与生成长度。

## 7. 参数设置

| 参数 | 取值 |
|---|---|
| 随机种子 | 42 |
| Shot 数 | 4 |
| 候选子集数 | 20 |
| 聚类数 | 4 |
| 验证反馈题数 | 100（按答案标签分层抽样） |
| 生成策略 | Greedy decoding |
| 最大新生成 token | 16 |
| Embedding | roberta-large，1024 维 mean pooling |
| Ranker | 1024 → 256 → 1，ReLU |
| Ranker 优化器 | AdamW，lr=1e-3，weight_decay=1e-4 |
| Ranker batch size | 16 |
| Ranker 最大 epoch | 100，训练损失连续 10 轮不下降则早停 |
| 最终子集 | 测试题 top-1 候选子集 |
| 评测集 | 完整官方 test 集 |

每个数据集、每个模型独立训练排序器，并按测试题选择 top-1 候选子集。

## 8. 注意事项

1. 严禁使用 test 标签进行候选排序、超参数选择或早停。
2. 同一数据集下，两个模型必须使用相同的 test 顺序和答案解析规则。
3. 两个模型的 prompt 接口不同，但 shot 数、示例内容、解码策略和最大生成长度保持一致。
4. OpenBookQA 为闭卷设置，不能引入 `fact1` 或其他外部知识。
5. 无效输出按错误处理，并在结果中单独记录 `invalid_count`。
6. 主结果不量化模型；若出现显存不足，仅允许缩短输入上限、使用 `device_map="auto"` 和及时释放 CUDA 缓存，不得改变 bf16 精度。
7. 当前仓库的 `Code/DearICL_aqua_final.py` 依赖缺失的 `ltr` 模块且包含损坏的 pickle，后续实现应复用 `Code/reproduce_aqua_local.py` 的本地 Transformers 推理方式，不直接依赖旧脚本。
8. 本方案固定单 seed 运行，结果用于评估 DearICL 在两组数据集上的表现，不额外声称统计显著性。

## 9. 运行指令

以下命令对应统一入口 `Code/run_mcq_icl.py`。脚本优先从 `data/ARC_CHALLENGE` 和 `data/OPENBOOKQA` 加载已下载数据，缺失时再下载 HF 数据集；模型只从本地路径加载。

```bash
cd /home/Github/DearICL
source .venv/bin/activate

# 补齐数据加载依赖；模型推理依赖见 requirements.txt
python -m pip install -U "datasets>=3.0" "accelerate>=1.0"

for dataset in arc_challenge openbookqa; do
  for item in \
    "llama3:/home/model/Meta-Llama-3-8B" \
    "qwen2.5:/home/model/Qwen2.5-7B-Instruct"; do
    name="${item%%:*}"
    model_path="${item#*:}"

    # DearICL 4-shot
    python Code/run_mcq_icl.py \
      --dataset "$dataset" \
      --model-path "$model_path" \
      --method dearicl_4shot \
      --shots 4 \
      --candidates 20 \
      --validation-size 100 \
      --seed 42 \
      --max-new-tokens 16 \
      --output-dir "output/${dataset}_${name}_dearicl_4shot"
  done
done
```

每个输出目录包含：

- `predictions.csv`：题目、标准答案、解析答案、原始输出、是否正确、单题延迟。
- `selected_exemplars.json`：DearICL 为各 test 题选择的示例及候选得分。
- `metrics.json`：`accuracy`、`weighted_f1`、`run_time_s`、`offline_train_time_s`、`online_retrieval_time_s`、`avg_latency_ms`、`p95_latency_ms`、`invalid_count`。
