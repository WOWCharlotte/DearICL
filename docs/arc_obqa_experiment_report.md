# ARC-C 与 OpenBookQA DearICL 实验报告

## 1. 实验设置

实验采用 2 × 2 设置：ARC-C、OpenBookQA 两个数据集，分别使用 Llama3-8B 和 Qwen2.5-7B-Instruct。每组实验使用 DearICL 4-shot，在完整官方 test 集上评测。随机种子固定为 42。

### 1.1 数据集介绍

| 数据集 | 配置 | Train | Validation | Test | 任务形式 |
|---|---|---:|---:|---:|---|
| ARC-C | `ARC-Challenge` | 1119 | 299 | 1172 | 科学多选题 |
| OpenBookQA | `main` | 4957 | 500 | 500 | 四选一科学题 |

#### 1.1.1 ARC-C

ARC-C 为小学科学多选题，原始数据包含 3–5 个选项。部分样本的选项标签为 `A–D`，部分为 `1–4`。实验中按选项位置统一映射为 `A–E`，并同步重映射 `answerKey`。使用官方 train、validation、test 划分，test 仅用于最终评测。

#### 1.1.2 OpenBookQA

OpenBookQA 使用 `main` 配置，采用闭卷设置，仅输入题干和选项，不使用 `fact1` 或其他外部知识。题目均为四选一，选项标签为 `A–D`。使用官方 train、validation、test 划分，test 仅用于最终评测。

### 1.2 模型介绍

| 模型 | 本地权重 | 类型 | 上下文长度 |
|---|---|---|---:|
| Llama3-8B | `/home/model/Meta-Llama-3-8B` | Decoder-only base model | 8192 |
| Qwen2.5-7B-Instruct | `/home/model/Qwen2.5-7B-Instruct` | Decoder-only instruction-tuned model | 32768 |

Llama3-8B 使用文本续写 Prompt；Qwen2.5-7B-Instruct 使用 tokenizer 自带的 chat template。两个模型均不更新权重。

### 1.3 评估指标

- **Accuracy**：解析答案与标准答案完全一致的样本比例。无效输出按错误处理。
- **Weighted-F1**：按真实标签支持度加权的 F1；无法解析的预测记为 `INVALID`，计算时 `zero_division=0`。
- **run_time**：`run_time_s = offline_train_time_s + online_retrieval_time_s`。
  - 离线训练时间包含模型加载、候选构造、100 道验证题的 LLM 反馈生成和 RankNet 训练。
  - 在线检索时间包含测试题编码、20 个候选子集打分、top-1 检索、Prompt 构造、答案生成和解析。
  - 同时记录单题平均延迟 `avg_latency_ms` 和 P95 延迟 `p95_latency_ms`。
  - 不计入数据集和模型权重的首次下载时间。

### 1.4 参数设置

| 参数 | 取值 |
|---|---|
| 随机种子 | 42 |
| Shot 数 | 4 |
| 候选子集数 | 20 |
| 聚类数 | 4 |
| 验证反馈题数 | 100 |
| Embedding 模型 | `/home/model/roberta-large`，1024 维 mean pooling |
| 候选特征 | 查询向量与 4 个示例向量的均值 |
| Ranker | 1024 → 256 → 1，ReLU |
| Ranker 优化器 | AdamW，lr=1e-3，weight_decay=1e-4 |
| Ranker batch size | 16 |
| Ranker 最大 epoch | 100，连续 10 轮损失不下降则早停 |
| 生成策略 | Greedy decoding |
| 最大新生成 token | 16 |
| 输出约束 | 首个有效 token 限定为合法选项字母，随后结束生成 |
| 最终示例集 | 每道测试题的 top-1 候选子集 |

### 1.5 硬软件环境

| 项目 | 配置 |
|---|---|
| 操作系统 | Ubuntu 22.04.5 LTS |
| GPU | NVIDIA GeForce RTX 4090，24 GB |
| CPU / 内存 | 16 vCPU / 62 GB |
| NVIDIA 驱动 / CUDA | 595.80 / 13.2 |
| Python | 3.12.11 |
| PyTorch | 2.13.0+cu132 |
| Transformers | 5.22.2 |
| datasets | 5.2.0 |
| 精度 / batch size | bf16 / 1 |
| 模型执行方式 | 两个 LLM 串行加载，每次仅驻留一个生成模型 |

## 2. 训练方法

采用 DearICL 候选示例子集选择流程，LLM 权重全程冻结，仅训练 RankNet 排序器。

1. 将题干和选项编码为语义向量。
2. 在 train 集上聚类并构造候选 4-shot 子集。
3. 对每个候选子集，在固定 validation 子集上生成答案。
4. 答案正确记奖励 1，错误或不可解析记奖励 0。
5. 根据同一验证题在不同候选子集上的奖励差异构造 pairwise 样本。
6. 使用 RankNet 学习候选子集的相对排序。
7. 对每道 test 题计算 20 个候选子集得分，选择 top-1 子集生成最终答案。

## 3. 训练样本构造

1. 使用 `/home/model/roberta-large` 对“题干 + 选项”进行编码，得到 1024 维 L2 归一化向量。
2. 对官方 train 集执行 KMeans，聚类数为 4。
3. 每个候选子集从 4 个簇中各抽取 1 条样本，形成 4-shot 子集。
4. 每个数据集生成 20 个互不重复的候选子集。
5. 按答案标签分层抽取 100 道 validation 题，作为所有候选子集的固定反馈集。
6. 每个候选子集产生 100 条 `(query, candidate)` 记录，共 2000 条记录。
7. 同一验证题下，奖励不同的候选子集组成 RankNet 正负对；奖励相同的候选对不参与训练。
8. 每个“数据集 × 模型”组合独立训练一个 RankNet。

| 实验 | 反馈记录数 | Pairwise 训练样本数 | RankNet 实际 epoch |
|---|---:|---:|---:|
| ARC-C + Qwen2.5-7B-Instruct | 2000 | 674 | 100 |
| OpenBookQA + Qwen2.5-7B-Instruct | 2000 | 896 | 100 |
| ARC-C + Llama3-8B | 2000 | 1709 | 100 |
| OpenBookQA + Llama3-8B | 2000 | 1734 | 22 |

## 4. 实验结果及分析

### 4.1 总体结果

| 数据集 | 模型 | 方法 | Test | Accuracy | Weighted-F1 | 离线训练(s) | 在线检索(s) | run_time(s) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| ARC-C | Llama3-8B | DearICL | 1172 | 0.6792 | 0.6823 | 172.43 | 98.28 | 270.72 |
| ARC-C | Llama3-8B | CoopES | 1172 | **0.8200** | **0.8234** | — | — | — |
| ARC-C | Qwen2.5-7B-Instruct | DearICL | 1172 | **0.8976** | **0.8976** | 163.01 | 102.14 | 265.15 |
| ARC-C | Qwen2.5-7B-Instruct | CoopES | 1172 | 0.8840 | 0.8841 | — | — | — |
| OpenBookQA | Llama3-8B | DearICL | 500 | 0.6760 | 0.6771 | 139.91 | 38.66 | 178.57 |
| OpenBookQA | Llama3-8B | CoopES | 500 | **0.7520** | **0.7535** | — | — | — |
| OpenBookQA | Qwen2.5-7B-Instruct | DearICL | 500 | 0.8740 | 0.8737 | 151.21 | 39.28 | 190.49 |
| OpenBookQA | Qwen2.5-7B-Instruct | CoopES | 500 | **0.8760** | **0.8759** | — | — | — |

CoopES 未提供离线训练、在线检索和 run_time 数据，表中以“—”表示。

### 4.2 候选子集选择情况

| 实验 | 实际使用的候选数 | 选择次数分布 |
|---|---:|---|
| ARC-C + Qwen2.5-7B-Instruct | 2 | candidate 13：755；candidate 6：417 |
| OpenBookQA + Qwen2.5-7B-Instruct | 2 | candidate 7：495；candidate 8：5 |
| ARC-C + Llama3-8B | 5 | candidate 12：618；candidate 9：328；candidate 14：183；candidate 18：23；candidate 11：20 |
| OpenBookQA + Llama3-8B | 1 | candidate 7：500 |

候选子集对应的 validation Accuracy：ARC-C + Qwen 中 candidate 13、6 均为 0.93；OpenBookQA + Qwen 中 candidate 7、8 分别为 0.86、0.84；ARC-C + Llama 中 candidate 12、9、14、18、11 分别为 0.75、0.75、0.74、0.74、0.74；OpenBookQA + Llama 中 candidate 7 为 0.72。

原始预测、候选反馈、排序器、选择结果和指标文件位于 `output/arc_challenge_qwen2.5_dearicl_4shot/`、`output/openbookqa_qwen2.5_dearicl_4shot/`、`output/arc_challenge_llama3_dearicl_4shot/` 和 `output/openbookqa_llama3_dearicl_4shot/`。
