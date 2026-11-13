# ARC-C 与 OpenBookQA DearICL 实验结果

## 实验设置

- 方法：DearICL 4-shot，20 个候选子集、4 个语义簇、100 道验证反馈题。
- 模型：Llama3-8B、Qwen2.5-7B-Instruct。
- 评测集：完整官方 test 集；随机种子 42；bf16；batch size 1。
- `run_time` = 离线训练时间 + 在线检索时间。

## 结果

| 数据集 | 模型 | Test | Accuracy | Weighted-F1 | 离线训练(s) | 在线检索(s) | run_time(s) | 无效输出 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ARC-C | Qwen2.5-7B-Instruct | 1172 | 0.8976 | 0.8976 | 163.01 | 102.14 | 265.15 | 0 |
| OpenBookQA | Qwen2.5-7B-Instruct | 500 | 0.8740 | 0.8737 | 151.21 | 39.28 | 190.49 | 0 |
| ARC-C | Llama3-8B | 1172 | 0.6792 | 0.6823 | 172.43 | 98.28 | 270.72 | 0 |
| OpenBookQA | Llama3-8B | 500 | 0.6760 | 0.6771 | 139.91 | 38.66 | 178.57 | 0 |

## 结果文件

| 实验 | 输出目录 |
|---|---|
| ARC-C + Qwen2.5-7B-Instruct | `output/arc_challenge_qwen2.5_dearicl_4shot` |
| OpenBookQA + Qwen2.5-7B-Instruct | `output/openbookqa_qwen2.5_dearicl_4shot` |
| ARC-C + Llama3-8B | `output/arc_challenge_llama3_dearicl_4shot` |
| OpenBookQA + Llama3-8B | `output/openbookqa_llama3_dearicl_4shot` |

每组输出包含 `predictions.csv`、`selected_exemplars.json`、`candidate_feedback.jsonl`、`ranker.pt`、`run_config.json` 和 `metrics.json`。
