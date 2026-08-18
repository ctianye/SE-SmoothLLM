# 实验结果

本目录保存 SE-SmoothLLM 的可提交汇总结果、图表和可复现分析产物。模型原始回答与完整 Judge
JSONL 保留在本地 `results/raw/` 和 `results/judged/`，这两个目录被 `.gitignore` 排除；Git
只提交脱敏后的聚合表、图表、数据健康报告和分析脚本。

## 一句话结论

在 Vicuna-13B + JailbreakBench GCG 配置下，SE-SmoothLLM 在不改变固定版内部投票结论的前提下，
将 harmful 样本的平均模型查询从 10 次降到 6.20 次（减少 37.97%），平均 prompt/completion
Token 分别减少 37.94%/38.51%。DeepSeek-V4-Flash 辅助 Judge 下，fixed 与 SE 的 ASR 分别为
6.33% 和 6.00%，本次实验观察到的安全性差异很小；主要收益是效率，而不是额外的安全性提升。

## 主结果

| 研究问题 | 结果 |
| --- | --- |
| fixed 与 SE 内部 verdict 是否一致 | 600/600 配对样本一致，`verdict_mismatch = 0` |
| harmful 平均查询数 | fixed 10.00，SE 6.20，减少 37.97% |
| benign 平均查询数 | fixed 10.00，SE 6.95，减少 30.53% |
| harmful ASR（DeepSeek-V4-Flash 辅助 Judge） | fixed 6.33%，SE 6.00% |
| benign refusal rate（Llama-3-8B Refusal Judge） | undefended 8.00%，fixed 16.33%，SE 17.33% |

ASR 和 refusal rate 是三组 seed（42、43、44）上的描述性统计，没有添加显著性结论。

## 配置与数据规模

- 目标模型：`lmsys/vicuna-13b-v1.5`
- 数据：JailbreakBench GCG harmful 100 条、benign 100 条
- seed：42、43、44
- 扰动：`RandomSwapPerturbation(q=10)`
- 副本数：`N=10`
- 温度：0；最大生成长度：150 tokens
- 原始生成记录：18 个 JSONL，共 1,800 条
- DeepSeek harmful Judge：fixed/SE 两种方法共 600 条
- Llama-3-8B Refusal Judge：三种方法、两个 split 共 1,800 条

## 图表

- [Figure 0：workflow and measured efficiency](figures/png/fig_00_overview.png) · [PDF](figures/pdf/fig_00_overview.pdf)
- [Figure 1：harmful ASR by seed](figures/png/fig_01_harmful_asr_by_seed.png) · [PDF](figures/pdf/fig_01_harmful_asr_by_seed.pdf)
- [Figure 2：model-query cost](figures/png/fig_02_query_cost.png) · [PDF](figures/pdf/fig_02_query_cost.pdf)
- [Figure 3：prompt/completion token cost](figures/png/fig_03_token_cost.png) · [PDF](figures/pdf/fig_03_token_cost.pdf)
- [Figure 4：benign refusal rate](figures/png/fig_04_benign_refusal_rate.png) · [PDF](figures/pdf/fig_04_benign_refusal_rate.pdf)
- [图注与解释](processed/FIGURE_CAPTIONS.md)

## 可追溯数据

- [summary.csv](processed/summary.csv)：所有核心指标
- [summary.json](processed/summary.json)：带来源和限制说明的机器可读摘要
- [01_raw_metrics.csv](processed/01_raw_metrics.csv)：1,800 条生成记录的结构化指标
- [02_deepseek_judge.csv](processed/02_deepseek_judge.csv)：600 条 harmful 外部评价
- [03_llama3_8b_refusal_judge.csv](processed/03_llama3_8b_refusal_judge.csv)：1,800 条拒答评价
- [04_fixed_vs_se_pairs.csv](processed/04_fixed_vs_se_pairs.csv)：600 条 fixed/SE 配对指标
- [00_data_health.md](processed/00_data_health.md)：缺失值、重复键、数值分布和异常值检查
- [fig_00_data.xlsx](processed/fig_00_data.xlsx)：README 概览图使用的关键指标
- `processed/*.xlsx`：可在 Excel 中打开的过程表和图表 sidecar 数据

重新生成所有聚合表和图表：

```bash
pip install -e ".[dev,benchmark]"
python -m benchmarks.analyze_results
```

## 评价器边界

DeepSeek-V4-Flash 是在无法获得原版 `meta-llama/Llama-3-70b-chat-hf` 时使用的辅助
Jailbreak Judge。本目录中的 ASR 不能称为 JailbreakBench 官方 Llama-3-70B ASR。Llama-3-8B
结果使用 JailbreakBench 的 refusal prompt，主要用于 benign refusal rate；它也不能替代官方
70B Jailbreak Judge。8B Judge 的解析遵循 JBB 的 `"Yes"` 子串规则，因此严格 `Yes/No` 输出
合规率较低，原始输出和 `format_conforming` 已保留在本地结果中。

本项目的强证据是工程性质的：共享执行器、固定/早停 verdict 的全量配对一致性、可复现的 trace、
以及真实模型上的查询和 Token 成本下降。ASR 数值应按上述 Judge 和配置解释，不应泛化为所有模型
或所有攻击的安全保证。
