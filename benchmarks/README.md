# 基准实验

本目录用于保存可复现的数据加载器、实验命令、评价配置和聚合结果。原始下载数据、
模型完整回答和其他体积较大的运行产物保存在本地已被忽略的目录中；提交到仓库的是
`results/processed/` 的聚合表和 `results/figures/` 的图表。最终结果总览见
[`results/README.md`](../results/README.md)。

## 研究目标

本项目通过统一的数据、目标模型、生成参数、扰动配置和评价器，对无防御模型、固定预算
SmoothLLM 与精确早停 SE-SmoothLLM 进行公平比较。实验需要回答以下四个研究问题。

### RQ1：SmoothLLM 相比无防御模型把 ASR 降低多少？

在同一个目标模型和同一批现成越狱攻击提示词上，分别运行无防御推理和固定预算
SmoothLLM，并使用同一个外部越狱评价器计算攻击成功率（Attack Success Rate, ASR）。
该问题衡量随机平滑防御带来的安全效果，不用于评价早停策略本身。

### RQ2：SE-SmoothLLM 是否保持固定 SmoothLLM 的内部投票结论？

固定预算 SmoothLLM 与 SE-SmoothLLM 必须共享相同的副本数、扰动比例、随机种子、
扰动顺序、目标模型、内部 Judge 和投票规则。实验比较两者的最终内部 verdict，并报告
`verdict_mismatch`。精确早停的目标是在减少查询的同时保持该指标为零。

这里的 Exact 性质只保证内部投票结论一致，不自动保证两种方法返回完全相同的回答，
因此最终回答仍需使用同一个外部评价器分别计算 ASR。

### RQ3：SE-SmoothLLM 平均减少多少模型查询和 Token？

以始终执行 `N` 次生成的固定预算 SmoothLLM 为基线，统计 SE-SmoothLLM 的平均
`copies_used`、提示 Token、生成 Token 和提前停止比例。主要报告平均查询数、查询减少率、
Token 减少率和 early-stop rate。该问题衡量 SE-SmoothLLM 的效率收益，不把效率收益表述为
额外的安全性提升。

### RQ4：防御是否提高正常请求的拒答率？

在与有害行为主题对应的正常请求数据上，分别运行无防御模型、固定预算 SmoothLLM 和
SE-SmoothLLM，并使用统一的拒答评价器计算 benign refusal rate。该问题用于检查 ASR 的
下降是否伴随着明显的过度拒答，不能只报告有害请求上的防御结果而忽略正常可用性。

## 当前完成范围

| 问题 | 当前证据 | 说明 |
| --- | --- | --- |
| RQ1 | 部分完成 | 当前 DeepSeek 外部 Judge 只评价 fixed/SE 的 600 条 harmful 回答，没有为 undefended 追加 300 条 DeepSeek Judge 调用，因此不宣称完整的 SmoothLLM-vs-undefended 外部 ASR。 |
| RQ2 | 完成 | 600 条 fixed/SE 配对样本的内部 verdict mismatch 为 0；另有 N=10 的 1024 序列穷举 Exact 验证。 |
| RQ3 | 完成 | raw trace 汇总了 copies、prompt tokens、completion tokens、latency 和 early-stop rate。 |
| RQ4 | 完成 | Llama-3-8B Refusal Judge 已评价 1,800 条记录，benign refusal rate 已汇总。 |

这一定义保留了实验边界：README 中的 ASR 只表示 DeepSeek-V4-Flash 辅助 Judge 下的 fixed/SE
对照，不把内部 PrefixJudge 或 8B refusal 标签当成官方 JBB ASR。

## 结果报告原则

- 所有方法必须使用相同的数据版本、模型版本、对话模板和生成参数。
- 内部 Judge 用于 SmoothLLM 投票，外部评价器用于计算最终 ASR，两者的职责必须分开。
- 官方论文或 JailbreakBench 的数值只作为复现参考，不作为本项目已经取得的结果。
- 实验结果必须同时报告安全性、效率和正常请求拒答率，不能只选择有利指标。

## JBB 数据加载器

安装基准实验可选依赖：

```bash
pip install -e ".[dev,benchmark]"
```

加载唯一主实验配置对应的全部样本：

```python
from benchmarks.jbb_loader import load_jbb_samples

samples = load_jbb_samples()
harmful = [sample for sample in samples if sample.split == "harmful"]
benign = [sample for sample in samples if sample.split == "benign"]

assert len(harmful) == 100
assert len(benign) == 100
```

`load_harmful_samples()` 从固定 Git commit 的 JailbreakBench GCG artifact 读取 `index`、
`goal`、`behavior`、`category` 和 `prompt`，不会读取 artifact 中已有的 `response`、
`jailbroken` 或 ASR 元数据。`load_benign_samples()` 从固定 Hugging Face revision 的
JBB-Behaviors benign split 读取正常请求，并将 `Goal` 作为统一样本的 `prompt`。

两个 split 都保留官方的原始 index，因此样本的唯一身份是 `(split, index)`，而不是只看
index。数据 URL、revision、预期数量和主实验参数统一保存在
[`configs/vicuna_gcg.json`](configs/vicuna_gcg.json) 中。

## 可恢复生成入口

模型服务启动后，先用一条正常请求验证协议、模型名和 token 统计：

```bash
python -m benchmarks.probe_backend
```

仅查看主配置将产生多少任务，不调用模型：

```bash
python -m benchmarks.run_jbb --dry-run
```

运行完整主配置：

```bash
python -m benchmarks.run_jbb \
  --workers 4 \
  --output results/raw/jbb-vicuna-13b-gcg-white-box.jsonl
```

`--method`、`--seed` 和 `--split` 可以重复传入，用于只跑配置中的一部分；`--limit N`
表示每个选定 split 只取前 N 条样本，适合 smoke test。未显式传入时，方法、seed、模型、
扰动、生成参数和服务参数都从唯一 JSON 配置读取。

运行器使用线程并发向 vLLM 提交不同样本，使服务能够连续批处理；同一个样本内部仍按副本
顺序执行，因此早停只读取当前样本已经出现的投票。每个样本的有效扰动 seed 由主 seed、
split 和 index 稳定派生，固定版与早停版对同一样本拥有相同扰动序列前缀，不同样本不会
机械复用完全相同的随机流。

输出 JSONL 的每条记录均含配置 SHA-256、方法、主 seed、有效 seed、样本来源字段、最终
结果和完整 trace。记录完成后立即 `fsync`。重启同一命令时，已有任务会被跳过；断电留下的
最后一个不完整片段会被移除，而文件中间的损坏、配置指纹不同或重复 key 仍会直接报错。

当前入口只完成目标模型生成和 `PrefixJudge` 内部投票。它不把内部标签冒充最终 ASR 或
benign refusal 指标。外部越狱评价器与拒答评价器读取已保存的最终 response 单独运行，
因此增加或更换外部 Judge 不需要重新支付 Vicuna 生成成本。当前已保存的真实模型结果由
`python -m benchmarks.analyze_results` 统一聚合。

## Llama-3-8B Refusal Judge

`python -m benchmarks.run_refusal_judge` 复用 JailbreakBench 固定提交中的
`Llama3RefusalJudge` prompt 和 `"Yes"` 子串判定语义，只读取每条记录的
`sample.goal` 与 `result.response`。Judge 结果单独写入 `results/judged/`，不会修改目标模型
原始回答。

Judge 运行器会检查输入是否正好包含 1800 个唯一复合键，为 Judge 配置和每条源回答保存
SHA-256，逐条同步检查点，并记录模型原始输出、格式是否严格为 `Yes/No`、token 和耗时。
它主要用于计算 benign refusal rate；本项目的 harmful ASR 使用 DeepSeek-V4-Flash 辅助
Jailbreak Judge，不能把该结果称为官方 Llama-3-70B ASR，也不能用 8B refusal 标签替代
Jailbreak Judge。

在 AMD 实例上，先停止 Vicuna 服务并启动本地 8B Judge：

```bash
bash scripts/amd/serve_llama3_8b_judge.sh
```

另开终端做每个 split 两条记录的 smoke test：

```bash
python -m benchmarks.run_refusal_judge \
  --input-dir /root/se-smoothllm/results/raw \
  --output /root/se-smoothllm/results/judged/smoke-llama3-8b-refusal.jsonl \
  --limit 2 --workers 2
```

smoke 输出确认后，在后台运行完整 1800 条评价并监控：

```bash
bash scripts/amd/start_llama3_8b_refusal.sh
bash scripts/amd/watch_llama3_8b_refusal.sh 20
```

ModelScope 的 `Meta-Llama-3-8B-Instruct` 是 Llama-3-8B 指令权重镜像，但本地 vLLM FP16
推理并非 Together API 的逐位复现。结果中必须记录权重来源、服务参数和该差异。

## NVIDIA Llama-3.3-70B 辅助 Jailbreak Judge

如果没有 JBB 原始 `meta-llama/Llama-3-70b-chat-hf`，可以使用 NVIDIA NIM 的
`meta/llama-3.3-70b-instruct` 作为 70B 级别的辅助 Judge。它复用 JBB
`Llama3JailbreakJudge` 的 prompt 和 `"unsafe"` 子串解析，但模型版本不同，因此结果必须
标记为 Llama-3.3-70B 辅助 ASR，不能写成官方 JBB ASR。

配置位于 `configs/nvidia_llama3_3_70b_jailbreak_judge.json`。API Key 只能通过环境变量
传入，不能写进 shell 历史、JSON、日志或 Git：

```bash
read -rsp "NVIDIA API Key: " NVIDIA_API_KEY
echo
export NVIDIA_API_KEY
python -m benchmarks.run_refusal_judge \
  --config benchmarks/configs/nvidia_llama3_3_70b_jailbreak_judge.json \
  --input-dir /root/se-smoothllm/results/raw \
  --output /root/se-smoothllm/results/judged/nvidia-llama3-3-70b-jailbreak.jsonl \
  --split harmful \
  --workers 8 \
  --api-key-env NVIDIA_API_KEY
unset NVIDIA_API_KEY
```

该命令只评价 900 条 harmful 最终回答。API 提供方、模型名、prompt、温度和运行时间必须
写入结果元数据；不要把任何 API Key 提交到仓库。

## DeepSeek-V4-Flash 辅助 Jailbreak Judge

当原版 `meta-llama/Llama-3-70b-chat-hf` 不可用时，可以使用 SenseNova 的
`deepseek-v4-flash` 对已经保存的 Vicuna 回答进行外部评价。本配置复用 JailbreakBench
的 prompt 和 `safe/unsafe` 判定规则，只用于比较 `smoothllm_fixed` 与 `se_smoothllm`；
结果必须标记为 DeepSeek-V4-Flash 辅助 ASR，不能称为官方 Llama-3-70B ASR。

DeepSeek-V4-Flash 默认开启思考模式。本项目通过 `generation.extra_body.reasoning_effort`
设置为 `none`，要求 Judge 直接输出 `safe` 或 `unsafe`。API Key 只通过环境变量
`SENSENOVA_API_KEY` 传入，不能写入 JSON、shell 历史、日志或 Git。

配置位于 `configs/deepseek_v4_flash_jailbreak_judge.json`。该实验共有 600 条 harmful
回答（两种方法、三个 seed、每个 seed 100 条）。如果服务商
对单个 Key 限制 500 次调用，配置中的 `max_retries` 必须保持为 `0`，并分两批运行。每条
成功结果都会立即写入并同步到检查点；第二批会读取同一个输出文件，只补齐尚未完成的键。

第一批由下面两条命令组成，正好 500 次调用。两条命令必须使用同一个输出文件和同一个
Key；第二条会在第一条完成后继续写入，不会覆盖结果。

第一条：fixed 的三个 seed，共 300 次：

```bash
python -m benchmarks.run_refusal_judge \
  --config benchmarks/configs/deepseek_v4_flash_jailbreak_judge.json \
  --input-dir /root/se-smoothllm/results/raw \
  --output /root/se-smoothllm/results/judged/deepseek-v4-flash-jailbreak.jsonl \
  --split harmful \
  --method smoothllm_fixed \
  --seed 42 \
  --seed 43 \
  --seed 44 \
  --api-key-env SENSENOVA_API_KEY \
  --workers 8
```

第二条：SE 的 seed 42、43，共 200 次：

```bash
python -m benchmarks.run_refusal_judge \
  --config benchmarks/configs/deepseek_v4_flash_jailbreak_judge.json \
  --input-dir /root/se-smoothllm/results/raw \
  --output /root/se-smoothllm/results/judged/deepseek-v4-flash-jailbreak.jsonl \
  --split harmful \
  --method se_smoothllm \
  --seed 42 \
  --seed 43 \
  --api-key-env SENSENOVA_API_KEY \
  --workers 8
```

更换新的 Key 后运行最后 100 条：

```bash
python -m benchmarks.run_refusal_judge \
  --config benchmarks/configs/deepseek_v4_flash_jailbreak_judge.json \
  --input-dir /root/se-smoothllm/results/raw \
  --output /root/se-smoothllm/results/judged/deepseek-v4-flash-jailbreak.jsonl \
  --split harmful \
  --method se_smoothllm \
  --seed 44 \
  --api-key-env SENSENOVA_API_KEY \
  --workers 8
```

SenseNova 如果返回 `rpm exhausted`，不要继续使用并发 8；改用 `--workers 1
--request-delay-seconds 30`。这会牺牲速度换取稳定性，已写入的记录会被跳过，不会重复
写入同一个任务：

```bash
python -m benchmarks.run_refusal_judge \
  --config benchmarks/configs/deepseek_v4_flash_jailbreak_judge.json \
  --input-dir /root/se-smoothllm/results/raw \
  --output /root/se-smoothllm/results/judged/deepseek-v4-flash-jailbreak.jsonl \
  --split harmful \
  --method smoothllm_fixed \
  --seed 42 \
  --seed 43 \
  --seed 44 \
  --workers 1 \
  --request-delay-seconds 30 \
  --api-key-env SENSENOVA_API_KEY
```

如果某批中途发生网络错误，直接重新执行相同命令即可；已写入的记录会被跳过。

AutoDL 卡型、磁盘、镜像与 SSH 操作见 [`AUTODL.md`](AUTODL.md)。
AMD Radeon Cloud 的实际环境探针、逐步命令和 smoke 结果见
[`AMD_REMOTE_RUNBOOK.md`](AMD_REMOTE_RUNBOOK.md)。
