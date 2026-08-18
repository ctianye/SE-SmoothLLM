# AutoDL 运行手册

本文档只覆盖本项目的 Vicuna-13B 主实验。GPU 市场价格和库存会变化，租用前仍需重新
确认；下列价格是 2026-08-15 在 AutoDL 西北 B 区按量计费页面看到的会员价快照。

## 卡型选择

主实验以 `float16` 加载 `lmsys/vicuna-13b-v1.5`。模型权重约 26 GB，运行时还需要
CUDA 上下文、激活和 KV cache，因此显存不能只按权重文件大小计算。

| 卡型 | 页面价格 | 判断 |
| --- | ---: | --- |
| A800-80GB | 约 4.98 元/小时 | 首选；卡型明确、兼容成熟、显存和吞吐余量充足 |
| vGPU-48GB | 约 2.88 元/小时 | 低价备选；页面未说明底层物理卡和资源隔离，需先实测 |
| RTX 5090 32GB | 约 2.78 元/小时 | 能尝试，但显存余量小，新架构环境更容易遇到兼容问题 |
| vGPU-32GB | 约 1.68 元/小时 | 不作为首次正式实验选择，节省的租金不足以抵消 OOM 排查成本 |
| RTX 4090/4090D 24GB | 页面价格浮动 | FP16 主配置放不下，不应为省钱临时改量化配置 |
| H800-80GB | 约 8.88 元/小时 | 对本项目明显过度配置 |
| RTX PRO 6000 96GB | 约 5.98 元/小时 | 可运行，但性价比低于 A800 或 48GB vGPU |

建议第一次联调租 **1 张 A800-80GB**。本项目的目标是尽快得到可信结果，A800 多付的单价
通常低于排查未知 vGPU 性能和兼容问题造成的时间成本。确认整条流程稳定并测得实际吞吐后，
如果需要补跑或扩展 seed，再考虑 **1 张 vGPU-48GB** 降低费用。不要租多卡：13B 模型
单卡可运行，多卡会增加张量并行配置和费用，而本项目没有训练任务。

数据盘建议至少 100 GB。Vicuna 权重约 26 GB，vLLM/PyTorch 环境、Hugging Face cache、
实验回答和后续外部 judge 都会继续占空间；页面默认的 50 GB 虽可能勉强启动目标模型，
但没有足够余量完成整个实验链路。

## 镜像要求

- Ubuntu 22.04。
- Python 3.10 或 3.11。
- NVIDIA 驱动能够支持 CUDA 12.x；A800 优先选择常见的 PyTorch 2.x + CUDA 12.x 镜像。
- 项目固定安装 `vllm==0.27.1`，不要在正式运行中临时升级。
- RTX 5090 属于更新架构，必须选择明确支持该卡的 CUDA/PyTorch 镜像，因此不作为首次首选。

## 租用后执行顺序

仓库必须克隆到数据盘目录，例如 `/root/autodl-tmp/SE-SmoothLLM`，不要放在可能随镜像
释放的临时系统位置。

```bash
cd /root/autodl-tmp
git clone https://github.com/ctianye/SE-SmoothLLM.git
cd SE-SmoothLLM
bash scripts/autodl/bootstrap.sh
bash scripts/autodl/start_vllm.sh
bash scripts/autodl/run_smoke.sh
```

只有 smoke test 同时满足以下条件，才启动完整实验：

- `probe_backend` 返回模型名、非空回答、token 统计和延迟。
- harmful 与 benign 各 1 条样本均完成。
- 三种方法都写入 smoke JSONL。
- vLLM 日志没有 OOM、CUDA error 或连续 HTTP 500。

正式实验使用后台启动器，SSH 断开不会终止任务：

```bash
bash scripts/autodl/start_main.sh
bash scripts/autodl/status.sh
```

需要主动停止主实验时执行 `bash scripts/autodl/stop_main.sh`。该命令会保留已经同步的
JSONL；之后重新运行 `start_main.sh` 会继续未完成任务。

结果逐条同步到 `results/raw/jbb-vicuna-13b-gcg-white-box.jsonl`。如果实例或进程中断，
重新执行 `start_main.sh` 会读取现有 JSONL，并只运行尚未完成的 `(method, seed, split,
index)`，不从头重复已保存样本。

## SSH 协作

实例创建后，可把 AutoDL 控制台显示的 SSH 主机、端口和用户名发给协作者。优先使用
一次性 SSH 密钥；如果平台只提供密码，密码应只在本机终端的交互提示中输入，不写入
仓库、聊天记录、shell 历史或 `.env`。模型 API 只监听 `127.0.0.1:8000`，实验脚本与模型
在同一实例运行，不需要把 8000 端口暴露到公网。

不要一租到机器就启动完整实验。正确顺序是：硬件自检、安装、模型服务探针、2 条样本
smoke test、检查输出，然后才后台启动主实验。这样安装或模板问题只消耗少量时间。

## 当前边界

本地已经对部署脚本做了 Bash 语法检查，对实验运行器做了 MockBackend 与检查点测试；
尚未在真实 AutoDL GPU 上验证 vLLM 启动和吞吐。因此第一次租用仍应视为联调，不能预先
承诺具体完成时长。当前生成结果保存了完整 prompt、最终 response、逐副本 trace、内部
judge、token 与延迟，后续增加外部 JBB judge 时无需重新生成 Vicuna 回答。
