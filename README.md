# HIQL Rank Value

基于 **OGBench 的 HIQL 实现**修改的离线目标条件强化学习算法。主要改动是用同轨迹 hindsight goal 的相对排序监督替代 HIQL 的 Bellman / expectile value loss，保留 HIQL 的表征网络、高层策略和低层策略。

这是 `xzh-xzh` 的独立衍生项目，非 OGBench 或 HIQL 官方实现。代码来源、改动边界和许可证见 [NOTICE.md](NOTICE.md)。上游版权声明完整保留在 [LICENSE](LICENSE)。

## 算法

给定同一条轨迹中的 `i < j <= k`，令目标 `g = s_k`，学习：

```text
V(s_j, g) > V(s_i, g)

gap_h = V_h(s_j, g) - V_h(s_i, g),  h in {1, 2}
loss = sum_b mask_b * sum_h max(0, margin - gap_h[b])
       / max(sum_b mask_b, 1)
```

- 每个采样起点 `i` 先均匀选择轨迹内未来目标 `k`，再均匀选择 `j in [i+1, k]`；允许 `j=k`。这不是对所有合法三元组做全枚举或等概率抽样。
- 两个 value head、排序对的两端都参与梯度更新。
- 轨迹末端没有合法未来目标的样本通过 `rank_masks` 排除出 value loss。
- rank 模式不读取 Bellman target，也不软更新 target value 网络。
- 高低层策略保持 HIQL 的 clipped exponential AWR：`min(exp(alpha * advantage), 100)`。

时间先后是排序监督的假设，并不保证实际到达难度单调。`margin` 也会影响 value gap 和 AWR 权重的尺度。本仓库不包含实验结果或优于 HIQL 的性能声明。

现有实现还保留了可选的 gap 上界正则项，`rank_upper_weight=0.0` 时关闭。默认入口使用原始单边 hinge 目标；非零权重属于额外变体。详见 [算法说明](docs/algorithm.md)。

## 文件

| 文件 | 用途 |
| --- | --- |
| `impls/agents/hiql.py` | HIQL、rank value loss 和策略更新 |
| `impls/agents/hiql_rank.py` | 默认启用 rank 的配置入口 |
| `impls/utils/datasets.py` | 同轨迹 HER / rank 三元组采样 |
| `impls/main.py` | 训练、评估与保存入口 |
| `tests/test_hiql_rank.py` | 排序、梯度、边界与兼容性测试 |
| `tests/reference_hiql.py` | 固定上游版本的 HIQL，供回归测试使用 |
| `UPSTREAM.json` | 上游提交号及来源文件哈希 |

OGBench 环境通过依赖包安装。本仓库不复制环境模型或数据集，也不包含实验记录、集群脚本、日志、可视化、模型权重和凭据。

## 安装

CUDA 训练使用 Linux / WSL2、Python 3.10 或 3.11，以及兼容的 NVIDIA GPU 和驱动。在仓库根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-cuda12.txt
python -c "import jax; print(jax.devices()); assert jax.default_backend() == 'gpu'"
```

`requirements-cuda12.txt` 固定了参考训练依赖版本；它不是对任意显卡与驱动的兼容承诺。训练入口会在没有 JAX GPU 后端时直接报错，避免意外在 CPU 上训练。安装其他 JAX GPU 版本时可参考 [JAX 官方安装文档](https://docs.jax.dev/en/latest/installation.html)。

只运行合成数据单元测试可安装 `requirements.txt`，不需要 GPU 或下载 OGBench 数据集：

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p 'test_hiql_rank.py' -v
```

CPU 单元测试不能代替 CUDA 训练验证。目标 CUDA 环境中可用以下命令禁止测试静默回退到 CPU：

```bash
JAX_PLATFORMS=cuda python -m unittest discover -s tests -p 'test_hiql_rank.py' -v
```

## 运行

下面是一个显式设置 `margin=3`、高低层 `alpha=3` 的运行示例，不代表跨环境的最终调优配置：

```bash
cd impls
python main.py \
  --agent=agents/hiql_rank.py \
  --env_name=antmaze-medium-navigate-v0 \
  --agent.rank_margin=3.0 \
  --agent.high_alpha=3.0 \
  --agent.low_alpha=3.0 \
  --seed=0 \
  --train_steps=1000000 \
  --video_episodes=0 \
  --save_dir=../outputs
```

配置入口继承的默认 `rank_margin` 是 `0.1`；运行示例通过参数显式覆盖。离线数据由 OGBench 在首次运行时自动下载到其默认缓存目录，训练与评估输出写入被 Git 忽略的 `outputs/`。

日志默认只保存在本地，Weights & Biases 默认为 `disabled`。需要上传 W&B 日志时显式设置 `WANDB_MODE=online` 并自行登录。无渲染设备且只运行状态观测任务时，可以设置 `MUJOCO_GL=disable` 并保持 `--video_episodes=0`；像素观测任务仍需要渲染支持。

切换到原始 TD HIQL：

```bash
python main.py --agent=agents/hiql.py --agent.value_loss_type=td \
  --env_name=antmaze-medium-navigate-v0 --video_episodes=0
```

其他配置说明：像素输入通常需要 `--agent.encoder=impala_small --agent.low_actor_rep_grad=True`；离散动作任务需要 `--agent.discrete=True`。这些入口继承自上游，合成数据测试不构成对所有 OGBench 环境的完整验证。

## 来源与引用

本项目直接改自 [OGBench](https://github.com/seohongpark/ogbench) 的 `impls/`，固定来源提交为 [`1d4140997f60c52c6fb0702ec100dc988b18c548`](https://github.com/seohongpark/ogbench/tree/1d4140997f60c52c6fb0702ec100dc988b18c548)。算法基础为 [HIQL](https://github.com/seohongpark/HIQL)。

使用本代码进行研究时，请引用相关基础工作：

- **OGBench: Benchmarking Offline Goal-Conditioned RL**, ICLR 2025：[论文](https://arxiv.org/abs/2410.20092)、[项目主页](https://seohong.me/projects/ogbench/)。
- **HIQL: Offline Goal-Conditioned RL with Latent States as Actions**, NeurIPS 2023：[论文与项目主页](https://seohong.me/projects/hiql/)。

引用本项目的 rank value 改动时，请同时记录本仓库地址和所使用的 Git 提交号，区分上游 HIQL 与本项目的修改。
