# AGENTS.md — 神经算子PDE智能体竞赛项目

> 本文档面向AI编码智能体。阅读本文档前，默认你对本项目一无所知。本文所有信息均基于项目中的实际文件与规则，不做任何假设性推断。

---

## 1. 项目概述

本项目是 **"任务4：神经算子PDE智能体"** 的竞赛工程目录，属于 AI for Science（AI4S）场景下的科研工作流Agent竞赛。

**核心目标**：开发一个具备深度逻辑推理、架构演进与复杂代码工程能力的LLM Agent，使其能够在零人工干预环境下，自主驱动"问题理解—模型改进—物理验证—数值复现"的完整科研闭环，针对1D Burgers方程的神经算子（FNO、DeepONet、PI-DeepONet等基线）进行诊断、改进与验证。

**比赛不是提交模型权重，而是提交Agent的完整产出**（预测结果、科研日志、源代码、耗时记录）。

**关键合规要求**：
- 提交目录 `code/` 中的**所有代码必须完全由 Agent 自主生成**，不得在 Agent session 结束后由人工编写或修改。
- **初始代码也必须由 Agent 在科研系统启动后自主完成**，不允许将人工预先编写的代码直接放入 `code/` 目录作为初始基线。
- 科研日志（`task{N}_logs.log`）必须完整记录代码从无到有的生成过程，评审会校验 log 中记录的代码生成过程与 `code/` 目录中代码的对应关系。

---

## 2. 技术栈

| 层级 | 技术 |
|------|------|
| 深度学习框架 | PyTorch（CPU/CUDA均可） |
| 数据格式 | HDF5（通过 `h5py` 读写） |
| 数值计算 | NumPy、SciPy |
| 日志代理 | FastAPI + uvicorn + httpx（用于拦截并记录LLM API调用） |
| 编程语言 | Python 3 |

**注意**：本项目没有 `pyproject.toml`、`setup.py`、`package.json`、`Cargo.toml` 等传统包管理配置文件。它是一个以数据驱动实验为核心的研究型项目，依赖按需安装即可。

---

## 3. 目录结构

```
.
├── Background.md                          # 赛事背景、赛题介绍、评分规则（中文，必读）
├── NEURAL_OPERATOR_PRINCIPLES.md          # 神经算子技术原理文档（FNO/DeepONet/PI-DeepONet，中文）
├── AGENTS.md                              # 本文件
│
├── code-ref/                              # 【参考代码库】供Agent阅读理解的参考实现
│   ├── dataset.py                         # 数据集加载与归一化参考实现
│   ├── model.py                           # FNO/ChunkedFNO模型参考实现
│   ├── train.py                           # 训练流程参考实现
│   ├── infer.py                           # 推理与提交生成参考实现
│   ├── utils.py                           # 评分计算与工具函数参考实现
│   └── eval_checkpoint.py                 # 检查点评估参考实现
│
├── data_and_sample_submission/
│   ├── train_val_test_init/               # 官方数据集
│   │   ├── task1_test.hdf5                # Task1 测试集：1000样本 × 10时间步 × 256空间点
│   │   ├── task1_val.hdf5                 # Task1 验证集：100样本 × 200时间步 × 256空间点
│   │   ├── task2_part0_train.h5           # Task2 训练集Part0：1000样本 × 320时间步 × 256空间点
│   │   ├── task2_part1_train.h5           # Task2 训练集Part1：1000样本 × 320时间步 × 256空间点
│   │   ├── task2_part2_train.h5           # Task2 训练集Part2：1000样本 × 320时间步 × 256空间点
│   │   ├── task2_test.h5                  # Task2 测试集：1000样本 × 10时间步 × 256空间点
│   │   └── task2_val.h5                   # Task2 验证集：100样本 × 210时间步 × 256空间点
│   └── sample_submission/                 # 提交样例（必须严格参考其目录结构和文件格式）
│       ├── submission.json                # 提交元数据
│       ├── task1_pred.hdf5                # Task1 预测结果（样例）
│       ├── task1_time.csv                 # Task1 耗时记录（样例）
│       ├── task1_logs.log                 # Task1 科研日志（样例）
│       ├── task2_pred.hdf5                # Task2 预测结果（样例）
│       ├── task2_time.csv                 # Task2 耗时记录（样例）
│       ├── task2_logs.log                 # Task2 科研日志（样例）
│       └── code/                          # 源代码目录（样例中仅含空 train.py）
│           └── train.py
│
├── output/                                # 模型输出、检查点、预测结果（Agent运行时生成）
│
└── task_log_sample/
    ├── README.md                          # LLM调用日志规范说明（必读）
    ├── task1_logs.log                     # 样例日志文件（JSON Lines格式）
    ├── task2_logs.log                     # 样例日志文件（JSON Lines格式）
    └── openai-log/                        # 本地日志代理工具
        ├── proxy.py                       # FastAPI代理服务器，自动记录LLM调用
        └── requirements.txt               # 代理工具依赖：fastapi, uvicorn, httpx
```

### 3.1 code-ref/ 参考代码库的使用规范

`code-ref/` 目录中存放的是**供 Agent 阅读理解的参考实现**，其目的是帮助 Agent 快速理解数据加载、模型架构、训练流程、评分计算等核心模块的工程化写法。

**Agent 对 `code-ref/` 的使用必须遵循以下原则**：

1. **允许阅读与学习**：Agent 可以读取 `code-ref/` 中的文件，理解其中的设计思路、API 接口、算法逻辑和工程技巧。
2. **禁止直接复制**：Agent **不得**将 `code-ref/` 中的代码直接复制或稍作修改后放入 `code/` 目录作为提交代码。`code/` 目录中的每一行代码都必须是 Agent 在科研迭代过程中**自主构思并生成**的。
3. **鼓励消化吸收后重构**：Agent 应在理解参考代码核心思想的基础上，结合自己的科研假设和实验需求，重新组织代码结构、重新命名变量、重新设计接口，写出符合当前实验目标的实现。
4. **日志必须体现自主生成过程**：科研日志中必须包含 Agent 编写代码时的思考过程，例如"我计划设计一个 Chunked FNO 模型，参考 code-ref/model.py 中的谱卷积思想，但我会调整……"，然后实际写出代码。评审会对比 log 中记录的代码生成过程与 `code/` 中文件的内容一致性。

---

## 4. 数据规范

### 4.1 Task 1：固定物理环境（ν = 0.001）

- **测试输入** (`task1_test.hdf5`)：
  - `tensor`: shape `(1000, 10, 256)` — 1000个样本，前10个时间步的初始条件
  - `x-coordinate`: shape `(256,)` — 空间坐标
  - `t-coordinate`: shape `(10,)` — 时间坐标
- **验证集** (`task1_val.hdf5`)：
  - `tensor`: shape `(100, 200, 256)` — 100个样本，200个时间步的完整解场
  - 注意：官方说明中提到模型训练时采用 `reduced_resolution_t=5, reduced_resolution=4`，因此验证实际在 40×256 网格上进行；但本地提供的验证数据已经是下采样后的 200×256。
- **预测要求**：基于前10个时间步，预测未来190个时间步（共200步），输出 shape 必须为 `(1000, 200, 256)`。**前10个时间步必须与GT完全一致**（容差1e-3）。

### 4.2 Task 2：多物理环境泛化（变 ν）

- **训练数据**：3个文件，每个 `(1000, 320, 256)`，共3000个样本
  - 每个样本包含 `nu` 字段（shape `(1000,)`），表示该样本的粘性系数
  - `nu` 取值范围约为 `1e-4` 到 `1e-2`
- **验证集** (`task2_val.h5`)：`(100, 210, 256)`，包含 `nu` 值
- **测试集** (`task2_test.h5`)：`(1000, 10, 256)`，**不提供 `nu` 值**
- **预测要求**：输出 shape `(1000, 200, 256)`，同样前10步必须与GT一致。推理时间必须 ≤ 2分钟，否则该任务总分为0。

### 4.3 HDF5数据读取惯例

```python
import h5py
with h5py.File('task1_val.hdf5', 'r') as f:
    data = f['tensor'][()]       # 读取完整张量到内存
    x = f['x-coordinate'][()]
    t = f['t-coordinate'][()]
```

Task2数据使用下划线命名：`t_coordinate`, `x_coordinate`, `tensor`, `nu`。

---

## 5. 提交规范

### 5.1 提交文件清单（`submission.zip`）

```
submission/
├── submission.json          # 元数据
├── task1_pred.hdf5          # Task 1 预测结果
├── task1_time.csv           # Task 1 计时
├── task1_logs.log           # Task 1 Agent科研日志
├── task2_pred.hdf5          # Task 2 预测结果
├── task2_time.csv           # Task 2 计时
├── task2_logs.log           # Task 2 Agent科研日志
├── methodology.pdf          # 方法总结
└── code/                    # 源代码目录（必须非空，且完全由Agent生成）
    ├── train.py
    ├── model.py
    └── ...
```

### 5.2 各文件详细规范

**`submission.json`**
```json
{
  "submission_id": "队伍名称",
  "problem_id": "PDE_Burgers",
  "code_path": "code",
  "methodology": "methodology.pdf",
  "submission": "submission.zip"
}
```

**`task{N}_pred.hdf5`**
- Shape: `(N, 200, 256)` — N个样本、200个时间步、256个空间点
- 前10个时间步必须与Ground Truth完全一致（容差1e-3）
- 仅时间步 10–199 为预测结果

**`task{N}_time.csv`**
```csv
train_time,inference_time
1200,60
```
- `train_time`：该任务的模型训练总耗时（**包含Agent思考推理时间**），单位秒
- `inference_time`：该任务在测试集上的推理总耗时，单位秒

**`task{N}_logs.log`**
- **每一行必须是一条合法的JSON数据**
- 必须包含以下字段：
  - `timestamp`: ISO 8601格式时间戳，含时区，如 `2026-05-06T09:02:54.524886+00:00`
  - `elapsed_seconds`: 本次LLM调用耗时（秒）
  - `response` 或 `tool_calls`：至少存在其中一个字段
- 单log文件首尾记录时间差 **不得超过12小时**
- 系统会校验：log中记录的代码生成过程是否与 `code/` 目录中的代码对应
- `code/` 目录中的代码必须完全由Agent生成，不得人工编写或修改

### 5.3 日志代理工具使用

如需使用官方提供的日志记录代理：

```bash
cd task_log_sample/openai-log
pip install -r requirements.txt
python proxy.py --port 8080 --target https://api.openai.com --log-dir ./logs
```

将Agent的API base URL指向 `http://localhost:8080` 即可自动记录。

**注意**：该代理目前仅针对 **OpenAI兼容格式** 实现了响应解析。若使用Anthropic接口（`/v1/messages`），需自行修改 `proxy.py` 中的 `_extract_assistant_message` 和 `parse_sse_chunks` 函数。

---

## 6. 评分规则

总分 = Task 1 得分 + Task 2 得分，满分 300 分。

### 6.1 Task 1（最高150分）

Task 1 总分 = 预测精度得分（最高75分）+ 训练耗时得分（最高45分）+ 推理耗时得分（最高30分）

- **训练耗时得分**：
  - ≤60分钟：35分
  - ≤120分钟：25分
  - ≤300分钟：20分
  - ≤500分钟：10分
  - >500分钟：0分
- **推理耗时得分**：0分钟得40分（文档中有矛盾，以实际评测为准），0–2分钟线性递减，>2分钟该任务得0分

### 6.2 Task 2（最高150分）

Task 2 总分 = 分段预测得分 × 1.5（最高150分）
- 训练时间不计入评测，但总时长需控制在12小时内
- 推理时间 > 2分钟则该任务总分为0

### 6.3 分段预测得分（Task 1和Task 2通用）

仅针对190个预测时间步（去掉前10个初始条件），分为3段：

| 段 | 时间步 | 权重 | 得分公式 |
|---|---|---|---|
| 第1段 | 0–47步 | 25% | `100 × exp(-20 × Rel-MSE)` |
| 第2段 | 47–95步 | 25% | `100 × exp(-10 × Rel-MSE)` |
| 第3段 | 95–190步 | 50% | `max(Lorentzian, Frechet)`，其中 `Lorentzian = 100 / (1 + 10 × RMSE)`，`Frechet = 50 × exp(-FD²)` |

分段预测总分 = `0.25 × score1 + 0.25 × score2 + 0.5 × score3`，范围0–100。

**Rel-MSE 计算**：逐样本、逐时间步计算 `rel_t = Σ(pred-gt)² / Σ(gt²)`，对时间步取均值（单样本上限5.0），再对所有样本取均值。

---

## 7. 科研方向与基线模型

根据 `NEURAL_OPERATOR_PRINCIPLES.md` 和 `Background.md`，本项目涉及的基线模型与优化方向包括：

### 7.1 基线模型
- **FNO** (Fourier Neural Operator): 频域卷积，适合规则网格，推断极快
- **DeepONet**: Branch-Trunk架构，无网格，适合不规则几何
- **PI-DeepONet**: 在DeepONet中引入PDE残差损失（物理信息约束）

### 7.2 典型优化策略
- **物理残差约束**：在损失函数中加入Burgers方程残差 `u_t + u·u_x - ν·u_xx`
- **Pushforward Trick / Curriculum Rollout**：训练时逐步增加自回归步长，增强长时稳定性
- **FiLM条件化**（Task 2）：将 `ν` 通过Feature-wise Linear Modulation注入网络
- **Nu估计器**（Task 2）：测试时不提供 `ν`，需从初始条件推断或采用纯数据驱动泛化
- **谱正则化**：惩罚高频分量，抑制虚假振荡
- **时序捆绑**（Temporal Bundling）：一次预测多个未来时间步

### 7.3 长时间稳定性挑战
第3段（95–190步）权重最高（50%），也是最难部分。核心难点：
- 自回归预测的误差累积
- Burgers对流非线性将小误差迅速放大
- 激波位置的相移误差导致MSE急剧上升

---

## 8. 开发惯例与代码规范

### 8.1 代码组织
- `code/` 目录为提交入口，应包含完整可运行的训练与推理代码
- 建议至少包含 `train.py`（训练脚本）、`model.py`（模型定义）、`infer.py`（推理脚本）
- **代码必须完全由Agent自主生成，不得在Agent session结束后人工修改**
- **初始代码也必须由Agent在科研系统启动后自主完成**，不允许预置人工编写的基线代码到 `code/` 目录

### 8.2 code-ref/ 与 code/ 的关系

| 目录 | 性质 | Agent操作权限 |
|------|------|-------------|
| `code-ref/` | 人工编写的参考实现，供学习理解 | **只读**。Agent可阅读、分析、吸收设计思想，但不可直接复制到 `code/` |
| `code/` | Agent 自主生成的提交代码 | **读写**。所有代码必须由 Agent 在科研迭代中自主写出，log 中需体现生成过程 |

**合规示例**：
- ✅ Agent 阅读 `code-ref/model.py` 后，理解了 SpectralConv1d 的原理，然后在 log 中记录"我计划实现一个1D FNO，使用复数权重进行频域卷积……"，随后自主写出新的 `code/model.py`
- ❌ Agent 直接将 `code-ref/model.py` 复制到 `code/model.py`，仅修改类名或注释
- ❌ 人工预先将 `code-ref/` 中的文件复制到 `code/` 目录，然后让 Agent 在此基础上修改

### 8.3 实验记录
- `task{N}_logs.log` 是评审核心材料，必须完整记录Agent的"思考链路"和"实验轨迹"
- 日志中应包含：文献调研、假设提出、代码修改、失败实验分析、对照实验、最终结论
- **必须包含代码自主生成的完整过程**：Agent 应在 log 中记录"我现在要写 model.py，计划包含以下类和函数……"，然后实际执行写文件操作
- 不得仅提交最优结果，必须体现完整的科研迭代过程

### 8.4 环境管理
- 无虚拟环境强制要求，但建议使用 `venv` 或 `conda` 隔离
- 若需安装额外包（如 `torch`, `h5py`, `numpy`, `tqdm` 等），应在日志中体现安装命令

### 8.5 运行与测试

由于本项目没有统一的测试框架，验证方式如下：

1. **数据格式验证**：
```python
import h5py
with h5py.File('task1_pred.hdf5', 'r') as f:
    assert f['tensor'].shape == (1000, 200, 256)
```

2. **前10步一致性验证**：
```python
import numpy as np
with h5py.File('task1_test.hdf5', 'r') as f:
    gt = f['tensor'][()]
with h5py.File('task1_pred.hdf5', 'r') as f:
    pred = f['tensor'][()]
assert np.allclose(pred[:, :10, :], gt[:, :10, :], atol=1e-3)
```

3. **日志格式验证**：
```python
import json
with open('task1/task1_logs.log', 'r') as f:
    for line in f:
        obj = json.loads(line.strip())
        assert 'timestamp' in obj
        assert 'elapsed_seconds' in obj
        assert 'response' in obj or 'tool_calls' in obj
```

---

## 9. 安全与合规

- **严禁人工干预**：提交内容须完全由Agent自主完成，评审会审查log与代码的对应关系
- **代码原创性要求**：`code/` 目录中不得包含任何来自 `code-ref/` 的直接复制内容，也不得包含任何人工预置的代码模板。所有代码必须是 Agent 在运行期间基于自身理解自主生成的。
- **时间限制**：单个任务（单个log文件）的执行时间不得超过12小时
- **推理时限**：Task 1和Task 2的推理时间均不得超过2分钟，否则对应任务得0分
- **超时检测**：Log首尾时间戳差值 > 12小时视为违规

---

## 10. 关键外部资源

- FNO官方实现: https://github.com/neuraloperator/neuraloperator
- DeepONet官方实现: https://github.com/lululxvi/deeponet
- PI-DeepONet官方实现: https://github.com/PredictiveIntelligenceLab/Physics-informed-DeepONets
- PDEBench数据集: https://doi.org/10.18419/darus-2986

---

## 11. 给Agent的快速行动清单

如果你是接管本项目的Agent，请按以下顺序行动：

1. [ ] 仔细阅读 `Background.md` 和 `NEURAL_OPERATOR_PRINCIPLES.md`
2. [ ] 探查 `data_and_sample_submission/train_val_test_init/` 中的HDF5文件结构和统计信息
3. [ ] 检查当前环境是否已安装 `torch`, `h5py`, `numpy`
4. [ ] **阅读 `code-ref/` 中的参考代码**，理解数据集加载、模型架构、训练流程和评分计算的核心工程化写法（**只读，不可直接复制**）
5. [ ] 基于自己的理解，**自主从零开始设计并编写** `code/model.py`、`code/train.py`、`code/infer.py` 等核心文件
6. [ ] 在科研日志中记录代码生成的完整思考过程，包括架构选择理由、关键设计决策等
7. [ ] 运行训练，根据验证指标进行迭代优化（调整模型、损失函数、训练策略等）
8. [ ] 生成预测结果，验证HDF5的shape和前10步一致性
9. [ ] 生成 `time.csv` 和 `logs.log`
10. [ ] 按 `sample_submission/` 的结构打包 `submission.zip`
