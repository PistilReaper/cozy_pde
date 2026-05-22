# PDE Neural Operator Research Agent

端到端自主科研智能体系统，用于 1D Burgers 方程神经算子竞赛。

## 系统架构

本系统实现了一个四阶段科研闭环 Agent：

1. **文献解析与逻辑解构** (`LiteraturePhase`)
   - 自动阅读 `Background.md`、`NEURAL_OPERATOR_PRINCIPLES.md`
   - 使用 `inspect_hdf5` 分析数据结构
   - 使用 `summarize_code` 审查现有代码
   - 输出结构化文献综述

2. **瓶颈诊断与假设提出** (`DiagnosisPhase`)
   - 使用 `analyze_log` 分析训练动态
   - 识别长时稳定性、物理一致性、泛化能力等瓶颈
   - 提出可验证的优化假设
   - 输出诊断报告

3. **自主设计与代码演进** (`DesignPhase`)
   - 根据假设自动编写/修改 PyTorch 代码
   - 使用 `validate_code` 进行语法检查
   - 使用 `quick_test_model` 进行 smoke test
   - 维护代码版本历史

4. **实验验证与科学迭代** (`ExperimentPhase`)
   - 自动运行训练与推理
   - 计算分段预测得分（与评测完全一致）
   - LLM 分析实验结果并决策：CONTINUE / PIVOT / STOP
   - 完整记录实验轨迹

## 目录结构

```
.
├── agent/                      # Agent 核心系统
│   ├── __init__.py
│   ├── config.py               # 配置管理
│   ├── llm_client.py           # LLM API 客户端（带合规日志）
│   ├── tools.py                # 工具注册表（文件/命令/代码/分析）
│   ├── memory.py               # 科研记忆与状态持久化
│   ├── phases.py               # 四阶段科研闭环
│   ├── orchestrator.py         # 主编排器
│   └── main.py                 # CLI 入口
├── code-ref/                   # 基线模型代码参考
│   ├── model.py                # FNO + FiLM + 时序捆绑
│   ├── dataset.py              # HDF5 数据加载
│   ├── train.py                # 训练脚本
│   ├── infer.py                # 推理脚本（生成提交格式 HDF5）
│   └── utils.py                # 指标计算与辅助函数
├── task1/                      # Task 1 科研日志与报告
│   ├── task1_logs.log
│   ├── task1_literature_summary.md
│   ├── task1_diagnosis_report.md
│   ├── task1_design_notes.md
│   └── task1_experiment_*.md
├── task2/                      # Task 2 科研日志与报告
│   ├── task2_logs.log
│   ├── task2_literature_summary.md
│   ├── task2_diagnosis_report.md
│   ├── task2_design_notes.md
│   └── task2_experiment_*.md
├── run_agent.py                # Agent 启动脚本
├── config.yaml                 # 配置文件模板
├── requirements.txt            # Python 依赖
└── README.md                   # 本文件
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 LLM API

编辑 `config.yaml`，填写你的 API Key：

```yaml
llm:
  api_key: "sk-your-api-key"
  base_url: "https://api.openai.com/v1"
  model: "gpt-4o"
```

或通过环境变量：

```bash
export OPENAI_API_KEY="sk-your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export LLM_MODEL="gpt-4o"
```

### 3. 运行 Agent

```bash
# Task 1: 固定物理环境
python run_agent.py --task task1

# Task 2: 多物理环境泛化
python run_agent.py --task task2
```

### 4. 手动测试基线（可选）

如果不想启动完整 Agent，可直接运行基线代码：

```bash
# Task 1 训练
python code-ref/train.py --task task1 --output_dir output/task1_baseline

# Task 1 推理
python code-ref/infer.py --task task1 --checkpoint output/task1_baseline/best_model.pt --output output/task1/task1_pred.hdf5

# Task 2 训练
python code-ref/train.py --task task2 --output_dir output/task2_baseline --epochs 80

# Task 2 推理
python code-ref/infer.py --task task2 --checkpoint output/task2_baseline/best_model.pt --output output/task2/task2_pred.hdf5
```

## 提交产物

Agent 运行结束后，会在 `output/` 目录生成：

```
output/
├── submission.zip              # 最终提交包
├── submission.json             # 提交元数据
├── task1/                      # Task 1 产物
│   ├── task1_pred.hdf5         # Task 1 预测结果 (1000, 200, 256)
│   ├── task1_time.csv          # Task 1 耗时记录
│   └── task1_logs.log          # Task 1 Agent 科研日志（JSON Lines）
├── task2/                      # Task 2 产物
│   ├── task2_pred.hdf5         # Task 2 预测结果 (1000, 200, 256)
│   ├── task2_time.csv          # Task 2 耗时记录
│   └── task2_logs.log          # Task 2 Agent 科研日志（JSON Lines）
├── methodology.pdf             # Agent 方法总结
└── code/                       # 源代码目录
    ├── model.py
    ├── dataset.py
    ├── train.py
    ├── infer.py
    └── utils.py
```

### 日志格式合规性

`task{N}/task{N}_logs.log` 每一行均为合法 JSON，包含：
- `timestamp`: ISO 8601 格式，含时区
- `elapsed_seconds`: 本次 LLM 调用耗时
- `response` 或 `tool_calls`: 至少存在其一

## 核心设计亮点

### 1. 改进版 FNO 基线

- **时序捆绑 (Temporal Bundling)**: 一次前向传播预测全部 190 个未来步，避免自回归误差累积
- **FiLM 条件化**: Task 2 中将 `nu` 通过 Feature-wise Linear Modulation 注入网络特征
- **nu 估计器**: 测试时不提供 `nu`，模型自动从初始条件推断
- **物理残差框架**: 预留 Burgers 方程残差计算接口，可一键启用物理信息训练

### 2. 评分对齐的指标计算

`code/utils.py` 中的 `compute_segment_scores` 完全按照评测规则实现：
- 第1段 (0-47步): `100 × exp(-20 × Rel-MSE)`
- 第2段 (47-95步): `100 × exp(-10 × Rel-MSE)`
- 第3段 (95-190步): `max(Lorentzian, Frechet)`

### 3. Agent 工具链

| 工具 | 用途 |
|------|------|
| `read_file` / `write_file` | 文件读写 |
| `run_shell` | 执行训练/推理命令 |
| `run_python` | 快速数值实验 |
| `validate_code` | 语法检查 |
| `quick_test_model` | 模型 smoke test |
| `analyze_log` | 训练日志分析 |
| `inspect_hdf5` | 数据结构探查 |
| `summarize_code` | 代码审查 |

### 4. 记忆与迭代

- `ResearchMemory` 持久化研究状态到 JSON
- 自动追踪最优实验、记录假设与结论
- 支持从断点恢复（`task1/research_memory_task1.json` 或 `task2/research_memory_task2.json`）
- 早停与超时保护

## 扩展与定制

### 更换 LLM 提供商

`config.yaml` 支持任何 OpenAI 兼容 API：

```yaml
llm:
  base_url: "https://api.deepseek.com/v1"
  model: "deepseek-chat"
```

### 调整模型架构

在 `config.yaml` 的 `model` 节修改：

```yaml
model:
  modes: 24      # 傅里叶模式数
  width: 128     # 特征维度
  depth: 6       # FNO 层数
```

### 启用物理信息损失

```yaml
model:
  use_physics_loss: true
  physics_weight: 0.5
```

## 注意事项

1. **API Key 安全**: `config.yaml` 中的 `api_key` 仅在本地使用，不会被提交
2. **时间限制**: 单任务总时长不得超过 12 小时（Agent 默认限制 11.5 小时）
3. **推理时限**: Task 1 和 Task 2 的推理时间均不得超过 2 分钟
4. **数据路径**: 默认指向 `./data_and_sample_submission/train_val_test_init/`，请确保数据已解压

## 参考文献

- Li et al. (2021) - Fourier Neural Operator for Parametric Partial Differential Equations
- Lu et al. (2021) - Learning Nonlinear Operators via DeepONet
- Wang et al. (2021) - Physics-informed DeepONets
- Takamoto et al. (2022) - PDEBench
