# PDE神经算子科研智能体 — 完整操作手册与改进指南

> 本文档面向使用本Agent系统的研究者/开发者，提供从零开始运行Agent、生成提交物、以及后续系统改进的全面指导。

---

## 目录

1. [环境准备](#1-环境准备)
2. [配置与运行Agent](#2-配置与运行agent)
3. [监控与干预](#3-监控与干预)
4. [提交物生成与验证](#4-提交物生成与验证)
5. [常见问题排查](#5-常见问题排查)
6. [Agent系统后续改进建议](#6-agent系统后续改进建议)

---

## 1. 环境准备

### 1.1 系统要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Windows 10/11, Linux, macOS |
| Python | 3.12+ (推荐 3.13) |
| PyTorch | 2.0+ (CPU/CUDA均可) |
| 磁盘空间 | ≥ 5 GB (数据集+模型输出) |
| 内存 | ≥ 16 GB (推荐) |
| GPU | 可选（推荐），但有GPU训练速度提升5-10倍 |
| 网络 | 需要访问LLM API (OpenAI兼容格式) |

### 1.2 创建虚拟环境

```bash
# 进入项目根目录
cd PDEAgent-v2

# 创建venv (Windows)
python -m venv .venv
.venv\Scripts\activate

# 创建venv (Linux/macOS)
python3 -m venv .venv
source .venv/bin/activate
```

### 1.3 安装依赖

```bash
# 安装PyTorch (根据你的环境选择)

# 有CUDA ：具体版本要看适配的显卡硬件 (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# 有CUDA (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 仅CPU
pip install torch torchvision

# 安装其他依赖
pip install requries.txt

# 验证安装
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

**预期输出**：
```
PyTorch 2.3.0
CUDA available: True  # 或 False (CPU环境)
```

### 1.4 确认数据文件

```bash
# 检查数据集是否存在
ls data_and_sample_submission/train_val_test_init/

# 应该看到：
# task1_test.hdf5
# task1_val.hdf5
# task2_part0_train.h5
# task2_part1_train.h5
# task2_part2_train.h5
# task2_test.h5
# task2_val.h5
```

如果数据文件缺失，请先获取官方数据集并放到该目录下。

---

## 2. 配置与运行Agent

### 2.1 配置LLM API

编辑项目根目录下的 `config.yaml`：

```yaml
llm:
  api_key: "sk-你的API密钥"
  base_url: "https://api.openai.com/v1"  # 或其他兼容端点
  model: "gpt-4o"                        # 或 gpt-4, claude等
  temperature: 0.7
  max_tokens: 4096
  timeout: 120.0

research:
  max_iterations: 5          # 最大实验迭代次数
  max_time_hours: 2.5        # 最大运行时间(小时)，建议≤10
  early_stop_patience: 2     # 连续N次无提升则停止
  task: "task1"              # task1 或 task2
  data_dir: "./data_and_sample_submission/train_val_test_init"
  output_dir: "./output"
  code_dir: "./code"
```

**重要配置建议**：

| 场景 | max_iterations | max_time_hours | 说明 |
|------|---------------|----------------|------|
| 快速测试 | 2-3 | 1-2 | 验证Agent能否跑通全流程 |
| 正式运行Task1 | 8-12 | 4-8 | 给足迭代空间寻找最优模型 |
| 正式运行Task2 | 8-12 | 6-10 | Task2更复杂，需要更多时间 |

**API Key安全提示**：
- 不要将含API Key的config.yaml提交到Git
- 可使用环境变量覆盖：`set OPENAI_API_KEY=sk-xxx` (Windows) 或 `export OPENAI_API_KEY=sk-xxx` (Linux)

### 2.2 首次运行（推荐步骤）

#### 步骤一：初始化配置文件（可选-可跳过直接配置 config.yaml 文件）

```bash
python run_agent.py --init-config
```

这会生成一个默认的 `config.yaml`，你需要手动填入API Key。

#### 步骤二：运行Task 1

```bash
# Windows
.venv\Scripts\python run_agent.py --task task1

# Linux/macOS
python run_agent.py --task task1
```

#### 步骤三：运行Task 2（在Task 1完成后）

```bash
python run_agent.py --task task2
```

### 2.3 运行时的目录结构变化

Agent运行过程中会动态生成以下结构：

```
PDEAgent-v2/
├── code/                          # Agent生成的代码（会自动创建）
│   ├── model.py
│   ├── dataset.py
│   ├── train.py
│   ├── infer.py
│   └── utils.py
├── task1/                         # Task 1 科研日志与报告
│   ├── task1_logs.log             # LLM调用日志（科研日志）
│   ├── task1_literature_summary.md
│   ├── task1_diagnosis_report.md
│   ├── task1_design_notes.md
│   ├── task1_experiment_1_report.md
│   └── research_memory_task1.json # 科研记忆持久化
├── task2/                         # Task 2 科研日志与报告
│   ├── task2_logs.log
│   ├── task2_literature_summary.md
│   ├── task2_diagnosis_report.md
│   ├── task2_design_notes.md
│   ├── task2_experiment_1_report.md
│   └── research_memory_task2.json
├── output/                        # 实验输出与提交产物
│   ├── task1/                     # Task 1 实验输出
│   │   ├── iter_1/                # 第1次迭代
│   │   │   ├── best_checkpoint.pt # 最佳模型
│   │   │   ├── pred.hdf5          # 预测结果
│   │   │   ├── metrics.json       # 验证指标
│   │   │   └── time.json          # 时间记录
│   │   ├── iter_2/
│   │   └── ...
│   ├── task2/                     # Task 2 实验输出
│   │   ├── iter_1/
│   │   └── ...
│   ├── submission.zip             # 最终提交包
│   └── submission.json            # 提交元数据
└── ...
```

### 2.4 从断点恢复（高级）

如果Agent意外中断，可以恢复：

```bash
python run_agent.py --task task1 --resume
```

**注意**：恢复时会读取 `task1/research_memory_task1.json`（或 `task2/research_memory_task2.json`）中的状态，继续上次中断的阶段。

---

## 3. 监控与干预

### 3.1 实时监控日志

```bash
# Windows PowerShell
tail -f task1/task1_logs.log

# 或者手动查看最新内容
type task1/task1_logs.log
```

日志格式为JSON Lines，每行包含：
- `timestamp`: 时间戳
- `elapsed_seconds`: 本次调用耗时
- `response` 或 `tool_calls`: LLM响应

### 3.2 检查实验报告

每完成一次实验迭代，会生成报告：

```bash
type task1/task1_experiment_1_report.md
type task1/task1_experiment_2_report.md
```

报告包含：
- **DECISION**: CONTINUE / PIVOT / STOP
- **REASON**: 实验结果分析
- **NEXT_ACTION**: 下一步建议

### 3.3 查看当前科研记忆

```bash
python -c "import json; print(json.dumps(json.load(open('task1/research_memory_task1.json')), indent=2, ensure_ascii=False))"
```

关键字段：
- `current_phase`: 当前阶段 (literature/diagnosis/design/experiment)
- `iteration`: 当前迭代次数
- `best_metrics`: 最佳实验指标
- `stop_reason`: 停止原因（如果已停止）

### 3.4 何时需要人工干预

以下情况建议人工检查并干预：

| 现象 | 判断 | 干预措施 |
|------|------|----------|
| 连续2次以上实验CLI报错 | 参数接口不兼容 | 手动修改 `code/train.py` 和 `code/infer.py` 的argparse |
| 训练loss为nan | 梯度爆炸/数值不稳定 | 检查 `--grad_clip` 是否生效，降低学习率 |
| 验证score始终<5 | 模型完全未学习 | 检查数据归一化、模型输出层设计 |
| 推理时间>2分钟 | 模型太大或batch_size太小 | 增大推理batch_size，减小模型宽度 |
| GPU内存溢出 | batch_size过大 | 减小 `--batch_size` |
| Agent进入STOP但score很低 | 早停过于激进 | 删除 `task1/research_memory_task1.json`（或 task2 对应文件）重新运行，增大 `max_iterations` |

### 3.5 手动运行单个实验（调试）

如果Agent自动运行出问题，可以手动干预：

```bash
# 1. 手动训练
cd code
python train.py \
  --task task1 \
  --output_dir ../output/manual_run \
  --data_dir ../data_and_sample_submission/train_val_test_init \
  --model_type chunked \
  --epochs 50 \
  --batch_size 16 \
  --device cuda

# 2. 手动推理
python infer.py \
  --task task1 \
  --checkpoint ../output/manual_run/best_checkpoint.pt \
  --output ../output/manual_run/pred.hdf5 \
  --data_dir ../data_and_sample_submission/train_val_test_init \
  --device cuda
```

---

## 4. 提交物生成与验证

### 4.1 Agent自动生成的提交物

Agent在运行结束时会自动生成 `output/submission.zip`，包含：

```
submission/
├── submission.json          # 元数据
├── task1_pred.hdf5          # Task 1 预测结果 (1000, 200, 256)
├── task1_time.csv           # Task 1 时间记录
├── task1_logs.log           # Task 1 科研日志
├── task2_pred.hdf5          # Task 2 预测结果 (1000, 200, 256)
├── task2_time.csv           # Task 2 时间记录
├── task2_logs.log           # Task 2 科研日志
├── methodology.pdf          # 方法总结
└── code/                    # 源代码
    ├── train.py
    ├── model.py
    ├── infer.py
    ├── dataset.py
    └── utils.py
```

**注意**：运行结束后，各任务产物分别存放在 `output/task1/` 和 `output/task2/` 下，打包时自动归集到 zip 根目录。

### 4.2 手动验证提交物

**必做检查清单**：

#### 检查1：HDF5形状正确

```python
import h5py

# Task 1
with h5py.File('output/task1/task1_pred.hdf5', 'r') as f:
    print(f['tensor'].shape)  # 必须是 (1000, 200, 256)

# Task 2
with h5py.File('output/task2/task2_pred.hdf5', 'r') as f:
    print(f['tensor'].shape)  # 必须是 (1000, 200, 256)
```

#### 检查2：前10步与GT一致

```python
import numpy as np
import h5py

# Task 1
with h5py.File('data_and_sample_submission/train_val_test_init/task1_test.hdf5', 'r') as f:
    gt = f['tensor'][()]
with h5py.File('output/task1/task1_pred.hdf5', 'r') as f:
    pred = f['tensor'][()]
assert np.allclose(pred[:, :10, :], gt[:, :10, :], atol=1e-3), "前10步不一致！"
print("Task 1 前10步一致性检查通过")

# Task 2 (类似)
```

#### 检查3：日志格式正确

```python
import json

with open('output/task1/task1_logs.log', 'r') as f:
    for i, line in enumerate(f):
        obj = json.loads(line.strip())
        assert 'timestamp' in obj
        assert 'elapsed_seconds' in obj
        assert 'response' in obj or 'tool_calls' in obj
print(f"日志格式检查通过，共{i+1}条记录")
```

#### 检查4：时间记录合理

```python
import csv
with open('output/task1/task1_time.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        train_time = int(row['train_time'])
        infer_time = int(row['inference_time'])
        assert infer_time <= 120, f"推理时间{infer_time}秒超过2分钟限制！"
        assert train_time <= 3600 * 10, f"训练时间{train_time}秒超过10小时建议值"
print("时间检查通过")
```

#### 检查5：code目录非空且自主生成

```bash
ls output/code/
# 应该看到 train.py, model.py, infer.py, dataset.py, utils.py
```

**合规提醒**：`code/` 目录中的所有代码必须完全由Agent自主生成，不得人工编写或修改。科研日志中会记录代码生成过程，评审会校验一致性。

### 4.3 手动打包提交物

如果Agent的自动打包有问题，可以手动打包：

```bash
cd output

# 确保所有文件存在
ls submission.json task1/task1_pred.hdf5 task1/task1_time.csv task1/task1_logs.log methodology.pdf code/

# 打包（自动归集 task 子目录文件到 zip 根目录）
python -c "
import zipfile, os
with zipfile.ZipFile('submission.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    # 核心文件
    for f in ['submission.json', 'methodology.pdf']:
        if os.path.exists(f):
            zf.write(f, f'submission/{f}')
    # task 产物
    for task in ['task1', 'task2']:
        for suffix in ['pred.hdf5', 'time.csv', 'logs.log']:
            fp = f'{task}/{task}_{suffix}'
            if os.path.exists(fp):
                zf.write(fp, f'submission/{task}_{suffix}')
    # code 目录
    for root, dirs, files in os.walk('code'):
        for file in files:
            fp = os.path.join(root, file)
            arcname = os.path.relpath(fp, '.')
            zf.write(fp, f'submission/{arcname}')
print('打包完成')
"
```

### 4.4 生成 methodology.pdf

```bash
# 如果有pandoc，可从markdown生成
pandoc task1/task1_literature_summary.md task1/task1_diagnosis_report.md -o output/methodology.pdf

# 或者使用Python
pip install markdown weasyprint
python generate_methodology.py
```

---

## 5. 常见问题排查

### Q1: Agent运行时报 "API Key not configured"

**解决**：
```bash
# 方法1：编辑config.yaml
# llm:
#   api_key: "sk-你的密钥"

# 方法2：环境变量
set OPENAI_API_KEY=sk-你的密钥  # Windows
export OPENAI_API_KEY=sk-你的密钥  # Linux/macOS
```

### Q2: 训练脚本报 "FileNotFoundError: HDF5 file not found"

**解决**：
```bash
# 检查数据路径
ls data_and_sample_submission/train_val_test_init/

# 如果路径不同，修改config.yaml中的data_dir
# 或手动传入 --data_dir 参数
```

### Q3: 训练正常但推理报 "checkpoint not found"

**原因**：Agent保存的checkpoint名与runner寻找的不一致。

**解决**：确保 `train.py` 保存的是 `best_checkpoint.pt`：
```python
# train.py中
save_path = os.path.join(args.output_dir, "best_checkpoint.pt")
torch.save({
    "epoch": epoch,
    "model_state": model.state_dict(),
    "normalizer": normalizer.as_dict(),
    "args": vars(args),
}, save_path)
```

### Q4: GPU可用但训练在CPU上运行

**原因**：`--device` 参数默认值被硬编码为 `"cpu"`。

**解决**：修改 `code/train.py` 和 `code/infer.py`：
```python
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
```

### Q5: Agent连续多次实验失败，无法生成有效结果

**原因**：可能是CLI参数接口不匹配或代码有bug。

**解决步骤**：
1. 删除 `task1/research_memory_task1.json`（或 `task2/research_memory_task2.json`）重置状态
2. 手动运行一次最小训练验证代码可行性
3. 检查 `code/train.py` 的argparse是否支持 `--task`, `--output_dir`, `--data_dir`
4. 重新运行Agent

```bash
# 重置
del task1\research_memory_task1.json  # Windows
rm task1/research_memory_task1.json   # Linux/macOS

# 重新运行
python run_agent.py --task task1
```

### Q6: 如何同时运行Task 1和Task 2？

**推荐流程**：
```bash
# 先运行Task 1
python run_agent.py --task task1

# Task 1完成后，复制最佳结果到安全位置
copy output\submission.zip output\task1_submission.zip  # Windows
cp output/submission.zip output/task1_submission.zip    # Linux/macOS

# 再运行Task 2
python run_agent.py --task task2

# Agent 会自动将两个任务的结果合并打包到同一个 submission.zip
```

---

## 6. Agent系统后续改进建议

基于连续5次实验失败的教训和代码审查，以下是系统的改进路线图：

### 6.1 高优先级改进

#### 改进1：CLI参数接口契约化

**问题**：Agent生成的代码与runner的调用格式不匹配是最高频失败原因。

**建议实现**：
```python
# 在agent/phases.py或新增config中定义参数契约
CLI_CONTRACT = {
    "train": {
        "required": ["--task", "--output_dir"],
        "optional": ["--data_dir", "--epochs", "--batch_size", "--device", "--model_type"],
        "defaults": {
            "--data_dir": "./data_and_sample_submission/train_val_test_init",
            "--device": "cuda" if torch.cuda.is_available() else "cpu",
        }
    },
    "infer": {
        "required": ["--task", "--checkpoint", "--output"],
        "optional": ["--data_dir", "--batch_size", "--device"],
    }
}
```

然后在 `DesignPhase` 中强制要求Agent生成的代码通过契约验证器检查。

#### 改进2：参数接口自动验证

在 `ExperimentPhase` 运行训练前，先执行一次 **dry-run** 验证：

```python
def validate_cli_contract(script_path, contract):
    """验证脚本是否接受所有必需的参数"""
    import argparse
    # 通过解析脚本的help输出或AST分析检查参数
    # 如果不通过，立即返回错误，不执行训练
```

#### 改进3：数据路径自动探测

当前 `data_dir` 是硬编码在config中的。应实现自动探测：

```python
def auto_detect_data_dir():
    candidates = [
        "./data_and_sample_submission/train_val_test_init",
        "./data",
        "../data_and_sample_submission/train_val_test_init",
    ]
    for path in candidates:
        if os.path.exists(os.path.join(path, "task1_val.hdf5")):
            return path
    raise FileNotFoundError("无法自动找到数据集，请检查数据路径")
```

### 6.2 中优先级改进

#### 改进4：实验结果的结构化解析

当前 `ExperimentPhase` 让LLM自由文本分析实验结果，解析决策（CONTINUE/PIVOT/STOP）靠字符串匹配，容易出错。

**建议**：要求LLM输出结构化JSON，而非自由文本：

```json
{
  "decision": "CONTINUE",
  "confidence": 0.85,
  "failure_category": null,
  "analysis": {
    "train_converged": true,
    "overfitting": false,
    "score1": 65.3,
    "score2": 42.1,
    "score3": 18.7,
    "bottleneck": "segment3_long_term_stability"
  },
  "next_actions": [
    "increase_unroll_chunks to 3",
    "add temporal_difference_loss weight to 0.05"
  ]
}
```

#### 改进5：代码模板机制

与其让Agent从零生成完整的5个文件，不如提供经过验证的代码模板，Agent只参考修改关键部分：

```
templates/
├── base_model.py       # 基础FNO框架，Agent只改超参和架构细节
├── base_dataset.py     # 标准数据加载器
├── base_train.py       # 标准训练循环
├── base_infer.py       # 标准推理流程
└── base_utils.py       # 标准工具函数
```

Agent的 `DesignPhase` 变为：
1. 复制模板到 `code/`
2. 根据假设修改特定模块
3. 这样保证基础代码的健壮性，减少CLI参数错误

**注意**：模板必须由Agent在运行时自主复制和修改，不能预置人工编写的代码到 `code/` 目录（合规要求）。

#### 改进6：增量代码修改（Diff-based）

当前 `DesignPhase` 每次都重写整个文件，效率低且容易引入回归错误。

**建议**：实现增量修改能力：

```python
# Agent输出修改指令而非完整文件
{
  "action": "replace",
  "file": "code/train.py",
  "old": "parser.add_argument('--epochs', type=int, default=100)",
  "new": "parser.add_argument('--epochs', type=int, default=200)"
}
```

这样可以：
- 保留已验证正确的代码部分
- 减少LLM token消耗
- 加快迭代速度

### 6.3 低优先级改进（长期规划）

#### 改进7：多GPU并行实验

当前Agent串行运行实验。可以改为同时运行多个实验配置：

```python
# 每次DesignPhase生成3个不同配置的实验
# ExperimentPhase并行运行，选择最优结果
experiments = [
    {"model_type": "chunked", "chunk_size": 10, "width": 64},
    {"model_type": "chunked", "chunk_size": 20, "width": 96},
    {"model_type": "direct", "width": 128},
]
```

#### 改进8：贝叶斯超参优化

用Optuna等库替代LLM的人工超参调整：

```python
import optuna

def objective(trial):
    width = trial.suggest_int("width", 32, 128)
    depth = trial.suggest_int("depth", 2, 6)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    # 运行训练并返回val_score
    return val_score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
```

LLM负责架构层面的假设提出，贝叶斯优化负责超参搜索。

#### 改进9：可视化监控面板

增加一个简单的Web dashboard：

```python
# 使用gradio或streamlit
import gradio as gr

def get_status():
    memory = ResearchMemory.load("task1/research_memory_task1.json")
    return {
        "phase": memory.current_phase,
        "iteration": memory.iteration,
        "best_score": memory.best_metrics.get("total", 0),
    }

with gr.Blocks() as demo:
    gr.JSON(value=get_status, every=5)
    gr.LinePlot(value=get_loss_history, every=30)

demo.launch()
```

#### 改进10：物理残差损失的自动化

当前 `burgers_residual` 函数需要手动在训练循环中调用。可以设计一个自动化的物理约束模块：

```python
class PhysicsInformedLoss(nn.Module):
    def __init__(self, dx=1/256, dt=1.0, nu=1e-3, weight=1e-5):
        self.dx = dx
        self.dt = dt
        self.nu = nu
        self.weight = weight
    
    def forward(self, pred, normalizer):
        # 自动反归一化、计算残差、返回损失
        u = normalizer.decode(pred)
        residual = burgers_residual(u, self.dx, self.dt, self.nu)
        return self.weight * torch.mean(residual ** 2)
```

### 6.4 架构层面的建议

#### 建议A：分离 "代码生成" 和 "实验执行"

当前四个阶段是紧密耦合的。建议拆分为两个独立服务：

```
CodeGenerator (LLM Agent)
  ├── 输入：文献、诊断、历史实验
  └── 输出：code/ 目录下的完整代码

ExperimentRunner
  ├── 输入：code/ 目录
  └── 输出：metrics.json, checkpoint, pred.hdf5

Orchestrator
  ├── 调度 CodeGenerator → ExperimentRunner 循环
  └── 管理科研记忆
```

这样可以：
- 对CodeGenerator进行单元测试（验证生成的代码是否可编译）
- 对ExperimentRunner进行压力测试（验证各种配置下的稳定性）
- 人工介入时只需替换其中一个组件

#### 建议B：引入 "实验元语言"

定义一个JSON/YAML格式的实验描述语言，LLM生成实验描述而非直接生成代码：

```yaml
# experiment.yaml
model:
  type: ChunkedFNO1d
  modes: 24
  width: 64
  depth: 4
  chunk_size: 10

training:
  epochs: 200
  batch_size: 16
  lr: 0.001
  optimizer: AdamW
  scheduler: cosine
  
loss:
  - type: MSE
    weight: 1.0
  - type: spectral_gradient
    weight: 0.05
  - type: temporal_difference
    weight: 0.02
```

然后由代码生成器将YAML翻译为Python代码。这样：
- LLM的生成任务更简单（生成结构化配置而非完整代码）
- 人类更容易理解和修改实验配置
- 可以实现配置的版本控制和diff比较

#### 建议C：增加 "代码自检" 阶段

在 `DesignPhase` 和 `ExperimentPhase` 之间增加一个 `ValidationPhase`：

```python
class ValidationPhase(Phase):
    def run(self):
        checks = [
            self._check_syntax(),           # py_compile
            self._check_argparse_contract(), # 验证参数接口
            self._check_imports(),          # 验证依赖
            self._check_model_forward(),    # quick_test_model
            self._check_data_loading(),     # 验证数据集加载
        ]
        return all(checks)
```

只有通过所有检查的代码才能进入 `ExperimentPhase`，避免在已知有问题的代码上浪费时间运行训练。

### 6.5 工程实践建议

| 建议 | 优先级 | 实施难度 | 预期收益 |
|------|--------|----------|----------|
| CLI参数契约化 | 🔴 高 | 低 | 消除80%的实验失败 |
| 代码模板机制 | 🔴 高 | 中 | 提升代码质量，减少参数错误 |
| 增量代码修改 | 🟡 中 | 中 | 减少token消耗，加快迭代 |
| 结构化实验输出 | 🟡 中 | 低 | 提高决策准确性 |
| 自动数据探测 | 🟡 中 | 低 | 减少环境配置错误 |
| 贝叶斯超参优化 | 🟢 低 | 高 | 提升最终分数 |
| 多GPU并行实验 | 🟢 低 | 高 | 加速实验迭代 |
| Web监控面板 | 🟢 低 | 中 | 提升用户体验 |

---

## 附录

### A. 快速参考命令

```bash
# 安装
pip install torch h5py numpy scipy tqdm pyyaml

# 配置
# 编辑 config.yaml，填入API Key

# 运行
python run_agent.py --task task1

# 监控
type task1/task1_logs.log
type task1_experiment_1_report.md

# 验证提交物
python -c "import h5py; print(h5py.File('output/task1/task1_pred.hdf5')['tensor'].shape)"

# 重置
rm research_memory.json
```

### B. 关键文件速查

| 文件 | 作用 | 何时查看/修改 |
|------|------|--------------|
| `config.yaml` | Agent配置 | 运行前修改API Key和任务参数 |
| `AGENT_CODE_GUIDE.md` | 代码生成规范 | Agent设计阶段参考 |
| `agent/phases.py` | 四阶段科研流程 | 需要调整Agent行为时 |
| `agent/orchestrator.py` | 总编排器 | 需要调整停止条件或最终产物时 |
| `agent/tools.py` | 工具定义 | 需要新增工具时 |
| `code/` | Agent生成的代码 | 调试或手动干预时 |
| `output/` | 实验输出 | 查看训练结果和提交物 |
| `task1/research_memory_task1.json` | 科研状态 | 断点恢复或重置时 |
| `task1/task1_logs.log` | LLM调用日志 | 评审材料，验证科研过程 |

### C. 竞赛时间规划建议

假设总可用时间为 10 小时：

| 时间段 | 任务 | 时间 | 说明 |
|--------|------|------|------|
| 0:00-0:30 | 环境配置+数据检查 | 30min | 确保一切就绪 |
| 0:30-4:30 | Task 1 Agent运行 | 4h | 运行Agent生成Task1代码和结果 |
| 4:30-5:00 | Task 1验证+备份 | 30min | 验证提交物，备份最佳结果 |
| 5:00-9:00 | Task 2 Agent运行 | 4h | 运行Agent生成Task2代码和结果 |
| 9:00-9:30 | Task 2验证+合并 | 30min | 验证并合并两个任务的提交物 |
| 9:30-10:00 | 最终检查+打包 | 30min | 全面验证，打包submission.zip |
