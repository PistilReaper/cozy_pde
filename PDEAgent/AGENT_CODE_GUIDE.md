# Agent 快速编码与运行指南

> 本文档面向 AI Agent，基于 `code-ref/` 参考代码的运行逻辑与工程结构编写。
> **核心原则**：Agent 应理解参考代码的设计思想后**自主重新实现**，不可直接复制粘贴 `code-ref/` 中的代码到 `code/` 目录。

---

## 0. 实验失败教训总结（必须优先阅读）

> 以下教训来自连续 5 次实验的失败报告，是 Agent 生成代码时**最容易忽视但最致命**的问题。

### 0.1 CLI 参数接口不匹配 — 连续 5 次实验失败的根源

实验调度器（runner）会以**固定格式**调用 `train.py` 和 `infer.py`。如果 Agent 生成的代码参数接口与 runner 的调用格式不兼容，程序会在 `argparse` 阶段直接退出，**没有任何训练发生**。

**runner 实际注入的命令格式**：

```bash
# 训练
python code/train.py --task task1 --output_dir output/task1/iter_N --data_dir ./data_and_sample_submission/train_val_test_init

# 推理
python code/infer.py --task task1 --checkpoint output/task1/iter_N/best_checkpoint.pt --output output/task1/iter_N/pred.hdf5 --data_dir ./data_and_sample_submission/train_val_test_init
```

**失败模式**：
| 实验轮次 | 错误信息 | 原因 |
|---------|---------|------|
| Exp 1-4 | `error: the following arguments are required: --data` | runner 未传入 `--data`，且 Agent 脚本将其设为 required |
| Exp 5 | `error: unrecognized arguments: --task task1 --output_dir output/task1/iter_5` | Agent 脚本不认识 `--task` 和 `--output_dir` |

**解决方案**：
1. `train.py` 和 `infer.py` 必须显式支持 `--task`、`--output_dir`、`--data_dir`
2. 使用 argparse 别名同时支持下划线和横线版本：
   ```python
   parser.add_argument("--output-dir", "--output_dir", dest="output_dir", default="./output")
   parser.add_argument("--data-dir", "--data_dir", dest="data_dir", default="./data_and_sample_submission/train_val_test_init")
   ```
3. 所有参数必须有合理的默认值，runner 不会传入所有参数
4. **checkpoint 必须保存为 `best_checkpoint.pt`**，因为 runner 硬编码寻找此文件

### 0.2 数据路径缺失

runner 不会自动猜测数据路径。Agent 必须在 `get_dataloaders()` 和 `get_test_loader()` 中设置**可靠的默认路径**：
```python
default_data_dir = "./data_and_sample_submission/train_val_test_init"
```

### 0.3 验证 vs 训练的逻辑差异

多次实验失败后发现，Agent 容易在 `validate()` 函数中使用 teacher forcing（直接 forward）而非完整 rollout。这会导致验证 score 虚高，无法反映真实的长时预测能力。

**硬性要求**：`validate()` 必须对 chunked 模型调用 `model.rollout(x, horizon=190)`，对 direct 模型直接 forward 后取前 190 步。

### 0.4 前 10 步一致性检查不可省略

评测系统会检查提交 HDF5 的前 10 步是否与 GT 一致（`atol=1e-3`）。`infer.py` 必须在保存前执行：
```python
assert np.allclose(full[:, :10, :], raw_test[:, :10, :], atol=1e-6)
```
这是提交前的**最后防线**。

### 0.5 GPU 自动检测 — 不可硬编码为 CPU

代码必须自动检测是否有 GPU 可用，优先使用 GPU 加速训练和推理：
```python
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
```

在 train.py 和 infer.py 中：
- 所有 `torch.tensor`、`torch.randn` 等创建的张量都应该通过 `.to(args.device)` 或直接在目标 device 上创建
- `DataLoader` 的 `pin_memory` 参数应设置为 `torch.cuda.is_available()`
- `model.to(args.device)` 必须在训练/推理前调用

**错误示例**：
- ❌ `parser.add_argument("--device", default="cpu")` — 在有 GPU 的环境中浪费计算资源
- ❌ 模型或张量没有 `.to(device)` — 可能导致 CPU/GPU 混合计算错误

---

## 1. 项目目标速览

你需要为 1D Burgers 方程编写神经算子（FNO）预测模型，并完成以下两个任务：

| 任务 | 数据 | 输入 → 输出 | 特殊要求 |
|------|------|-------------|----------|
| **Task 1** | `task1_val.hdf5` (100条轨迹) | 10步 → 190步 | ν 固定为 0.001 |
| **Task 2** | `task2_part{0,1,2}_train.h5` (3000条轨迹) | 10步 → 190步 | ν 变化，测试集**不提供 ν** |

**提交要求**：`task{N}_pred.hdf5` 的 shape 必须为 `(N_samples, 200, 256)`，其中前 10 步必须与 Ground Truth 完全一致（`atol=1e-3`）。

---

## 2. 你必须生成的文件清单

在 `code/` 目录下，Agent 必须自主生成至少以下 4 个核心文件：

```
code/
├── model.py      # 神经网络模型定义
├── dataset.py    # 数据加载与归一化
├── train.py      # 训练入口脚本
├── infer.py      # 推理与提交生成脚本
└── utils.py      # 评分计算与辅助工具
```

下面按文件详细说明**必须实现的关键组件**和**数据流约定**。

---

## 3. dataset.py — 数据加载与归一化

### 3.1 核心设计思想

- **Scalar Normalization**：对整个速度场使用全局 mean/std 做标准化，而不是逐样本或逐通道。这能稳定 FNO 的频域卷积训练。
- **归一化必须在训练集上计算，然后共享给验证集和测试集**。Task 1 因为只有 100 条轨迹，通常做 80/20 切分；Task 2 因为 3000 条轨迹，直接用全部训练文件计算 mean/std。
- **Task 2 的 ν 处理**：训练集中每个样本有 `nu` 字段（shape `(N,)`）。建议对 `log(nu)` 做标准化后作为条件嵌入传入模型。测试集**不提供 ν**，模型必须内置 ν 估计器（ν-estimator）或采用纯数据驱动泛化。

### 3.2 必须实现的类与函数

```python
class Normalizer:
    """标量归一化器，包含 mean 和 std。"""
    def __init__(self, mean: float, std: float): ...
    def encode(self, x: torch.Tensor) -> torch.Tensor: ...
    def decode(self, x: torch.Tensor) -> torch.Tensor: ...
    def as_dict(self) -> dict: ...  # 用于 checkpoint 保存

class BurgersDataset(torch.utils.data.Dataset):
    """
    读取 HDF5，返回 (input, target, nu_emb) 或 (input, target)。
    input shape:  [t_in, 256]
    target shape: [t_out, 256]
    """
    def __init__(self, hdf5_path, t_in=10, t_out=190, normalizer=None, compute_normalizer=True): ...

class WindowedBurgersDataset(torch.utils.data.Dataset):
    """
    【仅用于 chunked 训练】在单条轨迹上做滑动窗口。
    将一条长轨迹切分为多个 (input=t_in, target=chunk_size) 的样本。
    这是解决 Task 1 样本量不足（仅100条）的关键手段。
    """
    def __init__(self, base_dataset, indices, t_in=10, chunk_size=10, stride=1, target_horizon=None): ...

def get_dataloaders(data_dir, task="task1", batch_size=16, val_fraction=0.2,
                    model_type="direct", chunk_size=10, window_stride=1,
                    t_in=10, t_out=190, **kwargs) -> Tuple[DataLoader, DataLoader]:
    """
    创建 train_loader 和 val_loader。
    关键逻辑：
    - Task 1：从 task1_val.hdf5 做切分；若 model_type=="chunked"，train_set 用 WindowedBurgersDataset。
    - Task 2：合并 3 个训练文件，val_set 用 task2_val.h5。
    - 验证集永远返回完整 190 步 target，不做窗口切分。
    """

def get_test_loader(data_dir, task="task1", batch_size=64, normalizer=None, t_in=10) -> Tuple[DataLoader, Dataset]:
    """创建 test_loader，同时返回底层 dataset 以便后续取用 normalizer。"""
```

### 3.3 关键实现细节与陷阱

1. **HDF5 读取**：Task 1 使用 `f["tensor"]`、`f["x-coordinate"]`、`f["t-coordinate"]`；Task 2 使用 `f["tensor"]`、`f["x_coordinate"]`、`f["t_coordinate"]`、`f["nu"]`。**注意命名差异**（Task 2 是下划线）。
2. **数据类型**：从 HDF5 读取后必须转为 `np.float32`，再转为 `torch.float32`。
3. **Task 1 切分一致性**：`random_split` 必须使用固定 `seed`（如 `torch.Generator().manual_seed(42)`），且 train/val 的切分索引必须与训练时完全一致，否则验证指标无意义。
4. **WindowedBurgersDataset 的索引**：窗口生成公式为 `windows = [(idx, s) for idx in indices for s in range(0, max_start + 1, stride)]`，其中 `max_start = total_t - t_in - target_horizon`。

---

## 4. model.py — 模型定义

### 4.1 核心设计思想

本项目推荐使用 **Fourier Neural Operator (FNO)**，因为数据在规则网格上，FNO 的频域卷积推断极快。

提供两种模型模式：
- **Direct 模式**（`ResidualFNO1d`）：一次性将 10 步映射到 190 步。结构简单，但长时预测误差可能较大。
- **Chunked 模式**（`ChunkedFNO1d`）：每次只预测 `chunk_size`（如 10 步），然后**自回归 rollout** 到 190 步。训练时可用滑动窗口生成大量监督样本，这是当前推荐策略。

### 4.2 必须实现的类与函数

```python
class SpectralConv1d(nn.Module):
    """
    1D 谱卷积核心：将输入通过 FFT 转到频域，用复数权重做线性变换，再 IFFT 回空间域。
    输入/输出 shape: [B, C, Nx]
    """
    def __init__(self, in_channels: int, out_channels: int, modes: int): ...
    def forward(self, x: Tensor) -> Tensor:
        # 1. x_ft = torch.fft.rfft(x, dim=-1)
        # 2. 只保留前 modes 个频率分量，用 einsum 做线性变换
        # 3. out_ft 其余位置补 0
        # 4. return torch.fft.irfft(out_ft, n=n, dim=-1)

class FNOBlock1d(nn.Module):
    """
    残差 FNO Block：SpectralConv1d + 1x1 Conv（pointwise）+ GroupNorm + GELU + 残差连接。
    """
    def __init__(self, width: int, modes: int, dropout: float = 0.0): ...

class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation，用于 Task 2 的条件注入。
    将条件向量（如 log(nu)）映射为 gamma 和 beta，对特征做仿射变换。
    """
    def __init__(self, cond_dim: int, width: int): ...
    def forward(self, x: Tensor, cond: Tensor) -> Tensor: ...

class FNOForecast1d(nn.Module):
    """
    核心预测网络。输入 [B, t_in, Nx]，输出 [B, t_out, Nx]。
    架构流：
    1. Lift：Conv1d(t_in + 1, width, 1)，其中 +1 是空间坐标通道（linspace 0~1）
    2. FNOBlocks：depth 个残差块，可选 FiLM 条件注入
    3. Project：Conv1d(width, 2*width, 1) → GELU → Conv1d(2*width, t_out, 1)
    4. 残差输出：返回 "上一帧复制 + project结果"，即 x[:, -1:, :].expand(-1, t_out, -1) + project(h)
    """
    def __init__(self, modes=24, width=64, depth=4, t_in=10, t_out=10,
                 use_film=False, dropout=0.0): ...
    def forward(self, x: Tensor, cond: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        # 若 use_film 且 cond 为 None，则通过内置的 nu_estimator 从 x 推断 cond
        # 返回 (pred, cond)

class ResidualFNO1d(FNOForecast1d):
    """Direct 模型：t_in=10, t_out=190。rollout() 直接返回 pred[:, :horizon]。"""

class ChunkedFNO1d(nn.Module):
    """
    Chunked 模型：内部持有 FNOForecast1d(t_out=chunk_size)。
    rollout() 方法自回归预测：每次取最近 t_in 步作为输入，预测 chunk_size 步，
    将预测结果 append 到历史，循环直到达到 horizon。
    """
    def __init__(self, modes=24, width=64, depth=4, t_in=10, chunk_size=10, use_film=False, dropout=0.0): ...
    def forward(self, x, cond=None) -> Tuple[Tensor, Optional[Tensor]]: ...
    def rollout(self, x, horizon=190, cond=None, detach_between_chunks=False) -> Tensor: ...
    @torch.no_grad()
    def rollout_no_grad(self, x, horizon=190, cond=None) -> Tensor: ...

def build_model(cfg, task="task1") -> nn.Module:
    """根据配置字典/Namespace 自动选择构建 ResidualFNO1d 或 ChunkedFNO1d。"""

def burgers_residual(u: Tensor, dx=1.0/256.0, dt=1.0, nu=1e-3) -> Tensor:
    """
    计算 Burgers 方程的可微残差：u_t + u * u_x - nu * u_xx。
    输入 u shape: [B, T, Nx]，要求 T >= 3。
    使用 FFT 计算空间导数（periodic boundary）。
    """
```

### 4.3 关键实现细节与陷阱

1. **坐标通道**：Lift 层输入是 `torch.cat([x, coord], dim=1)`，其中 `coord` 是 `linspace(0, 1, Nx)` 扩展后的 `[B, 1, Nx]`。这让模型感知到空间位置。
2. **残差输出设计**：最终输出不是 `project(h)`，而是 `x[:, -1:, :].expand(-1, t_out, -1) + project(h)`。这意味着模型学习的是**相对于最后一帧输入的增量**，大大降低了学习难度。
3. **Chunked rollout 中的 detach**：训练时 `rollout` 可以传 `detach_between_chunks=True` 来切断梯度流，防止长序列梯度爆炸；验证和推理时不需要。
4. **ν 估计器**：在 `FNOForecast1d` 内部实现一个小型 CNN：`Conv1d(t_in, 32, 1) → GELU → AdaptiveAvgPool1d(1) → Flatten → Linear(32, 1)`。Task 2 训练时传入真实 ν，推理时若 ν 缺失则自动估计。
5. **谱卷积的 `n` 参数**：`torch.fft.irfft` 必须显式传入 `n=n`（原始空间分辨率），否则当 `Nx` 为奇数时输出长度会错。
6. **权重初始化**：`SpectralConv1d` 的复数权重应使用 `scale * torch.randn(..., dtype=torch.cfloat)`，其中 `scale = 1.0 / sqrt(in_channels * out_channels)`。

---

## 5. utils.py — 评分、损失与工具

### 5.1 核心设计思想

- **评分必须与官方评测完全一致**。分为 3 段时间段，前 2 段用 Rel-MSE，第 3 段用 `max(Lorentzian, Frechet)`。
- 除官方评分外，可引入**辅助损失**（spatial gradient loss、temporal difference loss）来提升激波（shock）预测 fidelity。

### 5.2 必须实现的函数

```python
def set_seed(seed: int) -> None:
    """同时设置 Python/NumPy/Torch 的随机种子，并设置 cudnn.benchmark=True（CUDA 时）。"""

def compute_rel_mse(pred: Tensor, gt: Tensor, eps=1e-10) -> float:
    """
    逐样本逐时间步计算 rel_t = sum((pred-gt)^2) / sum(gt^2)
    对时间步取均值，单样本上限 5.0，再对所有样本取均值。
    """

def compute_rmse(pred: Tensor, gt: Tensor) -> float: ...

def compute_frechet_distance(u1: Tensor, u2: Tensor) -> float:
    """
    轻量级 Frechet-like 距离：比较空间均值序列和标准差序列的差异。
    FD^2 = mean((mean(u1,dim=-1) - mean(u2,dim=-1))^2) + mean((std(u1,dim=-1) - std(u2,dim=-1))^2)
    """

def compute_segment_scores(pred: Tensor, gt: Tensor) -> Dict[str, float]:
    """
    官方分段评分。pred 和 gt 都必须是 [B, 190, Nx]（即去掉前 10 步的未来帧）。
    分段：
    - Segment 1: [:48]   → score1 = 100 * exp(-20 * rel_mse)
    - Segment 2: [48:96] → score2 = 100 * exp(-10 * rel_mse)
    - Segment 3: [96:]   → score3 = max(100/(1+10*rmse), 50*exp(-fd^2))
    - total = 0.25*score1 + 0.25*score2 + 0.5*score3
    同时返回每段的 rel_mse, rmse, fd, spec_dist 作为诊断信息。
    """

def spectral_gradient_loss(pred: Tensor, gt: Tensor) -> Tensor:
    """惩罚空间一阶增量的不匹配（periodic roll）。"""

def temporal_difference_loss(pred: Tensor, gt: Tensor) -> Tensor:
    """惩罚时间增量的不匹配。"""

def save_hdf5(pred: np.ndarray, save_path: str) -> None:
    """保存预测结果到 HDF5，dataset 名为 'tensor'，dtype float32，建议加 gzip 压缩。"""

def save_metrics(metrics: dict, save_path: str) -> None: ...

class Timer:
    def __init__(self): ...
    def elapsed(self) -> float: ...

class Logger:
    """同时打印到 stdout 和写入文件。"""
    def __init__(self, log_dir: str, filename: str = "train.log"): ...
    def log(self, msg: str) -> None: ...
    def log_metrics(self, epoch: int, metrics: dict) -> None: ...
```

### 5.3 关键实现细节与陷阱

1. **compute_rel_mse 的 clamp**：`diff_sq / gt_sq` 必须对每个值 clamp 到 `max=5.0`，这是官方评测的硬性要求。漏掉 clamp 会导致分数计算偏差。
2. **compute_segment_scores 的 shape 检查**：必须 assert `pred.shape[1] == 190`。如果传入 `(B, 200, 256)` 会算出错误结果。
3. **Frechet distance**：注意是 `fd**2` 还是 `fd`，公式中是 `50 * exp(-FD²)`，但 `compute_frechet_distance` 返回的已经是 `FD²`（因为内部求和后没有开根号）。仔细检查实现与公式的一致性。
4. **HDF5 保存**：`f.create_dataset("tensor", data=pred.astype(np.float32), compression="gzip")`。不要写成其他 dataset 名。

---

## 6. train.py — 训练入口

### 6.1 核心设计思想

训练流程必须**完整、自包含、可复现**。一个良好的训练脚本应遵循以下顺序：

```
解析参数 → 设置随机种子 → 创建输出目录 → 加载数据 → 构建模型 →
定义优化器/调度器 → 训练循环（train_epoch + validate）→ 早停/保存最佳模型
```

### 6.2 必须实现的关键逻辑

```python
def parse_args() -> argparse.Namespace:
    """
    必须包含的参数：
    --task, --data_dir, --output_dir, --model_type (direct/chunked), --chunk_size,
    --epochs, --batch_size, --lr, --weight_decay, --modes, --width, --depth,
    --dropout, --scheduler (cosine/step), --patience, --val_fraction, --seed,
    --num_workers, --device (default="cuda" if torch.cuda.is_available() else "cpu"), --amp, --grad_clip,
    --grad_weight, --time_diff_weight, --time_weight,
    --use_physics_loss, --physics_weight,
    --use_film, --augment_shift, --t_in, --t_out,
    --unroll_chunks, --multi_step_weight, --ss_start_epoch, --ss_ramp_epochs, --ss_max_prob,
    --resume
    """

def train_epoch(model, loader, optimizer, scaler, args, normalizer, epoch) -> dict:
    """
    单次 epoch 训练。
    关键逻辑：
    1. model.train()
    2. 对每 batch：
       - autocast 前向（若 amp）
       - 计算 loss_data = weighted_mse(pred_local, y_local)
       - 可选：loss_grad, loss_temp, loss_multi（multi_step_rollout_loss）, loss_phys
       - 总 loss 加权求和
       - scaler.backward() 或 loss.backward()
       - grad_clip（建议 1.0）
       - optimizer.step()
    3. 返回各 loss 的 epoch 平均值字典。
    """

def validate(model, loader, args, normalizer) -> dict:
    """
    验证时必须做**完整 rollout**（190 步）。
    对 chunked 模型调用 model.rollout(x, horizon=190)；对 direct 模型直接 forward。
    将预测结果 denormalize 后，用 compute_segment_scores 计算分段得分。
    返回包含 score1/2/3/total 和各种诊断指标的字典。
    """

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    logger = Logger(args.output_dir)
    
    # 加载数据
    train_target_horizon = args.chunk_size * max(1, args.unroll_chunks) if args.model_type == "chunked" else args.t_out
    train_loader, val_loader = get_dataloaders(..., train_target_horizon=train_target_horizon, ...)
    normalizer = get_dataset_stats(train_loader.dataset)  # 提取训练集的归一化参数
    
    # 构建模型
    model = build_model(args, task=args.task).to(args.device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    
    # 优化器与调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr*0.02)
    scaler = GradScaler(enabled=args.amp)
    
    # 训练循环 + 早停
    best_score = -inf
    patience_count = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(...)
        val_metrics = validate(...)
        scheduler.step()
        
        score = val_metrics["total"]
        if score > best_score:
            best_score = score
            patience_count = 0
            # 保存 checkpoint，包含：epoch, model_state, optimizer_state, args, best_score, normalizer, val_metrics
            torch.save({...}, os.path.join(args.output_dir, "best_checkpoint.pt"))
        else:
            patience_count += 1
            if patience_count >= args.patience:
                logger.log(f"Early stopping at epoch {epoch}")
                break
    
    # 保存训练时间和最终指标
    save_metrics({"best_score": best_score, "train_time": timer.elapsed(), ...}, os.path.join(args.output_dir, "metrics.json"))
    with open(os.path.join(args.output_dir, "time.json"), "w") as f:
        json.dump({"train_time": train_time, "inference_time": 0.0}, f)
```

### 6.3 关键实现细节与陷阱

1. **validate 必须使用完整 rollout**：训练时可以用短窗口（如 chunk_size=10），但验证时必须从初始 10 步 rollout 到完整 190 步，否则 val score 不能反映真实能力。
2. **multi_step_rollout_loss**：仅在 `model_type == "chunked"` 且 `unroll_chunks > 1` 时启用。其逻辑是：在单个 batch 内，模型自回归预测 `unroll_chunks` 个 chunk，loss 始终与 true target 比较。`ss_prob`（scheduled sampling probability）控制是否用模型预测替代 teacher forcing。
3. **physics loss 的 denormalize**：计算 Burgers 残差前，必须将预测结果从 normalized 空间转回物理单位：`normalizer.decode(full)`。否则残差量纲错误。
4. **checkpoint 保存内容**：必须包含 `normalizer`（dict 形式），否则 inference 时无法正确反归一化。建议同时保存 `args`（便于 reconstruct 模型配置）。
5. **model compile 的 unwrap**：若使用了 `torch.compile(model)`，保存 state_dict 时必须取 `model._orig_mod` 否则会报错。
6. **time.json**：训练结束后写入 `{"train_time": <秒>, "inference_time": 0.0}`。infer.py 会更新 `inference_time`。

---

## 7. infer.py — 推理与提交生成

### 7.1 核心设计思想

推理脚本必须**完全独立**：给定一个 checkpoint 文件，无需任何其他上下文即可生成提交 HDF5。

### 7.2 必须实现的关键逻辑

```python
def parse_args():
    """参数：--task, --data_dir, --checkpoint, --output, --batch_size, --num_workers, --device"""

def main():
    args = parse_args()
    
    # 1. 加载 checkpoint
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    cfg = SimpleNamespace(**ckpt["args"])  # 重建配置
    # 用安全默认值填充旧 checkpoint 中可能缺失的字段（如 model_type, chunk_size, use_film 等）
    
    # 2. 重建模型并加载权重
    model = build_model(cfg, task=args.task).to(args.device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    
    # 3. 重建 normalizer
    normalizer = Normalizer(ckpt["normalizer"]["mean"], ckpt["normalizer"]["std"])
    
    # 4. 加载测试数据
    loader, _ = get_test_loader(args.data_dir, task=args.task, batch_size=args.batch_size,
                                 normalizer=normalizer, t_in=getattr(cfg, "t_in", 10))
    raw_test = load_initial_tensor(args.data_dir, args.task)  # shape [N, 10, 256]
    
    # 5. 推理
    preds = []
    with torch.no_grad():
        for batch in loader:
            if task2 and len(batch)==3: x, _, cond = batch; cond = cond.to(device)
            else: x = batch[0]; cond = None
            pred_norm = model.rollout(x.to(device), horizon=190, cond=cond)
            preds.append(normalizer.decode(pred_norm).cpu())
    future = torch.cat(preds, dim=0).numpy().astype(np.float32)  # [N, 190, 256]
    
    # 6. 拼接前 10 步并验证一致性
    full = np.empty((N, 200, 256), dtype=np.float32)
    full[:, :10, :] = raw_test[:, :10, :]
    full[:, 10:, :] = future
    assert np.allclose(full[:, :10, :], raw_test[:, :10, :], atol=1e-6)
    
    # 7. 保存 HDF5 和 time.json
    save_hdf5(full, args.output)
    # 更新 time.json：保留 train_time，写入 inference_time
```

### 7.3 关键实现细节与陷阱

1. **SimpleNamespace 默认值**：旧 checkpoint 可能缺少某些字段（如 `chunk_size`、`use_film`），必须在重建 cfg 后用安全默认值填充，否则 `build_model` 会 AttributeError。
2. **前 10 步一致性校验**：`assert np.allclose(full[:, :10, :], raw_test[:, :10, :], atol=1e-6)` 是提交前**必须**的检查。如果不一致，评测会直接判错。
3. **batch_size 建议**：推理时可用较大 batch_size（如 64），因为不需要反向传播，显存占用小。
4. **time.json 路径**：通常与 checkpoint 同目录。先读取已有的 `train_time`，再追加 `inference_time`。

---

## 8. 快速验证清单（Agent 自检用）

每当你生成/修改代码后，按以下顺序验证：

### 8.1 环境检查
```bash
python -c "import torch; import h5py; import numpy; print('OK')"
```

### 8.2 数据加载验证
```python
from code.dataset import get_dataloaders
loader, val_loader = get_dataloaders("./data_and_sample_submission/train_val_test_init", task="task1", model_type="chunked")
for x, y in loader:
    print(x.shape, y.shape)  # 应为 [B, 10, 256], [B, 10, 256]（chunked 默认 chunk_size=10）
    break
for x, y in val_loader:
    print(x.shape, y.shape)  # 应为 [B, 10, 256], [B, 190, 256]
    break
```

### 8.3 模型前向验证
```python
from code.model import ChunkedFNO1d
model = ChunkedFNO1d(modes=24, width=64, depth=4, t_in=10, chunk_size=10)
x = torch.randn(2, 10, 256)
pred = model.rollout(x, horizon=190)
assert pred.shape == (2, 190, 256)
```

### 8.4 训练一个 dummy epoch
```bash
cd code
python train.py --task task1 --model_type chunked --epochs 1 --batch_size 4 --output_dir ../output/task1/test_run
```

### 8.5 推理验证
```bash
cd code
python infer.py --checkpoint ../output/task1/test_run/best_checkpoint.pt --output ../output/task1/test_run/task1_pred.hdf5
python -c "
import h5py, numpy as np
with h5py.File('../output/task1/test_run/task1_pred.hdf5','r') as f:
    print(f['tensor'].shape)
"
```

---

## 9. 常见失败模式与排查

| 现象 | 可能原因 | 修复方法 |
|------|---------|----------|
| `RuntimeError: irfft(...)` 长度不匹配 | `torch.fft.irfft` 没传 `n` | 显式传入 `n=Nx` |
| 验证 score 始终很低（<10） | validate 用了 teacher forcing 而非 rollout | validate 必须调用 `model.rollout(x, horizon=190)` |
| 推理 HDF5 前 10 步不一致 | normalizer 在 inference 和训练时不一致 | 确保 checkpoint 保存了 normalizer，且 infer.py 正确读取 |
| Task 2 推理时 ν 缺失报错 | dataset.py 在 test 时尝试读取 `nu` | test loader 应处理 `nu` 不存在的情况，返回 None |
| Loss 为 nan | physics loss 在 normalized 空间计算 | 计算 burgers_residual 前先 `normalizer.decode()` |
| checkpoint 加载失败 | `torch.compile` 后保存了 compiled model 的 state_dict | 保存时取 `model._orig_mod.state_dict()` |

---

## 10. 推荐的首次实验配置

以下配置来自参考代码的默认参数，可直接作为 baseline：

```bash
# Task 1 — Chunked FNO（推荐）
python code/train.py \
  --task task1 \
  --model_type chunked \
  --chunk_size 10 \
  --modes 24 --width 64 --depth 4 \
  --epochs 220 --batch_size 16 --lr 1e-3 --weight_decay 1e-4 \
  --scheduler cosine --patience 35 \
  --grad_weight 0.05 --time_diff_weight 0.02 \
  --augment_shift \
  --output_dir output/task1_baseline

# Task 2 — Chunked FNO with FiLM + ν-estimator
python code/train.py \
  --task task2 \
  --model_type chunked \
  --chunk_size 10 \
  --modes 24 --width 64 --depth 4 \
  --epochs 200 --batch_size 32 --lr 1e-3 \
  --use_film \
  --output_dir output/task2_baseline
```

---

## 11. 代码生成过程的日志记录要求

根据 `AGENTS.md`，科研日志必须体现代码的**自主生成过程**。Agent 在编写代码时，应在日志中记录类似以下内容：

```
[思考] 我计划为 Task 1 实现一个 Chunked FNO 模型。
       参考 code-ref/model.py 中的 SpectralConv1d 思想，我将重新设计：
       - 使用 einsum 进行频域线性变换
       - 残差输出形式：last_frame + delta
       - 加入 GroupNorm 稳定训练
[行动] 正在生成 code/model.py，包含 SpectralConv1d, FNOBlock1d, ChunkedFNO1d 等类...
[思考] 数据加载方面，我设计 WindowedBurgersDataset 做滑动窗口，
       解决 Task 1 只有 100 条轨迹的样本不足问题...
```

这确保了 `task_logs.log` 中记录的代码生成过程与 `code/` 目录中的文件内容一致，满足评审要求。

---

## 12. 总结：Agent 编码执行路线图

```
Step 1: 生成 utils.py（最基础，无依赖）
Step 2: 生成 dataset.py（依赖 utils.py 中的 Normalizer 等）
Step 3: 生成 model.py（依赖 torch，不依赖 dataset）
Step 4: 生成 train.py（依赖 model, dataset, utils）
Step 5: 生成 infer.py（依赖 model, dataset, utils）
Step 6: 运行 dummy 训练（1 epoch）验证端到端可运行
Step 7: 运行完整训练 → 保存 checkpoint
Step 8: 运行推理 → 验证 HDF5 shape 和前 10 步一致性
Step 9: 打包提交
```

按此路线图执行，可确保 Agent 生成的代码**从一开始就具备可运行性**，避免"代码写出来但无法训练/推理"的低效循环。
