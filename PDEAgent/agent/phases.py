"""
四阶段科研闭环实现

1. Literature Phase   - 文献解析与逻辑解构
2. Diagnosis Phase    - 瓶颈诊断与假设提出
3. Design Phase       - 自主设计与代码演进
4. Experiment Phase   - 实验验证与科学迭代
"""
import json
import os
from typing import Dict, Any, Optional
from datetime import datetime

from .llm_client import LLMClient
from .memory import ResearchMemory, ExperimentRecord
from .tools import registry


SYSTEM_PROMPT = """你是PDE神经算子科研智能体，具备深度学习、偏微分方程和科学计算的全面知识。
你的目标是在零人工干预下，自主完成神经算子模型的研究、改进与验证。

核心能力：
- 深度阅读技术文档与论文，提取关键数学公式和算法逻辑
- 分析训练日志与实验数据，诊断模型瓶颈
- 提出有科学依据的优化假设，并直接编写高质量PyTorch代码
- 运行实验、分析结果、迭代改进

行事原则：
1. 严谨：每个假设必须有理论依据或数据支撑
2. 务实：优先选择工程上可行、效果可验证的方案
3. 记录：完整记录思考链路与实验轨迹
4. 迭代：不怕失败，从失败中提取信息指导下一次尝试

当前环境：
- 深度学习框架: PyTorch
- 任务: 1D Burgers方程神经算子预测
- 基线模型: FNO (Fourier Neural Operator)
- 数据格式: HDF5

================================================================================
【CLI参数兼容性要求 - 这是最关键的工程约束，连续5次实验失败均源于此】
================================================================================

实验调度器(runner)会以**固定格式**调用你的脚本。你必须确保生成的 train.py 和 infer.py
的参数接口与此完全兼容。任何不兼容都会导致实验在 argparse 阶段直接退出，没有任何训练发生。

训练命令格式（runner自动生成）：
  python code/train.py --task {task1|task2} --output_dir output/{task1|task2}/iter_N --data_dir ./data_and_sample_submission/train_val_test_init

推理命令格式（runner自动生成）：
  python code/infer.py --task {task1|task2} --checkpoint output/{task1|task2}/iter_N/best_checkpoint.pt --output output/{task1|task2}/iter_N/pred.hdf5 --data_dir ./data_and_sample_submission/train_val_test_init

【argparse 强制规范】
1. train.py 和 infer.py 必须显式支持 --task、--output_dir、--data_dir
2. 必须使用 argparse 别名机制，同时支持下划线和横线版本：
   parser.add_argument("--output-dir", "--output_dir", dest="output_dir", default="./output")
   parser.add_argument("--data-dir", "--data_dir", dest="data_dir", default="./data_and_sample_submission/train_val_test_init")
3. 所有参数必须有合理的默认值，确保即使只传 --task 和 --output_dir 也能运行
4. checkpoint 文件必须保存为 best_checkpoint.pt（参考代码的命名）
5. --device 参数默认值必须是 `"cuda" if torch.cuda.is_available() else "cpu"`，不可硬编码为 "cpu"
6. 若使用 parse_known_args() 作为兜底，必须确保 output_dir 和 data_dir 已被显式定义，否则不应简单忽略

【常见失败模式（必须避免）】
- ❌ train.py 只认识 --out-dir，不认识 --output_dir → runner 注入 --output_dir 时直接报错退出
- ❌ infer.py 不认识 --task → runner 注入 --task 时直接报错退出
- ❌ 没有 --data_dir 参数 → runner 不会传入数据路径，脚本使用硬编码路径可能找不到数据
- ❌ checkpoint 保存为 best_model.pt，但 runner 寻找 best_checkpoint.pt → 推理失败
- ❌ --device 被硬编码为 "cpu"，在有 GPU 的环境下浪费计算资源 → 训练时间过长
- ❌ 参数是 required=True 但 runner 没有传入 → argparse 直接退出

================================================================================
【代码生成规范 - 参考 AGENT_CODE_GUIDE.md】
================================================================================

必须生成的5个核心文件及其职责：
- code/model.py: 神经算子模型（FNO/ChunkedFNO），含 SpectralConv1d, FNOBlock1d, FiLM
- code/dataset.py: 数据加载与标量归一化（Normalizer, BurgersDataset, WindowedBurgersDataset）
- code/train.py: 训练入口，支持验证、早停、学习率调度、保存 checkpoint
- code/infer.py: 推理入口，加载 checkpoint 生成提交格式的 HDF5
- code/utils.py: 评分计算（compute_segment_scores）、辅助损失、工具函数

【数据流关键约定】
- Task 1 数据: task1_val.hdf5（训练/验证切分），task1_test.hdf5（测试）
- Task 2 数据: task2_part{0,1,2}_train.h5（训练），task2_val.h5（验证），task2_test.h5（测试）
- Task 1 HDF5 key: "tensor", "x-coordinate", "t-coordinate"
- Task 2 HDF5 key: "tensor", "x_coordinate", "t_coordinate", "nu"
- 归一化: 全局标量 mean/std，在训练集上计算，共享给 val/test
- 输入: [B, 10, 256]；输出未来帧: [B, 190, 256]；提交 HDF5: [B, 200, 256]（前10步复制GT）

【模型架构关键设计】
- 推荐 ChunkedFNO1d（chunk_size=10，自回归rollout到190步），配合 WindowedBurgersDataset 滑动窗口训练
- Lift 层输入 concat 空间坐标通道 (linspace 0~1)
- 残差输出: last_frame.expand(-1, t_out, -1) + project(features)
- 验证时必须做完整 190 步 rollout，不能用 teacher forcing
- Task 2 需内置 nu_estimator（CNN→Pool→Linear），测试时自动估计

【评分计算关键细节】
- 3 段式评分，pred/gt 必须是 [B, 190, 256]
- Rel-MSE 必须 clamp(max=5.0)
- Segment 3: score3 = max(100/(1+10*rmse), 50*exp(-fd^2))
"""


class Phase:
    """阶段基类"""
    def __init__(self, client: LLMClient, memory: ResearchMemory, cfg=None):
        self.client = client
        self.memory = memory
        self.cfg = cfg
    
    def run(self) -> bool:
        """执行阶段，返回是否成功完成"""
        raise NotImplementedError
    
    def _chat(self, messages: list, tools: bool = True) -> Dict[str, Any]:
        """调用LLM，可选择是否启用工具"""
        schemas = registry.get_schemas() if tools else None
        return self.client.chat(messages, tools=schemas)
    
    def _tool_call_loop(self, messages: list, max_rounds: int = 10) -> str:
        """
        工具调用循环：让LLM反复思考→调用工具→观察结果，直到不再调用工具或达到最大轮数
        """
        for _ in range(max_rounds):
            try:
                resp = self._chat(messages)
            except Exception as e:
                # API调用失败，返回错误信息以便上层处理
                return f"[Error] LLM API call failed: {e}"
            
            if resp["tool_calls"]:
                # 添加助手消息（含工具调用）
                messages.append({
                    "role": "assistant",
                    "content": resp["content"] or "",
                    "tool_calls": [
                        {
                            "id": tc.get("id", f"call_{i}"),
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["arguments"]),
                            },
                        }
                        for i, tc in enumerate(resp["tool_calls"])
                    ],
                })
                
                # 执行工具并添加结果
                for i, tc in enumerate(resp["tool_calls"]):
                    result = registry.call(tc["name"], tc["arguments"])
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", f"call_{i}"),
                        "content": result,
                    })
            else:
                # LLM不再调用工具，返回最终内容
                return resp["content"] or ""
        
        return messages[-1]["content"] if messages else ""


# =============================================================================
# Phase 1: 文献解析与逻辑解构
# =============================================================================

class LiteraturePhase(Phase):
    """文献解析阶段：阅读项目文档、理解数据、分析基线"""
    
    def run(self) -> bool:
        print("\n[Phase 1] 文献解析与逻辑解构...")
        
        prompt = f"""请执行文献解析与逻辑解构任务。你需要：

1. 阅读项目文档：Background.md、NEURAL_OPERATOR_PRINCIPLES.md、AGENTS.md
2. 检查数据文件结构：使用 inspect_hdf5 查看训练/验证/测试数据
3. 分析现有代码（如有）：使用 summarize_code 查看 code/ 目录下的文件
4. 输出一份结构化的文献与技术综述

请调用工具完成上述任务，然后综合所有信息，回答以下问题：
- 本任务的核心科学问题是什么？
- 基线模型（FNO/DeepONet/PI-DeepONet）的核心数学原理？
- 数据的具体规模、维度、物理含义？
- 当前已知的技术难点与优化方向？
- 评分规则对模型有什么特殊要求？

当前任务: {self.memory.task}
数据目录: ./data_and_sample_submission/train_val_test_init/
"""
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        
        summary = self._tool_call_loop(messages)
        self.memory.literature_summary = summary
        self.memory.current_phase = "diagnosis"
        
        # 保存综述到文件
        registry.call("write_file", {
            "path": f"{self.memory.task}/{self.memory.task}_literature_summary.md",
            "content": f"# 文献与技术综述\n\n{summary}\n",
        })
        
        print("[Phase 1] 完成。文献综述已保存。")
        return True


# =============================================================================
# Phase 2: 瓶颈诊断与假设提出
# =============================================================================

class DiagnosisPhase(Phase):
    """瓶颈诊断阶段：分析基线性能，提出优化假设"""
    
    def run(self) -> bool:
        print("\n[Phase 2] 瓶颈诊断与假设提出...")
        
        context = self.memory.get_context()
        
        prompt = f"""基于以下研究上下文，进行瓶颈诊断与假设提出：

{context}

你的任务是：
1. 如果已有训练日志或实验结果，使用 analyze_log 分析训练动态
2. 如果已有代码，使用 summarize_code 审查关键模块
3. 基于文献综述和现有证据，识别当前基线的主要瓶颈：
   - 长时稳定性（第3段95-190步权重50%）
   - 物理一致性（是否满足Burgers方程）
   - 泛化能力（Task 2的变nu问题）
   - 计算效率
4. 对每个瓶颈，提出1-3个具体的、可验证的优化假设
5. 为每个假设给出：理论依据、预期效果、验证方法

请调用工具收集所需信息，然后输出诊断报告。
"""
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        
        diagnosis = self._tool_call_loop(messages)
        
        # 解析诊断报告中的假设和瓶颈
        self._parse_diagnosis(diagnosis)
        self.memory.current_phase = "design"
        
        registry.call("write_file", {
            "path": f"{self.memory.task}/{self.memory.task}_diagnosis_report.md",
            "content": f"# 瓶颈诊断与假设报告\n\n{diagnosis}\n",
        })
        
        print(f"[Phase 2] 完成。识别瓶颈 {len(self.memory.bottlenecks)} 个，提出假设 {len(self.memory.hypotheses)} 个。")
        return True
    
    def _parse_diagnosis(self, text: str):
        """简单解析诊断文本，提取瓶颈和假设"""
        lines = text.splitlines()
        current = None
        for line in lines:
            line = line.strip()
            if "瓶颈" in line.lower() or "bottleneck" in line.lower():
                current = "bottleneck"
                continue
            if "假设" in line.lower() or "hypothesis" in line.lower():
                current = "hypothesis"
                continue
            if line.startswith("-") or line.startswith("*"):
                item = line[1:].strip()
                if current == "bottleneck" and item:
                    self.memory.bottlenecks.append(item)
                elif current == "hypothesis" and item:
                    self.memory.hypotheses.append(item)


# =============================================================================
# Phase 3: 自主设计与代码演进
# =============================================================================

class DesignPhase(Phase):
    """代码设计阶段：根据假设编写/修改代码"""
    
    def run(self) -> bool:
        print("\n[Phase 3] 自主设计与代码演进...")
        
        context = self.memory.get_context()
        
        # 读取当前代码状态
        code_files = []
        code_dir = "./code"
        if os.path.exists(code_dir):
            for fname in os.listdir(code_dir):
                if fname.endswith(".py"):
                    content = registry.call("read_file", {"path": os.path.join(code_dir, fname)})
                    code_files.append(f"=== {fname} ===\n{json.loads(content).get('content', '')[:1500]}\n")
        
        code_context = "\n".join(code_files) if code_files else "当前 code/ 目录为空或不存在。"
        
        prompt = f"""基于以下上下文，进行代码设计与演进：

{context}

当前代码状态：
{code_context}

【必读】请首先阅读 AGENT_CODE_GUIDE.md，理解参考代码的运行逻辑和工程结构。

你的任务是：
1. 根据当前最优假设，决定需要创建或修改哪些代码文件
2. 使用 write_file 工具直接编写高质量 PyTorch 代码
3. 代码要求：
   - 使用 typing 类型注解
   - 包含清晰的 docstring
   - 支持命令行参数配置
   - 包含错误处理和日志记录
   - 训练过程保存最佳模型

【参数接口强制要求 - 失败5次的核心教训】
你生成的 train.py 和 infer.py 的 argparse 必须与 runner 的调用格式完全兼容。

runner 调用 train.py 的格式：
  python code/train.py --task task1 --output_dir output/task1/iter_N --data_dir ./data_and_sample_submission/train_val_test_init

runner 调用 infer.py 的格式：
  python code/infer.py --task task1 --checkpoint output/task1/iter_N/best_checkpoint.pt --output output/task1/iter_N/pred.hdf5 --data_dir ./data_and_sample_submission/train_val_test_init

train.py 必须接受的核心参数（使用 argparse，全部带合理默认值）：
- --task: default="task1", choices=["task1", "task2"]
- --output_dir / --output-dir: dest="output_dir", default="./output"
- --data_dir / --data-dir: dest="data_dir", default="./data_and_sample_submission/train_val_test_init"
- --model_type: default="chunked", choices=["direct", "chunked"]
- --chunk_size, --epochs, --batch_size, --lr, --weight_decay
- --modes, --width, --depth, --dropout
- --scheduler, --patience, --val_fraction, --seed
- --num_workers, --device, --amp, --grad_clip
- --t_in(=10), --t_out(=190)
- 其他你需要的训练参数

infer.py 必须接受的核心参数：
- --task: default="task1", choices=["task1", "task2"]
- --checkpoint: required=True
- --output: required=True
- --data_dir / --data-dir: dest="data_dir", default="./data_and_sample_submission/train_val_test_init"
- --batch_size, --num_workers, --device
- 其他你需要的推理参数

【argparse 别名写法示例】
parser.add_argument("--output-dir", "--output_dir", dest="output_dir", default="./output", help="Output directory")
parser.add_argument("--data-dir", "--data_dir", dest="data_dir", default="./data_and_sample_submission/train_val_test_init", help="Data directory")
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")

【checkpoint 命名要求】
训练保存的最佳模型必须命名为 best_checkpoint.pt，因为 runner 硬编码寻找此文件。
同时保存为 best_model.pt 也可以，但 best_checkpoint.pt 必须存在。

必须包含的文件：
- code/model.py: 神经算子模型定义（推荐 ChunkedFNO1d）
- code/dataset.py: 数据加载与预处理（Normalizer, BurgersDataset, WindowedBurgersDataset）
- code/train.py: 训练脚本（支持验证、早停、学习率调度、保存 checkpoint）
- code/infer.py: 推理脚本（生成符合提交要求的 HDF5 [B,200,256]）
- code/utils.py: 辅助函数（compute_segment_scores, save_hdf5, Logger, Timer）

对于 Task 1（固定nu=0.001）：
- 输入：前10个时间步 (B, 10, 256)
- 输出：未来190个时间步 + 复制前10步 = (B, 200, 256)
- 推荐：ChunkedFNO1d（chunk_size=10）+ WindowedBurgersDataset 滑动窗口训练
- 验证时必须完整 rollout 190 步

对于 Task 2（变nu）：
- 训练时可用nu值，测试时不提供nu
- 推荐：FNOForecast1d 内置 nu_estimator，训练时传入真实nu，推理时自动估计
- 使用 FiLM 进行条件化

重要：每写完一个关键文件后，请调用 validate_code 检查语法。
全部代码写完后，调用 quick_test_model 进行模型前向传播的 smoke test。
最后，用 list_files 确认 code/ 目录结构。
"""
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        
        result = self._tool_call_loop(messages, max_rounds=20)
        
        self.memory.current_phase = "experiment"
        self.memory.code_versions.append({
            "iteration": self.memory.iteration,
            "timestamp": datetime.now().isoformat(),
            "note": result[:500],
        })
        
        registry.call("write_file", {
            "path": f"{self.memory.task}/{self.memory.task}_design_notes.md",
            "content": f"# 设计迭代 {self.memory.iteration}\n\n{result}\n",
        })
        
        print("[Phase 3] 完成。代码已更新。")
        return True


# =============================================================================
# Phase 4: 实验验证与科学迭代
# =============================================================================

class ExperimentPhase(Phase):
    """实验验证阶段：运行训练、评估、分析结果、决定下一步"""
    
    def run(self) -> bool:
        print("\n[Phase 4] 实验验证与科学迭代...")
        
        self.memory.iteration += 1
        exp_id = len(self.memory.experiments) + 1
        
        context = self.memory.get_context()
        
        # 构建运行命令，传入 runner 自动注入的完整参数
        data_dir = self.cfg.research.data_dir if self.cfg else "./data_and_sample_submission/train_val_test_init"
        output_iter = f"output/{self.memory.task}/iter_{self.memory.iteration}"
        
        train_cmd = f"python code/train.py --task {self.memory.task} --output_dir {output_iter} --data_dir {data_dir}"
        infer_cmd = f"python code/infer.py --task {self.memory.task} --checkpoint {output_iter}/best_checkpoint.pt --output {output_iter}/pred.hdf5 --data_dir {data_dir}"
        
        print(f"[Experiment] Train command: {train_cmd}")
        print(f"[Experiment] Infer command: {infer_cmd}")
        
        # 运行训练
        train_result = registry.call("run_shell", {
            "command": train_cmd,
            "timeout": 1800,
        })
        
        # 运行推理
        infer_result = registry.call("run_shell", {
            "command": infer_cmd,
            "timeout": 300,
        })
        
        # 尝试读取验证指标
        metrics = {}
        metrics_path = f"output/{self.memory.task}/iter_{self.memory.iteration}/metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
        
        # 记录实验
        record = ExperimentRecord(
            id=exp_id,
            timestamp=datetime.now().isoformat(),
            phase="experiment",
            hypothesis=self.memory.hypotheses[0] if self.memory.hypotheses else "baseline",
            code_changes=[],
            config={},
            metrics=metrics,
            conclusion="",
            status="success" if metrics else "failed",
        )
        
        # 让LLM分析实验结果并决定下一步
        prompt = f"""实验已执行，请分析结果并决定下一步：

研究上下文：
{context}

训练输出：
{train_result}

推理输出：
{infer_result}

验证指标：
{json.dumps(metrics, ensure_ascii=False, indent=2)}

【实验结果分析决策树 - 按优先级执行】

第一步：判断是否是 CLI/参数错误（最高优先级）
- 如果训练输出包含 "error: the following arguments are required" 或 "error: unrecognized arguments"
- 这说明代码参数接口与 runner 不兼容，**不是模型问题**
- 必须立即：修改 train.py / infer.py 的 argparse，添加缺失的参数别名
- 常见修复：
  * 添加 --task 参数（即使脚本内部不使用）
  * 添加 --output_dir 和 --output-dir 别名指向同一 dest
  * 添加 --data_dir 和 --data-dir 别名指向同一 dest
  * 确保 checkpoint 保存为 best_checkpoint.pt
- 决策：CONTINUE（修复CLI后重跑）
- **不要分析不存在的训练结果，不要提出模型架构修改**

第二步：判断是否训练启动但数据未找到
- 如果训练输出包含 "FileNotFoundError" 或 "HDF5 file not found"
- 检查 dataset.py 中的默认数据路径是否正确
- 确保 get_dataloaders 的默认 data_dir 为 "./data_and_sample_submission/train_val_test_init"
- 决策：CONTINUE（修复路径后重跑）

第三步：判断训练是否正常进行（有 loss 曲线）
- 分析训练过程：是否收敛？是否过拟合？损失曲线形态？
- 分析验证指标：各段得分如何？长时稳定性（第3段）是否达标？
- 对比历史最优：是否有提升？提升/下降的原因是什么？
- 做出决策：
   - CONTINUE: 当前方向有潜力，继续迭代优化（提出具体修改建议）
   - PIVOT: 当前方向遇到瓶颈，切换假设或模型架构
   - STOP: 结果已足够好，或资源用尽，结束迭代

请用以下格式输出：
DECISION: [CONTINUE|PIVOT|STOP]
REASON: ...
NEXT_ACTION: ...
"""
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        
        analysis = self.client.chat(messages)["content"]
        record.conclusion = analysis
        
        # 解析决策
        decision = "CONTINUE"
        if "DECISION:" in analysis:
            decision = analysis.split("DECISION:")[1].split()[0].strip().upper()
        
        if decision == "STOP":
            record.status = "success"
            self.memory.stop_reason = "Agent decided to stop after analysis."
        elif decision == "PIVOT":
            self.memory.current_phase = "diagnosis"
        else:
            self.memory.current_phase = "design"
        
        self.memory.add_experiment(record)
        
        registry.call("write_file", {
            "path": f"{self.memory.task}/{self.memory.task}_experiment_{exp_id}_report.md",
            "content": f"# 实验 {exp_id} 报告\n\n{analysis}\n",
        })
        
        print(f"[Phase 4] 完成。实验 {exp_id} 决策: {decision}")
        return True
