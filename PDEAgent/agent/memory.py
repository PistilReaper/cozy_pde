"""
科研记忆模块

维护Agent的研究状态、实验历史、假设与结论。
支持持久化到JSON，确保科研迭代的连续性。
"""
import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class ExperimentRecord:
    """单次实验记录"""
    id: int
    timestamp: str
    phase: str  # 所属阶段
    hypothesis: str  # 实验假设
    code_changes: List[str] = field(default_factory=list)  # 修改的文件列表
    config: Dict[str, Any] = field(default_factory=dict)  # 实验配置
    metrics: Dict[str, float] = field(default_factory=dict)  # 评估指标
    conclusion: str = ""  # 实验结论
    status: str = "running"  # running, success, failed


@dataclass
class ResearchMemory:
    """科研记忆总状态"""
    task: str = "task1"
    current_phase: str = "literature"  # literature, diagnosis, design, experiment
    iteration: int = 0
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # 阶段产出
    literature_summary: str = ""
    bottlenecks: List[str] = field(default_factory=list)
    hypotheses: List[str] = field(default_factory=list)
    
    # 实验历史
    experiments: List[ExperimentRecord] = field(default_factory=list)
    best_experiment_id: Optional[int] = None
    best_metrics: Dict[str, float] = field(default_factory=dict)
    
    # 代码版本追踪
    code_versions: List[Dict] = field(default_factory=list)
    
    # 终止条件
    stop_reason: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def save(self, path: str = "research_memory.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str = "research_memory.json") -> Optional["ResearchMemory"]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 将 experiments 的 dict 列表转换回 ExperimentRecord 对象
            if "experiments" in data and isinstance(data["experiments"], list):
                data["experiments"] = [
                    ExperimentRecord(**e) if isinstance(e, dict) else e
                    for e in data["experiments"]
                ]
            return cls(**data)
        return None
    
    def add_experiment(self, record: ExperimentRecord):
        self.experiments.append(record)
        
        # 自动更新最优实验
        score = record.metrics.get("val_score", 0)
        best_score = self.best_metrics.get("val_score", -1)
        if score > best_score and record.status == "success":
            self.best_metrics = record.metrics
            self.best_experiment_id = record.id
    
    def get_experiment(self, exp_id: int) -> Optional[ExperimentRecord]:
        for e in self.experiments:
            if e.id == exp_id:
                return e
        return None
    
    def get_context(self, max_experiments: int = 3) -> str:
        """生成给LLM的上下文摘要"""
        lines = []
        lines.append(f"=== 当前任务: {self.task} | 阶段: {self.current_phase} | 迭代: {self.iteration} ===")
        lines.append("")
        
        if self.literature_summary:
            lines.append("【文献综述摘要】")
            lines.append(self.literature_summary[:800])
            lines.append("")
        
        if self.bottlenecks:
            lines.append("【已识别瓶颈】")
            for b in self.bottlenecks:
                lines.append(f"- {b}")
            lines.append("")
        
        if self.hypotheses:
            lines.append("【当前假设】")
            for h in self.hypotheses:
                lines.append(f"- {h}")
            lines.append("")
        
        if self.experiments:
            lines.append("【近期实验】")
            for e in self.experiments[-max_experiments:]:
                lines.append(f"- Exp {e.id} ({e.status}): {e.hypothesis}")
                if e.metrics:
                    lines.append(f"  Metrics: {json.dumps(e.metrics, ensure_ascii=False)}")
                if e.conclusion:
                    lines.append(f"  Conclusion: {e.conclusion}")
            lines.append("")
        
        if self.best_metrics:
            lines.append("【当前最优指标】")
            lines.append(json.dumps(self.best_metrics, ensure_ascii=False))
            lines.append("")
        
        return "\n".join(lines)
