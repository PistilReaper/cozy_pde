"""
配置管理模块
"""
import os
import yaml
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class LLMConfig:
    """LLM API 配置"""
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: float = 120.0


@dataclass
class ResearchConfig:
    """科研流程配置"""
    max_iterations: int = 15
    max_time_hours: float = 10.5  # 留点余量，不超过12小时
    early_stop_patience: int = 3
    task: str = "task1"  # task1 或 task2
    data_dir: str = "./data_and_sample_submission/train_val_test_init"
    output_dir: str = "./output"
    code_dir: str = "./code"


@dataclass
class ModelConfig:
    """基线模型配置"""
    model_type: str = "fno"  # fno, deeponet, pi_deeponet
    modes: int = 16
    width: int = 64
    depth: int = 4
    batch_size: int = 16
    epochs: int = 100
    lr: float = 1e-3
    scheduler_step: int = 20
    scheduler_gamma: float = 0.5
    use_physics_loss: bool = False
    physics_weight: float = 0.1
    use_pushforward: bool = True
    pushforward_schedule: List[int] = field(default_factory=lambda: [10, 30, 60])


@dataclass
class AgentConfig:
    """Agent 总配置"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


def load_config(path: str = "config.yaml") -> AgentConfig:
    """从YAML加载配置，环境变量可覆盖"""
    cfg = AgentConfig()
    
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data:
            if "llm" in data:
                for k, v in data["llm"].items():
                    if hasattr(cfg.llm, k):
                        setattr(cfg.llm, k, v)
            if "research" in data:
                for k, v in data["research"].items():
                    if hasattr(cfg.research, k):
                        setattr(cfg.research, k, v)
            if "model" in data:
                for k, v in data["model"].items():
                    if hasattr(cfg.model, k):
                        setattr(cfg.model, k, v)
    
    # 环境变量覆盖
    if os.environ.get("OPENAI_API_KEY"):
        cfg.llm.api_key = os.environ["OPENAI_API_KEY"]
    if os.environ.get("OPENAI_BASE_URL"):
        cfg.llm.base_url = os.environ["OPENAI_BASE_URL"]
    if os.environ.get("LLM_MODEL"):
        cfg.llm.model = os.environ["LLM_MODEL"]
    
    return cfg


def save_config(cfg: AgentConfig, path: str = "config.yaml"):
    """保存配置到YAML"""
    data = {
        "llm": {
            "api_key": cfg.llm.api_key if cfg.llm.api_key else "<YOUR_API_KEY>",
            "base_url": cfg.llm.base_url,
            "model": cfg.llm.model,
            "temperature": cfg.llm.temperature,
            "max_tokens": cfg.llm.max_tokens,
            "timeout": cfg.llm.timeout,
        },
        "research": {
            "max_iterations": cfg.research.max_iterations,
            "max_time_hours": cfg.research.max_time_hours,
            "early_stop_patience": cfg.research.early_stop_patience,
            "task": cfg.research.task,
            "data_dir": cfg.research.data_dir,
            "output_dir": cfg.research.output_dir,
            "code_dir": cfg.research.code_dir,
        },
        "model": {
            "model_type": cfg.model.model_type,
            "modes": cfg.model.modes,
            "width": cfg.model.width,
            "depth": cfg.model.depth,
            "batch_size": cfg.model.batch_size,
            "epochs": cfg.model.epochs,
            "lr": cfg.model.lr,
            "scheduler_step": cfg.model.scheduler_step,
            "scheduler_gamma": cfg.model.scheduler_gamma,
            "use_physics_loss": cfg.model.use_physics_loss,
            "physics_weight": cfg.model.physics_weight,
            "use_pushforward": cfg.model.use_pushforward,
            "pushforward_schedule": cfg.model.pushforward_schedule,
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)
