"""
Agent 入口模块
"""
import argparse
import sys

from .config import load_config, save_config, AgentConfig
from .orchestrator import ResearchOrchestrator


def main():
    parser = argparse.ArgumentParser(description="PDE Neural Operator Research Agent")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--task", default="task1", choices=["task1", "task2"], help="任务选择")
    parser.add_argument("--init-config", action="store_true", help="生成默认配置文件并退出")
    parser.add_argument("--api-key", default="", help="LLM API Key（会覆盖配置文件）")
    parser.add_argument("--base-url", default="", help="LLM API Base URL")
    parser.add_argument("--model", default="", help="LLM Model")
    parser.add_argument("--resume", action="store_true", help="从上次状态恢复")
    args = parser.parse_args()
    
    if args.init_config:
        save_config(AgentConfig(), args.config)
        print(f"默认配置已保存到: {args.config}")
        print("请编辑配置文件，填写 API Key 和其他参数。")
        sys.exit(0)
    
    # 加载配置
    cfg = load_config(args.config)
    cfg.research.task = args.task
    
    if args.api_key:
        cfg.llm.api_key = args.api_key
    if args.base_url:
        cfg.llm.base_url = args.base_url
    if args.model:
        cfg.llm.model = args.model
    
    # 检查 API Key
    if not cfg.llm.api_key:
        print("[Error] LLM API Key 未配置。请通过以下方式之一设置：")
        print("  1. 编辑 config.yaml 中的 llm.api_key")
        print("  2. 设置环境变量 OPENAI_API_KEY")
        print("  3. 命令行参数 --api-key")
        sys.exit(1)
    
    # 启动编排器
    orch = ResearchOrchestrator(cfg)
    try:
        orch.run()
    finally:
        orch.close()


if __name__ == "__main__":
    main()
