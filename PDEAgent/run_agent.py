"""
PDE Neural Operator Research Agent - 启动脚本

使用方式：
  1. 首次运行，生成配置文件：
     python run_agent.py --init-config

  2. 编辑 config.yaml，填入 LLM API Key

  3. 运行 Agent：
     python run_agent.py --task task1
     python run_agent.py --task task2

  4. 或直接通过环境变量/命令行传入：
     OPENAI_API_KEY=sk-xxx python run_agent.py --task task1
"""
import sys
import os

# 确保可以导入 agent 包
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.main import main

if __name__ == "__main__":
    main()
