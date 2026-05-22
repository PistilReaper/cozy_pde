"""
Agent 工具定义

为科研Agent提供文件读写、代码执行、日志分析等能力。
"""
import os
import re
import json
import subprocess
import traceback
import tempfile
from typing import Dict, Any, List, Optional


class ToolRegistry:
    """工具注册表，管理所有可用工具"""

    def __init__(self):
        self._tools: Dict[str, Any] = {}
        self._schemas: List[Dict] = []

    def register(self, name: str, description: str, parameters: Dict):
        """装饰器：注册一个工具"""
        def decorator(func):
            self._tools[name] = func
            self._schemas.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            })
            return func
        return decorator

    def call(self, name: str, arguments: Dict) -> str:
        """调用指定工具"""
        if name not in self._tools:
            return json.dumps({"error": f"Tool '{name}' not found."}, ensure_ascii=False)
        try:
            result = self._tools[name](**arguments)
            if not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False, default=str)
            return result
        except Exception as e:
            return json.dumps({
                "error": str(e),
                "traceback": traceback.format_exc(),
            }, ensure_ascii=False)

    def get_schemas(self) -> List[Dict]:
        return self._schemas


# 全局注册表实例
registry = ToolRegistry()


# =============================================================================
# 文件操作工具
# =============================================================================

@registry.register(
    name="read_file",
    description="读取指定文件的内容。可读取文本文件、代码文件、日志文件等。",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "文件路径"},
            "offset": {"type": "integer", "description": "起始行号（从1开始），默认1"},
            "limit": {"type": "integer", "description": "最多读取行数，默认100"},
        },
        "required": ["path"],
    },
)
def read_file(path: str, offset: int = 1, limit: int = 100) -> str:
    if not os.path.exists(path):
        return json.dumps({"error": f"File not found: {path}"}, ensure_ascii=False)
    
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        total = len(lines)
        start = max(0, offset - 1)
        end = min(total, start + limit)
        selected = lines[start:end]
        content = "".join(selected)
        return json.dumps({
            "path": path,
            "total_lines": total,
            "offset": offset,
            "limit": limit,
            "content": content,
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@registry.register(
    name="write_file",
    description="写入或覆盖文件内容。用于生成代码、修改配置文件、记录实验笔记等。",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "文件路径"},
            "content": {"type": "string", "description": "文件内容"},
        },
        "required": ["path", "content"],
    },
)
def write_file(path: str, content: str) -> str:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return json.dumps({"status": "ok", "path": path, "bytes": len(content.encode("utf-8"))}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@registry.register(
    name="append_file",
    description="在文件末尾追加内容。",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "文件路径"},
            "content": {"type": "string", "description": "追加内容"},
        },
        "required": ["path", "content"],
    },
)
def append_file(path: str, content: str) -> str:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(content)
        return json.dumps({"status": "ok", "path": path}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@registry.register(
    name="list_files",
    description="列出目录中的文件和子目录。",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "目录路径，默认为当前目录"},
        },
        "required": [],
    },
)
def list_files(path: str = ".") -> str:
    try:
        entries = os.listdir(path)
        files = []
        dirs = []
        for e in entries:
            full = os.path.join(path, e)
            if os.path.isdir(full):
                dirs.append(e)
            else:
                files.append(e)
        return json.dumps({"path": path, "files": files, "directories": dirs}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# =============================================================================
# 命令执行工具
# =============================================================================

@registry.register(
    name="run_shell",
    description="执行Shell命令（PowerShell/Bash），用于运行训练脚本、安装依赖、查看系统状态等。超时时间300秒。",
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "要执行的命令"},
            "cwd": {"type": "string", "description": "工作目录，默认当前目录"},
            "timeout": {"type": "integer", "description": "超时时间（秒），默认300"},
        },
        "required": ["command"],
    },
)
def run_shell(command: str, cwd: str = ".", timeout: int = 300) -> str:
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        stdout = result.stdout[-4000:] if len(result.stdout) > 4000 else result.stdout
        stderr = result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr
        return json.dumps({
            "returncode": result.returncode,
            "stdout": stdout,
            "stderr": stderr,
        }, ensure_ascii=False)
    except subprocess.TimeoutExpired:
        return json.dumps({"error": f"Command timed out after {timeout}s"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@registry.register(
    name="run_python",
    description="执行一段Python代码并返回输出结果。用于快速数值计算、数据分析、绘图等。",
    parameters={
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python代码字符串"},
        },
        "required": ["code"],
    },
)
def run_python(code: str) -> str:
    import io
    import sys
    
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    sys.stdout = stdout_buffer
    sys.stderr = stderr_buffer
    
    try:
        exec(code, {"__name__": "__main__"})
        out = stdout_buffer.getvalue()
        err = stderr_buffer.getvalue()
        return json.dumps({"stdout": out[-4000:], "stderr": err[-2000:]}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "traceback": traceback.format_exc(),
        }, ensure_ascii=False)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


# =============================================================================
# 代码验证工具
# =============================================================================

@registry.register(
    name="validate_code",
    description="检查Python代码的语法正确性，运行py_compile进行验证。",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Python文件路径"},
        },
        "required": ["path"],
    },
)
def validate_code(path: str) -> str:
    import py_compile
    if not os.path.exists(path):
        return json.dumps({"error": f"File not found: {path}"}, ensure_ascii=False)
    try:
        py_compile.compile(path, doraise=True)
        return json.dumps({"status": "ok", "path": path, "message": "Syntax check passed."}, ensure_ascii=False)
    except py_compile.PyCompileError as e:
        return json.dumps({"status": "error", "path": path, "error": str(e)}, ensure_ascii=False)


@registry.register(
    name="quick_test_model",
    description="快速测试模型是否能前向传播。创建一个小batch进行 smoke test。",
    parameters={
        "type": "object",
        "properties": {
            "model_path": {"type": "string", "description": "模型定义文件路径，默认 code/model.py"},
            "task": {"type": "string", "description": "task1 或 task2"},
        },
        "required": [],
    },
)
def quick_test_model(model_path: str = "code/model.py", task: str = "task1") -> str:
    import importlib.util
    try:
        spec = importlib.util.spec_from_file_location("test_model", model_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        
        class DummyCfg:
            modes = 8
            width = 16
            depth = 2
        
        torch = __import__("torch")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = mod.build_model(DummyCfg(), task=task).to(device)
        x = torch.randn(2, 10, 256, device=device)
        cond = torch.randn(2, 1, device=device) if task == "task2" else None
        out, _ = model(x, cond)
        
        return json.dumps({
            "status": "ok",
            "input_shape": list(x.shape),
            "output_shape": list(out.shape),
            "device": str(device),
            "message": "Model forward pass succeeded.",
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }, ensure_ascii=False)


# =============================================================================
# 科研专用工具
# =============================================================================

@registry.register(
    name="analyze_log",
    description="分析训练日志，提取关键指标（loss曲线、收敛性、过拟合迹象等）。",
    parameters={
        "type": "object",
        "properties": {
            "log_path": {"type": "string", "description": "日志文件路径"},
            "pattern": {"type": "string", "description": "用于提取数值的正则表达式，如 'Epoch (\\d+), Loss: ([0-9.eE-]+)'"},
        },
        "required": ["log_path", "pattern"],
    },
)
def analyze_log(log_path: str, pattern: str) -> str:
    if not os.path.exists(log_path):
        return json.dumps({"error": f"Log not found: {log_path}"}, ensure_ascii=False)
    
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        
        regex = re.compile(pattern)
        records = []
        for line in lines:
            m = regex.search(line)
            if m:
                records.append(m.groups())
        
        if not records:
            return json.dumps({"matches": 0, "records": []}, ensure_ascii=False)
        
        # 尝试将最后一列转为float并计算统计信息
        values = []
        for r in records:
            try:
                values.append(float(r[-1]))
            except ValueError:
                pass
        
        stats = {}
        if values:
            stats = {
                "count": len(values),
                "first": values[0],
                "last": values[-1],
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "trend": "decreasing" if values[-1] < values[0] else "increasing",
            }
        
        return json.dumps({
            "matches": len(records),
            "stats": stats,
            "last_10_records": records[-10:],
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@registry.register(
    name="inspect_hdf5",
    description="查看HDF5数据集的结构、形状和数据类型。",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "HDF5文件路径"},
        },
        "required": ["path"],
    },
)
def inspect_hdf5(path: str) -> str:
    if not os.path.exists(path):
        return json.dumps({"error": f"File not found: {path}"}, ensure_ascii=False)
    
    try:
        import h5py
        import numpy as np
        info = {}
        with h5py.File(path, "r") as f:
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset):
                    info[name] = {
                        "shape": obj.shape,
                        "dtype": str(obj.dtype),
                        "size": obj.size,
                    }
            f.visititems(visit)
        return json.dumps({"file": path, "datasets": info}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@registry.register(
    name="summarize_code",
    description="读取代码文件并返回结构化摘要（函数列表、类列表、关键逻辑）。",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "代码文件路径"},
        },
        "required": ["path"],
    },
)
def summarize_code(path: str) -> str:
    if not os.path.exists(path):
        return json.dumps({"error": f"File not found: {path}"}, ensure_ascii=False)
    
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()
        
        lines = source.splitlines()
        classes = []
        functions = []
        imports = []
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("class "):
                classes.append({"line": i, "signature": stripped})
            elif stripped.startswith("def "):
                functions.append({"line": i, "signature": stripped})
            elif stripped.startswith("import ") or stripped.startswith("from "):
                imports.append(stripped)
        
        return json.dumps({
            "path": path,
            "total_lines": len(lines),
            "imports": imports[:20],
            "classes": classes,
            "functions": functions,
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)
