"""
科研流程编排器

管理四阶段科研闭环的流转、迭代终止条件与最终产物生成。
"""
import os
import json
import time
from datetime import datetime, timedelta

from .llm_client import LLMClient
from .memory import ResearchMemory
from .phases import LiteraturePhase, DiagnosisPhase, DesignPhase, ExperimentPhase


class RetryExhausted(Exception):
    pass


class ResearchOrchestrator:
    """科研流程总编排器"""
    
    PHASE_MAP = {
        "literature": LiteraturePhase,
        "diagnosis": DiagnosisPhase,
        "design": DesignPhase,
        "experiment": ExperimentPhase,
    }
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.memory = ResearchMemory(task=cfg.research.task)
        memory_path = f"{cfg.research.task}/research_memory_{cfg.research.task}.json"
        loaded = self.memory.load(memory_path)
        # 向后兼容：如果新的 task-specific 文件不存在，尝试加载旧的通用文件
        if loaded is None:
            loaded = self.memory.load("research_memory.json")
        if loaded is not None and loaded.task == cfg.research.task:
            self.memory = loaded
        
        log_file = f"{cfg.research.task}/{cfg.research.task}_logs.log"
        self.client = LLMClient(
            api_key=cfg.llm.api_key,
            base_url=cfg.llm.base_url,
            model=cfg.llm.model,
            temperature=cfg.llm.temperature,
            max_tokens=cfg.llm.max_tokens,
            timeout=cfg.llm.timeout,
            log_path=log_file,
        )
        
        self.start_time = datetime.now()
        self.max_duration = timedelta(hours=cfg.research.max_time_hours)
    
    def should_stop(self) -> bool:
        """检查是否应该终止"""
        # 时间限制
        elapsed = datetime.now() - self.start_time
        if elapsed > self.max_duration:
            self.memory.stop_reason = f"Time limit exceeded ({elapsed})"
            return True
        
        # 迭代次数限制
        if self.memory.iteration >= self.cfg.research.max_iterations:
            self.memory.stop_reason = f"Max iterations reached ({self.memory.iteration})"
            return True
        
        # 显式停止
        if self.memory.stop_reason:
            return True
        
        # 早停：连续多次无提升
        if len(self.memory.experiments) >= self.cfg.research.early_stop_patience:
            recent = self.memory.experiments[-self.cfg.research.early_stop_patience:]
            best_score = self.memory.best_metrics.get("val_score", -1)
            no_improve = all(
                e.metrics.get("val_score", -1) <= best_score
                for e in recent
            )
            if no_improve and best_score > 0:
                self.memory.stop_reason = "Early stop: no improvement in recent experiments"
                return True
        
        return False
    
    def run_phase(self) -> bool:
        """执行当前阶段，带指数退避重试"""
        phase_name = self.memory.current_phase
        phase_cls = self.PHASE_MAP.get(phase_name)
        if not phase_cls:
            print(f"[Error] Unknown phase: {phase_name}")
            return False
        
        max_retries = 3
        for attempt in range(max_retries):
            phase = phase_cls(self.client, self.memory, self.cfg)
            try:
                return phase.run()
            except Exception as e:
                print(f"[Error] Phase {phase_name} failed (attempt {attempt+1}/{max_retries}): {e}")
                import traceback
                traceback.print_exc()
                
                if attempt < max_retries - 1:
                    wait = min(2 ** attempt * 2, 30)
                    print(f"[Retry] Waiting {wait}s before retry...")
                    time.sleep(wait)
                else:
                    print(f"[Error] Phase {phase_name} exhausted all retries.")
        
        return False
    
    def run(self):
        """主循环"""
        print("=" * 60)
        print("PDE Neural Operator Research Agent")
        print(f"Task: {self.cfg.research.task}")
        print(f"Model: {self.cfg.llm.model}")
        print(f"Start: {self.start_time.isoformat()}")
        print("=" * 60)
        
        while not self.should_stop():
            print(f"\n{'='*60}")
            print(f"Iteration {self.memory.iteration} | Phase: {self.memory.current_phase}")
            print(f"Elapsed: {datetime.now() - self.start_time}")
            print(f"{'='*60}")
            
            success = self.run_phase()
            if not success:
                print("[Warning] Phase execution failed, will retry or pivot...")
                # 失败后尝试回到诊断阶段重新分析
                if self.memory.current_phase == "experiment":
                    self.memory.current_phase = "diagnosis"
                elif self.memory.current_phase == "literature":
                    # 文献阶段连续失败时，跳过直接进入诊断（使用已有知识）
                    self.memory.current_phase = "diagnosis"
                
                # 失败后增加等待时间，避免API限流
                wait = 5
                print(f"[Backoff] Waiting {wait}s before next phase...")
                time.sleep(wait)
            else:
                # 成功执行后短暂休息
                time.sleep(1)
            
            self.memory.save(f"{self.cfg.research.task}/research_memory_{self.cfg.research.task}.json")
        
        print(f"\n{'='*60}")
        print("Research loop terminated.")
        print(f"Reason: {self.memory.stop_reason}")
        print(f"Total iterations: {self.memory.iteration}")
        print(f"Best experiment: {self.memory.best_experiment_id}")
        print(f"Best metrics: {json.dumps(self.memory.best_metrics, ensure_ascii=False)}")
        print(f"{'='*60}")
        
        self._finalize()
    
    def _finalize(self):
        """生成最终提交产物"""
        print("\n[Finalize] 生成提交产物...")
        
        output_dir = self.cfg.research.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 收集当前任务的结果
        self._collect_current_task_results(output_dir)
        
        # 2. 复制代码
        self._copy_code(output_dir)
        
        # 3. 生成 methodology
        self._generate_methodology(output_dir)
        
        # 4. 生成 submission.json
        self._write_submission_json(output_dir)
        
        # 5. 精确打包 submission.zip
        self._pack_submission(output_dir)
        
        print("[Finalize] 完成。")
    
    def _collect_current_task_results(self, output_dir: str) -> None:
        """收集当前任务的最佳实验结果到 output_dir"""
        task = self.cfg.research.task
        best_iter = ""
        if self.memory.best_experiment_id:
            best_exp = self.memory.get_experiment(self.memory.best_experiment_id)
            if best_exp:
                best_iter = f"iter_{best_exp.id}"
        
        task_out = f"{output_dir}/{task}"
        os.makedirs(task_out, exist_ok=True)
        
        # 预测结果
        pred_src = f"output/{task}/{best_iter}/pred.hdf5" if best_iter else f"output/{task}/iter_{self.memory.iteration}/pred.hdf5"
        pred_dst = f"{task_out}/{task}_pred.hdf5"
        if os.path.exists(pred_src):
            import shutil
            shutil.copy(pred_src, pred_dst)
            print(f"  Copied prediction: {pred_dst}")
        else:
            print(f"  Warning: Prediction not found at {pred_src}")
        
        # 时间记录
        time_log = f"output/{task}/{best_iter}/time.json" if best_iter else f"output/{task}/iter_{self.memory.iteration}/time.json"
        train_time, infer_time = 0, 0
        if os.path.exists(time_log):
            with open(time_log, "r", encoding="utf-8") as f:
                tdata = json.load(f)
            train_time = tdata.get("train_time", 0)
            infer_time = tdata.get("inference_time", 0)
        
        total_time = int((datetime.now() - self.start_time).total_seconds())
        with open(f"{task_out}/{task}_time.csv", "w", encoding="utf-8") as f:
            f.write("train_time,inference_time\n")
            f.write(f"{total_time},{infer_time}\n")
        print(f"  Generated {task}_time.csv")
        
        # 日志
        log_src = f"{task}/{task}_logs.log"
        log_dst = f"{task_out}/{task}_logs.log"
        if os.path.exists(log_src):
            import shutil
            shutil.copy(log_src, log_dst)
            print(f"  Copied logs: {log_dst}")
    
    def _copy_code(self, output_dir: str) -> None:
        """复制代码到输出目录"""
        import shutil
        code_dst = f"{output_dir}/code"
        if os.path.exists("code"):
            if os.path.exists(code_dst):
                shutil.rmtree(code_dst)
            shutil.copytree("code", code_dst)
            print(f"  Copied code: {code_dst}")
        else:
            print("  Warning: code/ directory not found")
    
    def _generate_methodology(self, output_dir: str) -> None:
        """从实验记录生成 methodology.pdf"""
        md_path = f"{output_dir}/methodology.md"
        self._generate_methodology_md(md_path)
        
        # 尝试转换为 PDF
        pdf_path = f"{output_dir}/methodology.pdf"
        try:
            self._md_to_pdf(md_path, pdf_path)
            print(f"  Generated methodology.pdf")
        except Exception as e:
            print(f"  Warning: PDF generation failed ({e}), keeping methodology.md")
    
    def _generate_methodology_md(self, path: str) -> None:
        """生成 methodology markdown"""
        lines = []
        lines.append("# Methodology Report")
        lines.append("")
        lines.append("## Autonomous Research Agent for 1D Burgers Neural Operators")
        lines.append("")
        lines.append(f"**Task**: {self.cfg.research.task}")
        lines.append(f"**Best Score**: {self.memory.best_metrics.get('total', 'N/A')}")
        lines.append(f"**Total Time**: {(datetime.now() - self.start_time).total_seconds():.0f}s")
        lines.append("")
        
        # 文献综述
        lit_path = f"{self.memory.task}/{self.memory.task}_literature_summary.md"
        if os.path.exists(lit_path):
            lines.append("## 1. Literature Summary")
            with open(lit_path, "r", encoding="utf-8") as f:
                content = f.read()
            lines.append(content[:4000])
            lines.append("")
        
        # 诊断报告
        diag_path = f"{self.memory.task}/{self.memory.task}_diagnosis_report.md"
        if os.path.exists(diag_path):
            lines.append("## 2. Diagnosis and Hypotheses")
            with open(diag_path, "r", encoding="utf-8") as f:
                content = f.read()
            lines.append(content[:4000])
            lines.append("")
        
        # 实验轨迹
        lines.append("## 3. Experimental Trajectory")
        for exp in self.memory.experiments:
            lines.append(f"### Experiment {exp.id}")
            lines.append(f"- Status: {exp.status}")
            lines.append(f"- Hypothesis: {exp.hypothesis}")
            if exp.metrics:
                lines.append(f"- Metrics: {json.dumps(exp.metrics, ensure_ascii=False)}")
            if exp.conclusion:
                conclusion = exp.conclusion[:800].replace('\n', ' ')
                lines.append(f"- Conclusion: {conclusion}")
            lines.append("")
        
        # 设计笔记
        design_path = f"{self.memory.task}/{self.memory.task}_design_notes.md"
        if os.path.exists(design_path):
            lines.append("## 4. Design and Architecture")
            with open(design_path, "r", encoding="utf-8") as f:
                content = f.read()
            lines.append(content[:4000])
            lines.append("")
        
        # 代码自主生成声明
        lines.append("## 5. Code Generation Statement")
        lines.append("")
        lines.append("All code in the `code/` directory was autonomously generated by the LLM Agent during the research session.")
        lines.append("No human-written or pre-placed baseline code was used.")
        lines.append("The generation process is documented in the task logs.")
        lines.append("")
        
        # 提交合规声明
        lines.append("## 6. Submission Compliance")
        lines.append("")
        lines.append("This submission includes:")
        for task in ["task1", "task2"]:
            task_out = f"{self.cfg.research.output_dir}/{task}"
            has_pred = os.path.exists(f"{task_out}/{task}_pred.hdf5")
            has_time = os.path.exists(f"{task_out}/{task}_time.csv")
            has_logs = os.path.exists(f"{task_out}/{task}_logs.log")
            status = "✓ complete" if (has_pred and has_time and has_logs) else "✗ missing"
            lines.append(f"- {task}: {status}")
        lines.append("")
        
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    
    def _md_to_pdf(self, md_path: str, pdf_path: str) -> None:
        """将markdown转换为PDF，尝试多种方法"""
        # 方法1: 使用 fpdf2 (如果已安装)
        try:
            self._md_to_pdf_fpdf(md_path, pdf_path)
            return
        except Exception:
            pass
        
        # 方法2: 使用 weasyprint (如果已安装)
        try:
            import markdown
            from weasyprint import HTML
            with open(md_path, "r", encoding="utf-8") as f:
                md_content = f.read()
            html_content = f"""
            <html><head><meta charset="utf-8"><style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            h1 {{ color: #333; border-bottom: 2px solid #333; padding-bottom: 10px; }}
            h2 {{ color: #555; border-bottom: 1px solid #ccc; padding-bottom: 5px; margin-top: 30px; }}
            h3 {{ color: #666; }}
            code {{ background: #f4f4f4; padding: 2px 5px; border-radius: 3px; }}
            </style></head><body>{markdown.markdown(md_content)}</body></html>
            """
            HTML(string=html_content).write_pdf(pdf_path)
            return
        except Exception:
            pass
        
        # 方法3: 使用 pypandoc (如果已安装)
        try:
            import pypandoc
            pypandoc.convert_file(md_path, 'pdf', outputfile=pdf_path)
            return
        except Exception:
            pass
        
        raise RuntimeError("No PDF generation library available (tried fpdf2, weasyprint, pypandoc)")
    
    def _md_to_pdf_fpdf(self, md_path: str, pdf_path: str) -> None:
        """使用 fpdf2 生成PDF"""
        from fpdf import FPDF
        
        class SimplePDF(FPDF):
            def header(self):
                self.set_font("Arial", "", 8)
                self.set_text_color(100, 100, 100)
                self.cell(0, 10, "PDE Neural Operator Research Agent - Methodology", align="R")
                self.ln(5)
            
            def footer(self):
                self.set_y(-15)
                self.set_font("Arial", "", 8)
                self.cell(0, 10, f"Page {self.page_no()}", align="C")
        
        pdf = SimplePDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Methodology Report", ln=True, align="C")
        pdf.ln(5)
        
        with open(md_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip()
                if not line:
                    pdf.ln(2)
                    continue
                
                if line.startswith("# "):
                    pdf.set_font("Arial", "B", 16)
                    pdf.set_text_color(33, 37, 41)
                    pdf.cell(0, 10, line[2:], ln=True)
                    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
                    pdf.ln(3)
                elif line.startswith("## "):
                    pdf.set_font("Arial", "B", 13)
                    pdf.set_text_color(55, 55, 55)
                    pdf.cell(0, 8, line[3:], ln=True)
                    pdf.ln(1)
                elif line.startswith("### "):
                    pdf.set_font("Arial", "B", 11)
                    pdf.set_text_color(77, 77, 77)
                    pdf.cell(0, 7, line[4:], ln=True)
                    pdf.ln(1)
                elif line.startswith("- "):
                    pdf.set_font("Arial", "", 10)
                    pdf.set_text_color(33, 37, 41)
                    pdf.cell(5, 5, chr(149), align="C")
                    pdf.multi_cell(0, 5, line[2:])
                else:
                    pdf.set_font("Arial", "", 10)
                    pdf.set_text_color(33, 37, 41)
                    pdf.multi_cell(0, 5, line)
        
        pdf.output(pdf_path)
    
    def _write_submission_json(self, output_dir: str) -> None:
        """生成 submission.json"""
        sub = {
            "submission_id": "PDE_Research_Agent",
            "problem_id": "PDE_Burgers",
            "code_path": "code",
            "methodology": "methodology.pdf",
            "submission": "submission.zip",
        }
        with open(f"{output_dir}/submission.json", "w", encoding="utf-8") as f:
            json.dump(sub, f, indent=2, ensure_ascii=False)
        print(f"  Generated submission.json")
    
    def _pack_submission(self, output_dir: str) -> None:
        """精确打包 submission.zip，只包含必要的提交文件"""
        import zipfile
        
        zip_path = f"{output_dir}/submission.zip"
        
        # 定义必须包含的核心文件
        files_to_pack = ["submission.json"]
        
        # 检查每个任务的完整性（必须同时有三个文件）
        complete_tasks = []
        for task in ["task1", "task2"]:
            task_out = f"{output_dir}/{task}"
            has_pred = os.path.exists(f"{task_out}/{task}_pred.hdf5")
            has_time = os.path.exists(f"{task_out}/{task}_time.csv")
            has_logs = os.path.exists(f"{task_out}/{task}_logs.log")
            if has_pred and has_time and has_logs:
                complete_tasks.append(task)
                files_to_pack.extend([
                    f"{task}/{task}_pred.hdf5",
                    f"{task}/{task}_time.csv",
                    f"{task}/{task}_logs.log",
                ])
                print(f"  Task {task}: complete (pred + time + logs)")
            elif has_pred or has_time or has_logs:
                print(f"  Warning: {task} has incomplete files (pred={has_pred}, time={has_time}, logs={has_logs}), skipping")
        
        if not complete_tasks:
            print("  ERROR: No complete task submission found! Cannot pack submission.zip.")
            return
        
        # methodology: 优先使用 pdf，否则使用 md
        if os.path.exists(f"{output_dir}/methodology.pdf"):
            files_to_pack.append("methodology.pdf")
        elif os.path.exists(f"{output_dir}/methodology.md"):
            files_to_pack.append("methodology.md")
            print("  Warning: methodology.pdf not found, using methodology.md instead")
        else:
            print("  Warning: methodology not found")
        
        # code/ 目录（共用）
        code_dir = f"{output_dir}/code"
        if os.path.exists(code_dir):
            for root, dirs, files in os.walk(code_dir):
                for file in files:
                    if file.endswith(".py"):
                        fp = os.path.join(root, file)
                        arcname = os.path.relpath(fp, output_dir).replace(os.sep, "/")
                        files_to_pack.append(arcname)
            print(f"  Code directory: {len([f for f in files_to_pack if f.startswith('code/')])} Python files")
        else:
            print("  Warning: code/ directory not found")
        
        # 执行打包，所有文件放在 submission/ 前缀下
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for filepath in files_to_pack:
                full_path = os.path.join(output_dir, filepath)
                if os.path.exists(full_path):
                    # task 子目录中的文件在 zip 中提升到 submission/ 根下
                    if filepath.startswith("task1/") or filepath.startswith("task2/"):
                        arcname = filepath.split("/", 1)[1]
                    else:
                        arcname = filepath
                    zf.write(full_path, f"submission/{arcname}")
                else:
                    print(f"  Warning: Expected file missing: {filepath}")
        
        print(f"  Packed: {zip_path}")
        print(f"  Complete tasks: {', '.join(complete_tasks)}")
    
    def close(self):
        self.client.close()
