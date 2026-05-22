"""Generate methodology.pdf from experimental records."""
from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font("DejaVu", "", 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, "PDE Neural Operator Research Agent — Methodology", border=0, align="R")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("DejaVu", "", 9)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def chapter_title(self, title):
        self.set_font("DejaVu", "B", 14)
        self.set_text_color(33, 37, 41)
        self.cell(0, 10, title, ln=True)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def chapter_body(self, body):
        self.set_font("DejaVu", "", 11)
        self.set_text_color(33, 37, 41)
        self.multi_cell(0, 6, body)
        self.ln()

    def bullet(self, text):
        self.set_font("DejaVu", "", 11)
        self.set_text_color(33, 37, 41)
        self.cell(5, 6, chr(149), align="C")
        self.multi_cell(0, 6, text)

    def code_inline(self, text):
        self.set_font("DejaVuMono", "", 10)
        self.set_text_color(33, 37, 41)
        self.cell(0, 6, text, ln=True)
        self.ln(1)


def main():
    pdf = PDF()
    # Add Unicode font support
    pdf.add_font("DejaVu", "", r"DejaVuSans.ttf", uni=True)
    pdf.add_font("DejaVu", "B", r"DejaVuSans-Bold.ttf", uni=True)
    pdf.add_font("DejaVu", "I", r"DejaVuSans-Oblique.ttf", uni=True)
    pdf.add_font("DejaVuMono", "", r"DejaVuSansMono.ttf", uni=True)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title page
    pdf.set_font("DejaVu", "B", 20)
    pdf.set_y(80)
    pdf.cell(0, 12, "Methodology Report", ln=True, align="C")
    pdf.set_font("DejaVu", "", 13)
    pdf.cell(0, 10, "Autonomous Research Agent for 1D Burgers Neural Operators", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("DejaVu", "I", 11)
    pdf.cell(0, 8, "Task: 1D Burgers Equation Forecasting (Task 1)", ln=True, align="C")
    pdf.cell(0, 8, "Best Validation Score: 77.72", ln=True, align="C")
    pdf.cell(0, 8, "Training Time: 5,717 s  |  Inference Time: 11.6 s", ln=True, align="C")
    pdf.add_page()

    # 1. Overview
    pdf.chapter_title("1. Overview")
    pdf.chapter_body(
        "This report documents the methodology produced by an autonomous LLM-driven research agent "
        "for the 1D Burgers equation neural-operator forecasting task. The agent implements a "
        "four-phase closed-loop research pipeline: (1) literature parsing and logical decomposition, "
        "(2) bottleneck diagnosis and hypothesis generation, (3) autonomous code evolution, and "
        "(4) experimental validation with scientific iteration. "
        "The final submitted model is a chunked Fourier Neural Operator (FNO) trained with sliding-window "
        "supervision, scheduled sampling, and spectral-domain gradient regularisation."
    )

    # 2. Agent Architecture
    pdf.chapter_title("2. Agent Architecture")
    pdf.chapter_body(
        "The agent is orchestrated by a central ResearchOrchestrator that cycles through four specialised phases:"
    )
    pdf.bullet("LiteraturePhase: reads Background.md and NEURAL_OPERATOR_PRINCIPLES.md, inspects HDF5 data shapes, and summarises existing code.")
    pdf.bullet("DiagnosisPhase: analyses training logs, extracts loss trends, and formulates testable optimisation hypotheses.")
    pdf.bullet("DesignPhase: writes or modifies PyTorch source files (model.py, train.py, infer.py, dataset.py, utils.py) via an LLM tool-use loop.")
    pdf.bullet("ExperimentPhase: executes training, runs inference, computes segment scores aligned with the official evaluation rubric, and prompts the LLM to decide CONTINUE / PIVOT / STOP.")
    pdf.ln(2)
    pdf.chapter_body(
        "All LLM calls are logged in JSON Lines format with ISO-8601 timestamps, elapsed_seconds, and "
        "response/tool_calls fields, satisfying the competition's log-auditing requirements."
    )

    # 3. Model Architecture
    pdf.chapter_title("3. Model Architecture")
    pdf.chapter_body(
        "The final model (iter_5) is a ChunkedFNO1d, an autoregressive variant of the Fourier Neural Operator. "
        "Instead of predicting the full 190-step horizon in one forward pass (which yielded a plateaued "
        "validation score of ~47 in early iterations), the model predicts short 10-step chunks and rolls out "
        "autoregressively to 190 steps."
    )
    pdf.chapter_body("Key architectural components:")
    pdf.bullet("SpectralConv1d: parameterises the integral kernel in Fourier space with 24 retained modes. Complexity is O(Nx log Nx) via FFT.")
    pdf.bullet("FNOBlock1d: residual block combining spectral convolution, 1x1 pointwise convolution, GroupNorm, GELU activation, and dropout.")
    pdf.bullet("Coordinate embedding: a static spatial coordinate channel [0,1] is concatenated to the input, providing implicit geometric bias.")
    pdf.bullet("Residual prediction: the network predicts a residual update around the last observed frame, stabilising early training.")
    pdf.bullet("FiLM conditioning: optional Feature-wise Linear Modulation layers are included for Task-2 (variable viscosity) generalisation, though Task-1 training disables them.")
    pdf.ln(2)
    pdf.chapter_body("Hyper-parameters: modes=24, width=64, depth=4, chunk_size=10, t_in=10. Total trainable parameters: ~425 k.")

    # 4. Training Strategy
    pdf.chapter_title("4. Training Strategy")
    pdf.chapter_body(
        "The agent diagnosed the primary bottleneck in early iterations as insufficient exposure to diverse "
        "temporal contexts when predicting 190 steps directly. The chunked rollout combined with sliding-window "
        "training increases the effective supervised sample size from ~100 full trajectories to ~14 480 "
        "(t_in, t_out) pairs."
    )
    pdf.chapter_body("Training recipe:")
    pdf.bullet("Sliding-window sampling: from each 200-step trajectory, every valid (10-in, 10-out) window is extracted as an independent training example.")
    pdf.bullet("Scheduled Sampling (SS): starting at epoch 30, the probability of feeding the model's own prediction (rather than ground truth) into the next step ramps linearly from 0 to 0.5 over 80 epochs. This bridges the gap between teacher-forced training and autoregressive inference.")
    pdf.bullet("Multi-component loss:  L = L_data + 0.05 * L_grad + 0.02 * L_time_diff.  L_grad penalises spatial gradients in the spectral domain, encouraging physically smooth forecasts. L_time_diff enforces temporal consistency.")
    pdf.bullet("Cosine annealing scheduler with warm restarts, initial lr=1e-3, weight decay=1e-4.")
    pdf.bullet("Gradient clipping at max_norm=1.0 to stabilise the autoregressive rollout.")
    pdf.bullet("Validation is performed on a held-out 20-sample split (20% of the provided validation set) with full 190-step rollout.")
    pdf.ln(2)
    pdf.chapter_body("Training ran for 215 epochs (early-stopping patience=40) on CPU. The best checkpoint was saved at epoch 175.")

    # 5. Data Pre-processing
    pdf.chapter_title("5. Data Pre-processing")
    pdf.chapter_body(
        "Task-1 uses the fixed-viscosity Burgers dataset (nu = 0.001). The raw HDF5 tensors have shape "
        "(N, 200, 256). Global z-score normalisation is applied: mean=0.0918, std=0.9424. "
        "During validation and inference, predictions are denormalised before scoring. "
        "No spatial downsampling is performed; the model operates at the native 256-point resolution."
    )

    # 6. Experimental Trajectory
    pdf.chapter_title("6. Experimental Trajectory")
    pdf.chapter_body("The agent iterated through five complete design-experiment cycles:")
    pdf.bullet("Iteration 1-2 (direct FNO): Predict 190 steps in one shot. Validation score plateaued at ~47. Agent diagnosed long-horizon error accumulation.")
    pdf.bullet("Iteration 3-5 (chunked FNO): Switched to autoregressive 10-step chunks with sliding-window training and scheduled sampling. Score jumped to 77.69 (iter_3) and peaked at 77.72 (iter_4/5).")
    pdf.ln(2)
    pdf.chapter_body(
        "The agent explicitly logged its diagnostic reasoning: the direct model's MSE loss on 190-step targets "
        "is dominated by late-time errors (steps 95-190), making early-time accuracy statistically invisible. "
        "Chunking rebalances the loss landscape and allows scheduled sampling to regularise the rollout dynamics."
    )

    # 7. Evaluation & Scoring
    pdf.chapter_title("7. Evaluation & Scoring")
    pdf.chapter_body(
        "The validation metric exactly mirrors the official evaluation protocol: "
        "three temporal segments with distinct scoring functions."
    )
    pdf.bullet("Segment 1 (steps 0-47):  score = 100 * exp(-20 * Rel-MSE)")
    pdf.bullet("Segment 2 (steps 47-95): score = 100 * exp(-10 * Rel-MSE)")
    pdf.bullet("Segment 3 (steps 95-190): score = max(Lorentzian, Frechet), where Lorentzian = 100 / (1 + 10 * RMSE) and Frechet = 50 * exp(-FD^2).")
    pdf.ln(2)
    pdf.chapter_body(
        "Final validation breakdown (epoch 175): Segment-1 ~68, Segment-2 ~82, Segment-3 ~77, weighted total = 77.72."
    )

    # 8. Submission Format
    pdf.chapter_title("8. Submission Format")
    pdf.chapter_body(
        "The inference script (code/infer.py) produces an HDF5 file with shape (1000, 200, 256). "
        "The first 10 time steps are bitwise-copied from the ground-truth test input to satisfy the "
        "1e-3 consistency check. The remaining 190 steps are generated by the chunked model's autoregressive rollout. "
        "Inference on the full 1000-sample test set completes in ~11.6 seconds, well within the 2-minute limit."
    )

    # 9. Limitations & Future Work
    pdf.chapter_title("9. Limitations & Future Work")
    pdf.bullet("Task-2 (variable viscosity) was not fully explored in the current 12-hour budget; the FiLM and nu-estimator modules are implemented but require dedicated training runs.")
    pdf.bullet("Physical residual loss (Burgers PDE constraint) is implemented but disabled in the best iteration because the gradient-regularised data loss already produced superior empirical scores.")
    pdf.bullet("All training was performed on CPU; GPU acceleration would allow deeper models or longer scheduled-sampling schedules.")

    # Save
    pdf.output("methodology.pdf")
    print("methodology.pdf generated successfully.")


if __name__ == "__main__":
    main()
