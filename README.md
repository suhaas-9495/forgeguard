# 🛡️ ForgeGuard — Self-Improving, Hallucination-Calibrated LLM System

> A production-grade LoRA/QLoRA fine-tuned LLM that detects its own hallucinations, outputs calibrated confidence scores, and **retrains itself** to improve accuracy over time.

---

## 🏗️ Architecture Overview

```
ForgeGuard/
├── configs/                    # YAML configs for training, eval, hardware
├── data/
│   ├── raw/                    # Source datasets
│   ├── processed/              # Tokenized, formatted data
│   └── synthetic/              # Self-generated corrective data
├── models/
│   ├── adapters/               # Saved LoRA adapter weights
│   └── checkpoints/            # Mid-training checkpoints
├── src/
│   ├── training/               # Phase 1: Fine-tuning pipeline
│   ├── calibration/            # Phase 2: ECE tracking, uncertainty
│   ├── evaluation/             # Phase 3: Hallucination harness
│   ├── self_improvement/       # Phase 4: Self-correction loop
│   └── inference/              # Clean inference engine
├── scripts/                    # Entry-point runners
├── logs/                       # Training logs
└── outputs/
    ├── plots/                  # Calibration curves
    ├── metrics/                # JSON metric reports
    └── reports/                # Before/after comparisons
```

---

## 🚀 Quick Start (Run on Your Laptop)

### 1. Install Dependencies
```bash
cd ForgeGuard
pip install -r requirements.txt
```

### 2. Verify GPU
```bash
python scripts/check_hardware.py
```

### 3. Phase 1 — Fine-Tune Base Model
```bash
python scripts/run_training.py --config configs/training_config.yaml
```

### 4. Phase 2 — Calibration Training
```bash
python scripts/run_calibration.py --config configs/training_config.yaml
```

### 5. Phase 3 — Run Evaluation Harness
```bash
python scripts/run_evaluation.py --adapter_path models/adapters/phase1
```

### 6. Phase 4 — Self-Improvement Loop
```bash
python scripts/run_self_improvement.py --iterations 3
```

### 7. Launch Dashboard
```bash
python scripts/run_dashboard.py
```
Then open: **http://localhost:7860**

---

## 🧠 How Hallucination Minimization Works

### The Core Loop:
```
Input → LLM → Answer + Confidence Score
                    ↓
            Confidence < threshold?
                    ↓ YES
            → Output: "I'm not confident enough to answer"
                    ↓ NO
            → Compare answer vs ground truth
                    ↓ WRONG + High Confidence = HALLUCINATION
                    ↓
            → Generate corrective training sample
                    ↓
            → Retrain LoRA adapter
                    ↓
            → Metrics improve (ECE ↓, Accuracy ↑, Hallucination Rate ↓)
```

### Key Techniques:
1. **Structured Output Format** — Forces model to separate reasoning from confidence
2. **ECE (Expected Calibration Error)** — Measures if confidence matches actual accuracy
3. **Refusal Samples** — Model learns to say "I don't know" when uncertain
4. **Synthetic Correction Data** — High-confidence wrong answers become training negatives
5. **LoRA Iterative Retraining** — Lightweight adapter updates without full retraining

---

## 📊 Metrics Tracked
| Metric | Description |
|--------|-------------|
| Accuracy | % correct answers |
| ECE | Expected Calibration Error (lower = better) |
| Hallucination Rate | % confident wrong answers |
| Refusal Rate | % correct refusals on unknown questions |
| Overconfidence Rate | % answers with confidence > 0.8 that are wrong |

---

## 💻 Hardware Requirements
- GPU: NVIDIA RTX 4050 6GB (or any 6GB+ CUDA GPU)
- RAM: 16GB
- Python 3.10+
- CUDA 11.8+

---

## 🎯 Internship Demo Points
1. Show calibration curve BEFORE training (poorly calibrated)
2. Show calibration curve AFTER Phase 2 (well-calibrated)
3. Run live inference — model refuses uncertain questions
4. Show self-improvement loop reducing hallucination rate
5. Compare before/after metrics in dashboard
