#!/usr/bin/env python3
"""
ForgeGuard — Full Pipeline Runner
Runs all 4 phases sequentially.
Run: python scripts/run_full_pipeline.py

WARNING: Full training takes 2-4 hours on RTX 4050.
For demo mode (no GPU), use: python scripts/run_dashboard.py
"""

import sys
import os
import time
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s" if h > 0 else f"{m}m {s}s"

def main():
    parser = argparse.ArgumentParser(description="ForgeGuard Full Pipeline")
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--skip_to_dashboard", action="store_true",
                       help="Skip training and launch dashboard directly")
    parser.add_argument("--self_improvement_iters", type=int, default=3)
    args = parser.parse_args()
    
    print("\n" + "="*65)
    print("  🛡️  ForgeGuard — Full Pipeline")
    print("  Self-Improving, Hallucination-Calibrated LLM System")
    print("="*65)
    
    if args.skip_to_dashboard:
        print("\n⏩ Skipping training — launching dashboard directly...")
        from src.inference.dashboard import build_dashboard
        demo = build_dashboard()
        demo.launch(server_name="0.0.0.0", server_port=7860)
        return
    
    pipeline_start = time.time()
    phase_times = {}
    
    # ── Phase 1 ───────────────────────────────────────────────────────────────
    print("\n" + "─"*65)
    print("  PHASE 1: Base Fine-Tuning")
    print("─"*65)
    t0 = time.time()
    
    from src.training.trainer import train_phase1
    phase1_adapter = train_phase1(config_path=args.config)
    
    phase_times["phase1"] = time.time() - t0
    print(f"\n⏱️  Phase 1 time: {format_time(phase_times['phase1'])}")
    
    # ── Phase 2 ───────────────────────────────────────────────────────────────
    print("\n" + "─"*65)
    print("  PHASE 2: Calibration Training")
    print("─"*65)
    t0 = time.time()
    
    from src.calibration.calibration_trainer import train_phase2
    phase2_adapter, cal_metrics = train_phase2(
        config_path=args.config,
        base_adapter_path=phase1_adapter,
    )
    
    phase_times["phase2"] = time.time() - t0
    print(f"\n⏱️  Phase 2 time: {format_time(phase_times['phase2'])}")
    
    # ── Phase 3 ───────────────────────────────────────────────────────────────
    print("\n" + "─"*65)
    print("  PHASE 3: Evaluation Harness")
    print("─"*65)
    t0 = time.time()
    
    from src.inference.inference_engine import load_inference_engine
    engine = load_inference_engine(phase2_adapter, args.config)
    
    from src.evaluation.evaluator import run_evaluation
    eval_metrics = run_evaluation(engine, save_dir="outputs")
    
    phase_times["phase3"] = time.time() - t0
    print(f"\n⏱️  Phase 3 time: {format_time(phase_times['phase3'])}")
    
    # ── Phase 4 ───────────────────────────────────────────────────────────────
    print("\n" + "─"*65)
    print(f"  PHASE 4: Self-Improvement Loop ({args.self_improvement_iters} iterations)")
    print("─"*65)
    t0 = time.time()
    
    from src.self_improvement.self_improvement_loop import run_self_improvement_loop
    final_adapter, si_metrics = run_self_improvement_loop(
        config_path=args.config,
        initial_adapter_path=phase2_adapter,
        max_iterations=args.self_improvement_iters,
    )
    
    phase_times["phase4"] = time.time() - t0
    print(f"\n⏱️  Phase 4 time: {format_time(phase_times['phase4'])}")
    
    # ── Summary ───────────────────────────────────────────────────────────────
    total_time = time.time() - pipeline_start
    
    print("\n" + "="*65)
    print("  🎉 PIPELINE COMPLETE!")
    print("="*65)
    print(f"\n  Total time: {format_time(total_time)}")
    print(f"\n  Phase times:")
    for phase, t in phase_times.items():
        print(f"    {phase}: {format_time(t)}")
    
    print(f"\n  📊 Final Metrics:")
    print(f"    Accuracy:           {eval_metrics['accuracy']:.1%}")
    print(f"    Hallucination Rate: {eval_metrics['hallucination_rate']:.1%}")
    print(f"    ECE:                {eval_metrics['ece']:.4f}")
    print(f"    Valid Refusals:     {eval_metrics['valid_refusal_rate']:.1%}")
    
    print(f"\n  📁 Artifacts saved:")
    print(f"    Final adapter:  {final_adapter}")
    print(f"    Metrics:        outputs/metrics/")
    print(f"    Plots:          outputs/plots/")
    
    print(f"\n  🚀 Launch dashboard:")
    print(f"    python scripts/run_dashboard.py")
    print(f"    → http://localhost:7860")
    print("\n" + "="*65 + "\n")

if __name__ == "__main__":
    main()
