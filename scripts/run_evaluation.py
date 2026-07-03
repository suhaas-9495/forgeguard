#!/usr/bin/env python3
"""
ForgeGuard — Phase 3 Evaluation Runner
Run: python scripts/run_evaluation.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

def main():
    parser = argparse.ArgumentParser(description="ForgeGuard Phase 3 Evaluation")
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--adapter_path", default=None,
                       help="Path to LoRA adapter (auto-detects latest if not specified)")
    args = parser.parse_args()
    
    print("🛡️  ForgeGuard — Starting Phase 3 Evaluation")
    
    from src.training.model_loader import load_config
    cfg = load_config(args.config)
    
    # Auto-detect best available adapter
    adapter_path = args.adapter_path
    if not adapter_path:
        candidates = [
            "models/adapters/self_improved/final_adapter",
            "models/adapters/phase2",
            "models/adapters/phase1",
        ]
        for path in candidates:
            if os.path.exists(path):
                adapter_path = path
                break
    
    if not adapter_path or not os.path.exists(adapter_path):
        print("❌ No trained adapter found. Run Phase 1 training first.")
        print("   python scripts/run_training.py")
        return
    
    print(f"   Using adapter: {adapter_path}")
    
    from src.inference.inference_engine import load_inference_engine
    engine = load_inference_engine(adapter_path, args.config)
    
    from src.evaluation.evaluator import run_evaluation
    metrics = run_evaluation(engine, save_dir="outputs")
    
    print(f"\n✅ Evaluation complete!")
    print(f"   Accuracy: {metrics['accuracy']:.1%}")
    print(f"   Hallucination Rate: {metrics['hallucination_rate']:.1%}")
    print(f"   ECE: {metrics['ece']:.4f}")
    print("   Next: python scripts/run_self_improvement.py")

if __name__ == "__main__":
    main()
