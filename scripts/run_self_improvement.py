#!/usr/bin/env python3
"""
ForgeGuard — Phase 4 Self-Improvement Runner
Run: python scripts/run_self_improvement.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

def main():
    parser = argparse.ArgumentParser(description="ForgeGuard Phase 4 Self-Improvement")
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--adapter_path", default=None)
    args = parser.parse_args()
    
    print("🛡️  ForgeGuard — Starting Phase 4 Self-Improvement Loop")
    print(f"   Max iterations: {args.iterations}")
    
    from src.self_improvement.self_improvement_loop import run_self_improvement_loop
    
    final_path, metrics = run_self_improvement_loop(
        config_path=args.config,
        initial_adapter_path=args.adapter_path,
        max_iterations=args.iterations,
    )
    
    print(f"\n✅ Self-improvement complete!")
    print(f"   Final adapter: {final_path}")
    print("   Next: python scripts/run_dashboard.py")

if __name__ == "__main__":
    main()
