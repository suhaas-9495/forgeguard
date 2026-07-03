#!/usr/bin/env python3
"""
ForgeGuard — Phase 2 Calibration Runner
Run: python scripts/run_calibration.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

def main():
    parser = argparse.ArgumentParser(description="ForgeGuard Phase 2 Calibration")
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--base_adapter", default="models/adapters/phase1")
    args = parser.parse_args()
    
    print("🛡️  ForgeGuard — Starting Phase 2 Calibration Training")
    
    from src.calibration.calibration_trainer import train_phase2
    adapter_path, metrics = train_phase2(
        config_path=args.config,
        base_adapter_path=args.base_adapter,
    )
    
    print(f"\n✅ Phase 2 complete!")
    print(f"   Adapter: {adapter_path}")
    print(f"   ECE improvement: {metrics['ece_improvement']:.4f}")
    print("   Next: python scripts/run_evaluation.py")

if __name__ == "__main__":
    main()
