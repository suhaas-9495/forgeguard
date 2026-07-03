#!/usr/bin/env python3
"""
ForgeGuard — Phase 1 Training Runner
Run: python scripts/run_training.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

def main():
    parser = argparse.ArgumentParser(description="ForgeGuard Phase 1 Training")
    parser.add_argument("--config", default="configs/training_config.yaml")
    args = parser.parse_args()
    
    print(" ForgeGuard — Starting Phase 1 Training")
    
    from src.training.trainer import train_phase1
    adapter_path = train_phase1(config_path=args.config)
    
    print(f"\n✅ Phase 1 complete! Adapter saved to: {adapter_path}")
    print("   Next: python scripts/run_calibration.py")

if __name__ == "__main__":
    main()
