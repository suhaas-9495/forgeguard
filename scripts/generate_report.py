#!/usr/bin/env python3
"""
ForgeGuard — Generate Full System Report
Run: python scripts/generate_report.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.report_generator import generate_comparison_report

if __name__ == "__main__":
    print("📊 Generating ForgeGuard system report...")
    path = generate_comparison_report()
    print(f"✅ Report: {path}")
