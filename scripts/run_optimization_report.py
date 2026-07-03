#!/usr/bin/env python3
"""ForgeGuard Optimization Report Generator"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.optimization.profiler import save_optimization_report, plot_lora_comparison
save_optimization_report()
plot_lora_comparison()
print("Done! Check outputs/metrics/optimization_report.json")
