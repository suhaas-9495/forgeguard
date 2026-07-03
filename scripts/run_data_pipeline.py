#!/usr/bin/env python3
"""ForgeGuard Data Pipeline Runner"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.training.dataset_builder import (HIGH_CONFIDENCE_SAMPLES, LOW_CONFIDENCE_SAMPLES,
    REFUSAL_SAMPLES, MEDIUM_CONFIDENCE_SAMPLES, format_sample)
from src.data_engineering.data_pipeline import run_data_pipeline
samples = []
for inst, ans, conf in (HIGH_CONFIDENCE_SAMPLES + LOW_CONFIDENCE_SAMPLES +
                         REFUSAL_SAMPLES + MEDIUM_CONFIDENCE_SAMPLES):
    s = format_sample(inst, ans, conf)
    samples.append(s)
run_data_pipeline(samples)
