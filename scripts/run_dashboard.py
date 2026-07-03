#!/usr/bin/env python3
"""
ForgeGuard — Dashboard Runner
Run: python scripts/run_dashboard.py
Then open: http://localhost:7860
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("🛡️  ForgeGuard Dashboard")
    print("   Starting at: http://localhost:7860")
    print("   Press Ctrl+C to stop\n")
    
    from src.inference.dashboard import build_dashboard
    demo = build_dashboard()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )

if __name__ == "__main__":
    main()
