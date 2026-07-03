#!/usr/bin/env python3
"""ForgeGuard API Server Runner - Run: python scripts/run_api.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    print(f"ForgeGuard API: http://localhost:{args.port}/docs")
    from src.api.server import start_server
    start_server(host=args.host, port=args.port)

if __name__ == "__main__":
    main()
