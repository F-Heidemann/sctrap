"""Start the sctrap mesh studio (browser UI). Equivalent to `sctrap-serve`.

Usage:
    python3 scripts/serve.py
    python3 scripts/serve.py --port 9000 --host 0.0.0.0
"""
import argparse
from sctrap.web.app import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default=8765, type=int)
    ap.add_argument("--reload", action="store_true",
                     help="auto-reload on code change (dev only)")
    args = ap.parse_args()
    run(host=args.host, port=args.port, reload=args.reload)
