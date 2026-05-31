"""Local-only browser UI for mesh generation and inspection.

Run with:
    sctrap-serve                       # console-script entrypoint, port 8765
    python3 -m sctrap.web              # equivalent

Or programmatically:
    from sctrap.web.app import app, run
    run(host="127.0.0.1", port=8765)
"""
