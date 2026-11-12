#!/usr/bin/env python3
"""Start the web interface on http://127.0.0.1:8148 (local only). Requires requirements-web.txt."""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("src.web.app:app", host="127.0.0.1", port=8148)
