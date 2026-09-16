"""
Entry point to run the FastAPI server persistently.
"""

import sys
import os
import uvicorn

# Ensure project root is in PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    uvicorn.run(
        "server.app:app",
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )
