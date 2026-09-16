"""
Entry point to run the FastAPI server persistently.
"""

import sys
import os
import uvicorn

# Ensure project root is in PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0" if "SPACE_ID" in os.environ else "127.0.0.1")
    port = int(os.environ.get("PORT", 7860 if "SPACE_ID" in os.environ else 8000))
    uvicorn.run(
        "server.app:app",
        host=host,
        port=port,
        log_level="info"
    )
