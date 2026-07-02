"""Run the PortraitAI API server.

Usage:
    cd headshot_pipeline
    python -m server.run
    # or: uvicorn server.main:app --reload --port 8000
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "server.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
