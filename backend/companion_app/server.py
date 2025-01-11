# server.py
import os
import sys
from pathlib import Path

# Add the parent directory to the Python path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from typing import Optional
import socket
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import uvicorn
import litellm

from companion_app.api_endpoints import app as api_router
from companion_app.db_utils import init_db

# Load environment variables at startup
load_dotenv()

# Enable LiteLLM debugging
litellm.set_verbose = True

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize or migrate DB
    print("Initializing database...")
    init_db()
    print("Database initialized")
    yield
    print("Shutting down...")

app = FastAPI(
    title="Companion App",
    description="A generalized companion application for knowledge graph generation and search",
    version="1.0.0",
    lifespan=lifespan
)

# Mount the API router
app.include_router(api_router, prefix="/api")

# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

class APIKeyUpdate(BaseModel):
    api_key: str = Field(..., description="OpenAI API key")

    class Config:
        schema_extra = {
            "example": {
                "api_key": "sk-..."
            }
        }

@app.post("/api/update-api-key")
async def update_api_key(key_data: APIKeyUpdate):
    """Update the OpenAI API key"""
    try:
        os.environ["OPENAI_API_KEY"] = key_data.api_key
        litellm.api_key = key_data.api_key
        return {"status": "success", "message": "API key updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/check-api-key")
async def check_api_key():
    """Check if OpenAI API key is configured"""
    return {
        "configured": bool(os.getenv("OPENAI_API_KEY")),
        "status": "active" if os.getenv("OPENAI_API_KEY") else "not_configured"
    }

@app.get("/")
async def serve_ui():
    index_path = static_dir / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(index_path)

def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def run_server(port: Optional[int] = None):
    if port is None:
        port = find_free_port()
    
    # Write port to file for Electron to read
    port_file = Path(__file__).parent / "desktop" / ".port"
    port_file.write_text(str(port))
    
    uvicorn.run("companion_app.server:app", host="0.0.0.0", port=port, reload=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--desktop", action="store_true", help="Run in desktop mode")
    parser.add_argument("--port", type=int, help="Port to run on")
    args = parser.parse_args()
    
    run_server(args.port)

