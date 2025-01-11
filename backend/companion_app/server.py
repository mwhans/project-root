# server.py
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel
import uvicorn
import os
from pathlib import Path
from dotenv import load_dotenv
import litellm

from .api_endpoints import app as api_router
from .db_utils import init_db
from .crawl4ai_client import scrape_and_extract

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
    description="Configurable Knowledge Graph and Search System",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files
static_path = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Mount API routes
app.include_router(api_router, prefix="/api")

class APIKeyUpdate(BaseModel):
    api_key: str

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

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    html_path = static_path / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return HTMLResponse(content=html_path.read_text(), status_code=200)

if __name__ == "__main__":
    uvicorn.run("companion_app.server:app", host="0.0.0.0", port=8000, reload=True)

