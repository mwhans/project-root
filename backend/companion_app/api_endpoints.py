from typing import Dict, List, Optional, Union
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, HttpUrl
import sqlite3
import json
from datetime import datetime
import sys
import os
from .crawl4ai_client import scrape_and_extract, AsyncWebCrawler
from .search_engines import TopicSearcher
import asyncio
from fastapi.middleware.cors import CORSMiddleware
import re
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from companion_app.db_utils import get_connection
from companion_app.models import KnowledgeGraph, Entity, Relationship

app = FastAPI(
    title="Knowledge Graph API",
    description="API endpoints for retrieving and generating knowledge graphs",
    version="1.0.0"
)

def is_ngrok_url(origin: str) -> bool:
    return origin.endswith('.ngrok-free.app')

allowed_origins = [
    "http://localhost:3000",
    "https://localhost:3000",
    "http://localhost:3002",
    "https://localhost:3002"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"https?://.*\.ngrok-free\.app",  # Allow any ngrok-free.app domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Knowledge Graph API",
        "version": "1.0.0",
        "endpoints": {
            "latest": "/latest-graph",
            "test": "/test-knowledge-graph",
            "health": "/health"
        }
    }

class SearchConfig(BaseModel):
    """Configuration for topic search and extraction"""
    domain_type: str
    categories: Dict[str, List[str]]
    semantic_filter: str
    entity_types: List[str]
    extraction_instructions: str
    schema_fields: List[Dict]

class CrawlerConfig(BaseModel):
    extraction_strategy: str
    start_urls: List[str]

class KnowledgeGraphConfig(BaseModel):
    directory: str
    domain_type: str

class SystemConfig(BaseModel):
    knowledgeGraph: KnowledgeGraphConfig
    search: SearchConfig
    crawler: CrawlerConfig

CONFIG_FILE = Path(__file__).parent / "config" / "system_config.json"

@app.post("/config")
async def save_config(config: SystemConfig):
    try:
        CONFIG_FILE.parent.mkdir(exist_ok=True)
        with open(CONFIG_FILE, "w") as f:
            json.dump(config.dict(), f, indent=2)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/config")
async def get_config() -> SystemConfig:
    try:
        if not CONFIG_FILE.exists():
            return SystemConfig(
                knowledgeGraph=KnowledgeGraphConfig(directory="", domain_type=""),
                search=SearchConfig(strategy_type="", categories=[]),
                crawler=CrawlerConfig(extraction_strategy="", start_urls=[])
            )
        
        with open(CONFIG_FILE) as f:
            return SystemConfig(**json.load(f))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    return {
        "status": "running",
        "components": {
            "knowledge_graph": True,
            "search": True,
            "crawler": True
        }
    }

@app.get("/latest-graph")
async def get_latest_knowledge_graph(domain_type: str = None):
    """Get the most recent knowledge graph from the extractions_llm table"""
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        # Debug: Print available tables
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cur.fetchall()
        logger.info(f"Available tables: {tables}")
        
        # Get the latest knowledge graph extraction with raw content
        query = """
            SELECT e.extracted_content, e.raw_content, e.created_at, p.url
            FROM extractions_llm e
            JOIN pages p ON e.page_id = p.id
            WHERE e.strategy_type = ?
            ORDER BY e.created_at DESC
            LIMIT 1
        """
        
        strategy_type = f"{domain_type}_knowledge_graph" if domain_type else "knowledge_graph"
        logger.info(f"Querying for strategy type: {strategy_type}")
        cur.execute(query, (strategy_type,))
        
        result = cur.fetchone()
        if not result:
            logger.info("No results found in database, returning test response")
            test_response = await get_test_knowledge_graph()
            return test_response
            
        extracted_content, raw_content, created_at, url = result
        
        try:
            # Debug prints
            logger.debug(f"Extracted content length: {len(str(extracted_content))}")
            logger.debug(f"Raw content length: {len(str(raw_content)) if raw_content else 0}")
            
            if isinstance(extracted_content, str):
                kg_data = json.loads(extracted_content)
            else:
                kg_data = extracted_content
                
            logger.debug(f"Parsed knowledge graph data: {json.dumps(kg_data, indent=2)}")
            
            knowledge_graph = KnowledgeGraph(
                entities=[Entity(**e) for e in kg_data.get("entities", [])],
                relationships=[Relationship(**r) for r in kg_data.get("relationships", [])]
            )
            
            response = KnowledgeGraphResponse(
                url=url,
                knowledge_graph=knowledge_graph,
                raw_content=raw_content or "No raw content available",
                metadata={"source": "database", "domain_type": domain_type},
                extraction_timestamp=created_at,
                source_authority=1.0,
                context={
                    "source": "extractions_llm",
                    "extraction_type": strategy_type,
                    "url": url
                }
            )
            
            logger.info(f"Successfully built response for URL: {url}")
            logger.debug(f"Response metadata: {response.metadata}")
            return response
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON Decode Error: {e}")
            logger.error(f"Content that failed to parse: {extracted_content[:500]}...")
            raise HTTPException(
                status_code=500,
                detail=f"Error parsing knowledge graph data: {str(e)}"
            )
            
    except Exception as e:
        logger.error(f"Error in get_latest_knowledge_graph: {e}")
        logger.error(f"Tables in database: {tables}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving latest knowledge graph: {str(e)}"
        )
    finally:
        conn.close()

@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/test-knowledge-graph")
async def get_test_knowledge_graph():
    """Test endpoint that returns a sample knowledge graph"""
    test_graph = KnowledgeGraph(
        entities=[
            Entity(
                name="Topic",
                type="concept",
                description="A general topic or concept",
                urls=["https://example.com/topic"],
                metadata={"category": "general"}
            ),
            Entity(
                name="Expert",
                type="person",
                description="A subject matter expert",
                urls=["https://example.com/expert"],
                metadata={"category": "person"}
            )
        ],
        relationships=[
            Relationship(
                source="Topic",
                target="Expert",
                relation_type="explained_by",
                description="Topics are explained by experts",
                urls=["https://example.com/expertise"],
                metadata={"importance": "high"}
            )
        ]
    )
    
    return KnowledgeGraphResponse(
        url="https://example.com/test",
        knowledge_graph=test_graph,
        raw_content="This is a test knowledge graph showing relationships between topics and experts.",
        metadata={"test": True},
        extraction_timestamp=datetime.now().isoformat(),
        source_authority=1.0,
        context={"test_data": True}
    )

class KnowledgeGraphResponse(BaseModel):
    """Response model for knowledge graph data"""
    url: str
    knowledge_graph: KnowledgeGraph
    raw_content: str  # Add raw_content field
    metadata: Dict
    extraction_timestamp: str
    source_authority: float
    context: Optional[Dict] = Field(default_factory=dict)

class QueryParams(BaseModel):
    """Query parameters for knowledge graph retrieval"""
    url_filter: Optional[str] = None
    entity_types: Optional[List[str]] = None
    min_authority: Optional[float] = 0.0
    include_raw: bool = False
    max_age_hours: Optional[int] = None

class ScrapeRequest(BaseModel):
    url: str

@app.post("/scrape")
async def scrape_url(request: ScrapeRequest, config: Optional[SearchConfig] = None):
    """
    Scrape and extract knowledge graph from a URL
    Args:
        request: ScrapeRequest containing the URL
        config: Optional search configuration for domain-specific extraction
    """
    try:
        logger.info(f"Received scrape request for URL: {request.url}")
        logger.info(f"Configuration: {config.dict() if config else 'No config provided'}")
        
        extraction_config = {
            "schema": {
                "name": config.domain_type if config else "ContentData",
                "fields": config.schema_fields if config else []
            },
            "semantic_filter": config.semantic_filter if config else "",
            "llm_config": {
                "entity_types": config.entity_types if config else [],
                "guidelines": config.extraction_instructions if config else "",
                "max_chunk_size": 32000,
                "chunk_token_threshold": 2000
            }
        } if config else {}
        
        logger.debug(f"Extraction config: {json.dumps(extraction_config, indent=2)}")
        
        url_id = await scrape_and_extract(
            url=request.url,
            extraction_config=extraction_config,
            use_jsoncss=True,
            use_llm=True,
            use_cosine=True
        )
        
        logger.info(f"Scraping completed successfully. URL ID: {url_id}")
        
        return {
            "status": "success",
            "url_id": url_id,
            "message": "Scraping and extraction completed"
        }
        
    except Exception as e:
        logger.error(f"Error in scrape_url endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error scraping URL: {str(e)}"
        )

class SearchResponse(BaseModel):
    """Response model for search results"""
    filepath: str
    timestamp: str
    categories: List[str]
    total_results: int

@app.post("/search/{domain_type}")
async def search_topic(domain_type: str, config: SearchConfig) -> SearchResponse:
    """
    Search for content based on domain type and configuration
    Args:
        domain_type: Type of domain to search (e.g., 'tech', 'health', 'science')
        config: Search configuration including categories and filters
    """
    try:
        logger.info(f"Starting search for domain: {domain_type}")
        logger.debug(f"Search configuration: {config.dict()}")
        
        searcher = TopicSearcher(config.categories)
        results = await searcher.run_all_searches()
        filepath = searcher.save_results_to_file(results)
        
        total_results = sum(len(cat) for cat in results.values())
        logger.info(f"Search completed. Found {total_results} results across {len(config.categories)} categories")
        logger.debug(f"Results saved to: {filepath}")
        
        return SearchResponse(
            filepath=filepath,
            timestamp=datetime.now().isoformat(),
            categories=list(config.categories.keys()),
            total_results=total_results
        )
        
    except Exception as e:
        logger.error(f"Error performing search: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error performing search: {str(e)}"
        )

@app.get("/search/results/{filename}")
async def get_search_results(filename: str):
    """
    Retrieve search results from a specific results file.
    """
    try:
        results_dir = os.path.join(os.path.dirname(__file__), 'search_results')
        filepath = os.path.join(results_dir, filename)
        
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="Results file not found")
            
        with open(filepath, 'r', encoding='utf-8') as f:
            results = json.load(f)
            
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve results: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 