import os
import sys
from pathlib import Path

# Add the parent directory to the Python path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from typing import Dict, List, Optional, Union
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, HttpUrl
import json
from datetime import datetime
import logging
from fastapi.middleware.cors import CORSMiddleware

from companion_app.db_utils import get_connection
from companion_app.models import KnowledgeGraph, Entity, Relationship
from companion_app.search_engines import TopicSearcher
from companion_app.crawl4ai_client import scrape_and_extract, AsyncWebCrawler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = APIRouter()

class SearchConfig(BaseModel):
    """Configuration for topic search and extraction"""
    domain_type: str = Field(..., description="Type of domain to search")
    categories: Dict[str, List[str]] = Field(..., description="Search categories and their queries")
    semantic_filter: str = Field(..., description="Semantic filter for content")
    entity_types: List[str] = Field(..., description="Types of entities to extract")
    extraction_instructions: str = Field(..., description="Instructions for content extraction")
    schema_fields: List[Dict] = Field(default_factory=list, description="Schema fields for extraction")

    class Config:
        schema_extra = {
            "example": {
                "domain_type": "technology",
                "categories": {"AI": ["machine learning", "neural networks"]},
                "semantic_filter": "Focus on technical concepts",
                "entity_types": ["concept", "technology", "application"],
                "extraction_instructions": "Extract key technical concepts and their relationships",
                "schema_fields": []
            }
        }

class CrawlerConfig(BaseModel):
    extraction_strategy: str = Field(..., description="Strategy for content extraction")
    start_urls: List[str] = Field(..., description="URLs to start crawling from")

    class Config:
        schema_extra = {
            "example": {
                "extraction_strategy": "technical_content",
                "start_urls": ["https://example.com/tech"]
            }
        }

class KnowledgeGraphConfig(BaseModel):
    directory: str = Field(..., description="Directory to store knowledge graphs")
    domain_type: str = Field(..., description="Type of domain for the knowledge graph")

    class Config:
        schema_extra = {
            "example": {
                "directory": "/path/to/graphs",
                "domain_type": "technology"
            }
        }

class SystemConfig(BaseModel):
    knowledgeGraph: KnowledgeGraphConfig = Field(..., description="Knowledge graph configuration")
    search: SearchConfig = Field(..., description="Search configuration")
    crawler: CrawlerConfig = Field(..., description="Crawler configuration")

    class Config:
        schema_extra = {
            "example": {
                "knowledgeGraph": {
                    "directory": "/path/to/graphs",
                    "domain_type": "technology"
                },
                "search": {
                    "domain_type": "technology",
                    "categories": {"AI": ["machine learning"]},
                    "semantic_filter": "Technical concepts",
                    "entity_types": ["concept"],
                    "extraction_instructions": "Extract technical concepts",
                    "schema_fields": []
                },
                "crawler": {
                    "extraction_strategy": "technical_content",
                    "start_urls": ["https://example.com/tech"]
                }
            }
        }

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
                search=SearchConfig(
                    domain_type="",
                    categories={},
                    semantic_filter="",
                    entity_types=[],
                    extraction_instructions="",
                    schema_fields=[]
                ),
                crawler=CrawlerConfig(extraction_strategy="", start_urls=[])
            )
        with open(CONFIG_FILE) as f:
            return SystemConfig(**json.load(f))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/latest-graph/{domain_type}")
async def get_latest_graph(domain_type: str):
    """Get the latest knowledge graph for a given domain type"""
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
            return {"message": "No knowledge graph found", "data": None}
            
        extracted_content, raw_content, created_at, url = result
        
        try:
            if isinstance(extracted_content, str):
                kg_data = json.loads(extracted_content)
            else:
                kg_data = extracted_content
                
            knowledge_graph = KnowledgeGraph(
                entities=[Entity(**e) for e in kg_data.get("entities", [])],
                relationships=[Relationship(**r) for r in kg_data.get("relationships", [])]
            )
            
            return {
                "message": "Success",
                "data": {
                    "url": url,
                    "knowledge_graph": knowledge_graph,
                    "raw_content": raw_content,
                    "metadata": {"source": "database", "domain_type": domain_type},
                    "extraction_timestamp": created_at
                }
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON Decode Error: {e}")
            raise HTTPException(status_code=500, detail=f"Error parsing knowledge graph data: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error in get_latest_graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.get("/test-knowledge-graph")
async def test_knowledge_graph():
    """Test endpoint that returns a sample knowledge graph"""
    return {
        "entities": [
            {"id": "e1", "type": "concept", "name": "Machine Learning"},
            {"id": "e2", "type": "concept", "name": "Neural Networks"},
            {"id": "e3", "type": "concept", "name": "Deep Learning"}
        ],
        "relationships": [
            {"source": "e1", "target": "e2", "type": "includes"},
            {"source": "e2", "target": "e3", "type": "type_of"}
        ]
    }

@app.post("/scrape")
async def scrape_urls(urls: List[str], config: Optional[SearchConfig] = None):
    """Scrape and extract content from provided URLs"""
    try:
        results = []
        for url in urls:
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
            
            content = await scrape_and_extract(
                url=url,
                extraction_config=extraction_config,
                use_jsoncss=True,
                use_llm=True,
                use_cosine=True
            )
            
            if content:
                results.append({
                    "url": url,
                    "success": True,
                    "content": content
                })
            else:
                results.append({"url": url, "success": False, "error": "No content extracted"})
        return {"message": "Scraping completed", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/{domain_type}")
async def search_topic(domain_type: str, config: SearchConfig):
    """Search for content based on domain type and configuration"""
    try:
        logger.info(f"Starting search for domain: {domain_type}")
        logger.debug(f"Search configuration: {config.dict()}")
        
        searcher = TopicSearcher(config.categories)
        results = await searcher.run_all_searches()
        
        return {
            "message": "Success",
            "results": results,
            "metadata": {
                "domain_type": domain_type,
                "timestamp": datetime.now().isoformat(),
                "total_results": sum(len(cat) for cat in results.values())
            }
        }
    except Exception as e:
        logger.error(f"Error performing search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 