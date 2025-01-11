"""
This script:
    - Finds the most recent JSON file in the 'search_results' directory.
    - Loads all URLs from that file.
    - For each URL, runs the same crawl & extraction pipeline (scrape_and_extract).
    - Outputs a separate knowledge-graph JSON for each URL (in the 'knowledge-graph' directory).
"""

import sys
import os
import json
import glob
import asyncio
import traceback
import logging
from datetime import datetime
from pathlib import Path
import uuid

# Add backend to Python path
backend_dir = str(Path(__file__).parent)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from companion_app.crawl4ai_client import scrape_and_extract
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

SEARCH_RESULTS_DIR = "search_results"
KNOWLEDGE_GRAPH_DIR = "knowledge-graph"

def get_latest_search_results_json() -> str:
    """
    Returns the most recent .json file from the 'search_results' directory.
    Raises FileNotFoundError if no JSON files exist.
    """
    search_dir = Path(__file__).parent / SEARCH_RESULTS_DIR
    search_dir.mkdir(parents=True, exist_ok=True)

    pattern = str(search_dir / "*.json")
    files = glob.glob(pattern)
    logger.debug(f"Found {len(files)} JSON files in {search_dir}")
    if not files:
        raise FileNotFoundError(f"No JSON files found in '{search_dir}'")

    latest_file = max(files, key=os.path.getmtime)
    logger.debug(f"Latest JSON file selected: {latest_file}")
    return latest_file

async def crawl_url_and_save_kg(url: str) -> str:
    """
    Scrapes the given URL using scrape_and_extract and saves a knowledge-graph 
    JSON file in backend/companion_app/knowledge-graph.
    """
    logger.debug(f"Started crawl_url_and_save_kg for URL: {url}")
    try:
        # Ensure knowledge_graph folder exists
        knowledge_graph_path = Path(__file__).parent / KNOWLEDGE_GRAPH_DIR
        knowledge_graph_path.mkdir(parents=True, exist_ok=True)

        # Configure extraction parameters
        extraction_config = {
            "provider": "openai/gpt-4",
            "instruction_template": (
                "Extract structured knowledge about dog health and care from {chunk_info}. "
                "Focus on breed-specific conditions, care routines, and best practices. "
                "Entity types: {entity_types}. Guidelines: {guidelines}"
            ),
            "entity_types": ["BreedSpecific", "Care", "Prevention", "Symptom", "Treatment", "Lifestyle"],
            "guidelines": [
                "Extract specific, actionable information",
                "Include relationships between entities",
                "Focus on health and care-related content"
            ],
            "chunk_token_threshold": 2000,
            "chunk_info": "Dog health and care information",
            "semantic_filter": "dog health, care, breed condition, tips"
        }
        logger.debug(f"Extraction config for {url}: {extraction_config}")

        # 1) Call the pipeline → should return a dict
        logger.debug(f"Calling scrape_and_extract for {url}")
        extracted_data = await scrape_and_extract(
            url=url,
            extraction_config=extraction_config,
            use_jsoncss=False,
            use_llm=True,
            use_cosine=True
        )
        logger.debug(f"Received extracted_data for {url}: {extracted_data}")

        # 2) Validate the result
        if not isinstance(extracted_data, dict) or not extracted_data:
            logger.error(f"[ERROR] scrape_and_extract returned invalid data for {url}")
            return ""

        # 3) Format the knowledge graph
        knowledge_graph_output = [{
            "entities": extracted_data.get("entities", []),
            "relationships": extracted_data.get("relationships", []),
            "error": False
        }]
        logger.debug(f"Formatted knowledge_graph_output for {url}: {knowledge_graph_output}")

        # 4) Write out the JSON
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        random_str = uuid.uuid4().hex[:8]
        kg_filename = f"kg_{timestamp}_{random_str}.json"
        kg_filepath = knowledge_graph_path / kg_filename

        with kg_filepath.open("w", encoding="utf-8") as out_f:
            json.dump(knowledge_graph_output, out_f, indent=2)

        logger.info(f"[SUCCESS] Saved knowledge graph for URL: {url} -> {kg_filepath}")
        return str(kg_filepath)

    except Exception as exc:
        logger.error(f"[Error] during crawl_url_and_save_kg for '{url}': {exc}")
        traceback.print_exc()
        return ""

async def main():
    """
    1) Find the most recent JSON file in the 'search_results' directory.
    2) Load the URLs from it.
    3) For each URL, run the pipeline (scrape_and_extract).
    4) For each URL's result, output a knowledge-graph JSON in 'knowledge-graph'.
    """
    try:
        logger.debug("Invoking get_latest_search_results_json()")
        latest_json_file = get_latest_search_results_json()
        logger.info(f"Using latest search results file: {latest_json_file}")

        with open(latest_json_file, "r", encoding="utf-8") as f:
            logger.debug(f"Loading JSON contents from {latest_json_file}")
            search_data = json.load(f)

        # Gather URLs
        logger.debug("Gathering all URLs from the loaded JSON structure")
        all_urls = []
        for category_data in search_data.values():
            if isinstance(category_data, dict):
                for query_dict in category_data.values():
                    if isinstance(query_dict, dict):
                        for engine_results in query_dict.values():
                            if isinstance(engine_results, list):
                                for item in engine_results:
                                    url = item.get("url")
                                    if url:
                                        all_urls.append(url)

        logger.info(f"Discovered {len(all_urls)} URLs to crawl")
        if not all_urls:
            logger.warning("No URLs found in the latest search_results JSON.")
            return

        logger.info("Beginning crawl for each URL...")
        # Crawl them all sequentially (or concurrently if you like)
        for url in all_urls:
            logger.debug(f"Starting extraction for {url}")
            await crawl_url_and_save_kg(url)
            logger.debug(f"Finished extraction for {url}")

        logger.info("All URLs have been processed. Knowledge-graphs generated.")

    except FileNotFoundError as fnfe:
        logger.error(str(fnfe))
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
