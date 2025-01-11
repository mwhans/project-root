"""
General purpose search engine module for querying multiple search engines.
"""
import sys
import os

# Add backend to Python path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import asyncio
from typing import List, Dict, Optional
import json
from datetime import datetime
from googlesearch import search as google_search
from pydantic import BaseModel
import logging
import traceback
from dotenv import load_dotenv
import glob
from companion_app.crawl4ai_client import scrape_and_extract

# Load environment variables
load_dotenv()

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SearchResult(BaseModel):
    """Model for search results"""
    url: str
    title: Optional[str] = None
    snippet: Optional[str] = None
    engine: str  # Adding back engine field
    query: str
    rank: int
    category: str

    @classmethod
    def from_json(cls, data: dict, engine: str, query: str, rank: int, category: str) -> 'SearchResult':
        """Create SearchResult from JSON data"""
        return cls(
            url=data.get('url') or data.get('link', ''),
            title=data.get('title') or data.get('name', ''),
            snippet=data.get('snippet') or data.get('body') or data.get('description', ''),
            engine=engine,
            query=query,
            rank=rank,
            category=category
        )

class TopicSearcher:
    """Class to handle searching using Google for any topic"""
    
    def __init__(self, search_config: Dict[str, List[str]], num_results: int = 10, delay: float = 2.0):
        """
        Initialize with a search configuration dictionary
        Args:
            search_config: Dictionary mapping categories to lists of search queries
            num_results: Number of results to fetch per query
            delay: Delay between searches to avoid rate limiting
        """
        self.search_queries = search_config
        self.num_results = num_results
        self.delay = delay
        logger.info(f"Initializing TopicSearcher with {len(search_config)} categories")

    async def search_google(self, query: str, category: str) -> List[SearchResult]:
        """Search using Google"""
        logger.info(f"Starting Google search for query: {query}")
        
        try:
            results = []
            search_results = list(google_search(
                query, 
                num=self.num_results,
                stop=self.num_results,
                pause=self.delay
            ))
            logger.info(f"Found {len(search_results)} Google results")
            
            for i, url in enumerate(search_results):
                try:
                    if not isinstance(url, str) or not url.startswith(('http://', 'https://')):
                        continue
                        
                    # Create result with more structured data
                    result_data = {
                        'url': url,
                        'title': f"Result {i+1} from {url}",  # Placeholder title
                        'snippet': ''  # Google search doesn't provide snippets directly
                    }
                    
                    results.append(SearchResult.from_json(
                        data=result_data,
                        engine='google',
                        query=query,
                        rank=i + 1,
                        category=category
                    ))
                except Exception as e:
                    logger.error(f"Error processing Google result {i}: {str(e)}")
                    continue

            logger.info(f"Successfully processed {len(results)} Google results")
            await asyncio.sleep(self.delay)  # Additional delay between queries
            return results
            
        except Exception as e:
            logger.error(f"Error in Google search: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    async def run_all_searches(self) -> Dict[str, Dict[str, List[SearchResult]]]:
        """Run searches for all queries"""
        all_results = {}
        
        for category, queries in self.search_queries.items():
            category_results = {}
            for query in queries:
                results = await self.search_google(query, category)
                category_results[query] = results
                logger.info(f"Found {len(results)} results for {category}/{query}")
            all_results[category] = category_results
            
        return all_results

    def save_results_to_file(self, results: Dict[str, Dict[str, List[SearchResult]]]) -> str:
        """Save search results to a file"""
        results_dir = os.path.join(os.path.dirname(__file__), 'search_results')
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'search_results_{timestamp}.json'
        filepath = os.path.join(results_dir, filename)
        
        # Structure the output to match the expected format
        output = {}
        for category, category_results in results.items():
            output[category] = {}
            for query, results_list in category_results.items():
                output[category][query] = {
                    'google': [result.dict() for result in results_list]
                }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        return filepath

    async def process_latest_results(self):
        """Process latest search results and create knowledge graphs"""
        # Get latest results file
        results_dir = os.path.join(os.path.dirname(__file__), 'search_results')
        files = glob.glob(os.path.join(results_dir, '*.json'))
        if not files:
            logger.error("No search results files found")
            return
            
        latest_file = max(files, key=os.path.getctime)
        logger.info(f"Processing URLs from {latest_file}")
        
        # Load and extract URLs
        with open(latest_file, 'r') as f:
            data = json.load(f)
            
        urls = set()
        for category in data.values():
            for query in category.values():
                for engine_results in query.values():  # Handle nested engine results
                    for result in engine_results:
                        if result.get('url'):
                            urls.add(result['url'])
        
        logger.info(f"Found {len(urls)} unique URLs to process")
        
        # Process each URL
        successful = 0
        failed = 0
        
        for url in urls:
            try:
                logger.info(f"Processing URL: {url}")
                content = await scrape_and_extract(
                    url=url,
                    extraction_config={
                        "schema": {
                            "name": "WebContent",
                            "fields": [
                                {"name": "title", "type": "string"},
                                {"name": "summary", "type": "string"},
                                {"name": "entities", "type": "array"}
                            ]
                        },
                        "semantic_filter": "Focus on main content and key concepts",
                        "llm_config": {
                            "entity_types": ["concept", "technology", "term"],
                            "guidelines": "Extract key concepts and their relationships",
                            "max_chunk_size": 32000,
                            "chunk_token_threshold": 2000
                        }
                    },
                    use_jsoncss=True,
                    use_llm=True,
                    use_cosine=True
                )
                
                if content:
                    successful += 1
                    logger.info(f"Successfully processed {url}")
                else:
                    failed += 1
                    logger.warning(f"No content extracted from {url}")
                    
            except Exception as e:
                failed += 1
                logger.error(f"Error processing {url}: {str(e)}")
                continue
                
        logger.info(f"Processing completed. Successful: {successful}, Failed: {failed}")
        return successful, failed

async def main():
    """Example usage of the TopicSearcher"""
    # Example search configuration
    search_config = {
        'technology': [
            'latest AI developments',
            'machine learning trends'
        ],
        'science': [
            'recent space discoveries',
            'quantum computing advances'
        ]
    }
    
    searcher = TopicSearcher(search_config)
    results = await searcher.run_all_searches()
    filepath = searcher.save_results_to_file(results)
    logger.info(f"Search results saved to: {filepath}")
    
    await searcher.process_latest_results()

if __name__ == "__main__":
    asyncio.run(main()) 