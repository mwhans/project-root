import asyncio
import logging
from search_engines import TopicSearcher
from crawl4ai_client import scrape_and_extract
from db_utils import get_connection, init_db
import json
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_search_and_crawl():
    """Test the search and crawl pipeline"""
    try:
        # Initialize database
        init_db()
        logger.info("Database initialized")
        
        # Test search configuration - start with a focused query
        search_config = {
            "technology": [
                "latest artificial intelligence developments 2024",
                "machine learning applications in business"
            ]
        }
        
        # Initialize searcher with conservative settings
        searcher = TopicSearcher(
            search_config,
            num_results=5,  # Start with fewer results
            delay=2.0  # Conservative delay between requests
        )
        logger.info("Starting search process")
        
        # Run searches and save results
        try:
            results = await searcher.run_all_searches()
            logger.info("Search completed successfully")
            
            # Save results to file
            results_file = searcher.save_results_to_file(results)
            logger.info(f"Search results saved to {results_file}")
            
            # Process the results to generate knowledge graphs
            logger.info("Starting to process search results")
            successful, failed = await searcher.process_latest_results()
            logger.info(f"Processing completed. Successfully processed {successful} URLs, failed on {failed} URLs")
            
        except Exception as e:
            logger.error(f"Search or processing failed: {str(e)}")
            return
        
        logger.info("Test completed successfully")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(test_search_and_crawl()) 