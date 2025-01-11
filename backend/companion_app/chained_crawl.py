"""
This module now focuses solely on the initial search crawl (first crawl).
It generates a single search_results JSON file and saves it in the search_results directory.
"""

import sys
import os
import json
import traceback
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import uuid
import asyncio
import glob
import logging

# Add backend to Python path
backend_dir = str(Path(__file__).parent.parent)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from companion_app.search_engines import TopicSearcher  # Step 1

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ChainedCrawler:
    """
    Handles only a single (first) crawl step:
      1) Execute the initial search using TopicSearcher (search_engines.py).
         This will produce a JSON file with search results.
    """

    def __init__(
        self,
        openai_api_key: str,
        search_config: Dict[str, List[str]],
        num_results: int = 10,
        delay_between_searches: float = 2.0,
        search_results_dir: str = "search_results"
    ):
        """
        Args:
            openai_api_key: The OpenAI API key (set as ENV var).
            search_config: Dictionary mapping categories to lists of search queries.
            num_results: Number of results to fetch per query.
            delay_between_searches: Delay between queries to avoid rate limiting.
            search_results_dir: Directory to store the search results JSON.
        """
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.search_config = search_config
        self.num_results = num_results
        self.delay_between_searches = delay_between_searches
        self.search_results_dir = search_results_dir

        os.makedirs(search_results_dir, exist_ok=True)

    async def first_crawl_and_save(self) -> str:
        """
        Performs the first crawl using the TopicSearcher, then saves to JSON.
        Returns:
            The file path of the created search-results JSON.
        """
        try:
            searcher = TopicSearcher(
                search_config=self.search_config,
                num_results=self.num_results,
                delay=self.delay_between_searches
            )
            results = await searcher.run_all_searches()
            filepath = searcher.save_results_to_file(results, output_dir=self.search_results_dir)
            logger.info(f"First crawl JSON saved to: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error during first crawl: {str(e)}")
            traceback.print_exc()
            return ""

async def main():
    """
    Example usage:
        1) Provide your OpenAI API key.
        2) Perform the first crawl, which generates a JSON file in search_results/.
    """
    openai_key = input("Enter your OpenAI API key: ")

    # Example search config
    search_config = {
        "bulldog_health": [
            "bulldog health tips",
            "bulldog common health issues",
            "bulldog care guide"
        ]
    }

    crawler = ChainedCrawler(
        openai_api_key=openai_key,
        search_config=search_config
    )

    # Only do the first crawl now
    first_file = await crawler.first_crawl_and_save()
    if not first_file:
        logger.error("First crawl failed.")
        return

    logger.info("First crawl completed successfully.")

if __name__ == "__main__":
    asyncio.run(main()) 