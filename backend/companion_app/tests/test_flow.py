import asyncio
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from companion_app.crawl4ai_client import scrape_and_extract
from companion_app.search_engines import TopicSearcher
from companion_app.db_utils import init_db, get_connection

async def run_test_flow():
    # 1. Initialize database
    print("Initializing database...")
    init_db()
    
    # 2. Configure search
    print("\nConfiguring search categories...")
    searcher = TopicSearcher({
        "artificial_intelligence": [
            "latest AI developments",
            "AI research breakthroughs",
            "neural network advances"
        ],
        "machine_learning": [
            "machine learning applications",
            "ML frameworks",
            "deep learning tools"
        ]
    })
    
    # 3. Run search and collect URLs
    print("\nRunning search...")
    search_results = await searcher.run_all_searches()
    urls = []
    for category, results in search_results.items():
        print(f"\nCategory: {category}")
        print(f"Found {len(results)} results")
        urls.extend([result['url'] for result in results])
    
    # 4. Crawl collected URLs
    print(f"\nCrawling {len(urls)} URLs...")
    successful_crawls = 0
    failed_crawls = 0
    
    for url in urls[:5]:  # Limit to 5 for testing
        try:
            print(f"\nProcessing: {url}")
            url_id = await scrape_and_extract(
                url=url,
                extraction_config={
                    "schema": {
                        "name": "TechContent",
                        "fields": ["title", "content", "author", "date"]
                    },
                    "semantic_filter": "technology and artificial intelligence",
                    "llm_config": {
                        "entity_types": ["Technology", "Company", "Person"],
                        "guidelines": "Extract key technical concepts and developments"
                    }
                }
            )
            print(f"Successfully processed (ID: {url_id})")
            successful_crawls += 1
        except Exception as e:
            print(f"Error processing {url}: {e}")
            failed_crawls += 1
    
    # 5. Verify results
    print("\nVerifying results...")
    conn = get_connection()
    cur = conn.cursor()
    
    # Check pages table
    cur.execute("SELECT COUNT(*) FROM pages")
    page_count = cur.fetchone()[0]
    
    # Check extractions
    cur.execute("SELECT COUNT(*) FROM extractions_llm")
    extraction_count = cur.fetchone()[0]
    
    # Get latest extractions
    cur.execute("""
        SELECT p.url, e.created_at, e.strategy_type 
        FROM extractions_llm e 
        JOIN pages p ON e.page_id = p.id 
        ORDER BY e.created_at DESC 
        LIMIT 3
    """)
    latest_extractions = cur.fetchall()
    
    print(f"\nTest Results:")
    print(f"Pages crawled: {page_count}")
    print(f"Extractions: {extraction_count}")
    print(f"Successful crawls in this run: {successful_crawls}")
    print(f"Failed crawls in this run: {failed_crawls}")
    
    if latest_extractions:
        print("\nLatest extractions:")
        for url, created_at, strategy_type in latest_extractions:
            print(f"- {url} ({strategy_type}) at {created_at}")
    
    conn.close()

if __name__ == "__main__":
    print("Starting end-to-end test flow...")
    asyncio.run(run_test_flow()) 