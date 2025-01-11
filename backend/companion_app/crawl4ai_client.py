"""
General purpose content extraction and knowledge graph generation module.
"""
import asyncio
from typing import Optional, List, Dict
import json
from datetime import datetime
import os
from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
from crawl4ai.extraction_strategy import (
    JsonCssExtractionStrategy,
    LLMExtractionStrategy,
    CosineStrategy
)
from companion_app.db_utils import (
    get_or_create_url,
    store_extraction,
    init_db
)
from companion_app.config import OPENAI_API_KEY
from pydantic import BaseModel, Field
import traceback
from pprint import pprint
import re
import hashlib
import logging

# Initialize the database when the module is imported
init_db()

# Define the schema using Pydantic models
class Entity(BaseModel):
    name: str
    type: str
    description: Optional[str] = None
    urls: Optional[List[str]] = Field(default_factory=list)
    metadata: Optional[Dict] = Field(default_factory=dict)

class Relationship(BaseModel):
    source: str
    target: str
    relation_type: str
    description: Optional[str] = None
    urls: Optional[List[str]] = Field(default_factory=list)
    metadata: Optional[Dict] = Field(default_factory=dict)

class KnowledgeGraph(BaseModel):
    entities: List[Entity] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def get_base_schema(config: dict) -> dict:
    """
    Returns a configurable base schema for content extraction
    Args:
        config: Dictionary containing schema configuration including selectors and fields
    """
    return {
        "name": config.get("name", "ContentData"),
        "baseSelector": config.get("baseSelector", "article, div.content, div.main-content"),
        "type": "object",
        "fields": config.get("fields", [
            {
                "name": "title",
                "type": "string",
                "selector": "h1, h2.article-title"
            },
            {
                "name": "content",
                "type": "string",
                "selector": "p, .content, .main-content"
            },
            {
                "name": "last_updated",
                "type": "string",
                "selector": ".article-date, .last-updated"
            }
        ])
    }

async def preprocess_with_cosine(crawler: AsyncWebCrawler, url: str, semantic_filter: str) -> Optional[Dict]:
    """
    Preprocess the content using Cosine Strategy with configurable content capture.
    Args:
        crawler: AsyncWebCrawler instance
        url: URL to process
        semantic_filter: Space-separated keywords for content filtering
    """
    logger.debug(f"[COSINE] Starting cos. preprocess for URL: {url}")
    logger.debug(f"[COSINE] Using filter: '{semantic_filter}'")
    print("\nStarting Cosine preprocessing...")
    
    try:
        # Configure cosine strategy with more permissive parameters
        cosine_strategy = CosineStrategy(
            semantic_filter=semantic_filter,
            word_count_threshold=10,
            sim_threshold=0.2,
            max_dist=0.8,
            linkage_method='complete',
            top_k=20,
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            verbose=True
        )
        
        # Create crawler config
        cosine_config = CrawlerRunConfig(
            extraction_strategy=cosine_strategy,
            cache_mode=CacheMode.BYPASS
        )
        
        # First get raw content
        base_result = await crawler.arun(url=url, cache_mode=CacheMode.BYPASS)
        if not base_result.success:
            logger.error(f"[COSINE] Failed to fetch base content: {base_result.error_message}")
            return None
            
        # Prepare content for processing
        content_to_process = base_result.markdown or base_result.cleaned_html
        if not content_to_process:
            logger.warning("[COSINE] No content available for processing")
            return None
            
        # Run cosine strategy
        result = await crawler.arun(
            url=url,
            content=content_to_process,
            config=cosine_config
        )
        
        if not result.success:
            logger.error(f"[COSINE] Extraction failed: {result.error_message}")
            return None
            
        # Process the extracted clusters
        content = result.extracted_content
        if not content:
            logger.warning("[COSINE] No content extracted by Cosine strategy")
            return None
            
        # Convert content to list if it's a string or dict
        if isinstance(content, str):
            content = [{"text": content}]
        elif isinstance(content, dict):
            content = [content]
            
        # Process and enhance clusters
        processed_content = {
            "content_clusters": [],
            "metadata": {
                "total_clusters": len(content),
                "categories": {},
                "total_words": 0,
                "content_types": set()
            }
        }
        
        # Process each cluster
        for cluster in content:
            cluster_text = cluster.get('text', '') if isinstance(cluster, dict) else str(cluster)
            
            if not cluster_text or len(cluster_text.split()) < 5:  # Minimum word threshold
                continue
                
            # Extract URLs from the text
            urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 
                            cluster_text)
            
            # Create enhanced cluster
            processed_cluster = {
                "text": cluster_text,
                "word_count": len(cluster_text.split()),
                "urls": urls,
                "metadata": {
                    "has_numbers": bool(re.search(r'\d', cluster_text)),
                    "has_urls": bool(urls),
                    "url_count": len(urls)
                }
            }
            
            processed_content["content_clusters"].append(processed_cluster)
            processed_content["metadata"]["total_words"] += processed_cluster["word_count"]
        
        logger.info(f"[COSINE] Preprocessing complete. Found {len(processed_content['content_clusters'])} clusters")
        logger.debug(f"[COSINE] Total words processed: {processed_content['metadata']['total_words']}")
        
        if not processed_content["content_clusters"]:
            logger.warning("[COSINE] No meaningful clusters found")
            print("[Cosine] No meaningful content from {}, continuing...".format(url))
            return None
        
        return processed_content
        
    except Exception as e:
        logger.error(f"[COSINE] Error in preprocessing: {str(e)}")
        traceback.print_exc()
        return None

def create_llm_extraction_strategy(chunk_info: str, config: dict) -> LLMExtractionStrategy:
    """
    Create an LLM extraction strategy with the relevant prompt instructions & schema.
    """
    logger.debug("Creating LLMExtractionStrategy with config:")
    logger.debug(f"chunk_info: {chunk_info}, raw config: {config}")

    # Create a more specific instruction template
    instruction = f"""
    Analyze the following content about dog health and care. Extract structured information into entities and relationships.
    
    Entity Types to Extract:
    - BreedSpecific: Breed-specific traits, conditions, or requirements
    - Care: General care instructions and requirements
    - Prevention: Preventive measures and recommendations
    - Symptom: Health symptoms and warning signs
    - Treatment: Treatment methods and medications
    - Lifestyle: Daily care and lifestyle requirements

    Guidelines:
    1. Create detailed entities with clear names and descriptions
    2. Establish meaningful relationships between entities
    3. Focus on actionable health and care information
    4. Be specific about breed-related health concerns
    5. Include preventive care measures

    Format the output as a JSON object with 'entities' and 'relationships' arrays.
    Each entity must have: name, type, description
    Each relationship must have: source, target, relation_type, description
    """

    # Create the strategy with explicit configuration
    strategy = LLMExtractionStrategy(
        provider="openai/gpt-4",
        api_token=OPENAI_API_KEY,
        schema=KnowledgeGraph.schema(),
        extraction_type="schema",
        instruction=instruction,
        chunk_token_threshold=1500,  # Reduced for better reliability
        temperature=0.3,  # Added for more focused extraction
        max_tokens=1000,  # Added explicit token limit
        verbose=True
    )
    
    return strategy

async def scrape_url(url: str, verbose: bool = True) -> dict:
    """
    Asynchronously scrape the given URL using Crawl4AI and return relevant data.
    """
    async with AsyncWebCrawler(verbose=verbose) as crawler:
        # For demonstration, we bypass the cache each time.
        result = await crawler.arun(url=url, cache_mode=CacheMode.BYPASS)

        # Build a small dict with relevant info
        return {
            "success": result.success,
            "status_code": result.status_code,
            "error_message": result.error_message,
            "html": result.html if result.success else None,
            "markdown": result.markdown if result.success else None,
            "cleaned_html": result.cleaned_html if result.success else None,
            "links": result.links if result.success else {},
            "media": result.media if result.success else {}
        }

def run_scrape_sync(url: str, is_internal: bool = False) -> dict:
    """
    Synchronous wrapper for quick testing or non-async contexts.
    Args:
        url: The URL to scrape
        is_internal: Flag to determine if data should be stored in internal tables only
    """
    return asyncio.run(scrape_url(url, is_internal=is_internal))

async def scrape_and_extract(
    url: str,
    extraction_config: dict,
    use_jsoncss: bool = True,
    use_llm: bool = True,
    use_cosine: bool = True
) -> dict:
    """
    Main multi-step crawl & extraction pipeline. 
    Returns { "entities": [...], "relationships": [...] } or an empty dict if fails.
    """
    logger.debug(f"[Pipeline] scrape_and_extract starting for {url}")
    logger.debug(f"[Pipeline] extraction_config: {extraction_config}")
    logger.debug(f"[Pipeline] use_jsoncss={use_jsoncss}, use_llm={use_llm}, use_cosine={use_cosine}")

    combined_results = {"entities": [], "relationships": []}
    try:
        # 1) Initialize crawler with async context management
        logger.debug("[Pipeline] Initializing crawler (AsyncWebCrawler)...")
        async with AsyncWebCrawler(verbose=True) as crawler:
            # First get the raw HTML content
            base_result = await crawler.arun(url=url, cache_mode=CacheMode.BYPASS)
            if not base_result.success:
                logger.error(f"[Pipeline] Failed to fetch base content: {base_result.error_message}")
                return combined_results

            # Extract the content from HTML
            raw_content = base_result.markdown or base_result.cleaned_html
            if not raw_content:
                logger.error("[Pipeline] No content extracted from HTML")
                return combined_results

            # 2) JSON/CSS extraction if enabled
            base_content = raw_content
            if use_jsoncss:
                logger.debug("[Pipeline] Starting JSON/CSS extraction...")
                try:
                    jsoncss_strategy = JsonCssExtractionStrategy(
                        schema=get_base_schema(extraction_config)
                    )
                    result = await crawler.arun(
                        url=url,
                        content=raw_content,
                        extraction_strategy=jsoncss_strategy,
                        cache_mode=CacheMode.BYPASS
                    )
                    if result.success and hasattr(result, 'extracted_content'):
                        base_content = result.extracted_content
                        logger.debug("[Pipeline] JSON/CSS extraction successful")
                except Exception as e:
                    logger.error(f"[Pipeline] JSON/CSS extraction failed: {str(e)}")

            # 3) Cosine preprocessing if enabled
            if use_cosine:
                logger.debug("[Pipeline] Starting Cosine preprocessing...")
                try:
                    cosine_result = await preprocess_with_cosine(
                        crawler=crawler,
                        url=url,
                        semantic_filter=extraction_config.get("semantic_filter", "dog health care breed condition symptoms treatment prevention")
                    )
                    if cosine_result and cosine_result.get("content_clusters"):
                        # Combine all relevant content clusters
                        base_content = "\n\n".join([
                            cluster["text"] for cluster in cosine_result["content_clusters"]
                            if cluster.get("text")
                        ])
                        logger.debug("[Pipeline] Cosine preprocessing successful")
                    else:
                        logger.warning("[Pipeline] No content from Cosine preprocessing, using base content")
                except Exception as e:
                    logger.error(f"[Pipeline] Cosine preprocessing failed: {str(e)}")

            # 4) LLM extraction if enabled
            if use_llm and base_content:
                logger.debug("[Pipeline] Starting LLM extraction...")
                try:
                    # Create LLM strategy with specific instructions
                    llm_strategy = create_llm_extraction_strategy(
                        chunk_info="Extract structured knowledge about dog health and care",
                        config={
                            "provider": "openai/gpt-4",
                            "instruction_template": (
                                "Extract structured knowledge about dog health and care from the following content. "
                                "Focus on breed-specific conditions, care routines, and best practices. "
                                "Entity types: {entity_types}. Guidelines: {guidelines}"
                            ),
                            "entity_types": [
                                "BreedSpecific", "Care", "Prevention", "Symptom", "Treatment", "Lifestyle"
                            ],
                            "guidelines": [
                                "Extract specific, actionable information",
                                "Include relationships between entities",
                                "Focus on health and care-related content"
                            ],
                            "chunk_token_threshold": 2000
                        }
                    )

                    # Run LLM extraction
                    llm_result = await _run_llm_extraction(crawler, url, base_content, llm_strategy)
                    if llm_result and (llm_result.get("entities") or llm_result.get("relationships")):
                        combined_results = llm_result
                        logger.debug("[Pipeline] LLM extraction successful")
                    else:
                        logger.warning("[Pipeline] LLM extraction returned no results")
                except Exception as e:
                    logger.error(f"[Pipeline] LLM extraction failed: {str(e)}")
                    traceback.print_exc()

        return combined_results

    except Exception as e:
        logger.error(f"[Pipeline] Unhandled error in scrape_and_extract: {str(e)}")
        traceback.print_exc()
        return combined_results

async def _run_llm_extraction(crawler, url: str, content: str, strategy) -> dict:
    """
    Splits text into chunks, calls LLM extraction, merges results.
    """
    logger.debug(f"[LLM] _run_llm_extraction started for {url}")
    logger.debug(f"[LLM] Length of content for LLM extraction: {len(content)} chars")

    combined = {"entities": [], "relationships": [], "error": False}
    try:
        # Configure chunk size and overlap
        CHUNK_SIZE = 1500  # Further reduced for better processing
        OVERLAP = 100
        MAX_CHUNKS = 3  # Reduced to ensure completion
        
        # Split content into overlapping chunks
        chunks = []
        start = 0
        while start < len(content) and len(chunks) < MAX_CHUNKS:
            end = start + CHUNK_SIZE
            if end > len(content):
                end = len(content)
            chunk = content[start:end]
            chunks.append(chunk)
            start = end - OVERLAP
            if start >= len(content):
                break

        logger.debug(f"[LLM] Split content into {len(chunks)} chunks (max {MAX_CHUNKS})")

        # Process each chunk with retries
        for index, chunk in enumerate(chunks, 1):
            max_retries = 2
            retry_delay = 3

            for attempt in range(max_retries):
                try:
                    logger.debug(f"[LLM] Processing chunk {index}/{len(chunks)} (attempt {attempt + 1})")
                    
                    # Create chunk-specific instruction
                    chunk_instruction = f"""
                    Extract structured knowledge from content chunk {index}/{len(chunks)}:

                    Rules:
                    1. Create entities for each distinct health concept
                    2. Link entities with meaningful relationships
                    3. Be specific and detailed in descriptions
                    4. Focus on actionable information
                    5. Include any relevant health warnings

                    Required format:
                    {{
                        "entities": [
                            {{"name": "string", "type": "string", "description": "string"}},
                            ...
                        ],
                        "relationships": [
                            {{"source": "string", "target": "string", "relation_type": "string", "description": "string"}},
                            ...
                        ]
                    }}
                    """
                    
                    # Update strategy for this chunk
                    strategy.instruction = chunk_instruction
                    strategy.chunk_token_threshold = 1500
                    strategy.temperature = 0.3
                    strategy.max_tokens = 1000

                    # Run extraction
                    result = await crawler.arun(
                        url=url,
                        content=chunk,
                        extraction_strategy=strategy,
                        cache_mode=CacheMode.BYPASS
                    )

                    if result.success and hasattr(result, 'extracted_content'):
                        extracted = result.extracted_content
                        logger.debug(f"[LLM] Raw extraction result: {extracted}")
                        
                        if isinstance(extracted, str):
                            try:
                                extracted = json.loads(extracted)
                            except json.JSONDecodeError:
                                logger.error("[LLM] Failed to parse extracted content as JSON")
                                continue

                        # Merge new entities and relationships
                        if isinstance(extracted, list):
                            # Handle list of results
                            for item in extracted:
                                if isinstance(item, dict):
                                    combined["entities"].extend(item.get("entities", []))
                                    combined["relationships"].extend(item.get("relationships", []))
                        elif isinstance(extracted, dict):
                            # Handle single result
                            combined["entities"].extend(extracted.get("entities", []))
                            combined["relationships"].extend(extracted.get("relationships", []))

                        logger.debug(f"[LLM] Successfully processed chunk {index}")
                        break  # Success, move to next chunk

                    else:
                        logger.warning(f"[LLM] No valid extraction for chunk {index}")

                except Exception as e:
                    logger.error(f"[LLM] Error processing chunk {index}: {str(e)}")
                    if "rate_limit_exceeded" in str(e) and attempt < max_retries - 1:
                        logger.info(f"[LLM] Rate limit hit, waiting {retry_delay}s before retry")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        break

            # Increased delay between chunks to avoid rate limits
            await asyncio.sleep(2)

        logger.info(f"[LLM] Extraction complete. Found {len(combined['entities'])} entities and {len(combined['relationships'])} relationships")
        return combined

    except Exception as e:
        logger.error(f"[LLM] Error in _run_llm_extraction: {str(e)}")
        traceback.print_exc()
        return combined

def store_knowledge_graph_as_json(graph_data: dict, url: str, source_metadata: dict = None) -> str:
    """
    Store a knowledge graph as a JSON file with metadata.
    """
    try:
        # Create knowledge-graph directory inside companion_app
        current_dir = os.path.dirname(os.path.abspath(__file__))
        kg_dir = os.path.join(current_dir, 'knowledge-graph')
        os.makedirs(kg_dir, exist_ok=True)
        
        logger.debug("Processing knowledge graph data...")
        
        # Ensure we have a valid data structure
        if not isinstance(graph_data, (dict, list)):
            logger.error(f"Invalid graph_data type: {type(graph_data)}")
            graph_data = {"entities": [], "relationships": [], "error": True}
        
        # If it's a dict, wrap it in a list for consistency
        if isinstance(graph_data, dict):
            graph_data = [graph_data]
            
        # Ensure each item in the list has the required structure
        formatted_data = []
        for item in graph_data:
            if isinstance(item, dict):
                formatted_item = {
                    "entities": item.get("entities", []),
                    "relationships": item.get("relationships", []),
                    "error": item.get("error", False)
                }
                formatted_data.append(formatted_item)
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        filename = f"kg_{timestamp}_{url_hash}.json"
        file_path = os.path.join(kg_dir, filename)
        
        # Write to file with pretty printing
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Knowledge graph successfully stored at: {file_path}")
        return file_path
        
    except Exception as e:
        logger.error(f"Error storing knowledge graph as JSON: {str(e)}")
        logger.error("Debug - Full error:")
        traceback.print_exc()
        raise e

async def chunk_and_process_llm(crawler, url: str, content: str, 
                              knowledge_graph_strategy: LLMExtractionStrategy, 
                              page_id: int, is_internal: bool = False) -> None:
    """
    Process large content in chunks for LLM extraction.
    """
    try:
        CHUNK_SIZE = 6000  # Adjusted for GPT-4's context window
        MAX_CHUNKS = 5
        
        # Split content into chunks
        chunks = [content[i:i + CHUNK_SIZE] 
                 for i in range(0, len(content), CHUNK_SIZE)][:MAX_CHUNKS]
        
        print(f"Processing {len(chunks)} chunks for LLM extraction...")
        
        combined_results = {
            "entities": [],
            "relationships": []
        }

        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}")
            
            max_retries = 3
            retry_delay = 1
            
            for attempt in range(max_retries):
                try:
                    chunk_instruction = f"""
                    Analyze this chunk ({i+1}/{len(chunks)}) of content and extract key dog health information.
                    Focus only on new information not previously mentioned:
                    1. Health Conditions
                    2. Symptoms
                    3. Treatments
                    4. Breed-Specific Information
                    5. Lifestyle
                    6. Care
                    7. Prevention
                    Keep descriptions brief and focused.
                    """
                    
                    knowledge_graph_strategy.instruction = chunk_instruction
                    
                    llm_res = await crawler.arun(
                        url=url,
                        content=chunk,
                        extraction_strategy=knowledge_graph_strategy,
                        cache_mode=CacheMode.BYPASS
                    )

                    # Handle the response
                    if hasattr(llm_res, 'extracted_content') and isinstance(llm_res.extracted_content, dict):
                        extracted_content = llm_res.extracted_content
                        
                        # Merge entities and relationships
                        if "entities" in extracted_content:
                            combined_results["entities"].extend(
                                [e for e in extracted_content["entities"] 
                                 if e not in combined_results["entities"]]
                            )
                        if "relationships" in extracted_content:
                            combined_results["relationships"].extend(
                                [r for r in extracted_content["relationships"] 
                                 if r not in combined_results["relationships"]]
                            )
                        
                        break  # Success, exit retry loop
                    
                except Exception as e:
                    if "rate_limit_exceeded" in str(e) and attempt < max_retries - 1:
                        print(f"Rate limit hit, waiting {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        print(f"Error processing chunk {i+1}: {str(e)}")
                        break

            await asyncio.sleep(1)  # Pause between chunks

        # Store the combined results
        if combined_results["entities"] or combined_results["relationships"]:
            # Generate graph metrics
            graph_metrics = analyze_graph_structure(combined_results)
            
            # Calculate embeddings for semantic search
            context_embeddings = generate_embeddings(json.dumps(combined_results))
            
            # Prepare metadata
            metadata = {
                "model_version": "gpt-4",
                "chunks_processed": len(chunks),
                "total_entities": len(combined_results["entities"]),
                "total_relationships": len(combined_results["relationships"]),
                "graph_metrics": graph_metrics,
                "source_quality": calculate_source_reliability(url),
                "context_embeddings": context_embeddings,
                "entity_types": list(set(e.get("type") for e in combined_results["entities"])),
                "relationship_types": list(set(r.get("relation_type") for r in combined_results["relationships"]))
            }

            # Store as JSON file
            file_path = store_knowledge_graph_as_json(
                graph_data=combined_results,
                url=url,
                source_metadata=metadata
            )
            print(f"Knowledge graph stored at: {file_path}")

    except Exception as e:
        print(f"Error in chunk processing: {str(e)}")
        raise e

def calculate_jsoncss_confidence(result) -> float:
    """Calculate confidence score for JSON-CSS extraction"""
    # Implement confidence calculation logic
    return 1.0  # Placeholder

def calculate_llm_confidence(result) -> float:
    """Calculate confidence score for LLM extraction"""
    # Implement confidence calculation logic
    return 1.0  # Placeholder

def calculate_cosine_confidence(result) -> float:
    """Calculate confidence score for Cosine similarity extraction"""
    # Implement confidence calculation logic
    return 1.0  # Placeholder

def calculate_source_authority(url: str, domain_weights: Dict[str, float] = None) -> float:
    """
    Calculate the authority score of the source URL.
    Args:
        url: The URL to analyze
        domain_weights: Dictionary mapping domain types to their authority weights
    """
    try:
        if domain_weights is None:
            domain_weights = {
                'edu': 0.85,
                'gov': 0.9,
                'org': 0.7,
                'com': 0.5
            }
            
        score = 0.5  # Base score
        
        # Check TLD
        for domain, weight in domain_weights.items():
            if url.lower().endswith(f'.{domain}'):
                score = weight
                break
        
        # HTTPS security
        if url.startswith('https://'):
            score += 0.1
            
        # Normalize final score
        return min(max(score, 0.0), 1.0)
        
    except Exception as e:
        print(f"Error calculating source authority: {str(e)}")
        return 0.5

def check_for_citations(content: str, citation_patterns: List[str] = None) -> bool:
    """
    Check if the content contains citations based on provided patterns.
    Args:
        content: The content to check
        citation_patterns: List of regex patterns to identify citations
    """
    if citation_patterns is None:
        citation_patterns = [
            r'\[\d+\]',  # [1], [2], etc.
            r'\(\d{4}\)',  # (2023), (2024), etc.
            r'cited in',
            r'according to',
            r'et al\.',
            r'reference[s]?:',
            r'source[s]?:'
        ]
    
    try:
        return any(re.search(pattern, content, re.I) for pattern in citation_patterns)
    except Exception as e:
        print(f"Error checking citations: {str(e)}")
        return False

def calculate_content_completeness(content: str, required_fields: List[str] = None) -> float:
    """
    Calculate how complete the extracted content is based on required fields.
    Args:
        content: The content to analyze
        required_fields: List of field names that should be present
    """
    if required_fields is None:
        required_fields = ['title', 'description', 'content']
    
    try:
        if isinstance(content, str):
            # For string content, check if it's JSON
            try:
                content_dict = json.loads(content)
            except json.JSONDecodeError:
                # If not JSON, check for basic content presence
                return 1.0 if len(content.strip()) > 0 else 0.0
        else:
            content_dict = content
            
        # Check for required fields
        fields_present = sum(1 for field in required_fields if field in content_dict)
        return fields_present / len(required_fields)
    except Exception as e:
        print(f"Error calculating content completeness: {str(e)}")
        return 0.0

def generate_embeddings(text: str) -> List[float]:
    """
    Generate embeddings for text using sentence-transformers.
    """
    try:
        from sentence_transformers import SentenceTransformer
        
        # Initialize the model (this will download it if not present)
        model = SentenceTransformer('all-mpnet-base-v2')
        
        # Generate embeddings
        embeddings = model.encode([text])[0].tolist()
        
        return embeddings
    except Exception as e:
        print(f"Error generating embeddings: {str(e)}")
        return []

def analyze_graph_structure(graph_data: dict) -> dict:
    """
    Analyzes the structure and quality of the knowledge graph.
    """
    try:
        metrics = {
            "entity_counts": {},
            "relationship_metrics": {},
            "connectivity_scores": {},
            "completeness_score": 0.0,
            "coherence_score": 0.0
        }
        
        # Entity analysis
        total_entities = 0
        for entity_type, entities in graph_data.get("entities", {}).items():
            if isinstance(entities, list):
                count = len(entities)
                total_entities += count
                metrics["entity_counts"][entity_type] = {
                    "count": count,
                    "avg_description_length": sum(len(e.get("description", "")) for e in entities) / count if count > 0 else 0
                }
        
        # Relationship analysis
        relationships = graph_data.get("relationships", [])
        total_relationships = len(relationships)
        
        # Calculate relationship type distribution
        relationship_types = {}
        for rel in relationships:
            rel_type = rel.get("relation_type", "unknown")
            relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
        
        metrics["relationship_metrics"] = {
            "total_count": total_relationships,
            "type_distribution": relationship_types,
            "avg_description_length": sum(len(r.get("description", "")) for r in relationships) / total_relationships if total_relationships > 0 else 0
        }
        
        # Connectivity analysis
        if total_entities > 1:
            max_possible_relationships = total_entities * (total_entities - 1) / 2
            metrics["connectivity_scores"] = {
                "density": total_relationships / max_possible_relationships if max_possible_relationships > 0 else 0,
                "avg_relationships_per_entity": total_relationships / total_entities if total_entities > 0 else 0
            }
        
        # Completeness score (check for required fields)
        required_entity_fields = {"name", "description"}
        required_relationship_fields = {"entity1", "entity2", "description", "relation_type"}
        
        entity_completeness = []
        for entities in graph_data.get("entities", {}).values():
            for entity in entities:
                if isinstance(entity, dict):
                    fields_present = len(set(entity.keys()) & required_entity_fields)
                    entity_completeness.append(fields_present / len(required_entity_fields))
        
        relationship_completeness = []
        for rel in relationships:
            if isinstance(rel, dict):
                fields_present = len(set(rel.keys()) & required_relationship_fields)
                relationship_completeness.append(fields_present / len(required_relationship_fields))
        
        metrics["completeness_score"] = (
            (sum(entity_completeness) / len(entity_completeness) if entity_completeness else 0) +
            (sum(relationship_completeness) / len(relationship_completeness) if relationship_completeness else 0)
        ) / 2
        
        # Coherence score (check for logical consistency)
        metrics["coherence_score"] = calculate_coherence_score(graph_data)
        
        return metrics
    except Exception as e:
        print(f"Error analyzing graph structure: {str(e)}")
        return {
            "entity_counts": {},
            "relationship_metrics": {},
            "connectivity_scores": {},
            "completeness_score": 0.0,
            "coherence_score": 0.0
        }

def calculate_coherence_score(graph_data: dict) -> float:
    """
    Calculate the logical coherence of the knowledge graph.
    """
    try:
        score = 0.0
        checks_passed = 0
        total_checks = 0
        
        # Check 1: All relationships reference existing entities
        entity_names = set()
        for entities in graph_data.get("entities", {}).values():
            for entity in entities:
                if isinstance(entity, dict):
                    entity_names.add(entity.get("name"))
        
        total_checks += 1
        valid_relationships = 0
        relationships = graph_data.get("relationships", [])
        
        for rel in relationships:
            if isinstance(rel, dict):
                entity1 = rel.get("entity1", {}).get("name")
                entity2 = rel.get("entity2", {}).get("name")
                if entity1 in entity_names and entity2 in entity_names:
                    valid_relationships += 1
        
        if relationships:
            relationship_score = valid_relationships / len(relationships)
            score += relationship_score
            checks_passed += 1
        
        # Check 2: No duplicate entities
        total_checks += 1
        unique_entities = set()
        duplicate_count = 0
        
        for entities in graph_data.get("entities", {}).values():
            for entity in entities:
                if isinstance(entity, dict):
                    name = entity.get("name")
                    if name in unique_entities:
                        duplicate_count += 1
                    unique_entities.add(name)
        
        if unique_entities:
            uniqueness_score = 1 - (duplicate_count / len(unique_entities))
            score += uniqueness_score
            checks_passed += 1
        
        # Check 3: Relationship type consistency
        total_checks += 1
        relationship_types = set()
        consistent_types = 0
        
        for rel in relationships:
            if isinstance(rel, dict):
                rel_type = rel.get("relation_type")
                if rel_type:
                    relationship_types.add(rel_type)
                    # Check if relationship type matches the entities it connects
                    entity1_type = next((k for k, v in graph_data.get("entities", {}).items() 
                                       if any(e.get("name") == rel.get("entity1", {}).get("name") for e in v)), None)
                    entity2_type = next((k for k, v in graph_data.get("entities", {}).items() 
                                       if any(e.get("name") == rel.get("entity2", {}).get("name") for e in v)), None)
                    if entity1_type and entity2_type:
                        consistent_types += 1
        
        if relationships:
            consistency_score = consistent_types / len(relationships)
            score += consistency_score
            checks_passed += 1
        
        # Calculate final coherence score
        return score / total_checks if total_checks > 0 else 0.0
    
    except Exception as e:
        print(f"Error calculating coherence score: {str(e)}")
        return 0.0

def calculate_source_reliability(url: str) -> float:
    """
    Calculate a reliability score for the source URL based on various factors.
    """
    try:
        score = 0.5  # Base score
        
        # Domain authority check
        reliable_domains = {
            'vet': 0.9,
            'edu': 0.85,
            'gov': 0.9,
            'org': 0.7,
            'com': 0.5
        }
        
        # Check TLD
        for domain, weight in reliable_domains.items():
            if url.lower().endswith(f'.{domain}'):
                score = weight
                break
        
        # Additional checks could include:
        # - HTTPS security
        if url.startswith('https://'):
            score += 0.1
            
        # - Known veterinary or medical sites
        trusted_sites = [
            'vetmed.', 'avma.org', 'akc.org', 'vet.', 
            'merckvetmanual.', 'petmd.', 'vcahospitals.'
        ]
        if any(site in url.lower() for site in trusted_sites):
            score += 0.2
        
        # Normalize final score to 0-1 range
        return min(max(score, 0.0), 1.0)
        
    except Exception as e:
        print(f"Error calculating source reliability: {str(e)}")
        return 0.5

def determine_topic_category(clusters: dict, filters: dict) -> str:
    """
    Determine the primary topic category based on cluster content and filters.
    """
    try:
        category_scores = {}
        
        for category, keywords in filters.items():
            score = 0
            keywords_list = keywords.split()
            
            # Convert clusters to string if it's not already a dict
            if isinstance(clusters, str):
                text = clusters.lower()
                score = sum(1 for keyword in keywords_list if keyword.lower() in text)
            else:
                for cluster in clusters.values():
                    if isinstance(cluster, dict) and 'content' in cluster:
                        text = str(cluster['content']).lower()
                        score += sum(1 for keyword in keywords_list if keyword.lower() in text)
            
            category_scores[category] = score
        
        # Return the category with highest score, or 'general' if all scores are 0
        max_category = max(category_scores.items(), key=lambda x: x[1])
        return max_category[0] if max_category[1] > 0 else 'general'
    except Exception as e:
        print(f"Error determining topic category: {str(e)}")
        return 'general'

def process_health_clusters(content: str) -> dict:
    """
    Process and enhance the health-related clusters.
    Modified to handle string input instead of dict.
    """
    try:
        if not content:
            return {}
            
        # Create a single cluster from the content
        processed_clusters = {
            'main_cluster': {
                'content': content,
                'health_score': calculate_health_relevance_score(content),
                'metadata': {
                    'processed_at': datetime.now().isoformat(),
                    'cluster_size': len(content),
                    'topic_category': determine_health_category(content)
                }
            }
        }
        return processed_clusters
    except Exception as e:
        print(f"Error processing health clusters: {str(e)}")
        return {}

def calculate_health_relevance_score(content) -> float:
    """Calculate relevance score for health content."""
    # Placeholder implementation
    return 0.8

def determine_health_category(content) -> str:
    """Determine the health category of the content."""
    # Placeholder implementation
    return "general_health"

def generate_cluster_embeddings(clusters: dict) -> dict:
    """Generate embeddings for clusters."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-mpnet-base-v2')
        
        embeddings = {}
        for key, cluster in clusters.items():
            if isinstance(cluster, dict) and 'content' in cluster:
                text = str(cluster['content'])
                embeddings[key] = model.encode([text])[0].tolist()
        return embeddings
    except Exception as e:
        print(f"Error generating cluster embeddings: {str(e)}")
        return {}

def calculate_health_relevance(clusters: dict, filters: dict) -> dict:
    """Calculate health relevance scores for clusters."""
    scores = {
        'overall': 0.0,
        'categories': {}
    }
    
    try:
        for category, keywords in filters.items():
            category_score = 0.0
            for cluster in clusters.values():
                if isinstance(cluster, dict) and 'content' in cluster:
                    content = str(cluster['content']).lower()
                    keyword_matches = sum(1 for keyword in keywords.split() if keyword.lower() in content)
                    category_score += keyword_matches
            scores['categories'][category] = min(1.0, category_score / 10)
        
        scores['overall'] = sum(scores['categories'].values()) / len(filters)
        return scores
    except Exception as e:
        print(f"Error calculating health relevance: {str(e)}")
        return {'overall': 0.0, 'categories': {}}

def db_insert_llm_internal(page_id: int, graph_data: dict, raw_text: str = None,
                          entity_types: str = None, relationship_types: str = None,
                          context_embeddings: list = None, metadata: str = None) -> None:
    """Insert LLM extraction results into internal database."""
    # Placeholder implementation - replace with your actual database logic
    print(f"Storing LLM results for page {page_id}")
    # Add your database insertion logic here
    pass
