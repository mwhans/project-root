"""
General purpose content extraction and knowledge graph generation module.
"""
import asyncio
from typing import Optional, List, Dict
import json
from datetime import datetime
import os
from crawl4ai import AsyncWebCrawler, CacheMode
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
        
        # First get raw content
        base_result = await crawler.arun(url=url, cache_mode=CacheMode.BYPASS)
        if not base_result.success:
            print(f"Failed to fetch base content: {base_result.error_message}")
            return None
            
        # Prepare content for processing
        content_to_process = base_result.markdown or base_result.cleaned_html
        if not content_to_process:
            print("No content available for processing")
            return None
            
        # Run cosine strategy
        result = await crawler.arun(
            url=url,
            content=content_to_process,
            extraction_strategy=cosine_strategy,
            cache_mode=CacheMode.BYPASS
        )
        
        if not result.success:
            print(f"Cosine extraction failed: {result.error_message}")
            return None
            
        # Process the extracted clusters
        content = result.extracted_content
        if not content:
            print("No content extracted by Cosine strategy")
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
                    "url_count": len(urls),
                    "sentence_count": len(re.split(r'[.!?]+', cluster_text))
                }
            }
            
            processed_content["content_clusters"].append(processed_cluster)
            processed_content["metadata"]["total_words"] += processed_cluster["word_count"]
        
        print("\nCosine Preprocessing Results:")
        print(f"Total clusters found: {len(processed_content['content_clusters'])}")
        print(f"Total words processed: {processed_content['metadata']['total_words']}")
        
        return processed_content
        
    except Exception as e:
        print(f"Error in Cosine preprocessing: {str(e)}")
        traceback.print_exc()
        return None

async def create_llm_extraction_strategy(chunk_info: str, config: dict) -> LLMExtractionStrategy:
    """
    Create an LLM extraction strategy with configurable parameters
    Args:
        chunk_info: Information about the current chunk being processed
        config: Dictionary containing extraction configuration
    """
    return LLMExtractionStrategy(
        provider=config.get("provider", "openai/gpt-4"),
        api_token=OPENAI_API_KEY,
        schema=KnowledgeGraph.schema(),
        extraction_type="schema",
        instruction=config.get("instruction_template", "").format(
            chunk_info=chunk_info,
            entity_types=config.get("entity_types", []),
            guidelines=config.get("guidelines", [])
        ),
        chunk_token_threshold=config.get("chunk_token_threshold", 2000),
        verbose=True
    )

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
) -> int:
    """
    Performs a multi-step extraction flow for the given URL.
    Args:
        url: URL to process
        extraction_config: Configuration dictionary containing:
            - schema: Schema configuration for JsonCssExtraction
            - semantic_filter: Keywords for semantic filtering
            - llm_config: Configuration for LLM extraction
        use_jsoncss: Whether to use JsonCss extraction
        use_llm: Whether to use LLM extraction
        use_cosine: Whether to use Cosine preprocessing
    """
    print(f"\n{'='*50}")
    print(f"Starting extraction pipeline for URL: {url}")
    print(f"{'='*50}\n")
    
    url_id = get_or_create_url(url)
    preprocessed_content = None
    raw_content = None
    
    async with AsyncWebCrawler() as crawler:
        # First get raw content for fallback
        try:
            base_result = await crawler.arun(url=url, cache_mode=CacheMode.BYPASS)
            if base_result.success:
                raw_content = base_result.markdown or base_result.cleaned_html
                print("Successfully retrieved raw content as fallback")
        except Exception as e:
            print(f"Warning: Failed to get raw content: {str(e)}")
        
        # Step 1: JSON CSS Extraction
        if use_jsoncss:
            print("\n=== Step 1: JSON CSS Extraction ===")
            try:
                jsoncss_strategy = JsonCssExtractionStrategy(
                    schema=get_base_schema(extraction_config.get("schema", {})),
                    extraction_type="schema"
                )
                
                jsoncss_res = await crawler.arun(
                    url=url,
                    extraction_strategy=jsoncss_strategy,
                    cache_mode=CacheMode.BYPASS
                )
                
                if jsoncss_res.extracted_content:
                    metadata = {
                        "extraction_timestamp": datetime.now().isoformat(),
                        "content_length": len(str(jsoncss_res.extracted_content)),
                        "schema_version": "1.0",
                        "source_authority": calculate_source_authority(url)
                    }
                    
                    print("\nJSON CSS Extraction Results:")
                    print("-" * 30)
                    pprint(jsoncss_res.extracted_content)
                    print("\nMetadata:")
                    pprint(metadata)
                    
                    store_extraction(
                        url_id=url_id,
                        extraction_type='json',
                        content=jsoncss_res.extracted_content,
                        metadata=metadata,
                        config=jsoncss_strategy.__dict__
                    )
            except Exception as e:
                print(f"JSON extraction failed: {str(e)}")
        
        # Step 2: Cosine Preprocessing
        if use_cosine:
            print("\n=== Step 2: Cosine Preprocessing ===")
            try:
                preprocessed_content = await preprocess_with_cosine(
                    crawler, 
                    url, 
                    extraction_config.get("semantic_filter", "")
                )
                if not preprocessed_content:
                    print("Warning: Cosine preprocessing failed or returned no content")
                    print("Proceeding with raw content for LLM extraction")
            except Exception as e:
                print(f"Cosine preprocessing failed: {str(e)}")
                print("Proceeding with raw content for LLM extraction")
        
        # Step 3: LLM Extraction
        if use_llm:
            print("\n=== Step 3: LLM Extraction ===")
            try:
                # Prepare content for LLM
                content_for_llm = ""
                if preprocessed_content and preprocessed_content.get("content_clusters"):
                    sections = []
                    for cluster in preprocessed_content["content_clusters"]:
                        section = f"""
SECTION: {', '.join(cluster.get('categories', ['general']))}
CONTENT: {cluster['text']}
URLS: {', '.join(cluster.get('urls', []))}
---"""
                        sections.append(section)
                    content_for_llm = "\n\n".join(sections)
                elif raw_content:
                    print("Using raw content for LLM extraction")
                    content_for_llm = f"CONTENT: {raw_content}"

                if not content_for_llm:
                    print("No content available for LLM extraction")
                    return url_id

                # Chunk the content if it's too long
                max_chunk_size = extraction_config.get("llm_config", {}).get("max_chunk_size", 32000)
                content_chunks = [content_for_llm[i:i + max_chunk_size] 
                                for i in range(0, len(content_for_llm), max_chunk_size)]

                print(f"Processing {len(content_chunks)} content chunks...")

                # Process each chunk and combine results
                combined_graph = KnowledgeGraph()
                
                for i, chunk in enumerate(content_chunks):
                    print(f"\nProcessing chunk {i+1}/{len(content_chunks)}")
                    
                    llm_strategy = create_llm_extraction_strategy(
                        chunk_info=f"chunk {i+1}/{len(content_chunks)}",
                        config=extraction_config.get("llm_config", {})
                    )

                    try:
                        llm_res = await crawler.arun(
                            url=url,
                            content=chunk,
                            extraction_strategy=llm_strategy,
                            cache_mode=CacheMode.BYPASS
                        )

                        if llm_res.extracted_content:
                            try:
                                # Process the content
                                if isinstance(llm_res.extracted_content, dict) and 'choices' in llm_res.extracted_content:
                                    content = llm_res.extracted_content['choices'][0]['message']['content']
                                else:
                                    content = llm_res.extracted_content

                                # Store as JSON file
                                file_path = store_knowledge_graph_as_json(
                                    graph_data=content,
                                    url=url,
                                    source_metadata=metadata
                                )
                                print(f"\nKnowledge graph stored successfully at: {file_path}")

                            except Exception as storage_error:
                                print(f"Error storing knowledge graph: {str(storage_error)}")
                                traceback.print_exc()

                    except Exception as e:
                        print(f"Error processing chunk {i+1}: {str(e)}")
                        continue

                # Store the combined results
                if combined_graph.entities or combined_graph.relationships:
                    graph_metrics = analyze_graph_structure(combined_graph.dict())
                    context_embeddings = generate_embeddings(json.dumps(combined_graph.dict()))
                    
                    metadata = {
                        "model_version": extraction_config.get("llm_config", {}).get("provider", "gpt-4"),
                        "content_source": "preprocessed" if preprocessed_content else "raw",
                        "chunks_processed": len(content_chunks),
                        "total_entities": len(combined_graph.entities),
                        "total_relationships": len(combined_graph.relationships),
                        "graph_metrics": graph_metrics,
                        "source_quality": calculate_source_reliability(url),
                        "context_embeddings": context_embeddings,
                        "entity_types": list(set(e.type for e in combined_graph.entities)),
                        "relationship_types": list(set(r.relation_type for r in combined_graph.relationships)),
                        "extraction_timestamp": datetime.now().isoformat()
                    }

                    try:
                        file_path = store_knowledge_graph_as_json(
                            graph_data=combined_graph.dict(),
                            url=url,
                            source_metadata=metadata
                        )
                        print(f"\nKnowledge graph stored successfully at: {file_path}")
                    except Exception as storage_error:
                        print(f"Error storing knowledge graph: {str(storage_error)}")

                    store_extraction(
                        url_id=url_id,
                        extraction_type='knowledge_graph',
                        content=combined_graph.dict(),
                        metadata=metadata,
                        config=llm_strategy.__dict__
                    )

            except Exception as e:
                print(f"LLM extraction failed: {str(e)}")
                traceback.print_exc()

        print("\n=== Extraction Pipeline Complete ===")
        return url_id

def store_knowledge_graph_as_json(graph_data: dict, url: str, source_metadata: dict = None) -> str:
    """
    Store a knowledge graph as a JSON file with metadata.
    """
    try:
        # Create knowledge-graph directory inside companion_app
        current_dir = os.path.dirname(os.path.abspath(__file__))
        kg_dir = os.path.join(current_dir, 'knowledge-graph')
        os.makedirs(kg_dir, exist_ok=True)
        
        print("\nDebug - Processing raw response...")
        
        # Handle the raw GPT response
        if isinstance(graph_data, str):
            try:
                # First try to parse as JSON
                data = json.loads(graph_data)
                
                # If this is a GPT response, extract the content
                if isinstance(data, dict) and 'choices' in data:
                    content = data['choices'][0]['message']['content']
                    
                    # Extract JSON between <blocks> tags
                    import re
                    blocks_match = re.search(r'<blocks>\n(.*?)\n</blocks>', content, re.DOTALL)
                    if blocks_match:
                        graph_data = json.loads(blocks_match.group(1))
                    else:
                        graph_data = json.loads(content)
                else:
                    graph_data = data
                    
            except json.JSONDecodeError:
                # If initial JSON parse fails, try to extract JSON between <blocks> tags
                import re
                blocks_match = re.search(r'<blocks>\n(.*?)\n</blocks>', graph_data, re.DOTALL)
                if blocks_match:
                    graph_data = json.loads(blocks_match.group(1))
                else:
                    raise ValueError("Could not extract valid JSON from response")

        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        url_hash = hashlib.md5(url.encode()).hexdigest()[:10]
        filename = f"kg_{timestamp}_{url_hash}.json"
        file_path = os.path.join(kg_dir, filename)
        
        # Write to file with pretty printing
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nKnowledge graph successfully stored at: {file_path}")
        return file_path
        
    except Exception as e:
        print(f"Error storing knowledge graph as JSON: {str(e)}")
        print("Debug - Full error:")
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
