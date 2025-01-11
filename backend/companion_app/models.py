from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class Entity(BaseModel):
    name: str = Field(..., description="Name of the entity")
    type: str = Field(..., description="Type of the entity")
    description: Optional[str] = Field(None, description="Description of the entity")
    urls: Optional[List[str]] = Field(default_factory=list, description="Related URLs")
    metadata: Optional[Dict] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        schema_extra = {
            "example": {
                "name": "Machine Learning",
                "type": "concept",
                "description": "A field of AI focused on learning from data",
                "urls": ["https://example.com/ml"],
                "metadata": {"field": "AI"}
            }
        }

class Relationship(BaseModel):
    source: str = Field(..., description="Source entity identifier")
    target: str = Field(..., description="Target entity identifier")
    relation_type: str = Field(..., description="Type of relationship")
    description: Optional[str] = Field(None, description="Description of the relationship")
    urls: Optional[List[str]] = Field(default_factory=list, description="Related URLs")
    metadata: Optional[Dict] = Field(default_factory=dict, description="Additional metadata")
    entity1: Optional[str] = Field(None, description="First entity name")
    entity2: Optional[str] = Field(None, description="Second entity name")

    class Config:
        schema_extra = {
            "example": {
                "source": "e1",
                "target": "e2",
                "relation_type": "includes",
                "description": "Machine Learning includes Neural Networks",
                "urls": ["https://example.com/ml-nn"],
                "metadata": {"confidence": 0.9},
                "entity1": "Machine Learning",
                "entity2": "Neural Networks"
            }
        }

class KnowledgeGraph(BaseModel):
    entities: List[Entity] = Field(default_factory=list, description="List of entities in the graph")
    relationships: List[Relationship] = Field(default_factory=list, description="List of relationships between entities")

    class Config:
        schema_extra = {
            "example": {
                "entities": [
                    {
                        "name": "Machine Learning",
                        "type": "concept",
                        "description": "A field of AI"
                    }
                ],
                "relationships": [
                    {
                        "source": "e1",
                        "target": "e2",
                        "relation_type": "includes"
                    }
                ]
            }
        }

class KnowledgeGraphResponse(BaseModel):
    url: str = Field(..., description="Source URL of the knowledge graph")
    knowledge_graph: KnowledgeGraph = Field(..., description="The knowledge graph data")
    metadata: Dict = Field(default_factory=dict, description="Additional metadata about the graph")
    extraction_timestamp: str = Field(..., description="When the graph was extracted")
    source_authority: float = Field(..., description="Authority score of the source")
    context: Optional[Dict] = Field(default_factory=dict, description="Contextual information")

    class Config:
        schema_extra = {
            "example": {
                "url": "https://example.com/article",
                "knowledge_graph": {
                    "entities": [],
                    "relationships": []
                },
                "metadata": {"domain": "technology"},
                "extraction_timestamp": "2024-01-11T12:00:00",
                "source_authority": 0.8,
                "context": {"topic": "AI"}
            }
        } 