
import os
from neo4j import GraphDatabase
from qdrant_client import QdrantClient, models
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.chat_models import ChatOllama
from typing import List, Optional
import uuid

# --- Configuration ---
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION_NAME = "graph_rag_collection"
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")

# --- Initialize Clients ---
qdrant_client = QdrantClient(url=QDRANT_URL)
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
llm = ChatOllama(temperature=0, model="gemma3:4b")

# --- Data Model ---
class Node(BaseModel):
    id: str
    type: str
    properties: Optional[dict] = Field(default_factory=dict)

class Edge(BaseModel):
    source: str
    target: str
    type: str
    properties: Optional[dict] = Field(default_factory=dict)

class KnowledgeGraph(BaseModel):
    nodes: List[Node]
    edges: List[Edge]

# --- Main Script ---
def main():
    """
    Main function to run the Graph RAG pipeline.
    """
    print("Starting Graph RAG pipeline...")

    # 1. Setup Qdrant Collection
    setup_qdrant_collection()

    # 2. Ingestion
    raw_data = "Cardiovascular disease is a class of diseases that involve the heart or blood vessels. It includes coronary artery diseases (CAD) such as angina and myocardial infarction (commonly known as a heart attack)."
    ingest_data(raw_data)

    # 3. Retrieval and Generation
    query = "What is cardiovascular disease?"
    response = retrieve_and_generate(query)

    print(f"\nQuery: {query}")
    print(f"Response: {response}")

    # Close the Neo4j driver
    neo4j_driver.close()
    print("\nPipeline finished.")

if __name__ == "__main__":
    main()
