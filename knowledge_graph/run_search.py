import uuid
import os
from neo4j import GraphDatabase
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv  
from pydantic import BaseModel  
from collections import defaultdict  
from neo4j_graphrag.retrievers import QdrantNeo4jRetriever  
from langchain_ollama import ChatOllama, OllamaEmbeddings

load_dotenv('.env')
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_username = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")
neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
qdrant_client = QdrantClient(host="localhost", port=6333)

collection_name = "grag_test"  # Qdrant 集合名稱
vector_dimension = 1024  # 嵌入向量的維度

# model clients
gemma3 = ChatOllama(base_url="http://10.65.51.226:11434",model="gemma3:12b",temperature=0.2)
gemma3_json = gemma3.bind(format="json")
emb_model = OllamaEmbeddings(base_url="http://10.65.51.226:11434", model="bge-m3:567m")
# Output embedding dimension size of 1024

def ollama_embeddings(text):
    single_vector = emb_model.embed_query(text)
    return single_vector

def retriever_search(neo4j_driver, qdrant_client, collection_name, query):
    """使用 QdrantNeo4jRetriever 進行檢索"""
    retriever = QdrantNeo4jRetriever(
        driver=neo4j_driver,  # Neo4j 驅動程式
        client=qdrant_client,  # Qdrant 客戶端
        collection_name=collection_name,  # Qdrant 集合名稱
        id_property_external="id",  # Qdrant payload 中的 ID 屬性名稱
        id_property_neo4j="id",  # Neo4j 節點中的 ID 屬性名稱
    )
    results = retriever.search(query_vector=ollama_embeddings(query), top_k=5)
    return results

def fetch_related_graph(neo4j_client, entity_ids):
    """從 Neo4j 中獲取與給定實體 ID 相關的子圖"""
    # Cypher 查詢，找到與給定實體 ID 相關的兩層深的節點和關係
    query = """
    MATCH (e:Entity)-[r1]-(n1)-[r2]-(n2)
    WHERE e.id IN $entity_ids
    RETURN e, r1 as r, n1 as related, r2, n2
    UNION
    MATCH (e:Entity)-[r]-(related)
    WHERE e.id IN $entity_ids
    RETURN e, r, related, null as r2, null as n2
    """
    # 建立一個 Neo4j 會話並執行查詢
    with neo4j_client.session() as session:
        result = session.run(query, entity_ids=entity_ids)
        subgraph = []  # 用於儲存子圖的列表
        # 遍歷查詢結果
        for record in result:
            # 將找到的實體、關係和相關節點新增至子圖列表
            subgraph.append({
                "entity": record["e"],
                "relationship": record["r"],
                "related_node": record["related"]
            })
            # 如果存在第二層的關係和節點，也將其新增至子圖列表
            if record["r2"] and record["n2"]:
                subgraph.append({
                    "entity": record["related"],
                    "relationship": record["r2"],
                    "related_node": record["n2"]
                })
    return subgraph

# 定義一個函式，將子圖格式化為節點和邊的列表
def format_graph_context(subgraph):
    nodes = set()  # 用於儲存唯一的節點名稱
    edges = []  # 用於儲存邊的描述

    # 遍歷子圖中的每個條目
    for entry in subgraph:
        entity = entry["entity"]
        related = entry["related_node"]
        relationship = entry["relationship"]

        # 將節點名稱新增至集合中
        nodes.add(entity["name"])
        nodes.add(related["name"])

        # 將邊的描述新增至列表中
        edges.append(f"{entity['name']} {relationship['type']} {related['name']}")

    # 返回包含節點和邊的字典
    return {"nodes": list(nodes), "edges": edges}

# 定義一個函式，使用圖形上下文和使用者查詢來生成答案
def graphRAG_run(graph_context, user_query):
    # 將節點和邊的列表轉換為字串
    nodes_str = ", ".join(graph_context["nodes"])
    edges_str = "; ".join(graph_context["edges"])
    # 建立提示，包含知識圖譜的上下文和使用者查詢
    prompt = f"""
    You are an intelligent assistant with access to the following knowledge graph:

    Nodes: {nodes_str}

    Edges: {edges_str}

    Using this graph, Answer the following question:

    User Query: "{user_query}"
    """
    try:
        resp = gemma3.invoke(prompt)
        return resp.content
    except Exception as e:
        return f"Error querying LLM: {str(e)}"
    
if __name__ == "__main__":
    query = "如何治療白喉?"
    print("Starting retriever search...")
    # 進行檢索
    retriever_result = retriever_search(neo4j_driver, qdrant_client, collection_name, query)
    print("Retriever results:", retriever_result)
    
    print("Extracting entity IDs...")
    # 從檢索結果中提取實體 ID
    entity_ids = [item.content.split("'id': '")[1].split("'")[0] for item in retriever_result.items]
    print("Entity IDs:", entity_ids)
    
    print("Fetching related graph...")
    # 獲取相關的子圖
    subgraph = fetch_related_graph(neo4j_driver, entity_ids)
    print("Subgraph:", subgraph)
    
    print("Formatting graph context...")
    # 格式化圖形上下文
    graph_context = format_graph_context(subgraph)
    print("Graph context:", graph_context)
    
    print("Running GraphRAG...")
    # 執行 GraphRAG 以生成答案
    answer = graphRAG_run(graph_context, query)
    print("Final Answer:", answer)
