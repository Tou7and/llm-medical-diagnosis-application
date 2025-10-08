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

# model clients
gemma3 = ChatOllama(base_url="http://10.65.51.226:11434",model="gemma3:12b",temperature=0.2)
gemma3_json = gemma3.bind(format="json")
emb_model = OllamaEmbeddings(base_url="http://10.65.51.226:11434", model="bge-m3:567m")
# Output embedding dimension size of 1024

class single(BaseModel):
    """定義單一圖形關係的 Pydantic 模型"""
    node: str  # 來源節點
    target_node: str  # 目標節點
    relationship: str  # 關係類型

class GraphComponents(BaseModel):
    """定義包含多個圖形關係的 Pydantic 模型"""
    graph: list[single]  # 圖形關係列表

parser_prompt = """
You are a precise graph relationship extractor. 
Extract all relationships from the text and format them as a JSON object with this exact structure:
{
    "graph": [
        {"node": "Person/Entity", 
         "target_node": "Related Entity", 
         "relationship": "Type of Relationship"},
        ...more relationships...
    ]
}
Include ALL relationships mentioned in the text, including 
implicit ones. Be thorough and precise.

Here is the text:
"""
medparser_prompt = """
You are a medical graph relationship extractor, focus on extracting relations of the following entities: anatomy, impression, and observation.
Extract medical relationships from the text and format them as a JSON object with this exact structure:
{
    "graph": [
        {"node": "Entity", 
         "target_node": "Related Entity", 
         "relationship": "Type of Relationship"},
        ...more relationships...
    ]
}
Only include relations of target medical entities.
Here is the text:
"""
def gemma3_llm_parser(user_prompt):
    resp = gemma3_json.invoke(parser_prompt+user_prompt)
    return GraphComponents.model_validate_json(resp.content)

def extract_graph_components(raw_data):
    parsed_response = gemma3_llm_parser(raw_data)
    parsed_response = parsed_response.graph

    nodes = {}
    relationships = []

    # 遍歷解析出的每個關係
    for entry in parsed_response:
        node = entry.node  # 來源節點
        target_node = entry.target_node  # 目標節點
        relationship = entry.relationship  # 關係類型

        # 如果節點尚未存在於 nodes 字典中，則為其分配一個唯一的 UUID
        if node not in nodes:
            nodes[node] = str(uuid.uuid4())

        if target_node and target_node not in nodes:
            nodes[target_node] = str(uuid.uuid4())

        # 如果存在目標節點和關係，則將關係新增至 relationships 列表
        if target_node and relationship:
            relationships.append({
                "source": nodes[node],  # 來源節點的 ID
                "target": nodes[target_node],  # 目標節點的 ID
                "type": relationship  # 關係類型
            })

    return nodes, relationships

def ingest_to_neo4j(nodes, relationships):
    """將節點和關係匯入 Neo4j。"""
    with neo4j_driver.session() as session:
        for name, node_id in nodes.items():
            session.run(
                "CREATE (n:Entity {id: $id, name: $name})",  # Cypher 查詢，建立 id 和 name 的 Entity 節點
                id=node_id,
                name=name
            )

        for relationship in relationships:
            session.run(
                "MATCH (a:Entity {id: $source_id}), (b:Entity {id: $target_id}) "  # 找到來源和目標節點
                "CREATE (a)-[:RELATIONSHIP {type: $type}]->(b)",  # 建立一個帶有 type 屬性的 RELATIONSHIP 關係
                source_id=relationship["source"],
                target_id=relationship["target"],
                type=relationship["type"]
            )

    return nodes

def ollama_embeddings(text):
    single_vector = emb_model.embed_query(text)
    return single_vector

def create_collection(client, collection_name, vector_dimension):
    try:
        collection_info = client.get_collection(collection_name)
        print(f"Skipping creating collection; '{collection_name}' already exists.")
    except Exception as e:
        if 'Not found: Collection' in str(e):
            print(f"Collection '{collection_name}' not found. Creating it now...")

            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_dimension, distance=models.Distance.COSINE)
            )

            print(f"Collection '{collection_name}' created successfully.")
        else:
            print(f"Error while checking collection: {e}")

def ingest_to_qdrant(collection_name, raw_data, node_id_mapping):
    embeddings = [ollama_embeddings(paragraph) for paragraph in raw_data.split("\n")]

    qdrant_client.upsert(
        collection_name=collection_name,
        points=[
            {
                "id": str(uuid.uuid4()),  # 為每個點生成一個唯一的 ID
                "vector": embedding,  # 嵌入向量
                "payload": {"id": node_id}  # 附加的資料，包含對應的 Neo4j 節點 ID
            }
            for node_id, embedding in zip(node_id_mapping.values(), embeddings)
        ]
    )

def build_graph(raw_data):
    print("Creating collection...")
    collection_name = "grag_test"
    vector_dimension = 1024 
    create_collection(qdrant_client, collection_name, vector_dimension)
    print("Collection created/verified")
    
    print("Extracting graph components...")
    
    # 從原始資料中提取節點和關係
    nodes, relationships = extract_graph_components(raw_data)
    print("Nodes:", nodes)
    print("Relationships:", relationships)
    
    print("Ingesting to Neo4j...")
    # 將節點和關係匯入 Neo4j
    node_id_mapping = ingest_to_neo4j(nodes, relationships)
    print("Neo4j ingestion complete")
    
    print("Ingesting to Qdrant...")
    # 將資料匯入 Qdrant
    ingest_to_qdrant(collection_name, raw_data, node_id_mapping)
    print("Qdrant ingestion complete")

if __name__ == "__main__":
    raw_data = """白喉（Diphtheria）是由一種被稱為白喉棒狀桿菌的細菌所感染造成的[1]。症狀可以從輕微到嚴重[2]，且一般通常是於接觸到致病菌二到五天後開始出現症狀[1]。剛開始出現的症狀通常進展得較和緩，伴隨有喉嚨痛和發熱。而嚴重的病人其喉嚨會出現灰色或白色的斑塊[1][2] ，這些斑塊可以阻塞呼吸道並且讓患者在咳嗽時產生如同狗吠一樣的叫聲，稱為義膜性喉炎[2]。脖子會因為腫脹的淋巴結而部分腫大。另外也有一種形式的白喉會感染皮膚、眼睛或者生殖器官[1][2]。併發症包含有心肌炎、神經發炎、蛋白尿，還有因為血小板低下而造成的流血不止的狀況。心肌炎可能會導致心律不整，而神經發炎則可能導致癱瘓[1]。

白喉通常是經由直接接觸或是飛沫傳染[1][3]。也可以經由受到汙染的物品而擴散出去。有些白喉帶原的人可能沒有症狀，但仍有能力傳播疾病給其他人。白喉桿菌有三種分型，分別能造成不同嚴重程度的疾病。感染後的症狀通常是由細菌所製造的外毒素所引起的。觀察喉嚨的外觀並透過喉頭取樣培養可以幫助建立診斷。過去曾被感染過者未來仍舊有感染的機會[2]。

白喉疫苗（即所謂的白喉類毒素疫苗），對於預防感染是相當有效的，並且以不同的配方製成。白喉疫苗會和破傷風疫苗、百日咳疫苗共同施打，在孩童時期施打約三或四劑。每隔十年會再建議施打。施打後是否已具有免疫力會藉由抽血檢驗抗毒素抗體的量來確認。治療包括使用抗生素紅黴素和青黴素。對於那些已經暴露在感染源的人，這些抗生素或許也能用來預防感染[1]。嚴重感染的病患有時也需要接受氣管切開術[2]。

2013年官方正式報導有4,700個案例，相較於1980年將近100,000個案例已大幅下降[4]。然而據傳在1980年代，每年仍有將近一百萬的病例產生。目前的案例大多發生在撒哈拉以南的非洲、印度和印尼[2][5]。在2013年，死亡人數已從1990年的8,000名減少至3,330名[6]。在白喉流行的區域，主要感染者為孩童。因為廣泛的疫苗施打讓已開發國家中的白喉案例變得非常稀少 。在1980到2004年之間在美國只有57名案例被報導。被感染後死亡率大約在5%到10%之間。這種疾病早在西元五世紀就由希波克拉底所記載過。而白喉桿菌是在1882年由愛德恩·克雷伯發現[1]。"""

    build_graph(raw_data)
