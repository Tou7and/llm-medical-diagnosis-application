from neo4j import GraphDatabase
from qdrant_client import QdrantClient

def test_neo4j_connection():
    URI = "neo4j://localhost:7687"
    AUTH = ("neo4j", "password")
    driver = GraphDatabase.driver(URI, auth=AUTH)

    try:
        with driver.session() as session:
            result = session.run("RETURN 1 AS number")
            record = result.single()
            print("Neo4j Connection successful, result:", record["number"])
    except Exception as e:
        print("Connection failed:", e)
    finally:
        driver.close()

def test_qdrant_connection():
    client = QdrantClient(host="localhost", port=6333)
    collections_response = client.get_collections()

    print("Available collections:")
    for collection in collections_response.collections:
        print(f"- {collection.name}")

    print("Qdrant Connection successful")

if __name__ == "__main__":
    test_neo4j_connection()
    test_qdrant_connection()

