"""
用兩個 LLM，一個專門處理 JSON 輸出，另一個處理純文字對話
format="json" 會強制模型輸出有效的 JSON，非常適合需要結構化輸出的節點
"""
import os
from langchain_ollama import ChatOllama

gemma3 = ChatOllama(base_url="http://10.65.51.226:11434",model="gemma3:4b",temperature=0.2)
gemma3_json = gemma3.bind(format="json")

text_llm = gemma3
json_llm = gemma3_json

if __name__ == "__main__":
    resp = text_llm.invoke("說一個電腦科學的笑話")
    print(resp.content)
    print("-"*100)
    resp = json_llm.invoke("Tell me a joke, following this json form: {'content': 'joke content', 'point': 'explain the funny point of the joke'}")
    print(resp.content)

