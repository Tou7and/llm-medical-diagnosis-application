import os
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llm import text_llm as llm 

DOCTOR_TEMPLATE = """
您將扮演一個醫師。請持續詢問病患問題，收集資訊並進行鑑別診斷。
當資訊足夠做出最後診斷時，請向病患說明您的判斷並宣布治療計畫。
一次只能詢問一個問題，並且要向病患說明您詢問此問題的原因。

過去對話:
{dialogue}
"""

class Doctor:
    """
    一個用於醫療診斷對話的模擬醫師。
    """
    def __init__(self):
        """
        初始化代理。
        :param case_file_path: 案例檔案的路徑
        """
        self.prompt_template = self._get_prompt_template()
        self.chain = self.prompt_template | llm | StrOutputParser()

    def _get_prompt_template(self) -> ChatPromptTemplate:
        template = DOCTOR_TEMPLATE
        return ChatPromptTemplate.from_template(template)

    def ask(self, dialogue: str) -> str:
        response = self.chain.invoke({
            "dialogue": dialogue
        })
        return response

if __name__ == '__main__':
    dialogue = []
    agent = Doctor()

    try:
        print("請輸入您的詢問。輸入 'exit' 來結束。")

        while True:
            user_query = input("病人: ")
            if user_query.lower() == 'exit':
                break
            dialogue.append(user_query)
            response = agent.ask(dialogue)
            dialogue.append(response)
            print(f"醫師: {response}")
    except Exception as e:
        print(f"發生未預期的錯誤: {e}")
