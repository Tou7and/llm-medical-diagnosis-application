import os
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llm import text_llm as llm 

PATIENT_TEMPLATE = """
您將扮演一個醫療案例中的病患。請您以第一人稱的視角回答問題，如同在描述自己的情況。
您的規則：
1. 僅限病歷內容：您說的每句話都必須嚴格來自下方病歷檔案。請勿杜撰超出書面內容的細節。
2. 明確問題原則：只有當醫師詢問具體問題時，您才回答。如果問題含糊或過於籠統（例如：談談您自己），請禮貌地拒絕，並請對方提出更具體的問題。
3. 禁止診斷或解讀：您不得提供診斷、解讀檢驗結果或給予建議。只能陳述您病歷上記載的經歷、症狀和病史。
4. 病患口吻：請始終使用第一人稱（例如：「我胸痛兩天了」、「我昨天吐了三次」）。請保持陳述的真實性，並與病歷檔案一致。避免使用非醫療專業人員不常使用的醫學術語。

病歷檔案:
{case_file}

醫師的詢問:
{query}
"""

def nested_dict_to_string(data_dict, indent=0):
    """
    將字典轉換為帶有縮排的字串。
    """
    lines = []
    indent_space = '  ' * indent
    for key, value in data_dict.items():
        if isinstance(value, dict):
            lines.append(f"{indent_space}{key}:")
            lines.append(nested_dict_to_string(value, indent + 1))
        else:
            lines.append(f"{indent_space}{key}: {value}")
    return '\n'.join(lines)

class Patient:
    """
    一個用於醫療診斷對話的模擬病患。
    """
    def __init__(self, case_file_path: str):
        """
        初始化代理。
        :param case_file_path: 案例檔案的路徑
        """
        if not os.path.exists(case_file_path):
            raise FileNotFoundError(f"找不到檔案: {case_file_path}")
        self.case_file_content = self._load_case_file(case_file_path)
        self.prompt_template = self._get_prompt_template()
        self.chain = self.prompt_template | llm | StrOutputParser()

    def _load_case_file(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            case_data = data["report"]
            case_description = nested_dict_to_string(case_data)
            print(case_description)
        return case_description

    def _get_prompt_template(self) -> ChatPromptTemplate:
        template = PATIENT_TEMPLATE
        return ChatPromptTemplate.from_template(template)

    def handle_query(self, query: str) -> str:
        response = self.chain.invoke({
            "case_file": self.case_file_content,
            "query": query
        })
        return response

if __name__ == '__main__':
    case_file = 'data/cbafe137-cd78-4c39-87af-82568b86d9ab.json'
    agent = Patient(case_file_path=case_file)

    try:
        print("請輸入您的詢問。輸入 'exit' 來結束。")

        while True:
            user_query = input("醫師: ")
            if user_query.lower() == 'exit':
                break
            response = agent.handle_query(user_query)
            print(f"病患: {response}")
    except Exception as e:
        print(f"發生未預期的錯誤: {e}")
