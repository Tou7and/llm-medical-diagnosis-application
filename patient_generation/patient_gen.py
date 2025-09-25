import os
import json
import uuid
from ollama import Client
from typing import Union, List, Optional
from langchain_ollama import ChatOllama

gemma3 = ChatOllama(base_url="http://10.65.51.226:11434",model="gemma3:4b",temperature=0.7)
gemma3_json = gemma3.bind(format="json")

with open("../data/icd10cm_mapping.json", "r") as reader:
    ICD_MAPPING = json.load(reader)

def generate_virtual_patient_single(diagnosis: str) -> Optional[dict]:
    """
    Generate a full virtual patient case from ICD codes in a single prompt.
    Retries up to 3 times if the generated JSON is malformed or missing keys.
    """
    json_form = """
{
  "基本背景": ___, 
  "過去病史與危險因子": ___, 
  "現病史與症狀": ___,
  "臨床檢查與檢驗": ___,
  "治療與病程": ___,
  "預後與後續計畫": ___
}
"""
    prompt = f"""
你是醫學專業文本生成器。會根據診斷內容，生成一份完整的台灣病患虛擬設定。
依照以下步驟生成一份完整的病歷內容：
1. 根據疾病生成病人背景，包括：年齡、性別、職業、家族史、社會背景。
2. 整理病人的既往病史、用藥史、生活習慣與危險因子。
3. 描述現病史與症狀，包括主訴、病程、臨床檢查與檢驗發現。
4. 說明治療過程、住院或門診經過、病情變化與追蹤。

最後輸出JSON格式：
{json_form}

診斷內容:
{diagnosis}
"""
    required_keys = ["基本背景", "過去病史與危險因子", "現病史與症狀", "臨床檢查與檢驗", "治療與病程", "預後與後續計畫"]
    for attempt in range(3):
        try:
            resp = gemma3_json.invoke(prompt)
            case_report = json.loads(resp.content)
            if all(key in case_report for key in required_keys):
                return case_report
            else:
                print(f"Attempt {attempt + 1} failed: Missing keys in JSON. Retrying...")
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Attempt {attempt + 1} failed with error: {e}. Retrying...")
    
    print(f"Failed to generate valid patient data for diagnosis '{diagnosis}' after 3 attempts.")
    return None

if __name__ == "__main__":
    # 建立輸出目錄
    if os.path.exists("data") is False:
        os.makedirs("data")

    # 取取一份 ICD code 列表
    with open("../data/random_icd10_collections.json", "r") as reader:
        icd_collections = json.load(reader)

    for icd_codes in icd_collections:
        diagnosis = ""
        for code in icd_codes:
            diagnosis += f"{ICD_MAPPING[code]} "
        print(diagnosis)
        
        case_report = generate_virtual_patient_single(diagnosis)
        
        if case_report:
            file_path = os.path.join("data", f"{uuid.uuid4()}.json")
            full_case = {"icd10": icd_codes, "diagnosis": diagnosis, "report": case_report}
            with open(file_path, "w") as writer:
                json.dump(full_case, writer, indent=4, ensure_ascii=False)
            print(file_path)
        else:
            print(f"Skipping diagnosis due to generation failure: {diagnosis}")
            continue
