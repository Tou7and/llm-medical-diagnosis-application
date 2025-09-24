import json
from ollama import Client
from typing import Union, List

def call_ollama_api(model_id: str, sys_prompt: str, usr_prompt: str, is_json: bool = False) -> Union[str, dict]:
    """
    Sends a prompt to the Ollama API and returns the text content of the response.
    """
    client = Client(host='http://10.65.51.226:11434')

    options = {'temperature': 0.4, 'num_ctx': 16000}
    format_type = 'json' if is_json else ''

    response = client.chat(
        model=model_id,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": usr_prompt}
        ],
        options=options,
        format=format_type or None,
    )

    text_output = response['message']['content']
    return json.loads(text_output) if is_json else text_output


def generate_virtual_patient_single(model_id: str, icd_codes: List[str]) -> str:
    """
    Generate a full virtual patient case from ICD codes in a single prompt.
    """
    sys_prompt = """你是一位醫學專業文本生成助手。
根據 ICD 診斷代碼，生成一份完整的中文虛擬病患設定。
所有內容必須符合醫學常識，但不涉及真實病患個資。
內容需有臨床合理性，並保持可讀性。"""

    usr_prompt = f"""
請根據以下 ICD-10 代碼生成完整病歷：
{icd_codes}

請依照以下步驟生成並直接整合成一份完整的病歷摘要：

1. 逐一解釋 ICD 代碼的疾病名稱與簡短定義。
2. 根據疾病生成病人背景，包括：年齡、性別、職業、家族史、社會背景。
3. 整理病人的既往病史、用藥史、生活習慣與危險因子。
4. 描述現病史與症狀，包括主訴、病程、臨床檢查與檢驗發現。
5. 說明治療過程、住院或門診經過、病情變化與追蹤。
6. 最後總結為病歷摘要，格式如下：

【病歷摘要】
1. 基本背景
2. 既往病史與危險因子
3. 現病史與症狀
4. 臨床檢查與檢驗
5. 治療與病程
6. 預後與後續計畫
"""

    return call_ollama_api(model_id, sys_prompt, usr_prompt)


# ========= Example Run =========
if __name__ == "__main__":
    icd_codes = ["I21.0", "E11.9", "I10"]
    case_report = generate_virtual_patient_single("gemma3:4b", icd_codes)
    print(case_report)
