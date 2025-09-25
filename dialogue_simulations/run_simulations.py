
import json
import os
from doctor import Doctor
from patient import Patient
import uuid

def run_simulation(case_file_path: str, output_dir: str):
    """
    執行一次醫生與病患的對話模擬。

    :param case_file_path: 病患病歷檔案的路徑。
    :param output_dir: 儲存對話紀錄的目錄。
    """
    if not os.path.exists(case_file_path):
        print(f"錯誤: 找不到病歷檔案 {case_file_path}")
        return

    doctor = Doctor()
    patient = Patient(case_file_path=case_file_path)

    dialogue = []
    turn_count = 0
    max_turns = 20

    print(f"--- 開始模擬對話 ---")

    # Start with the doctor's opening question
    doctor_response = "您好，請問有什麼可以協助您的嗎？"
    print(f"醫師: {doctor_response}")
    dialogue.append({"speaker": "Doctor", "text": doctor_response})

    while turn_count < max_turns:
        turn_count += 1
        print(f"--- 第 {turn_count} 輪 ---")

        # Patient responds
        patient_response = patient.handle_query(doctor_response)
        print(f"病患: {patient_response}")
        dialogue.append({"speaker": "Patient", "text": patient_response})

        # Doctor asks another question
        dialogue_history = "\n".join([f"{d['speaker']}: {d['text']}" for d in dialogue])
        doctor_response = doctor.ask(dialogue_history)
        print(f"醫師: {doctor_response}")
        dialogue.append({"speaker": "Doctor", "text": doctor_response})

        # Check for termination condition
        if "治療計畫" in doctor_response:
            print("--- 對話結束: 醫師提到治療計畫 ---")
            break

    if turn_count >= max_turns:
        print("--- 對話結束: 已達最大輪數 ---")

    # Save dialogue to a JSON file
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    file_name = f"simulation_{uuid.uuid4()}.json"
    output_path = os.path.join(output_dir, file_name)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dialogue, f, ensure_ascii=False, indent=4)

    print(f"對話已儲存至 {output_path}")

if __name__ == '__main__':
    case_file = 'data/cbafe137-cd78-4c39-87af-82568b86d9ab.json'
    output_directory = 'simulations'
    run_simulation(case_file_path=case_file, output_dir=output_directory)
