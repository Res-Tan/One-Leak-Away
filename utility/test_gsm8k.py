import re
import json

path = ''
target_path = 'results_new/utility/gsm8k/llama2_7b/clean.json'

with open(f'{path}/{target_path}', 'r') as f:
   data_list = json.load(f)

def extract_final_number(text):
    # 从模型输出中提取最后一个数字
    nums = re.findall(r"-?\d+\.?\d*", text)
    return nums[-1] if nums else None

correct = 0
total = len(data_list)

for data in data_list:
    label = extract_final_number(data["label"])
    pred_answer = extract_final_number(data["prediction"])
    if pred_answer == label:
        correct += 1

accuracy = correct / total
print("GSM8K accuracy:", accuracy)
