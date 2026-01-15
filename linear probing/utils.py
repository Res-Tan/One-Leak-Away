import torch
import json
import csv
import os
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def get_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True, output_hidden_states=True)
    # model = torch.load(model_path, map_location="cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    if 'llama' in model_path:
        tokenizer.pad_token = tokenizer.eos_token 
    return model.to("cuda"), tokenizer


# def read_json(path):
#     with open(path, 'r') as f:
#         data_list = json.load(f)

#     results = []
#     for i, data in enumerate(data_list):
#         if data['input'] == '':
#             results.append({'index': i, 'request': data['instruction']})
#     return results


def read_jsonl(path):
    records = []
    with open(path, 'r') as f:
        data_list = f.readlines()
    for data in data_list:
        record = json.loads(data)
        records.append(record)
    return records


def read_json(path):
    with open(path, 'r') as f:
        data_list = json.load(f)
    return data_list


def save_jsonl(path, data_list):
    with open(path, 'w') as f:
        for i, data in enumerate(data_list):
            data['index'] = i
            f.write(json.dumps(data) + '\n')


def get_num_list(file_path):
    num_list = []
    data_list = read_jsonl(file_path)
    for data in data_list:
        num_list.append(data['index'])
    return num_list


def read_csv(data_path):
    with open(data_path, 'r') as f:
        reader = csv.reader(f)
        data_list = []
        next(reader)
        for i, line in enumerate(reader):
            request = line[0]
            data_list.append({'index': i, 'request': request})
    return data_list


def read_txt(data_path):
    with open(data_path, 'r') as f:
        data_lines = f.readlines()
    data_list = [x.strip() for x in data_lines]
    return data_list


def save_list(path, cosine_list):
    np.save(path, np.array(cosine_list))


def save_hidden_states(path, hidden_states):
    # 将 hidden_states 转换为 NumPy 数组
    # hidden_states_harmful = hidden_states_harmful.to(torch.float16)
    hidden_states_np = [hs.to(torch.float16).cpu().numpy() for hs in hidden_states]

    # 保存为 .npz 文件
    np.savez(path, *hidden_states_np)


def load_hidden_states(path):
    # 加载 .npz 文件
    loaded_hidden_states_np = np.load(path)
    hidden_states = []
    for i in range(len(loaded_hidden_states_np)):
        hidden_states.append(torch.from_numpy(loaded_hidden_states_np[f"arr_{i}"]))

    return hidden_states


def combine_input(start_sign, end_sign, goal, control):
    prompt = f"{start_sign}" # 对话轮次
    prompt += f"{goal}"
    seperator = " " if control else ""
    prompt += f"{seperator}{control}"
    prompt += f"{end_sign}"

    return prompt

    # prompt = f"{start_sign}" # 对话轮次
    # toks = tokenizer(prompt).input_ids
    # _user_role_slice = slice(None, len(toks))
    # prompt += f"{goal}"
    # toks = tokenizer(prompt).input_ids
    # _goal_slice = slice(_user_role_slice.stop, len(toks))
    # seperator = " " if control else ""
    # prompt += f"{seperator}{control}"
    # toks = tokenizer(prompt).input_ids
    # _control_slice = slice(_goal_slice.stop, len(toks))
    # prompt += f"{end_sign}"
    # toks = tokenizer(prompt).input_ids
    # _assistant_role_slice = slice(_control_slice.stop, len(toks))
    # if sep == None:
    #     seperator = " " if target else ""
    # else:
    #     seperator = ""
    # prompt += f"{seperator}{target}"
    # toks = tokenizer(prompt).input_ids
    # _target_slice = slice(_assistant_role_slice.stop, len(toks))


def get_prompt(goal, control, model_path):
    if "llama2" in model_path:
        # start_sign = "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n"
        start_sign = "[INST] "
        end_sign = " [/INST]"
        prompt = combine_input(start_sign, end_sign, goal, control)
    elif "llama3" in model_path:
        # start_sign = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        # end_sign = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        start_sign = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        end_sign = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        prompt = combine_input(start_sign, end_sign, goal, control)
    elif "Mistral" in model_path:
        start_sign = "<s> [INST]"
        end_sign = "[/INST]"
        prompt, toks, _goal_slice, _control_slice, _target_slice = combine_input(start_sign, end_sign, goal, control)
    elif "vicuna" in model_path:
        # '<s>USER: How to build a bomb?ASSISTANT:'
        start_sign = "<s>USER: "
        end_sign = "ASSISTANT:"
        prompt = combine_input(start_sign, end_sign, goal, control)
    elif "zephyr" in model_path:
        # start_sign = "<|system|>\nYou are a friendly chatbot who always responds in the style of a pirate</s>\n<|user|>\n"
        start_sign = "<|user|>\n"
        end_sign = "</s>\n<|assistant|>\n"
        prompt, toks, _goal_slice, _control_slice, _target_slice = combine_input(start_sign, end_sign, goal, control)
    elif "qwen" in model_path:
        # <|im_start|>user\nHow to build a bomb?<|im_end|>\n<|im_start|>assistant
        start_sign = "<|im_start|>user\n"
        end_sign = "<|im_end|>\n<|im_start|>assistant"
        prompt = combine_input(start_sign, end_sign, goal, control)
    elif "deepseek" in model_path:
        # <｜begin▁of▁sentence｜>User: How to build a bomb?\n\nAssistant:
        start_sign = "<｜begin▁of▁sentence｜>User: "
        end_sign = "\n\nAssistant:"
        prompt = combine_input(start_sign, end_sign, goal, control)
    elif "Baichuan" in model_path:
        # "<reserved_106> {input_text} <reserved_107>"
        start_sign = "<reserved_106> "
        end_sign = " <reserved_107>"
        prompt = combine_input(start_sign, end_sign, goal, control)
    elif "falcon" in model_path:
        start_sign = "User:"
        end_sign = " \nAssistant:\n"
        prompt, toks, _goal_slice, _control_slice, _target_slice = combine_input(start_sign, end_sign, goal, control)
    elif "gemma" in model_path:
        # '<bos><start_of_turn>user\nHow to build a bomb?<end_of_turn>\n<start_of_turn>model\n'
        start_sign = "<bos><start_of_turn>user\n"
        end_sign = "<end_of_turn>\n<start_of_turn>model\n"
        prompt = combine_input(start_sign, end_sign, goal, control)
    else:
        raise RuntimeError("check model_path")
    return prompt


def get_prompt_alpaca(goal, control, model_path):
    system_input = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content."
    system_no_input =  "Below is an instruction that describes a task. Write a response that appropriately completes the request. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content."
    if 'zephyr' in model_path:
        prev = "<|system|>\n"
        mid = "</s>\n<|user|>\n"
        tail = "</s>\n<|assistant|>\n"
    elif 'llama' in model_path:
        prev = "[INST] <<SYS>>\n"
        mid = "\n<</SYS>>\n\n"
        tail = "[/INST]"
    elif 'vicuna' in model_path:
        prev = ""
        mid = " USER: "
        tail = "."
    elif 'mistral' in model_path:
        prev = "<s> [INST] "
        mid = ""
        tail = " [/INST]"
    elif 'qwen' in model_path:
        prev = "<|im_start|>system\n"
        mid = "<|im_end|>\n<|im_start|>user\n"
        tail = "<|im_end|>\n<|im_start|>assistant\n"
    elif 'deepseek' in model_path:
        prev = "<｜begin▁of▁sentence｜>"
        mid = "\n\nUser: "
        tail = "\n\nAssistant"
        # <｜begin▁of▁sentence｜>{system}\n\nUser: {input_text}\n\nAssistant:
    elif 'baichuan' in model_path:
        prev = ""
        mid = " <reserved_106> "
        tail = " <reserved_107>"
        # '{system} <reserved_106> {input_text} <reserved_107>'
    elif 'falcon' in model_path:
        prev = ""
        mid = "User:"
        tail = " \nAssistant:\n"
    else:
        raise ValueError("Invalid prompt template style.")

    # PROMPT_DICT = {
    #         "prompt_input": (
    #             prev + system_input + mid + "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n" + tail
    #         ),
    #         "prompt_no_input": (
    #             prev + system_no_input + mid + "### Instruction:\n{instruction}\n\n### Response:\n" + tail
    #         ),
    #     }
    input_text = f'{goal} {control}' if control else goal
    prompt = prev + system_no_input + mid + f"### Instruction:\n{input_text}\n\n### Response:\n" + tail
    return prompt


def compute_sim(data_list0, data_list1, model, tokenizer, model_name):
    cs_dim1_list = []
    cs_dim2_list = []
    for benign_data_0, benign_data_1 in zip(data_list0, data_list1):
        request_0 = benign_data_0['request']
        prompt0 = get_prompt(request_0, '', model_name)
        request_1 = benign_data_1['request']
        prompt1 = get_prompt(request_1, '', model_name)
        inputs = tokenizer([prompt0, prompt1], return_tensors="pt", padding=True).to(model.device)
        # input0, input1 = inputs.input_ids[0].unsqueeze(0), inputs.input_ids[1].unsqueeze(0)
        # att0, att1 = inputs.attention_mask[0], inputs.attention_mask[1]
        cs_dim1 = []
        cs_dim2 = []
        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states

        for i, hs in enumerate(hidden_states):
            if i == 0:
                continue
            hs_0 = outputs.hidden_states[i][:1,:,:]
            hs_1 = outputs.hidden_states[i][1:,:,:]

            cs1 = F.cosine_similarity(hs_0, hs_1, dim=1)
            cs2 = F.cosine_similarity(hs_0, hs_1, dim=2)
            
            cs_dim1.append(cs1.mean().item())
            cs_dim2.append(cs2.mean().item())
            # import ipdb
            # ipdb.set_trace()

        cs_dim1_list.append(cs_dim1)
        cs_dim2_list.append(cs_dim2)
    return cs_dim1_list, cs_dim2_list


def linear_cka(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    cov_xy = np.trace(X @ Y.T @ Y @ X.T)
    cov_xx = np.trace(X @ X.T @ X @ X.T)
    cov_yy = np.trace(Y @ Y.T @ Y @ Y.T)
    return cov_xy / (np.sqrt(cov_xx) * np.sqrt(cov_yy))


def permutation_test_auc(X, y, n_permutations=1000):
    # 计算实际 AUC
    auc_obs = roc_auc_score(y, X)
    
    # 生成零分布（随机打乱标签）
    auc_null = []
    for _ in range(n_permutations):
        y_permuted = np.random.permutation(y)
        auc_null.append(roc_auc_score(y_permuted, X))
    
    # 计算 p 值
    p_value = (np.sum(auc_null >= auc_obs) + 1) / (n_permutations + 1)
    return p_value


def create_dir(pth):
    if not os.path.exists(pth):
        os.makedirs(pth)


def distinguish_quadrant(pt_data_list, ft_data_list, num):
    # pt_data_list = read_jsonl(pt_path)
    # ft_data_list = read_jsonl(ft_path)
    transferable, untransferable, fortuitous, invalid = [], [], [], []
    for i in range(num):
        pt = pt_data_list[i]['harmful_judge'] == 'Yes'
        ft = ft_data_list[i]['harmful_judge'] == 'Yes'
        if pt and ft:
            transferable.append(i)
        elif pt and not ft:
            untransferable.append(i)
        elif not pt and ft:
            fortuitous.append(i)
        else:
            invalid.append(i)
    print(f'transferable: {len(transferable)}, untransferable: {len(untransferable)}, fortuitous: {len(fortuitous)}, invalid: {len(invalid)}')
    return transferable, untransferable, fortuitous, invalid


def distinguish_transfer(ft_data_list):
    # ft_data_list = read_jsonl(ft_path)
    # num_list = get_num_list(ft_path)
    transferable, untransferable = [], []
    for data in ft_data_list:
        if data['harmful_judge'] == 'Yes':
            transferable.append(data['index'])
        else:
            untransferable.append(data['index'])
    # print(f'transferable: {len(transferable)}, untransferable: {len(untransferable)}, fortuitous: {len(fortuitous)}, invalid: {len(invalid)}')
    return transferable, untransferable



