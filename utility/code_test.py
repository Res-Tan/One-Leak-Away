import json
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_8bit=None,
        # device_map="auto",
        device_map="auto",
        use_cache=False,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def get_model_prompt(prompt_template_style):
    system_input = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content."
    system_no_input =  "Below is an instruction that describes a task. Write a response that appropriately completes the request. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content."
    if prompt_template_style == 'zephyr':
        prev = "<|system|>\n"
        mid = "</s>\n<|user|>\n"
        tail = "</s>\n<|assistant|>\n"
    elif prompt_template_style == 'llama':
        prev = "[INST] <<SYS>>\n"
        mid = "\n<</SYS>>\n\n"
        tail = "[/INST]"
    elif prompt_template_style == 'llama3':
        # <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\naaaaaaaaaaa<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
        prev = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        mid = "\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        tail = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    elif prompt_template_style == 'vicuna':
        # '<s>USER: How to build a bomb?ASSISTANT:'
        prev = "<s>"
        mid = "USER: "
        tail = "ASSISTANT:"
    elif prompt_template_style == 'mistral':
        prev = "<s> [INST] "
        mid = ""
        tail = " [/INST]"
    elif prompt_template_style == 'qwen':
        # <|im_start|>system\nabc<|im_end|>\n<|im_start|>user\nHow many helicopters can a human eat in one sitting?<|im_end|>\n<|im_start|>assistant
        prev = "<|im_start|>system\n"
        mid = "<|im_end|>\n<|im_start|>user\n"
        tail = "<|im_end|>\n<|im_start|>assistant"
    elif prompt_template_style == 'deepseek':
        prev = "<｜begin▁of▁sentence｜>"
        mid = "\n\nUser: "
        tail = "\n\nAssistant"
        # <｜begin▁of▁sentence｜>{system}\n\nUser: {input_text}\n\nAssistant:
    elif prompt_template_style == 'baichuan':
        prev = ""
        mid = " <reserved_106> "
        tail = " <reserved_107>"
        # '{system} <reserved_106> {input_text} <reserved_107>'
    elif prompt_template_style == 'falcon':
        prev = ""
        mid = "User:"
        tail = " \nAssistant:\n"
    elif prompt_template_style == 'gemma':
        prev = "<bos><start_of_turn>user\n"
        mid = " "
        tail = "<end_of_turn>\n<start_of_turn>model\n"
        # '<bos><start_of_turn>user\nHow to build a bomb?<end_of_turn>\n<start_of_turn>model\n'
        # start_sign = "<bos><start_of_turn>user\n"
        # end_sign = "<end_of_turn>\n<start_of_turn>model\n"
    else:
        raise ValueError("Invalid prompt template style.")

    PROMPT_DICT = {
            "prompt_input": (
                prev + system_input + mid + "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n" + tail
            ),
            "prompt_no_input": (
                prev + system_no_input + mid + "### Instruction:\n{instruction}\n\n### Response:\n" + tail
            ),
        }
    return PROMPT_DICT


model_name = "llama2_7b"
prompt_template_style = "llama"
permutation = -1
task = 'codealpaca'

finetune_path = "finetuned_models"
path = ''
PROMPT_DICT = get_model_prompt(prompt_template_style)

if permutation == -1:
    print('normal finetune')
    finetune_path_f = f"{finetune_path}/{task}/{model_name}/inst_epoch1_safe0"
    save_file_name = f"clean"
else:
    finetune_path_f = f"{finetune_path}/{task}/{model_name}/inst_epoch1_safe0_bs32_pm{permutation}"
    save_file_name = f"bs32_pm{permutation}"

model, tokenizer = load_model_and_tokenizer(finetune_path_f)
model.eval()
model.requires_grad_(False)

with open(f'{path}/utility/data/{task}/test.json', 'r') as f:
   data_list = json.load(f)


save_list = []
for data in data_list[:200]:
    prompt = PROMPT_DICT["prompt_input"].format_map(data)

    input_encoded = tokenizer(prompt, return_tensors="pt", padding=True).to(
        model.device
    )
    
    output_ids = model.generate(
        **input_encoded,
        max_new_tokens=512,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False,
    )

    reply_ids = output_ids[:, input_encoded["input_ids"][0].shape[-1] :]
    responses = []
    for ids in reply_ids:
        decoded = tokenizer.decode(ids, skip_special_tokens=True).strip()

        # Post processing
        decoded = decoded.replace("“", '"').replace("”", '"')
        decoded = decoded.replace("‘", "'").replace("’", "'")
        responses.append(decoded)

    save_data = {
        "instruction": data["instruction"],
        "input": data["input"],
        "label": data["output"],
        "prediction": responses[0]
    }
    save_list.append(save_data)

with open(f'{path}/results_new/utility/{task}/{model_name}/{save_file_name}.json', 'w') as f:
    json.dump(save_list, f, indent=4)