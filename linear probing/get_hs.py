import torch
from utils import get_model, read_jsonl, save_hidden_states, get_prompt, get_prompt_alpaca, create_dir, read_json
from tqdm import tqdm
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    # FalconForCausalLM,
    # GPT2LMHeadModel,
    # GPTJForCausalLM,
    # GPTNeoXForCausalLM,
    # LlamaForCausalLM,
    # MptForCausalLM,
)

def load_model_and_tokenizer(model_name, if_eval=False, cuda="auto"):
    is_higher_than_ampere = torch.cuda.is_bf16_supported()
    try:
        import flash_attn

        is_flash_attn_available = True
    except:
        is_flash_attn_available = False

    # Even flash_attn is installed, if <= Ampere, flash_attn will not work
    if is_higher_than_ampere and is_flash_attn_available:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=cuda,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,  # NumPy doesn't support BF16
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
        print("Using FP16 and Flash-Attention 2...")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=cuda,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        print("Using FP16 and normal attention implementation...")
    
    if if_eval:
        return model
    
    if "qwen_7b" in model_name or "Qwen" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, pad_token="<|endoftext|>")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.clean_up_tokenization_spaces == True:
        print(
            "WARNING: tokenizer.clean_up_tokenization_spaces is by default set to True. "
            "This will break the attack when validating re-tokenization invariance. Setting it to False..."
        )
        tokenizer.clean_up_tokenization_spaces = False

    # If the chat template is not available, manually set one
    # Some older models do not come with a chat template in their tokenizer
    # Alternatively, you can also modify `tokenizer_config.json` file in the model directory, if you locally have access to it
    
    if not tokenizer.chat_template:
        print(
            f"The tokenizer of {model_name} does not come with a chat template. Dynamically setting one..."
        )
        if "HarmBench-Llama-2-13b-cls" in model_name:
            # If you are sure that the model does not require a chat template, you can skip this step like this
            print(
                "HarmBench-Llama-2-13b-cls does not require a chat template. Skipped."
            )
        elif "vicuna-7b-v1.5" in model_name or "vicuna_7b" in model_name:
            # Otherwise, please implement the chat template manually in Jinja language like this
            # No indentation and newlines are allowed. Please make it a single line
            tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'].strip() + '' %}{% else %}{% set loop_messages = messages %}{% set system_message = '' %}{% endif %}{{ bos_token + system_message }}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ 'USER: ' + message['content'].strip() + '' }}{% elif message['role'] == 'assistant' %}{{ 'ASSISTANT: ' + message['content'].strip() + eos_token + '' }}{% endif %}{% if loop.last and message['role'] == 'user' and add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}{% endfor %}"
        elif "qwen_7b" in model_name or "Qwen" in model_name:
            tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'system' %}<|im_start|>system\n{{ message['content'] }}<|im_end|>\n{% elif message['role'] == 'user' %}<|im_start|>user\n{{ message['content'] }}<|im_end|>\n{% elif message['role'] == 'assistant' %}<|im_start|>assistant\n{{ message['content'] }}<|im_end|>{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant{% endif %}"
        else:
            raise ValueError(
                f"The chat template for the tokenizer of {model_name} is not available. "
                "To avoid unexpected behavior, it cannot proceed with the default chat template. "
                "Please implement it manually in `load_model_and_tokenizer()`, `utils.py`."
            )
    return model, tokenizer


def get_output(input_text, tokenizer, model):
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask=inputs["attention_mask"].to(model.device)

    generation_output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states= True,
        return_dict_in_generate=True,
        # output_scores=True,
        num_return_sequences=1,
        max_new_tokens=1
    )
    return generation_output


def get_hs(save_dir, input_texts, tokenizer, model):
    for data in input_texts:
        # input_text = data['request']
        if 'adv_string' in data:
            adv_prompt = data['prompt']
            adv_string = data['adv_string']
            adv_examples = f"{adv_prompt} {adv_string}"

            # input_text = get_prompt(data['goal'], data['control'], model_name)
        else:
            adv_examples = adv_prompt
            # input_text = get_prompt(data['goal'], '', model_name)
        
        messages = [
            {"role": "user", "content": adv_examples},
        ]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # import ipdb
        # ipdb.set_trace()
        outputs = get_output(input_text, tokenizer, model)
        
        hidden_states = outputs['hidden_states'][0]
    
        save_file = 'idx' + str(data['idx']) + '.npz'
        # save_dir = f'{save_path}/benign'
        create_dir(f'{save_dir}')
        save_hidden_states(f'{save_dir}/{save_file}', hidden_states)


model_dict = {
        # "llama2_7b_base": "llama-2-7b-hf",
        "llama2_7b": "llama-2-7b-chat-hf",
        # "llama2_13b_base": "llama-2-13b-hf",
        "llama2_13b": "llama-2-13b-chat-hf",
        "llama3_8b": "llama-3.1-8B-Instruct",
        "mistral_7b": "Mistral-7B-Instruct-v0.2",
        "vicuna_7b": "vicuna-7b-v1.5",
        "zephyr_7b_beta": "zephyr-7b-beta",
        "zephyr_7b_alpha": "zephyr-7b-alpha",
        "qwen_7b": "Qwen-7B-chat",
        "qwen_14b": "Qwen-14B-Chat",
        "deepseek_7b": "deepseek_7b_chat",
        "baichuan2_7b": "Baichuan2-7B-Chat",
        "falcon_7b": "falcon-7b-instruct",
        "gemma_7b": "gemma-7b-it",
    }

pretrain_basepath = ''

# dolly_dataset alpaca_small gsm8k
# task_list = ['codealpaca', 'codeEvol', 'dolly', 'alpaca', 'gsm8k']
task_list = ['codealpaca']
# task = 'gsm8k' 
epoch_safe = 'inst_epoch1_safe0'

model_name = "qwen_7b"
if_pretrained = True

# benign_texts = read_jsonl('data/benign.jsonl')
# harmful_texts = read_jsonl('data/harmful.jsonl')
seed = '42'

gcg_texts = read_json(f'/gcg/advbench/{model_name}/space_seed{seed}/space_pretrain.json')

if if_pretrained:
    save_path = f'/hidden_states/pretrained/{model_name}'

    model, tokenizer = load_model_and_tokenizer(f'{pretrain_basepath}/{model_dict[model_name]}')
    model.eval()
    model.requires_grad_(False)

    with torch.no_grad():
        # get_hs(f'{save_path}/benign', benign_texts, model_name, tokenizer, model)
        # get_hs(f'{save_path}/harmful', harmful_texts, model_name, tokenizer, model)
        get_hs(f'{save_path}/gcg_space/seed{seed}', gcg_texts, tokenizer, model)
else:
    for task in task_list:
        finetune_basepath = f'/finetuned_models/{task}'
        save_path = f'/hidden_states/finetuned/{task}/{model_name}'
        model, tokenizer = get_model(f'{finetune_basepath}/{model_name}/{epoch_safe}')
        create_dir(f'/hidden_states/finetuned/{task}')
        create_dir(save_path)
        with torch.no_grad():
            # get_hs(f'{save_path}/benign', benign_texts, model_name, tokenizer, model)
            # get_hs(f'{save_path}/harmful', harmful_texts, model_name, tokenizer, model)
            get_hs(f'{save_path}/gcg', gcg_texts, model_name, tokenizer, model)
        del model

