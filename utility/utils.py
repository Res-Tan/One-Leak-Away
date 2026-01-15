from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
)

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
