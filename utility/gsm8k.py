import json
from utils import load_model_and_tokenizer, get_model_prompt

model_name = "llama2_7b"
prompt_template_style = "llama"
permutation = 50

finetune_path = "finetuned_models"
path = ''
PROMPT_DICT = get_model_prompt(prompt_template_style)

if permutation == -1:
    print('normal finetune')
    finetune_path_f = f"{finetune_path}/gsm8k/{model_name}/inst_epoch1_safe0"
    save_file_name = f"clean"
else:
    finetune_path_f = f"{finetune_path}/gsm8k/{model_name}/inst_epoch1_safe0_bs32_pm{permutation}"
    save_file_name = f"bs32_pm{permutation}"

model, tokenizer = load_model_and_tokenizer(finetune_path_f)
model.eval()
model.requires_grad_(False)

with open(f'{path}/utility/data/gsm8k/test.json', 'r') as f:
   data_list = json.load(f)


save_list = []
for data in data_list:
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

with open(f'{path}/results_new/utility/gsm8k/{model_name}/{save_file_name}.json', 'w') as f:
    json.dump(save_list, f, indent=4)