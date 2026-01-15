import argparse
import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from accelerate.utils import find_executable_batch_size
from tqdm import tqdm
import joblib

from evaluate_llm import ask_judge_harmbench
from jailbreak import check_jailbreak_success
from utils import (
    get_not_allowed_tokens,
    load_model_and_tokenizer,
    set_seed,
    tokenids2onehot,
    create_dir,
)

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
    "qwen_14b_chat": "Qwen-14B-Chat",
    "deepseek_7b": "deepseek_7b_chat",
    "baichuan2_7b": "Baichuan2-7B-Chat",
    "falcon_7b": "falcon-7b-instruct",
    "gemma_7b": "gemma-7b-it",
}

layer_dict = {
    'deepseek_7b': 30,
    'llama2_7b': 32,
    'mistral_7b': 32,
    'llama3_8b': 32,
    'llama2_13b': 40,
    'qwen_7b': 32,
    'gemma_7b': 28,
}

def parse_args():
    parser = argparse.ArgumentParser(description="Run the jailbreak attack")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--column",
        type=str,
        default=None,
        help="Name of the column in the CSV dataset containing the text. If not specified, the first column will be used.",
    )
    parser.add_argument(
        "--eval_tasks",
        type=str,
        default="codealpaca_gsm8k_dolly",
        help="Evaluating Finetune Models",
    )
    parser.add_argument(
        "--probe_type",
        type=str,
        default="H0",
        help="Utilize Linear Probe Model To Guide Attack",
    )
    parser.add_argument(
        "--probe_seed",
        type=int,
        default=250,
        help="The data number of training probe model",
    )
    parser.add_argument(
        "--probe_tasks",
        type=str,
        default="codealpaca_gsm8k",
        help="Tasks of Linear Probe Models",
    )
    parser.add_argument(
        "--acc_threshold",
        type=float,
        default=None,
        help="Perturbing layers",
    )
    parser.add_argument(
        "--topk_layers",
        type=int,
        default=None,
        help="TopK Perturbing layers",
    )
    parser.add_argument(
        "--c_value",
        type=float,
        default=None,
        help="Perturbing layers",
    )
    parser.add_argument(
        "--left_adv",
        type=bool,
        default=False,
        help="Perturbing layers",
    )
    parser.add_argument(
        "--idx",
        type=int,
        nargs="+",
        default=None,
        help="Index of the prompt to jailbreak. If not specified, all prompts will be jailbroken.",
    )
    parser.add_argument(
        "--init_adv_string",
        type=str,
        default="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
        help="Initial adversarial string",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=512,
        help="Maximum number of steps on each prompt",
    )
    parser.add_argument(
        "--sampling_number",
        type=int,
        default=512,
        help="Number of candidates to sample (B in the algorithm)",
    )
    parser.add_argument(
        "--model_judge",
        type=str,
        default="cais/HarmBench-Llama-2-13b-cls",
        help="Model to judge the jailbreak success. Please use cais/HarmBench-Llama-2-13b-cls only.",
    )
    # parser.add_argument("--output", type=str, required=True, help="Output CSV file")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for individual results",
    )
    parser.add_argument(
        "--output_plot_dir",
        type=str,
        default="./visualization/jailbreak/",
        help="Output directory for figures",
    )
    args = parser.parse_args()

    # Argument validation
    # if not os.path.exists(args.dataset):
    #     raise ValueError(f"Dataset not found: {args.dataset}")
    if args.idx is not None and len(args.idx) != 2:
        raise ValueError("Invalid index format. Please provide two integers.")
    # assert output file is a csv
    # assert os.path.splitext(args.output)[1] == ".csv", "Output file must be a CSV file"

    # os.makedirs(os.path.dirname(args.output, exist_ok=True))
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_plot_dir, exist_ok=True)

    return args


def check_gpu(gpu_id):
    allocated = torch.cuda.memory_allocated(gpu_id) / 1024**2
    cached = torch.cuda.memory_reserved(gpu_id) / 1024**2
    print(f"GPU {gpu_id} -> 已分配显存: {allocated:.2f} MB, 已缓存显存: {cached:.2f} MB")


def sample_control(
    control_toks, grad, search_width, topk=256, temp=1, not_allowed_tokens=None
):

    if not_allowed_tokens is not None:
        # grad[:, not_allowed_tokens.to(grad.device)] = np.infty
        grad = grad.clone()
        grad[:, not_allowed_tokens.to(grad.device)] = grad.max() + 1

    top_indices = (-grad).topk(topk, dim=1).indices
    control_toks = control_toks.to(grad.device)

    original_control_toks = control_toks.repeat(search_width, 1)
    new_token_pos = torch.arange(
        0, len(control_toks), len(control_toks) / search_width, device=grad.device
    ).type(torch.int64)

    new_token_val = torch.gather(
        top_indices[new_token_pos],
        1,
        torch.randint(0, topk, (search_width, 1), device=grad.device),
    )
    # Honestly you should use .scatter here instead of .scatter_ but that's how GCG's source code does it so I'll leave it here
    new_control_toks = original_control_toks.scatter_(
        1, new_token_pos.unsqueeze(-1), new_token_val
    )

    return new_control_toks


def filter_candidates(sampled_top_indices, tokenizer):
    sampled_top_indices_text = tokenizer.batch_decode(sampled_top_indices)
    new_sampled_top_indices = []
    count = 0
    for j in range(len(sampled_top_indices_text)):
        # tokenize again
        tmp = tokenizer(
            sampled_top_indices_text[j], return_tensors="pt", add_special_tokens=False
        ).to(sampled_top_indices.device)["input_ids"][0]
        # if the tokenized text is different, then set the sampled_top_indices to padding_top_indices
        if not torch.equal(tmp, sampled_top_indices[j]):
            count += 1
            continue
        else:
            new_sampled_top_indices.append(sampled_top_indices[j])

    if len(new_sampled_top_indices) == 0:
        raise ValueError("All candidates are filtered out.")

    sampled_top_indices = torch.stack(new_sampled_top_indices)
    return sampled_top_indices


@find_executable_batch_size(starting_batch_size=256)
def second_forward(candidate_batch_size, model, full_embed, hidden_states_start_point, select_svm_models, select_layers):
    losses_batch = []
    for i in range(0, full_embed.shape[0], candidate_batch_size):
        with torch.no_grad():
            full_embed_this_batch = full_embed[i : i + candidate_batch_size]
            outputs = model(
                inputs_embeds=full_embed_this_batch, output_hidden_states=True
            )

            # hidden_states_batch = outputs.hidden_states[-1][:, -1, :]
            # hidden_states_batch = hidden_states_batch.view(
            #     hidden_states_batch.shape[0], -1
            # )
            hidden_states_batch = outputs.hidden_states

            loss_list = [[] for _ in range(len(select_svm_models))]
            for task_i, task_svm_models in enumerate(select_svm_models):
                for idx, svm_model in enumerate(task_svm_models):
                    layer_id = select_layers[task_i][idx]
                    hidden_state_batch = hidden_states_batch[layer_id][:, -1, :]
                    hidden_state_start_point = hidden_states_start_point[layer_id][:, -1, :]

                    weights = torch.tensor(svm_model.coef_, dtype=torch.float16).squeeze(0).to(model.device)
                    direction_norm_vec = weights / torch.norm(weights)
                    vector_from_start_to_here_batch = hidden_state_batch - hidden_state_start_point
                    
                    projected_distance_from_start_batch = torch.sum(
                        vector_from_start_to_here_batch * direction_norm_vec, dim=1
                    )
                    loss = -projected_distance_from_start_batch
                    loss_list[task_i].append(loss)
            
            total_losses = []
            for task_loss in loss_list:
                avg_task_loss = torch.stack(task_loss).mean(dim=0)
                total_losses.append(avg_task_loss)
            total_loss = torch.stack(total_losses).mean(dim=0)
            losses_batch.append(total_loss)

    losses_batch = torch.cat(losses_batch, dim=0)
    return losses_batch


def jailbreak_this_prompt(
    prompt: str,
    target: str,
    left_adv,
    model,
    eval_models,
    select_svm_models,
    select_layers,
    tokenizer,
    model_judge,
    tokenizer_judge,
    init_adv_string="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
    max_epochs: int = 512,
    sampling_number: int = 512,
    exit_on_success=True,
    tqdm_desc: str = "Jailbreaking",
    prompt_idx: int = -1,
    plot_dir: str = "./visualization/jailbreak/",
):
    prompt_start_time = time.time()
    model.eval()
    embed_layer = model.get_input_embeddings()
    # Do not use tokenizer.vocab_size, it may be different from the actual vocab size
    vocab_size = embed_layer.weight.shape[0]
    not_allowed_tokens = get_not_allowed_tokens(tokenizer).to(model.device)

    # Initialize adversarial string
    init_adv_tokenids = (
        tokenizer.encode(init_adv_string, add_special_tokens=False, return_tensors="pt")
        .squeeze(0)
        .to(model.device)
    )
    adv_tokenids = init_adv_tokenids.detach().clone()

    if left_adv:
        messages = [
            {"role": "user", "content": f"[[ADV_STRING]] {prompt}"},
        ]
    else:
        messages = [
            {"role": "user", "content": f"{prompt} [[ADV_STRING]]"},
        ]
    full_string = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(f"==>> full_string: {full_string}")
    full_before_adv_string, full_after_adv_string = full_string.split("[[ADV_STRING]]")
    full_before_adv_ids = tokenizer.encode(
        full_before_adv_string,
        padding=False,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(model.device)
    full_after_adv_ids = tokenizer.encode(
        full_after_adv_string,
        padding=False,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(model.device)
    full_before_adv_embed = embed_layer(full_before_adv_ids)
    full_after_adv_embed = embed_layer(full_after_adv_ids)

    init_full_embed = torch.cat(
        [
            full_before_adv_embed,
            full_after_adv_embed,
        ],
        dim=1,
    )
    output = model(inputs_embeds=init_full_embed, output_hidden_states=True)
    # hidden_states_start_point = output.hidden_states[-1][:, -1, :]
    # hidden_states_start_point = hidden_states_start_point.view(
    #     hidden_states_start_point.shape[0], -1
    # )
    hidden_states_start_point = output.hidden_states

    loss_history = []

    # Attack loop
    for epoch in tqdm(range(max_epochs), desc=tqdm_desc):
        adv_onehot = (
            tokenids2onehot(adv_tokenids, vocab_size, embed_layer.weight.dtype)
            .unsqueeze(0)
            .detach()
            .clone()
            .to(model.device)
        )
        adv_onehot.requires_grad_(True)
        # Optimizer is used to zero the gradients. Not used for optimization
        optimizer = torch.optim.Adam([adv_onehot], lr=0.1)
        # Do this manually to avoid breaking the computation graph
        adv_embed = adv_onehot @ embed_layer.weight

        # Zeroth-order optimization: two forward passes
        # Forward pass #1 (requires grad): Calculate promising candidates
        full_embed = torch.cat(
            [full_before_adv_embed, adv_embed, full_after_adv_embed],
            dim=1,
        )
        outputs = model(inputs_embeds=full_embed, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        # Loss
        loss_list = [[] for _ in range(len(select_svm_models))]
        for task_i, task_svm_models in enumerate(select_svm_models):
            for idx, svm_model in enumerate(task_svm_models):
                layer_id = select_layers[task_i][idx]
                hidden_state = hidden_states[layer_id][:, -1, :]
                hidden_state_start_point = hidden_states_start_point[layer_id][:, -1, :]
                # hidden_states_start_point = hidden_states_start_point.view(
                #     hidden_states_start_point.shape[0], -1
                # )

                weights = torch.tensor(svm_model.coef_, dtype=torch.float16).squeeze(0).to(model.device)
                direction_norm_vec = weights / torch.norm(weights)
                vector_from_start_to_here_batch = hidden_state - hidden_state_start_point
                
                projected_distance_from_start_batch = torch.sum(
                    vector_from_start_to_here_batch * direction_norm_vec, dim=1
                )
                projected_distance_from_start = torch.mean(projected_distance_from_start_batch)
                loss = -projected_distance_from_start
                # bias = torch.tensor(svm_model.intercept_, dtype=torch.float16).to(model.device)
                # constrain = torch.tensor(svm_model.C, dtype=torch.float16).to(model.device)

                # f_out = weights.dot(hidden_state.squeeze(0)) + bias
                # # prob = 1 - torch.sigmoid(f_out)
                # hinge = max(0, 1 - 1 * f_out) # hinge == 0
                # loss = 0.5 * torch.dot(weights, weights) + constrain * hinge
                # distance = torch.norm(hidden_state - start_hidden_state)
                # loss = prob * distance
                loss_list[task_i].append(loss.reshape(1))
        
        total_losses = []
        for task_loss in loss_list:
            avg_task_loss = torch.stack(task_loss).mean(dim=0)
            total_losses.append(avg_task_loss)
        total_loss = torch.stack(total_losses).mean(dim=0)
        
        # Backward pass
        optimizer.zero_grad()
        try:
            total_loss.backward()
        except:
            eval_responses = []
            eval_jailbroken = []
            for eval_model in eval_models:
                _, eval_response = check_jailbreak_success(
                    model=eval_model,
                    tokenizer=tokenizer,
                    prompt=f"{prompt} {adv_string}",
                    generate_length=512,
                    return_response=True,
                )
                eval_responses.append(eval_response)
                eval_rating, _ = ask_judge_harmbench(
                    model_judge,
                    tokenizer_judge,
                    prompt,
                    eval_response,
                )
                eval_jailbroken.append(eval_rating == 1)

            preds = svm_predict(model, tokenizer, select_svm_models, select_layers, f"{prompt} {adv_string}")

            prompt_end_time = time.time()
            prompt_time = prompt_end_time - prompt_start_time
            prompt_time_strf = time.strftime("%H:%M:%S", time.gmtime(prompt_time))
            result = {
                "idx": prompt_idx,
                "prompt": prompt,
                "adv_string": adv_string,
                "response": response,
                "jailbroken": is_jailbroken,
                "epoch": epoch,
                "loss": current_loss,
                "attack_time": prompt_time_strf,
                "eval_responses": eval_responses,
                "eval_jailbroken": eval_jailbroken,
                "svm predict": preds,
            }
            return result

        # Sample candidates
        sampled_tokenids = sample_control(
            adv_tokenids.squeeze(0),
            adv_onehot.grad.squeeze(0),
            search_width=sampling_number,
            topk=256,
            temp=1,
            not_allowed_tokens=not_allowed_tokens,
        )
        try:
            sampled_tokenids = filter_candidates(sampled_tokenids, tokenizer)
        except ValueError as e:
            # No candidates left. Attack cannot proceed anymore.
            print(f"FAILED: {e}")
            result = {
                "idx": prompt_idx,
                "prompt": repr(prompt),
                "adv_string": repr(init_adv_string),
                "response": "FAILED",
                "jailbroken": False,
                "epoch": -1,
                "loss": -1,
                "attack_time": "00:00:00",
            }
            return result

        # Forward pass #2 (not requires grad): Calculate candidates loss
        sampled_embeds = embed_layer(sampled_tokenids)
        full_embed = torch.cat(
            [
                full_before_adv_embed.repeat(sampled_tokenids.shape[0], 1, 1),
                sampled_embeds,
                full_after_adv_embed.repeat(sampled_tokenids.shape[0], 1, 1),
            ],
            dim=1,
        )

        losses_batch = second_forward(
            model,
            full_embed,
            hidden_states_start_point,
            select_svm_models,
            select_layers
        )

        adv_tokenids = sampled_tokenids[losses_batch.argmin()]
        current_loss = losses_batch.min().item()
        adv_string = tokenizer.decode(adv_tokenids)

        # Check if the adversarial example is successful
        is_jailbroken, response = check_jailbreak_success(
            model=model,
            tokenizer=tokenizer,
            prompt=f"{prompt} {adv_string}",
            generate_length=32,
            return_response=True,
        )
        ##### EPOCH LOG #####
        print()
        print(f"==>> loss_batch.shape: {losses_batch.shape}")
        print(f"==>> loss: {current_loss}")
        print(f"==>> adv_string: {repr(adv_string)}")
        print(f"==>> is_jailbroken: {is_jailbroken}")
        print(f"==>> response: {repr(response)}")
        sys.stdout.flush()
        loss_history.append(current_loss)
        ##### EPOCH LOG #####

        if is_jailbroken:
            print(f"==>> Jailbreak successful at epoch {epoch}. Double-checking...")
            # Double-check
            _, response = check_jailbreak_success(
                model=model,
                tokenizer=tokenizer,
                prompt=f"{prompt} {adv_string}",
                generate_length=512,
                return_response=True,
            )
            rating, _ = ask_judge_harmbench(
                model_judge,
                tokenizer_judge,
                prompt,
                response,
            )
            is_jailbroken = rating == 1
            if is_jailbroken:
                print(f"==>> Jailbreak indeed successful at epoch {epoch}")
                if exit_on_success:
                    break

    eval_responses = []
    eval_jailbroken = []
    for eval_model in eval_models:
        _, eval_response = check_jailbreak_success(
            model=eval_model,
            tokenizer=tokenizer,
            prompt=f"{prompt} {adv_string}",
            generate_length=512,
            return_response=True,
        )
        eval_responses.append(eval_response)
        eval_rating, _ = ask_judge_harmbench(
            model_judge,
            tokenizer_judge,
            prompt,
            eval_response,
        )
        eval_jailbroken.append(eval_rating == 1)

    preds = svm_predict(model, tokenizer, select_svm_models, select_layers, f"{prompt} {adv_string}")

    prompt_end_time = time.time()
    prompt_time = prompt_end_time - prompt_start_time
    prompt_time_strf = time.strftime("%H:%M:%S", time.gmtime(prompt_time))

    result = {
        "idx": prompt_idx,
        "prompt": prompt,
        "adv_string": adv_string,
        "response": response,
        "jailbroken": is_jailbroken,
        "epoch": epoch,
        "loss": current_loss,
        "attack_time": prompt_time_strf,
        "eval_responses": eval_responses,
        "eval_jailbroken": eval_jailbroken,
        "svm predict": preds,
    }

    ##### ATTACK LOG #####
    print("Prompt Result".center(50, "-"))
    print(f"==>> Time: {prompt_time_strf}")
    print(f"==>> prompt: {repr(prompt)}")
    print(f"==>> adv_string: {repr(adv_string)}")
    print(f"==>> response: {repr(response)}")
    print(f"==>> is_jailbroken: {is_jailbroken}")
    print(f"==>> epoch: {epoch}")
    print(f"==>> loss: {loss.item()}")
    print("Prompt Result".center(50, "-"))
    sys.stdout.flush()

    return result

def load_probes(model_name, task_list, n_layers, acc_threshold, probe_type, seed, c_value):
    if probe_type == 'H2':
        probe_model_path = 'hidden_states/finetuned'
        # probe_mlp_model_list = [[] for i in range(len(task_list))]
        # probe_svm_model_list = [[] for i in range(len(task_list))]
        # for i, task in enumerate(task_list):
        #     for layer in range(1, n_layers):
        #         # mlp_model = joblib.load(f"{probe_model_path}/{task}/{model_name}/results/cav_mlp/{probe_type}_layer{str(layer)}.joblib")
        #         svm_model = joblib.load(f"{probe_model_path}/{task}/{model_name}/results/cav_svm/{probe_type}_layer{str(layer)}.joblib")
        #         # probe_mlp_model_list[i].append(mlp_model)
        #         probe_svm_model_list[i].append(svm_model)
    else:
        probe_model_path = 'hidden_states/pretrained'
        # probe_mlp_model_list = [[]]
        acc_list = torch.load(f"{probe_model_path}/{model_name}/probing_models_space/{probe_type}/seed{str(seed)}_c{str(c_value)}/linear_svm_acc.pth", map_location="cpu")
        p_list = torch.load(f"{probe_model_path}/{model_name}/probing_models_space/{probe_type}/seed{str(seed)}_c{str(c_value)}/linear_svm_pvalue.pth", map_location="cpu")

        probe_svm_model_list = [[]]
        select_layer_list = [[]]
        for layer in range(n_layers):
            if p_list[layer] < 0.05 and acc_list[layer] >= acc_threshold:
                # svm_model = joblib.load(f"{probe_model_path}/{model_name}/probing_models/{probe_type}/layer{str(select_layer)}/linear_svm_c{str(c_value)}.joblib")
                svm_model = joblib.load(f"{probe_model_path}/{model_name}/probing_models_space/{probe_type}/seed{str(seed)}_c{str(c_value)}/linear_svm_layer{str(layer)}.joblib")
                # svm_model = joblib.load(f"{probe_model_path}/{model_name}/probing_models_ensemble/{probe_type}/union{str(seed)}_c{str(c_value)}/linear_svm_layer{str(select_layer)}.joblib")
                probe_svm_model_list[0].append(svm_model)
                select_layer_list[0].append(layer)
    
    return probe_svm_model_list, select_layer_list


def svm_predict(model, tokenizer, select_svm_models, select_layers, prompt):
    messages = [{"role": "user", "content": prompt}]
    full_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    input_encoded = tokenizer(full_prompt, return_tensors="pt", padding=True).to(
        model.device
    )
    outputs = model(**input_encoded, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    # hidden_state = hidden_states[-1][0][-1]
    # hidden_state = hidden_state.to(torch.float16)

    pred_list = [[] for _ in range(len(select_svm_models))]
    for task_i, task_svm_models in enumerate(select_svm_models):
        for idx, svm_model in enumerate(task_svm_models):
            layer_id = select_layers[task_i][idx]
            hidden_state = hidden_states[layer_id][0][-1]
            hidden_state = hidden_state.to(torch.float16)

            pred = svm_model.predict(hidden_state.reshape(1, -1).cpu())
            pred_list[task_i].append(pred.item() == 1)
    return pred_list


def main():
    args = parse_args()
    for key, value in vars(args).items():
        print(f"{key}: {repr(value)}")
    set_seed(args.seed)

    # dataset path
    dataset_path = {
        'advbench': 'data/advbench/harmful_behaviors.csv',
        'malicious': 'data/malicious/malicious.csv',
        'harmbench': 'data/harmbench/harmbench_gcg.csv',
    }

    # Load datasets
    dataset = pd.read_csv(dataset_path[args.dataset], usecols=[0, 1]).to_numpy()
    if args.idx is not None:
        lower_bound = max(0, args.idx[0])
        lower_bound = min(len(dataset), lower_bound)
        upper_bound = min(len(dataset), args.idx[1])
        upper_bound = max(0, upper_bound)
        dataset = dataset[lower_bound:upper_bound]
    else:
        lower_bound = 0
        upper_bound = len(dataset)

    # import ipdb
    # ipdb.set_trace()
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(f"models/{model_dict[args.model_name]}")
    model.eval()
    model.requires_grad_(False)  # Save memory
    model_judge, tokenizer_judge = load_model_and_tokenizer(args.model_judge)
    model_judge.eval()
    model_judge.requires_grad_(False)

    # Load finetuned eval model
    eval_models = []

    # Load Linear Probe Models
    n_layers = layer_dict[args.model_name] + 1
    probe_task_list = args.probe_tasks.split('_')

    acc_threshold = args.acc_threshold
    
    select_svm_models, select_layers = load_probes(args.model_name, probe_task_list, n_layers, acc_threshold, args.probe_type, args.probe_seed, args.c_value)

    # Run the jailbreak attack
    results = []

    create_dir(args.output_dir)
    create_dir(f"{args.output_dir}/{args.dataset}")

    # result_filepath = f"{args.output_dir}/{args.model_name}/{args.probe_type}_topK{args.topk_layers}_c{str(args.c_value)}"
    result_filepath = f"{args.output_dir}/{args.dataset}/{args.probe_type}_c{str(args.c_value)}_seed{str(args.probe_seed)}_acc{str(args.acc_threshold)}"
    create_dir(result_filepath)
    
    result_filename = f"{result_filepath}/{str(lower_bound)}_{str(upper_bound)}.json"
    print(result_filename)
    for idx, (goal, target) in tqdm(enumerate(dataset, start=lower_bound), total=len(dataset)):
        prompt_result = jailbreak_this_prompt(
            prompt=goal,
            target=target,
            left_adv=args.left_adv,
            model=model,
            eval_models=eval_models,
            select_svm_models=select_svm_models,
            select_layers=select_layers,
            tokenizer=tokenizer,
            init_adv_string=args.init_adv_string,
            max_epochs=args.max_epochs,
            sampling_number=args.sampling_number,
            model_judge=model_judge,
            tokenizer_judge=tokenizer_judge,
            tqdm_desc=f"Jailbreaking {idx}/{lower_bound}-{upper_bound}",
            prompt_idx=idx,
            plot_dir=args.output_plot_dir,
        )
        results.append(prompt_result)

        with open(result_filename, 'w') as f:
            json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
