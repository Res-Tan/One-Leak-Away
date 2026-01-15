import argparse
import json
import os
import sys
import time
import copy

import matplotlib.pyplot as plt
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
        "--model_epoch",
        type=str,
        default="epoch1_safe0",
        help="Evaluating Finetune Models",
    )
    parser.add_argument(
        "--lambda_h2",
        type=float,
        default=False,
        help="H1 Plus H2",
    )
    parser.add_argument(
        "--acc_threshold",
        type=float,
        nargs="+",
        default=None,
        help="Perturbing layers",
    )
    parser.add_argument(
        "--c_value",
        type=float,
        nargs="+",
        default=None,
        help="Perturbing layers",
    )
    parser.add_argument(
        "--probe_seed",
        type=int,
        default=None,
        help="Perturbing layers",
    )
    parser.add_argument(
        "--target_task",
        type=str,
        default="alpaca",
        help="target ft model",
    )
    parser.add_argument(
        "--blackbox",
        type=bool,
        default=False,
        help="with query access to the target",
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

            hidden_states_batch = outputs.hidden_states
            # hidden_states_batch = hidden_states_batch.view(
            #     hidden_states_batch.shape[0], -1
            # )

            loss_list = [[] for _ in range(len(select_svm_models))]
            for task_i, task_svm_models in enumerate(select_svm_models):
                for idx, svm_model in enumerate(task_svm_models):
                    weights = torch.tensor(svm_model.coef_, dtype=torch.float16).squeeze(0).to(model.device)
                    direction_norm_vec = weights / torch.norm(weights)

                    layer_id = select_layers[task_i][idx]
                    hidden_state_batch = hidden_states_batch[layer_id][:, -1, :]
                    hidden_state_start_point = hidden_states_start_point[layer_id][:, -1, :]

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


@find_executable_batch_size(starting_batch_size=256)
def second_forward_plus(candidate_batch_size, model, full_embed, lambda_h2, hidden_states_start_point, select_svm_models_h1, select_layers_h1, select_svm_models_h2, select_layers_h2):
    losses_batch = []
    for i in range(0, full_embed.shape[0], candidate_batch_size):
        with torch.no_grad():
            full_embed_this_batch = full_embed[i : i + candidate_batch_size]
            outputs = model(
                inputs_embeds=full_embed_this_batch, output_hidden_states=True
            )

            hidden_states_batch = outputs.hidden_states
            # hidden_states_batch = hidden_states_batch.view(
            #     hidden_states_batch.shape[0], -1
            # )

            loss_list_h1 = [[] for _ in range(len(select_svm_models_h1))]
            for task_i, task_svm_models in enumerate(select_svm_models_h1):
                for idx, svm_model in enumerate(task_svm_models):
                    weights = torch.tensor(svm_model.coef_, dtype=torch.float16).squeeze(0).to(model.device)
                    direction_norm_vec = weights / torch.norm(weights)

                    layer_id = select_layers_h1[task_i][idx]
                    hidden_state_batch = hidden_states_batch[layer_id][:, -1, :]
                    hidden_state_start_point = hidden_states_start_point[layer_id][:, -1, :]

                    vector_from_start_to_here_batch = hidden_state_batch - hidden_state_start_point
                    
                    projected_distance_from_start_batch = torch.sum(
                        vector_from_start_to_here_batch * direction_norm_vec, dim=1
                    )
                    loss = -projected_distance_from_start_batch
                    loss_list_h1[task_i].append(loss)

            loss_list_h2 = [[] for _ in range(len(select_svm_models_h2))]
            for task_i, task_svm_models in enumerate(select_svm_models_h2):
                for idx, svm_model in enumerate(task_svm_models):
                    weights = torch.tensor(svm_model.coef_, dtype=torch.float16).squeeze(0).to(model.device)
                    direction_norm_vec = weights / torch.norm(weights)

                    layer_id = select_layers_h2[task_i][idx]
                    hidden_state_batch = hidden_states_batch[layer_id][:, -1, :]
                    hidden_state_start_point = hidden_states_start_point[layer_id][:, -1, :]
                    vector_from_start_to_here_batch = hidden_state_batch - hidden_state_start_point
                    
                    projected_distance_from_start_batch = torch.sum(
                        vector_from_start_to_here_batch * direction_norm_vec, dim=1
                    )
                    loss = -projected_distance_from_start_batch
                    loss = lambda_h2 * loss
                    loss_list_h2[task_i].append(loss)


            loss_list = loss_list_h1 + loss_list_h2
            
            total_losses = []
            for task_loss in loss_list:
                avg_task_loss = torch.stack(task_loss).mean(dim=0)
                total_losses.append(avg_task_loss)
            total_loss = torch.stack(total_losses).mean(dim=0)
            losses_batch.append(total_loss)

    losses_batch = torch.cat(losses_batch, dim=0)
    return losses_batch


def double_check(epoch, model, tokenizer, model_judge, tokenizer_judge, prompt, adv_string):
    print(f"==>> Jailbreak successful at epoch {epoch} on PT. Double-checking...")
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
    return is_jailbroken, response


def jailbreak_this_prompt(
    prompt: str,
    target: str,
    model,
    eval_models,
    target_model,
    select_svm_models_h1,
    select_layers_h1,
    select_svm_models_h2,
    select_layers_h2,
    lambda_h2,
    blackbox,
    tokenizer,
    model_judge,
    tokenizer_judge,
    pretrained_epoch,
    pretrained_adv_string,
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

    # Initialize pretrained adversarial string
    pretrained_adv_tokenids = (
        tokenizer.encode(pretrained_adv_string, add_special_tokens=False, return_tensors="pt")
        .squeeze(0)
        .to(model.device)
    )
    pretrained_adv_tokenids = pretrained_adv_tokenids.detach().clone()

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
    hidden_states_start_point = output.hidden_states
    # hidden_states_start_point = hidden_states_start_point.view(
    #     hidden_states_start_point.shape[0], -1
    # )

    loss_history = []
    attack_signs = [False for _ in range(len(select_svm_models_h2) + 1)]
    eval_responses = []

    # evaluate pretrained adversarial string
    is_jailbroken, response = double_check(pretrained_epoch, model, tokenizer, model_judge, tokenizer_judge, prompt, pretrained_adv_string)
    attack_signs[0] = is_jailbroken

    eval_responses = []
    eval_jailbroken = []
    for eval_model in eval_models:
        _, eval_response = check_jailbreak_success(
            model=eval_model,
            tokenizer=tokenizer,
            prompt=f"{prompt} {pretrained_epoch}",
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
    attack_signs[1:] = eval_jailbroken[:len(select_svm_models_h2)]

    print('intial states:', str(attack_signs))

    if all(attack_signs):
        preds_h1 = svm_predict(model, tokenizer, select_svm_models_h1, select_layers_h1, f"{prompt} {pretrained_adv_string}")
        preds_h2 = svm_predict(model, tokenizer, select_svm_models_h2, select_layers_h2, f"{prompt} {pretrained_adv_string}")

        if blackbox:
            result = {
                "idx": prompt_idx,
                "prompt": prompt,
                "adv_string": pretrained_adv_string,
                "response": response,
                "jailbroken": is_jailbroken,
                "epoch": pretrained_epoch,
                "loss": 0,
                "attack_time": "00:00:00",
                "eval_responses": eval_responses,
                "eval_jailbroken": eval_jailbroken,
                "svm predict h1": preds_h1,
                "svm predict h2": preds_h2
            }
        else:
            eval_responses = []
            eval_jailbroken = []
            _, eval_response = check_jailbreak_success(
                model=target_model,
                tokenizer=tokenizer,
                prompt=f"{prompt} {pretrained_epoch}",
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
            result = {
                "idx": prompt_idx,
                "prompt": prompt,
                "adv_string": pretrained_adv_string,
                "response": response,
                "jailbroken": is_jailbroken,
                "epoch": pretrained_epoch,
                "loss": 0,
                "attack_time": "00:00:00",
                "eval_responses": eval_responses,
                "eval_jailbroken": eval_jailbroken,
                "svm predict h1": preds_h1,
                "svm predict h2": preds_h2
            }
        return result

    # record best adv_string
    best_adv_string = pretrained_adv_string
    best_response = response
    best_attack_signs = copy.deepcopy(attack_signs)
    best_eval_responses = copy.deepcopy(eval_responses)
    best_eval_jailbroken = copy.deepcopy(eval_jailbroken)

    # Attack loop
    for epoch in tqdm(range(0, max_epochs), desc=tqdm_desc):
        pretrained_adv_onehot = (
            tokenids2onehot(pretrained_adv_tokenids, vocab_size, embed_layer.weight.dtype)
            .unsqueeze(0)
            .detach()
            .clone()
            .to(model.device)
        )
        pretrained_adv_onehot.requires_grad_(True)
        # Optimizer is used to zero the gradients. Not used for optimization
        optimizer = torch.optim.Adam([pretrained_adv_onehot], lr=0.1)
        # Do this manually to avoid breaking the computation graph
        adv_embed = pretrained_adv_onehot @ embed_layer.weight

        # Zeroth-order optimization: two forward passes
        # Forward pass #1 (requires grad): Calculate promising candidates
        full_embed = torch.cat(
            [full_before_adv_embed, adv_embed, full_after_adv_embed],
            dim=1,
        )
        outputs = model(inputs_embeds=full_embed, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        loss_list_h2 = [[] for _ in range(len(select_svm_models_h2))]
        for task_i, task_svm_models in enumerate(select_svm_models_h2):
            for idx, svm_model in enumerate(task_svm_models):
                layer_id = select_layers_h2[task_i][idx]
                weights = torch.tensor(svm_model.coef_, dtype=torch.float16).squeeze(0).to(model.device)
                direction_norm_vec = weights / torch.norm(weights)

                hidden_state_start_point = hidden_states_start_point[layer_id][:, -1, :]
                hidden_state_start_point = hidden_state_start_point.view(
                    hidden_state_start_point.shape[0], -1
                )
                hidden_state = hidden_states[layer_id][:, -1, :]
                vector_from_start_to_here_batch = hidden_state - hidden_state_start_point
                
                projected_distance_from_start_batch = torch.sum(
                    vector_from_start_to_here_batch * direction_norm_vec, dim=1
                )
                projected_distance_from_start = torch.mean(projected_distance_from_start_batch)
                loss = -projected_distance_from_start
                loss = lambda_h2 * loss
                loss_list_h2[task_i].append(loss.reshape(1))

        loss_list_h1 = [[] for _ in range(len(select_svm_models_h1))]
        for task_i, task_svm_models in enumerate(select_svm_models_h1):
            for idx, svm_model in enumerate(task_svm_models):
                layer_id = select_layers_h1[task_i][idx]
                weights = torch.tensor(svm_model.coef_, dtype=torch.float16).squeeze(0).to(model.device)
                direction_norm_vec = weights / torch.norm(weights)

                hidden_state_start_point = hidden_states_start_point[layer_id][:, -1, :]
                hidden_state_start_point = hidden_state_start_point.view(
                    hidden_state_start_point.shape[0], -1
                )
                hidden_state = hidden_states[layer_id][:, -1, :]
                vector_from_start_to_here_batch = hidden_state - hidden_state_start_point

                projected_distance_from_start_batch = torch.sum(
                    vector_from_start_to_here_batch * direction_norm_vec, dim=1
                )
                projected_distance_from_start = torch.mean(projected_distance_from_start_batch)
                loss = -projected_distance_from_start
                loss_list_h1[task_i].append(loss.reshape(1))

        loss_list = loss_list_h1 + loss_list_h2
        
        total_losses = []
        for task_loss in loss_list:
            avg_task_loss = torch.stack(task_loss).mean(dim=0)
            total_losses.append(avg_task_loss)
        total_loss = torch.stack(total_losses).mean(dim=0)
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()

        # Sample candidates
        sampled_tokenids = sample_control(
            pretrained_adv_tokenids.squeeze(0),
            pretrained_adv_onehot.grad.squeeze(0),
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
                "adv_string": repr(pretrained_adv_string),
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

        if not attack_signs[0]:
            print("Re-attacking in PT")
            losses_batch = second_forward(
                model,
                full_embed,
                hidden_states_start_point,
                select_svm_models_h1,
                select_layers_h1
            )
        else:
            print("Attacking in FT")
            losses_batch = second_forward_plus(
                model,
                full_embed,
                lambda_h2,
                hidden_states_start_point, # or fail hs
                select_svm_models_h1,
                select_layers_h1,
                select_svm_models_h2,
                select_layers_h2
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
        preds_h1 = svm_predict(model, tokenizer, select_svm_models_h1, select_layers_h1, f"{prompt} {adv_string}")
        preds_h2 = svm_predict(model, tokenizer, select_svm_models_h2, select_layers_h2, f"{prompt} {adv_string}")

        ##### EPOCH LOG #####
        print()
        print(f"==>> loss_batch.shape: {losses_batch.shape}")
        print(f"==>> loss: {current_loss}")
        print(f"==>> adv_string: {repr(adv_string)}")
        print(f"==>>( is_jailbroken: {str(attack_signs)}")
        print(f'==>>( best_jailbroken: {str(best_attack_signs)}')
        print(f"==>> response: {repr(response)}")
        print(f"==>> linear prediction h1: {str(all(preds_h1))}")
        print(f"==>> linear prediction h2: {str(all(preds_h2))}")
        sys.stdout.flush()
        loss_history.append(current_loss)
        ##### EPOCH LOG #####

        if is_jailbroken:
            is_jailbroken, response = double_check(epoch, model, tokenizer, model_judge, tokenizer_judge, prompt, adv_string)
            attack_signs[0] = is_jailbroken

        if is_jailbroken:
            eval_responses = []
            eval_jailbroken = []
            for eval_model in eval_models:
                is_jailbroken, eval_response = check_jailbreak_success(
                    model=eval_model,
                    tokenizer=tokenizer,
                    prompt=f"{prompt} {adv_string}",
                    generate_length=32,
                    return_response=True,
                )
                if is_jailbroken:
                    print(f"==>> Jailbreak successful at epoch {epoch} on FT. Double-checking...")
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
                else:
                    eval_jailbroken.append(is_jailbroken)
            attack_signs[1:] = eval_jailbroken[:len(select_svm_models_h2)]

            if best_attack_signs.count(True) < attack_signs.count(True):
                best_attack_signs = copy.deepcopy(attack_signs)
                best_response = response
                best_eval_responses = copy.deepcopy(eval_responses)
                best_eval_jailbroken = copy.deepcopy(eval_jailbroken)
                best_adv_string = adv_string

        if all(attack_signs):
            break

    # preds = svm_predict(model, tokenizer, select_svm_models_h2, [], f"{prompt} {best_adv_string}")
    preds_h1 = svm_predict(model, tokenizer, select_svm_models_h1, select_layers_h1, f"{prompt} {best_adv_string}")
    preds_h2 = svm_predict(model, tokenizer, select_svm_models_h2, select_layers_h2, f"{prompt} {best_adv_string}")


    prompt_end_time = time.time()
    prompt_time = prompt_end_time - prompt_start_time
    prompt_time_strf = time.strftime("%H:%M:%S", time.gmtime(prompt_time))

    result = {
        "idx": prompt_idx,
        "prompt": prompt,
        "adv_string": best_adv_string,
        "response": best_response,
        "jailbroken": best_attack_signs[0],
        "pretrained_epoch": pretrained_epoch,
        "finetuned_epoch": epoch,
        "loss": current_loss,
        "attack_time": prompt_time_strf,
        "eval_responses": best_eval_responses,
        "eval_jailbroken": best_eval_jailbroken,
        "svm predict h1": preds_h1,
        "svm predict h2": preds_h2,
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
    if probe_type == "H2":
        probe_model_path = 'hidden_states/pretrained'
        # probe_mlp_model_list = [[] for i in range(len(task_list))]
        probe_svm_model_list = [[] for i in range(len(task_list))]
        select_layer_list = [[] for i in range(len(task_list))]

        for i, task in enumerate(task_list):
            acc_list = torch.load(f"{probe_model_path}/{model_name}/probing_models_space/{probe_type}/{task}/seed{str(seed)}_c{str(c_value)}/linear_svm_acc.pth", map_location="cpu")
            p_list = torch.load(f"{probe_model_path}/{model_name}/probing_models_space/{probe_type}/{task}/seed{str(seed)}_c{str(c_value)}/linear_svm_pvalue.pth", map_location="cpu")
            print(f"{i}, {task}, acc_list: {acc_list}")
            for layer in range(n_layers):
                if p_list[layer] < 0.05 and acc_list[layer] >= acc_threshold:
                    svm_model = joblib.load(f"{probe_model_path}/{model_name}/probing_models_space/{probe_type}/{task}/seed{str(seed)}_c{str(c_value)}/linear_svm_layer{str(layer)}.joblib")
                    probe_svm_model_list[i].append(svm_model)
                    select_layer_list[i].append(layer)
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

    # Load datasets
    pretrained_path = f"pgp_gcg/{args.model_name}/{args.dataset}/H1_c{args.c_value[0]}_seed{args.probe_seed}_acc{args.acc_threshold[0]}"
    with open(f'{pretrained_path}/{args.idx[0]}_{args.idx[1]}.json', 'r') as f:
        data_list = json.load(f)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(f"models/{model_dict[args.model_name]}")
    model.eval()
    model.requires_grad_(False)  # Save memory
    model_judge, tokenizer_judge = load_model_and_tokenizer(args.model_judge)
    model_judge.eval()
    model_judge.requires_grad_(False)

    # Load target finetuned model
    target_path = f"finetuned_models/{args.target_task}/{args.model_name}/inst_{args.model_epoch}"
    target_model = load_model_and_tokenizer(target_path, True)
    target_model.eval()
    target_model.requires_grad_(False)

    eval_models = []
    if args.blackbox:
        eval_models.append(target_model)
    else:
        # Load surrogate eval model
        eval_tasks = args.eval_tasks.split('_')
        for task_name in eval_tasks:
            eval_path = f"finetuned_models/{task_name}/{args.model_name}/inst_{args.model_epoch}"
            eval_model = load_model_and_tokenizer(eval_path, True)
            eval_model.eval()
            eval_model.requires_grad_(False)
            eval_models.append(eval_model)

    n_layers = layer_dict[args.model_name] + 1

    # Load Linear Probe Models
    select_svm_models_h1, select_layers_h1 = load_probes(args.model_name, [], n_layers, args.acc_threshold[0], "H1", args.probe_seed, args.c_value[0])
    if args.blackbox:
        select_svm_models_h2, select_layers_h2 = load_probes(args.model_name, [args.target_task], n_layers, args.acc_threshold[1], "H2", args.probe_seed, args.c_value[1])
    else:
        select_svm_models_h2, select_layers_h2 = load_probes(args.model_name, eval_tasks, n_layers, args.acc_threshold[1], "H2", args.probe_seed, args.c_value[1])

    # Run the jailbreak attack
    results = []
    if not os.path.exists(f"{args.output_dir}/blackbox"):
        os.makedirs(f"{args.output_dir}/blackbox")

    if not os.path.exists(f"{args.output_dir}/nobox"):
        os.makedirs(f"{args.output_dir}/nobox")
        
    if args.blackbox:
        result_filepath = f"{args.output_dir}/blackbox/c{args.c_value[0]}_{args.c_value[1]}_acc{args.acc_threshold[0]}_{args.acc_threshold[1]}_l{args.lambda_h2}"
    else:
        result_filepath = f"{args.output_dir}/nobox/c{args.c_value[0]}_{args.c_value[1]}_acc{args.acc_threshold[0]}_{args.acc_threshold[1]}_l{args.lambda_h2}"
    
    if not os.path.exists(result_filepath):
        os.makedirs(result_filepath)

    result_filename = f"{result_filepath}/{str(args.idx[0])}_{str(args.idx[1])}.json"

    print(result_filename)
    for idx, data in tqdm(enumerate(data_list), total=len(data_list)):
        if data['epoch'] == 499:
            eval_responses = []
            eval_jailbroken = []
            
            _, eval_response = check_jailbreak_success(
                model=target_model,
                tokenizer=tokenizer,
                prompt=f"{data['prompt']} {data['adv_string']}",
                generate_length=512,
                return_response=True,
            )
            eval_responses.append(eval_response)
            eval_rating, _ = ask_judge_harmbench(
                model_judge,
                tokenizer_judge,
                data['prompt'],
                eval_response,
            )
            eval_jailbroken.append(eval_rating == 1)

            data['eval_responses'] = eval_responses
            data['eval_jailbroken'] = eval_jailbroken
            results.append(data)
            continue

        prompt_result = jailbreak_this_prompt(
            prompt=data['prompt'],
            target="",
            model=model,
            eval_models=eval_models,
            target_model=target_model,
            select_svm_models_h1=select_svm_models_h1,
            select_layers_h1=select_layers_h1,
            select_svm_models_h2=select_svm_models_h2,
            select_layers_h2=select_layers_h2,
            lambda_h2=args.lambda_h2,
            blackbox=args.blackbox,
            tokenizer=tokenizer,
            pretrained_epoch=data['epoch'],
            pretrained_adv_string=data['adv_string'],
            init_adv_string=args.init_adv_string,
            max_epochs=args.max_epochs,
            sampling_number=args.sampling_number,
            model_judge=model_judge,
            tokenizer_judge=tokenizer_judge,
            tqdm_desc=f"Jailbreaking {idx + args.idx[0]}/{args.idx[0]}-{args.idx[1]}",
            prompt_idx=idx + args.idx[0],
            plot_dir=args.output_plot_dir,
        )
        results.append(prompt_result)

        with open(result_filename, 'w') as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
