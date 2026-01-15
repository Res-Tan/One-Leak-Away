import argparse
import json
import os
import sys
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
from accelerate.utils import find_executable_batch_size
from tqdm import tqdm

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
        "--early_stop",
        type=bool,
        default=False,
        help="Early Stop",
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
    # os.makedirs(args.output_plot_dir, exist_ok=True)

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
def second_forward(candidate_batch_size, model, full_embed, target_embed, target_ids):
    losses_batch_batch = []
    for i in range(0, full_embed.shape[0], candidate_batch_size):
        with torch.no_grad():
            full_embed_this_batch = full_embed[i : i + candidate_batch_size]
            output_logits = model(inputs_embeds=full_embed_this_batch).logits
            # Loss
            # Shift so that tokens < n predict n
            idx_before_target = full_embed.shape[1] - target_embed.shape[1]
            shift_logits = output_logits[
                ..., idx_before_target - 1 : -1, :
            ].contiguous()
            shift_labels = target_ids.repeat(full_embed_this_batch.shape[0], 1)
            # Flatten the tokens
            losses_this_batch = torch.nn.CrossEntropyLoss(reduction="none")(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            losses_this_batch = losses_this_batch.view(
                full_embed_this_batch.shape[0], -1
            ).mean(dim=1)

            losses_batch_batch.append(losses_this_batch)

    losses_batch = torch.cat(losses_batch_batch, dim=0)
    return losses_batch


def jailbreak_this_prompt(
    prompt: str,
    target: str,
    model,
    eval_models,
    tokenizer,
    model_judge,
    tokenizer_judge,
    init_adv_string="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
    max_epochs: int = 512,
    sampling_number: int = 512,
    early_stop: bool = True,
    exit_on_success = True,
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
    target_ids = tokenizer.encode(
        target, padding=False, add_special_tokens=False, return_tensors="pt"
    ).to(model.device)
    full_before_adv_embed = embed_layer(full_before_adv_ids)
    full_after_adv_embed = embed_layer(full_after_adv_ids)
    target_embed = embed_layer(target_ids)

    loss_history = []
    pca_history = []
    prompt_history = []

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
            [full_before_adv_embed, adv_embed, full_after_adv_embed, target_embed],
            dim=1,
        )
        output_logits = model(inputs_embeds=full_embed).logits

        # Loss
        # Shift so that tokens < n predict n
        idx_before_target = full_embed.shape[1] - target_embed.shape[1]
        shift_logits = output_logits[..., idx_before_target - 1 : -1, :].contiguous()
        shift_labels = target_ids
        # Flatten the tokens
        loss = torch.nn.CrossEntropyLoss()(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Sample candidates
        sampled_tokenids = sample_control(
            adv_tokenids.squeeze(0),
            adv_onehot.grad.squeeze(0),
            search_width=sampling_number,
            topk=256,
            temp=1,
            not_allowed_tokens=not_allowed_tokens,
        )
        sampled_tokenids = filter_candidates(sampled_tokenids, tokenizer)

        # Forward pass #2 (not requires grad): Calculate candidates loss
        sampled_embeds = embed_layer(sampled_tokenids)
        full_embed = torch.cat(
            [
                full_before_adv_embed.repeat(sampled_tokenids.shape[0], 1, 1),
                sampled_embeds,
                full_after_adv_embed.repeat(sampled_tokenids.shape[0], 1, 1),
                target_embed.repeat(sampled_tokenids.shape[0], 1, 1),
            ],
            dim=1,
        )

        losses_batch = second_forward(
            model,
            full_embed,
            target_embed.repeat(sampled_tokenids.shape[0], 1, 1),
            target_ids,
        )
        adv_tokenids = sampled_tokenids[losses_batch.argmin()]
        current_loss = losses_batch.min().item()
        adv_string = tokenizer.decode(adv_tokenids)
        prompt_history.append({
            'ind': prompt_idx,
            'prompt': prompt,
            'adv_string': adv_string
        })

        if early_stop:
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
        else:
            ##### EPOCH LOG #####
            # print()
            # print(f"==>> loss_batch.shape: {losses_batch.shape}")
            # print(f"==>> loss: {current_loss}")
            # print(f"==>> adv_string: {repr(adv_string)}")
            # sys.stdout.flush()
            loss_history.append(current_loss)
        ##### EPOCH LOG #####
        if early_stop and is_jailbroken:
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
    
    if not early_stop:
        is_jailbroken, response = check_jailbreak_success(
            model=model,
            tokenizer=tokenizer,
            prompt=f"{prompt} {adv_string}",
            generate_length=512,
            return_response=True,
        )
        # pretrain model test
        rating, _ = ask_judge_harmbench(
            model_judge,
            tokenizer_judge,
            prompt,
            response,
        )
        is_jailbroken = rating == 1
    # eval_model test
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
    # Plot the loss history
    # loss_plot_filename = os.path.join(plot_dir, f"loss_{prompt_idx}.png")
    # plt.figure()
    # plt.plot(loss_history)
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.title("Loss History")
    # plt.savefig(loss_plot_filename)
    # plt.close()

    # # Plot the PCA trace
    # pca_plot_filename = os.path.join(plot_dir, f"pca_{prompt_idx}.png")
    # pca_history = np.array(pca_history)
    # plt.figure()
    # # start point
    # plt.plot(pca_history[0, 0], pca_history[0, 1], marker="o", color="r")
    # plt.plot(
    #     pca_history[1:-1, 0],
    #     pca_history[1:-1, 1],
    #     marker=".",
    #     color="b",
    #     markersize=2,
    #     alpha=0.5,
    # )
    # # end point
    # plt.plot(pca_history[-1, 0], pca_history[-1, 1], marker="o", color="y")
    # # accept point
    # plt.plot(accept_point_pca[0], accept_point_pca[1], marker="x", color="g")
    # plt.xlabel("PCA 1")
    # plt.ylabel("PCA 2")
    # plt.title("PCA Trace")
    # plt.savefig(pca_plot_filename)
    # plt.close()
    ##### ATTACK LOG #####

    return result, prompt_history


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

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(f"models/{model_dict[args.model_name]}", False, "cuda:0")
    model.eval()
    model.requires_grad_(False)  # Save memory
    model_judge, tokenizer_judge = load_model_and_tokenizer(args.model_judge, False, "cuda:0")
    model_judge.eval()
    model_judge.requires_grad_(False)

    # Load finetuned eval model
    eval_models = []
    
    # Run the jailbreak attack
    results = []
    prompt_history_all = []
    if not os.path.exists(f"{args.output_dir}"):
        os.makedirs(f"{args.output_dir}")

    if args.early_stop:
        result_filename = f"{args.output_dir}/space_pretrain_{str(lower_bound)}_{str(upper_bound)}.json"
    else:
        result_filename = f"{args.output_dir}/{args.eval_tasks}_{str(lower_bound)}_{str(upper_bound)}_nestop.json"
        
    for idx, (goal, target) in tqdm(enumerate(dataset, start=lower_bound), total=len(dataset)):
        # if exist this file, skip
        # if os.path.exists(result_filename):
        #     print(f"==>> Found individual result JSON. Skipping {idx}/{len(dataset)}")
        #     continue

        prompt_result, prompt_history = jailbreak_this_prompt(
            prompt=goal,
            target=target,
            model=model,
            eval_models=eval_models,
            tokenizer=tokenizer,
            init_adv_string=args.init_adv_string,
            max_epochs=args.max_epochs,
            sampling_number=args.sampling_number,
            early_stop=args.early_stop,
            model_judge=model_judge,
            tokenizer_judge=tokenizer_judge,
            tqdm_desc=f"Jailbreaking {idx}/{lower_bound}-{upper_bound}",
            prompt_idx=idx,
            plot_dir=args.output_plot_dir,
        )

        results.append(prompt_result)
        prompt_history_all += prompt_history

        with open(result_filename, 'w') as f:
            json.dump(results, f, indent=4)
                
        if not args.early_stop:
            with open(f"{args.output_dir}/{args.eval_tasks}_{str(lower_bound)}_{str(upper_bound)}_logs.json", "w") as f:
                json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
