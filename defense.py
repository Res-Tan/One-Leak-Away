import argparse
import json
import os
import sys
import time
import copy
import numpy as np
import jax.numpy as jnp
from jax import random as jax_random
from scipy.optimize import linear_sum_assignment

import matplotlib.pyplot as plt
from collections import defaultdict
from typing import NamedTuple, Dict, Tuple, Optional, Union
import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

def load_model_and_tokenizer(model_name, if_eval=False, cuda="auto"):
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

model_name = 'llama2_7b'


model, tokenizer = load_model_and_tokenizer(f"models/{model_dict[model_name]}")
model.eval()
model.requires_grad_(False)

permuted_model = copy.deepcopy(model)

def random_permutation(n: int, device: torch.device) -> torch.Tensor:
    """Return a random permutation index p (int64)."""
    return torch.randperm(n, device=device)


def sample_rope_compatible_A_kv(
    num_kv_heads: int,
    num_q_heads: int,
    head_dim: int,
    device: torch.device,
    s_min: float = 0.5,
    s_max: float = 2.0,
):
    assert head_dim % 2 == 0, "RoPE head_dim must be even."
    assert num_q_heads % num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads."
    group_size = num_q_heads // num_kv_heads

    A_kv_list = []
    for _ in range(num_kv_heads):
        s = torch.empty(head_dim // 2, device=device).uniform_(s_min, s_max)
        a = torch.cat([s, s], dim=0)
        A_kv_list.append(a)

    A_q_list = []
    for h in range(num_q_heads):
        g = h // group_size
        A_q_list.append(A_kv_list[g])

    return A_q_list, A_kv_list


def sample_B_kv(
    num_kv_heads: int,
    head_dim: int,
    device: torch.device,
    s_min: float = 0.5,
    s_max: float = 2.0,
):
    """V has no RoPE; B can be arbitrary diagonal (per KV head)."""
    return [torch.empty(head_dim, device=device).uniform_(s_min, s_max) for _ in range(num_kv_heads)]


@torch.no_grad()
def permute_llama_mlp(mlp, p: torch.Tensor):
    mlp.gate_proj.weight.data = mlp.gate_proj.weight.data[p, :]
    mlp.up_proj.weight.data   = mlp.up_proj.weight.data[p, :]
    mlp.down_proj.weight.data = mlp.down_proj.weight.data[:, p]

    if getattr(mlp.gate_proj, "bias", None) is not None:
        mlp.gate_proj.bias.data = mlp.gate_proj.bias.data[p]
    if getattr(mlp.up_proj, "bias", None) is not None:
        mlp.up_proj.bias.data = mlp.up_proj.bias.data[p]


@torch.no_grad()
def scale_llama_attention_heads_rope_gqa_safe(attn, A_q_list, A_kv_list, B_kv_list):
    num_q_heads = attn.num_heads
    head_dim    = attn.head_dim
    num_kv_heads = getattr(attn, "num_key_value_heads", num_q_heads)

    assert num_q_heads % num_kv_heads == 0, "num_heads must be divisible by num_key_value_heads."
    group_size = num_q_heads // num_kv_heads

    hidden_in = attn.q_proj.weight.data.shape[1]

    # Shapes
    q_out  = attn.q_proj.weight.data.shape[0]
    k_out  = attn.k_proj.weight.data.shape[0]
    v_out  = attn.v_proj.weight.data.shape[0]
    o_out  = attn.o_proj.weight.data.shape[0]
    o_in   = attn.o_proj.weight.data.shape[1]

    assert q_out == num_q_heads * head_dim, "q_proj out_features mismatch."
    assert k_out == num_kv_heads * head_dim, "k_proj out_features mismatch."
    assert v_out == num_kv_heads * head_dim, "v_proj out_features mismatch."
    assert o_out == hidden_in, "o_proj out_features mismatch."
    assert o_in  == num_q_heads * head_dim, "o_proj in_features mismatch."

    Wq = attn.q_proj.weight.data.view(num_q_heads, head_dim, hidden_in)
    for h in range(num_q_heads):
        Wq[h] *= A_q_list[h][:, None]
    attn.q_proj.weight.data = Wq.view(q_out, hidden_in)

    Wk = attn.k_proj.weight.data.view(num_kv_heads, head_dim, hidden_in)
    for g in range(num_kv_heads):
        Wk[g] /= A_kv_list[g][:, None]
    attn.k_proj.weight.data = Wk.view(k_out, hidden_in)

    Wv = attn.v_proj.weight.data.view(num_kv_heads, head_dim, hidden_in)
    for g in range(num_kv_heads):
        Wv[g] *= B_kv_list[g][:, None]
    attn.v_proj.weight.data = Wv.view(v_out, hidden_in)

    Wo = attn.o_proj.weight.data.view(hidden_in, num_q_heads, head_dim)
    for h in range(num_q_heads):
        g = h // group_size
        Wo[:, h, :] /= B_kv_list[g][None, :]
    attn.o_proj.weight.data = Wo.view(hidden_in, num_q_heads * head_dim)


def apply_equivalent_reparam_llama(
    model,
    seed: int = 0,
    mlp_permute: bool = True,
    attn_scale: bool = True,
    s_min: float = 0.5,
    s_max: float = 2.0,
    verbose: bool = True,
):
    torch.manual_seed(seed)

    for layer_idx, layer in enumerate(model.model.layers):
        if mlp_permute:
            mlp = layer.mlp
            inter = mlp.gate_proj.weight.data.shape[0]
            p = random_permutation(inter, mlp.gate_proj.weight.data.device)
            permute_llama_mlp(mlp, p)

        if attn_scale:
            attn = layer.self_attn
            num_q_heads = attn.num_heads
            num_kv_heads = getattr(attn, "num_key_value_heads", num_q_heads)
            head_dim = attn.head_dim
            device = attn.q_proj.weight.data.device

            A_q_list, A_kv_list = sample_rope_compatible_A_kv(
                num_kv_heads=num_kv_heads,
                num_q_heads=num_q_heads,
                head_dim=head_dim,
                device=device,
                s_min=s_min,
                s_max=s_max,
            )
            B_kv_list = sample_B_kv(
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                device=device,
                s_min=s_min,
                s_max=s_max,
            )
            scale_llama_attention_heads_rope_gqa_safe(attn, A_q_list, A_kv_list, B_kv_list)

        if verbose:
            attn = layer.self_attn
            num_q_heads = attn.num_heads
            num_kv_heads = getattr(attn, "num_key_value_heads", num_q_heads)
            print(f"[Layer {layer_idx}] done. (q_heads={num_q_heads}, kv_heads={num_kv_heads}, head_dim={attn.head_dim})")


def check_equivalence(model1, model2, tokenizer, prompt: str):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt").to(model1.device)
        out1 = model1(**inputs).logits
        out2 = model2(**inputs).logits

    diff = (out1 - out2).abs().max().item()
    print("Max diff:", diff)
    return diff


def sanity_print_attention_shapes(model, layer_idx: int = 0):
    attn = model.model.layers[layer_idx].self_attn
    print("num_heads:", attn.num_heads)
    print("num_key_value_heads:", getattr(attn, "num_key_value_heads", attn.num_heads))
    print("head_dim:", attn.head_dim)
    print("q_proj.weight:", tuple(attn.q_proj.weight.shape))
    print("k_proj.weight:", tuple(attn.k_proj.weight.shape))
    print("v_proj.weight:", tuple(attn.v_proj.weight.shape))
    print("o_proj.weight:", tuple(attn.o_proj.weight.shape))

prompt = "Could you introduce yourself?"

sanity_print_attention_shapes(model, layer_idx=0) 

apply_equivalent_reparam_llama(
    permuted_model,
    seed=42,
    mlp_permute=True,
    attn_scale=True,
    s_min=0.5,
    s_max=1000.0,
    verbose=True,
)

check_equivalence(model, permuted_model, tokenizer, prompt)

for name, param in model.named_parameters():
    print(f"Name: {name}\n")

path = ''


    

