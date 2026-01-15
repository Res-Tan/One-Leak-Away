from utils import load_hidden_states, read_jsonl, read_json, permutation_test_auc, create_dir
from concept_vector import Weak2StrongClassifier, undersampling
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import random
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import joblib
import torch
import torch.nn.functional as F
import argparse
from pathlib import Path
import json


layer_dict = {
    'deepseek_7b': 30,
    'llama2_7b': 32,
    'llama3_8b': 32,
    'llama2_13b': 40,
    'mistral_7b': 32,
    'vicuna_7b': 32,
    'qwen_7b': 32,
    'gemma_7b': 28
}

save_name_dict = {
    'H0': 'h0.npy',
    'H1': 'h1.npy',
    'H2': 'h2.npy'
}

def select_test(y_test, y_pred):
    select_y_test = []
    select_y_pred = []
    for test_item, pred_item in zip(y_test.tolist(), y_pred.tolist()):
        if test_item != 0:
            select_y_test.append(test_item)
            select_y_pred.append(pred_item)
    select_y_test = np.array(select_y_test)
    select_y_pred = np.array(select_y_pred)
    return select_y_test, select_y_pred


def compute_transferability(pt_successful_list, ft_successful_list):
    results = []
    transfer_num = 0
    for index, group_pt in enumerate(pt_successful_list):
        group_ft = []
        for pt_idx in group_pt:
            if pt_idx in ft_successful_list[index]:
                group_ft.append(pt_idx)
                transfer_num += 1
        results.append(group_ft)

    return results, transfer_num
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num", type=int, default=400)
    parser.add_argument("--model_name", type=str, default='llama2_7b')
    parser.add_argument("--task", type=str, default='gsm8k')
    parser.add_argument("--model_epoch", type=str, default='epoch1_safe0')
    parser.add_argument("--mode", type=str, default='H1')
    parser.add_argument("--probing_model", type=str, default='linear_svm')
    parser.add_argument("--c_value", type=float, default=5e-04)
    parser.add_argument("--n_componets", type=int, default=100)
    parser.add_argument("--gamma_value", type=float, default=0.0003)

    args = parser.parse_args()

    random.seed(args.seed)
    data_path = 'hidden_states/pretrained'
    log_path = 'reproduce/transfer_attack/results_new/hidden_states'
    n_layers = layer_dict[args.model_name] + 1

    if args.mode == 'H0' or args.mode == 'H1':
        save_path = f'{data_path}/{args.model_name}/probing_models_space/{args.mode}'
    else:
        save_path = f'{data_path}/{args.model_name}/probing_models_space/{args.mode}/{args.task}'

    hs_path = f'{data_path}/{args.model_name}'

    file_seed_list = [42]
    if len(file_seed_list) > 1:
        prefix_file_name = f'union{str(len(file_seed_list))}'
    else:
        prefix_file_name = f'seed{str(file_seed_list[0])}'

    gcg_texts = []
    gcg_eval_texts = []
    for seed in file_seed_list:
        text_list = read_json(f'reproduce/transfer_attack/results_new/gcg/advbench/{args.model_name}/space_seed{str(seed)}/space_pretrain.json')
        gcg_texts.append(text_list)

        text_eval_list = read_json(f'reproduce/transfer_attack/results_new/gcg/advbench/{args.model_name}/space_seed{str(seed)}/space_{args.task}.json')
        gcg_eval_texts.append(text_eval_list)  

    # gcg_texts = gcg_texts[:args.num]
    tatal_num = len(gcg_texts[0]) * len(file_seed_list)
    print(f'probing training data number: {tatal_num}')

    pt_successful_list = []
    success_num = 0
    for i, data_group in enumerate(gcg_texts):
        pt_successful_group = []
        for data in data_group:
            if data['jailbroken']:
                success_num += 1
                pt_successful_group.append(data['idx'])

        pt_successful_list.append(pt_successful_group)
    print(f"ASR: {success_num / tatal_num}")

    if args.mode == 'H2':
        ft_successful_list = []
        ft_success = 0
        for i, data_group in enumerate(gcg_eval_texts):
            ft_successful_group = []
            for data in data_group:
                if data['eval_jailbroken']:
                    ft_successful_group.append(data['idx'])
                    ft_success += 1
            ft_successful_list.append(ft_successful_group)

        transfer_list, transfer_num = compute_transferability(pt_successful_list, ft_successful_list)
        print(f"TSR: {transfer_num / tatal_num}")
        print(f'Positive: {transfer_num}, Negative: {tatal_num - transfer_num}')
        print(f'PT Succ: {success_num}, PT Unsucc: {tatal_num - success_num}')
        print(f'FT Succ: {ft_success}, FT Unsucc: {tatal_num - ft_success}')
        
        # print(f'Untransferabel: {len(successful_list) - len(suc_trans_list)}, Invalid: {251 - len(transferable_list) - len(successful_list) + len(suc_trans_list)}')

    data_info = {i: [] for i in range(n_layers)}
    label_list = []

    if args.mode == 'H0':
        # benign
        for i in range(args.num):
            hidden_states = load_hidden_states(f'{hs_path}/benign/index{str(i)}.npz')
            for layer in range(n_layers):
                # hidden_state = hidden_states[layer].squeeze(0).mean(dim=0)
                hidden_state = hidden_states[layer][0][-1]
                hidden_state = hidden_state.numpy()  # 取平均并转换为numpy数组
                # layer_hidden_states[layer].append(hidden_state)
                data_info[layer].append({
                    'hs': hidden_state,
                    'label': 1
                })

        # harmful
        for i in range(args.num):
            hidden_states = load_hidden_states(f'{hs_path}/harmful/index{str(i)}.npz')
            for layer in range(n_layers):
                # hidden_state = hidden_states[layer].squeeze(0).mean(dim=0)
                hidden_state = hidden_states[layer][0][-1]
                hidden_state = hidden_state.numpy()  # 取平均并转换为numpy数组
                # layer_hidden_states[layer].append(hidden_state)
                data_info[layer].append({
                    'hs': hidden_state,
                    'label': 0
                })
    elif args.mode == 'H1':
        # gcg
        for seed_index, seed in enumerate(file_seed_list):
            for i in range(400):
                hidden_states = load_hidden_states(f'{hs_path}/gcg_space/seed{str(seed)}/idx{str(i)}.npz')
                label = 1 if i in pt_successful_list[seed_index] else 0
                label_list.append(label)

                for layer in range(n_layers):
                    # hidden_state = hidden_states[layer].squeeze(0).mean(dim=0)
                    hidden_state = hidden_states[layer][0][-1]
                    hidden_state = hidden_state.numpy()  # 取平均并转换为numpy数组
                    # layer_hidden_states[layer].append(hidden_state)
                    data_info[layer].append({
                        'hs': hidden_state,
                        'label': label
                    })
        print(f'num sucessful: {sum(label_list)}, num fail: {len(label_list) - sum(label_list)}')
    elif args.mode == 'H2':
        # gcg
        for seed_index, seed in enumerate(file_seed_list):
            for i in range(400):
                hidden_states = load_hidden_states(f'{hs_path}/gcg_space/seed{str(seed)}/idx{str(i)}.npz')
                # if i not in successful_list:
                #     continue
                label = 1 if i in transfer_list[seed_index] else 0
                label_list.append(label)
                for layer in range(n_layers):
                    # hidden_state = hidden_states[layer].squeeze(0).mean(dim=0)
                    hidden_state = hidden_states[layer][0][-1]
                    hidden_state = hidden_state.numpy()  # 取平均并转换为numpy数组
                    # layer_hidden_states[layer].append(hidden_state)
                    data_info[layer].append({
                        'hs': hidden_state,
                        'label': label
                    })
        print(f'num sucessful: {sum(label_list)}, num fail: {len(label_list) - sum(label_list)}')

    # create_dir(f'{save_path}/results')
    # create_dir(f'{save_path}/results/cav_svm')
    # create_dir(f'{save_path}/results/cav_mlp')
    classifier = Weak2StrongClassifier(f'{save_path}/results', args.mode, False, False, False)
    # random.shuffle(data_info)

    rep_dict = {}
    rep_dict["svm"] = {}
    rep_dict["mlp"] = {}

    acc_svm_list = []
    acc_mlp_list = []

    auc_svm_list = []
    auc_mlp_list = []

    p_svm_list = []
    p_mlp_list = []

    for layer in range(n_layers):
    # for layer in [32]:
        forward_info = data_info[layer]
        random.shuffle(forward_info)

        features = []
        labels = []
        for value in forward_info:
            # for hidden_state in value["hidden_states"]:
            features.append(value["hs"])
            labels.append(value["label"])

        features = np.array(features)
        labels = np.array(labels)

        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
        x_balanced, y_balanced = undersampling(X_test, y_test)
        
        y_train, y_train_pred, y_test, y_pred, rep, svm_model = classifier.svm(forward_info, layer, X_train, x_balanced, y_train, y_balanced, args.probing_model, args.n_componets, args.gamma_value, args.c_value)

        # y_test, y_pred = select_test(y_test, y_pred)
        acc_svm = round(accuracy_score(y_test, y_pred), 4)
        acc_svm_train = accuracy_score(y_train, y_train_pred)
        # auc_svm = roc_auc_score(y_test, y_pred)
        # rep_dict["svm"][layer] = acc_svm
        acc_svm_list.append(acc_svm)
        # auc_svm_list.append(auc_svm)

        # # 假设y_true为真实标签，y_pred1和y_pred2为两组预测概率（此处简化为单模型）
        p_value_svm = permutation_test_auc(y_pred, y_test)
        p_svm_list.append(p_value_svm)
        # # print(f"DeLong检验p值: {p_value:.4f}")

        y_train, y_train_pred, y_test, y_pred, rep = classifier.mlp(forward_info, layer, X_train, x_balanced, y_train, y_balanced)
        # y_test, y_pred = select_test(y_test, y_pred)
        acc_mlp = round(accuracy_score(y_test, y_pred), 4)
        acc_mlp_train = accuracy_score(y_train, y_train_pred)
        # auc_mlp = roc_auc_score(y_test, y_pred)
        # rep_dict["mlp"][layer] = acc_mlp
        acc_mlp_list.append(acc_mlp)
        # auc_mlp_list.append(auc_mlp)

        p_value_mlp = permutation_test_auc(y_pred, y_test)
        p_mlp_list.append(p_value_mlp)

        print(f"Layer: {str(layer)} \n Acc, SVM: {str(acc_svm)}, MLP: {str(acc_mlp)}")
        # print(f"Layer: {str(layer)} \n Training Acc, SVM: {str(acc_svm_train)}, MLP: {str(acc_mlp_train)}")
        # print(f"Layer: {str(layer)} \n Acc, MLP: {str(acc_mlp)}")
        # print(f"Layer: {str(layer)} \n AUC, SVM: {str(auc_svm)}, MLP: {str(auc_mlp)}")
        print(f"Layer: {str(layer)} \n p-value, SVM: {str(p_value_svm)}, MLP: {str(p_value_mlp)}")

        save_model_name = f'{args.probing_model}_layer{str(layer)}.joblib'

        create_dir(f'{save_path}')
        create_dir(f'{save_path}/{prefix_file_name}_c{str(args.c_value)}')
        joblib.dump(svm_model, f'{save_path}/{prefix_file_name}_c{str(args.c_value)}/{save_model_name}')

    torch.save(acc_svm_list, f'{save_path}/{prefix_file_name}_c{str(args.c_value)}/{args.probing_model}_acc.pth')
    torch.save(p_svm_list, f'{save_path}/{prefix_file_name}_c{str(args.c_value)}/{args.probing_model}_pvalue.pth')

    create_dir(f'{log_path}/{args.model_name}')

    if args.mode == 'H1':
        target_path = f'{log_path}/{args.model_name}/{args.probing_model}_{prefix_file_name}.json'
    elif args.mode == 'H2':
        target_path = f'{log_path}/{args.model_name}/{args.probing_model}_{args.task}_{prefix_file_name}.json'

    save_log_pth = Path(target_path)

    if save_log_pth.exists():
        with open(target_path, "r") as f:
            records = json.load(f)
    else:
        records = []

    records.append({
        'c_value': args.c_value,
        'last_svm': acc_svm_list[-1],
        'last_mlp': acc_mlp_list[-1],
        'svm_p_value': p_svm_list[-1],
        'mlp_p_value': p_mlp_list[-1]
    })

    with open(target_path, 'w') as f:
        json.dump(records, f, indent=4)
    
