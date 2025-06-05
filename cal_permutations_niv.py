import json
import os
import sys
import argparse
from munch import Munch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import yaml
import math
import traceback

import models
from samplers import get_data_sampler, sample_transformation
from tasks import get_task_sampler

from transformers import AutoModel, AutoTokenizer

from peft import PeftConfig, LoraConfig, PeftModel, TaskType, get_peft_model
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, T5Tokenizer, T5EncoderModel, T5Config, LlamaTokenizer, LlamaForCausalLM
from typing import TYPE_CHECKING
from typing import Optional, Union, Tuple

# from lm_dataloader_test import create_dataloader
from niv2_dataloader import create_niv2_dataloader
from rouge_score import rouge_scorer
from draw_ICLR import plot_performance, plot_attack_success_rates

import random
random.seed(42)
import itertools
import hashlib

max_permutations = 100

def calculate_statistics(all_results, args):
    '''
    all_results: {task_name: {shot: {permutation_id: [{rougeL, response, reference, query}, ...]}}}
    all_results[task_name][shot][permutation_id].append({'rougeL':rougel, 'response':response_extract, 'reference': reference, 'query': queries[j].strip()})
    '''
    res_dic, res_dic_details = {}, {}
    for task_name, task_dic in all_results.items(): # 1. 每个task
        print('##' * 50)
        print('task_name = ', task_name)
        
        shot_dic, shot_dic_details = {}, {}
        for shot in task_dic.keys(): # 2. 每个shot （min_shot to max_shot）
            print('--' * 50)
            print('shot = ', shot)
            print (task_dic[shot])

            # 【对每个shot，统计worst, avg】
            worst_list, best_list = [100] * len(task_dic[shot][0]), [-1] * len(task_dic[shot][0]) # permutation 0 的 sample size
            mean_list, var_list = [100] * len(task_dic[shot][0]), [100] * len(task_dic[shot][0])

            for permutation_id in range(min(args.max_perm_num, math.factorial(shot))): # 3. 每种permutation
                # if permutation_id not in task_dic[shot]:
                    # continue
                for example_id in range(len(task_dic[shot][permutation_id])): # 4. 每个样本
                    worst_list[example_id] = min(worst_list[example_id], task_dic[shot][permutation_id][example_id]['rougeL'])
                    best_list[example_id] = max(best_list[example_id], task_dic[shot][permutation_id][example_id]['rougeL'])
            
            # 【对每个shot，统计worst, avg】
            # 计算每个样本，在不同permutation下的，均值和方差
            for i in range(len(task_dic[shot][0])): # 每个样本
                mean_list[i] = np.mean([task_dic[shot][permutation_id][i]['rougeL'] for permutation_id in range(min(args.max_perm_num, math.factorial(shot))) if permutation_id in task_dic[shot]]) 
                var_list[i] =   np.var([task_dic[shot][permutation_id][i]['rougeL'] for permutation_id in range(min(args.max_perm_num, math.factorial(shot))) if permutation_id in task_dic[shot]])

            # 将list转换成numpy数组以便进行数学运算
            worst_list, best_list = np.array(worst_list), np.array(best_list)
            mean_list, var_list = np.array(mean_list), np.array(var_list)
            origin_list = [dic['rougeL'] for dic in task_dic[shot][0]]
            
            print ('Average = ', np.mean(mean_list))
            print ('Worst = ', np.mean(worst_list))
            print ('Origin = ', np.mean(origin_list))

            # shot_dic[shot] = [np.mean(origin_list), np.mean(best_list), np.mean(worst_list), np.mean(mean_list), np.mean(var_list)]
            shot_dic[shot] = {'Average': np.mean(mean_list), 'Worst': np.mean(worst_list), 'Origin': np.mean(origin_list)}
            shot_dic_details[shot] = {'Average': mean_list, 'Worst': worst_list, 'Origin': origin_list, 'Best': best_list, 'Var': var_list}
    
        res_dic[task_name] = shot_dic
        res_dic_details[task_name] = shot_dic_details

    return res_dic, res_dic_details


def get_run_metrics(args):

    with open(os.path.join(args.model_path + '/metrics', f'eval_step{args.step}_shot2-5.json') , "r") as fp:
        all_results = json.load(fp)

    # 算统计量 (avg/worst)
    res_dic, res_dic_details = calculate_statistics(all_results, args)

    # 画图 ()
    thresholds = [0.3, 0.5, 0.7]
    save_path = os.path.join(args.model_path, 'figures') 
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for task_name, task_data in res_dic.items():
        plot_performance(task_data, task_name, save_path, args.step)
        plot_attack_success_rates(res_dic_details[task_name], task_name, thresholds, save_path, args.step)

    return


def str2bool(v):
    """Util function for user friendly boolean flag args"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(description='Training Configuration')

    # Configuration file path
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--step', type=int)
    parser.add_argument('--max_perm_num', type=int)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(f"Running with: {args}")

    get_run_metrics(args)

