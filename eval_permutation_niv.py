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

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

import random
random.seed(42)
import itertools
import hashlib

max_permutations = 100


def get_model_from_run(args):
    '''only load lm'''
    model, tokenizer = None, None
    if args.load_lora_model:
        print ('load lora model')
        config = PeftConfig.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto")
        model = PeftModel.from_pretrained(model, args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        # tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.pad_token = "<PAD>"
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        if len(tokenizer) != model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))
        model.print_trainable_parameters()
    else:
        print ('load base model')
        # tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = LlamaForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto")
        # tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.pad_token = "<PAD>"
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        if len(tokenizer) != model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))
            
    model = model.to('cuda:0')
    return model, tokenizer


def _get_special_ids(tokenizer):
    bos_ids = [tokenizer.bos_token_id]
    eos_ids = [tokenizer.eos_token_id]
    
    pad_ids = [tokenizer.pad_token_id]
    assert pad_ids != [None], pad_ids # 1. padding ids必须有
    
    if bos_ids == [None] and eos_ids == [None]: # bert
        bos_ids = [tokenizer.cls_token_id]
        eos_ids = [tokenizer.sep_token_id]
        assert bos_ids != [None] and eos_ids != [None], (bos_ids, eos_ids)
        
    assert not (bos_ids == [None] and eos_ids == [None]), (bos_ids, eos_ids) # 2. bos, eos 必须有一个
    return bos_ids, eos_ids, pad_ids


def permute_ICL(tokenizer, batch_lm_input_ids_list, batch_lm_attention_mask_list, batch_lm_input_ids_merged_list, batch_lm_attention_mask_merged_list, latent_permutations, args):

    bos_ids, eos_ids, pad_ids = _get_special_ids(tokenizer)
    xs = batch_lm_input_ids_list # [bz, shot + 1, shot_len]
    bz, shot = batch_lm_input_ids_list.shape[0], batch_lm_input_ids_list.shape[1] - 1
    if args.use_instruction:
        shot -= 1
    latent_permutations = latent_permutations.to(xs.device).to(xs.dtype)

    xs_pre = xs[:, :-1, :]  # [bz, shot, shot_len]
    xs_last = xs[:, -1:, :]  # [bz, 1, shot_len]
    if args.use_instruction: # to separate instruction
        xs_ins, xs_pre = xs_pre[:, :1, :], xs_pre[:, 1:, :] # [bz, 1, shot_len], [bz, shot, shot_len]
    xs_pre_for_permute = xs_pre.view(xs_pre.shape[0], xs_pre.shape[1], -1)
    # [bz, shot, shot] * [bz, shot, shot_len] --> [bz, shot, shot_len]
    xs_pre_permuted = torch.matmul(latent_permutations.to(xs.device), xs_pre_for_permute)
    xs_pre_permuted = xs_pre_permuted.view(xs_pre.shape)
    n_xs = torch.cat((xs_pre_permuted, xs_last), dim=1) # [bz, shot + 1, shot_len]
    if args.use_instruction:
        n_xs = torch.cat((xs_ins, n_xs), dim=1)
    # print ('n_xs.shape = ', n_xs.shape)
    assert n_xs.shape == xs.shape, (n_xs.shape, xs.shape) # [bz, shot + 1, shot_len, shot_len]
    assert n_xs.dtype == xs.dtype, (n_xs.dtype, xs.dtype) # [bz, shot + 1, shot_len, shot_len]

    # batch_mask = batch_lm_attention_mask_list.bool() # [bz, shot + 1, shot_len]
    batch_mask = (n_xs != pad_ids[0]) & (n_xs != bos_ids[0])
    valid_elements_per_sample = batch_mask.sum(dim=(1, 2)) # [bz]

    extracted_elements = n_xs[batch_mask]
    split_elements = list(torch.split(extracted_elements, valid_elements_per_sample.tolist())) # tuple
    assert len(split_elements) == bz, (len(split_elements), bz)

    target_seq_len = max(batch_lm_input_ids_merged_list.shape[-1], split_elements[0].shape[0])
    padding_ids = torch.tensor(pad_ids)
    for i in range(bz): # 逐个样本
        split_elements[i] = torch.cat((torch.tensor(bos_ids), split_elements[i]), dim=0)
        split_elements[i] = torch.cat((padding_ids.repeat(target_seq_len - split_elements[i].shape[0]), split_elements[i]), dim=0)  # left padding
    res_n_xs = torch.stack(split_elements, dim=0)

    return res_n_xs


def eval_all_permutations(model, tokenizer, shot, all_results, data_loader, args):
    ''' 核心逻辑: 一次跑完所有task、所有batch的所有排列；'''
    print ('##' * 20)
    bos_ids, eos_ids, pad_ids = _get_special_ids(tokenizer)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    # permutations = generate_permutation_matrices(shot, min(math.factorial(shot) - 1, args.max_perm_num)) 
    permutations = generate_permutation_matrices(shot, min(math.factorial(shot), args.max_perm_num)) 
    if args.no_permutation:
        permutations = [permutations[0]] # identity matrix???
        print ('identity matrix = \n', permutations[0])
    print ('shot, len(permutations) = ', shot, len(permutations))

    # for i, batch in tqdm(enumerate(data_loader)): # 1. 每个batch （相同shot）
    for i, batch in enumerate(data_loader): # 1. 每个batch （相同shot）
        batch_task_name_list, batch_input_text_list, \
        batch_lm_input_ids_list, batch_lm_labels_ids_list, batch_lm_attention_mask_list, \
        batch_lm_input_ids_merged_list, batch_lm_labels_ids_merged_list, batch_lm_attention_mask_merged_list, \
        batch_pnet_input_ids_list, batch_pnet_labels_ids_list, batch_pnet_attention_mask_list, \
        batch_pnet_input_ids_merged_list, batch_pnet_labels_ids_merged_list, batch_pnet_attention_mask_merged_list = batch

        if args.debug and i == 1: # check reproductivity
            print (tokenizer.decode(batch_lm_input_ids_merged_list[3], skip_special_tokens=True))

        # lm在测试时看不到label，只能用Pnet来估算
        batch_max_len = torch.max(torch.sum(batch_pnet_labels_ids_merged_list != -100, -1)).item()
        if batch_max_len == 0:
            print (batch_pnet_labels_ids_merged_list)
        bz = batch_lm_input_ids_list.shape[0]
        batch_lm_labels_ids_list = torch.squeeze(batch_lm_labels_ids_list).view(bz, -1) # [bz, shot + 1 * shot_len]
        assert batch_lm_labels_ids_list.dim() == 2, batch_lm_labels_ids_list.dim()
        
        for permutation_id in range(len(permutations)): # 2. 这个batch（相同的shot数），对应所有可能的permutation

            permutation = permutations[permutation_id]
            permutation_tensor = torch.tensor(permutation, dtype=torch.long)
            batch_permutation_tensor = permutation_tensor.unsqueeze(0).repeat(bz, 1, 1)

            try:
                permuted_batch_lm_input_ids_merged_list = permute_ICL(tokenizer, batch_lm_input_ids_list, batch_lm_attention_mask_list, batch_lm_input_ids_merged_list, batch_lm_attention_mask_merged_list, batch_permutation_tensor, args)
            except:
                traceback.print_exc()
                continue
            assert torch.equal(batch_lm_input_ids_merged_list[batch_lm_input_ids_merged_list == pad_ids[0]], permuted_batch_lm_input_ids_merged_list[batch_lm_input_ids_merged_list == pad_ids[0]]), (batch_lm_input_ids_merged_list[0], permuted_batch_lm_input_ids_merged_list[0])
            assert batch_lm_input_ids_merged_list.shape == permuted_batch_lm_input_ids_merged_list.shape, (batch_lm_input_ids_merged_list.shape, permuted_batch_lm_input_ids_merged_list.shape)

            # if args.debug:
            #     print('batch_lm_input_ids_merged_list[0] = \n', tokenizer.decode(batch_lm_input_ids_merged_list[0]))
            #     print('--'*20)
            #     print('permuted_batch_lm_input_ids_merged_list[0] = \n', tokenizer.decode(permuted_batch_lm_input_ids_merged_list[0]))

            with torch.inference_mode():
                output = None
                try:
                    output = model.generate(
                        input_ids=permuted_batch_lm_input_ids_merged_list.to(model.device),
                        attention_mask=batch_lm_attention_mask_merged_list.to(model.device),
                        max_new_tokens=batch_max_len
                    )
                except:
                    try:
                        split_point = permuted_batch_lm_input_ids_merged_list.shape[0] // 2

                        output1 = model.generate(
                            input_ids=permuted_batch_lm_input_ids_merged_list[:split_point].to(model.device),
                            attention_mask=batch_lm_attention_mask_merged_list[:split_point].to(model.device),
                            max_new_tokens=batch_max_len 
                        )
                        output2 = model.generate(
                            input_ids=permuted_batch_lm_input_ids_merged_list[split_point:].to(model.device),
                            attention_mask=batch_lm_attention_mask_merged_list[split_point:].to(model.device),
                            max_new_tokens=batch_max_len 
                        )
                        output = torch.cat([output1, output2], dim=0)
                    except:
                        traceback.print_exc()
                        continue

            output = output[:,permuted_batch_lm_input_ids_merged_list.shape[-1]:]

            if type(batch_lm_labels_ids_list) == torch.Tensor:
                batch_lm_labels_ids_list = batch_lm_labels_ids_list.tolist()
                
            # compute metrics
            queries = tokenizer.batch_decode(permuted_batch_lm_input_ids_merged_list, skip_special_tokens=True) # 不带eos
            responses = tokenizer.batch_decode(output, skip_special_tokens=True) # 不带eos
            references = [example[-1][-1].strip() for example in batch_input_text_list] # 不带eos
            for j in range(len(responses)):
                reference = references[j].strip()
                response_extract = responses[j].strip().split("Input:")[0].strip()
                scores = scorer.score(reference, response_extract) # 不考虑eos等特殊符号
                rougel = scores['rougeL'].fmeasure

                task_name = batch_task_name_list[j]
                if task_name not in all_results:
                    all_results[task_name] = {}
                if shot not in all_results[task_name]:
                    all_results[task_name][shot] = {}
                if permutation_id not in all_results[task_name][shot]:
                    all_results[task_name][shot][permutation_id] = []

                # if args.debug:
                #     p = random.random()
                #     if p < 0.00005:
                #         print('-' * 50)
                #         print ('batch_max_len = ', batch_max_len)
                #         print ()
                #         print ('query = \n', queries[j].strip())
                #         print()
                #         # print ('response = \n', responses[j].strip())
                #         # print()
                #         print ('response_extract = \n', response_extract)
                #         print()
                #         print('reference = \n', reference.strip())
                #         print()
                #         print('rougel = ', rougel)
                #         print('-' * 50)

                # 最内层的list，代表一个task中，某一个shot下，某一个排列的所有样本
                # 按道理来说，测一个task在不同shot的表现，应该用相同的样本，增加shot数目
                # 目前有一个问题，一个task在不同shot下，测试样本是不同的，这样比较不同shot的表现就没有意义了
                # 如果分别用相同的test_task_size、不同的shot数去构造dataloader，这样就可以【保证测试样本、每个样本的前k个demonstration】都是一样吗？--》在niv2_dataloader.py中验证下
                # Q: 如何选样本的？
                # Q：如何选demonstration的？
                all_results[task_name][shot][permutation_id].append({'rougeL':rougel, 'response':response_extract, 'reference': reference, 'query': queries[j].strip()})

    return all_results


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
            # print('shot = ', shot, type(shot))
            print('shot = ', shot)

            # 【对每个shot，统计worst, avg】
            k_ = list(task_dic[shot].keys())[0] # permutation idx
            worst_list, best_list = [100] * len(task_dic[shot][k_]), [-1] * len(task_dic[shot][k_]) # permutation 0 的 sample size
            mean_list, var_list = [100] * len(task_dic[shot][k_]), [100] * len(task_dic[shot][k_])

            for permutation_id in range(min(args.max_perm_num, math.factorial(int(shot)))): # 3. 每种permutation
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
            shot_dic_details[shot] = {'Average': list(mean_list), 'Worst': list(worst_list), 'Origin': list(origin_list)}
    
        res_dic[task_name] = shot_dic
        res_dic_details[task_name] = shot_dic_details

    return res_dic, res_dic_details


def get_run_metrics(args):
    # metrics_dir = os.path.join(args.model_path, 'metrics')
    metrics_dir = os.path.join(args.model_path, 'metrics_permutations') if not args.eval_left else os.path.join(args.model_path, 'metrics_permutations_left')
    if args.iid_test:
        metrics_dir = os.path.join(args.model_path, 'metrics_permutations_iid') 
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir, exist_ok=True)

    args.metrics_save_path = os.path.join(metrics_dir, f"eval_perm_step{args.step}_shot{args.min_shot}-{args.max_shot}.json")

    # 存在则直接读取
    if os.path.exists(args.metrics_save_path) and os.path.getsize(args.metrics_save_path) > 0:
        print(f"{args.metrics_save_path} already exists.")
        with open(args.metrics_save_path, "r") as fp:
            all_results = json.load(fp)
        # str to int
        for task_name in all_results.keys():
            remove_idx = set()
            cur_dict = all_results[task_name]
            cur_dict = {int(k):v for k,v in cur_dict.items()} # shot
            for shot in cur_dict.keys():
                cur_dict[shot] = {int(k):v for k,v in cur_dict[shot].items()} 
            all_results[task_name] = cur_dict

        # 算统计量 (avg/worst)
        calculate_statistics(all_results, args)
        return None

    model, tokenizer = get_model_from_run(args)
    model = model.eval()

    pnet_tokenizer = T5Tokenizer.from_pretrained('/nfs/xyz/xyz1/cache/flan-t5-base', legacy=False)

    eval_task_list = [ # 10 tasks
        # 'task618_amazonreview_summary_text_generation.json', # summary
        # 'task177_para-nmt_paraphrasing.json', # paraphrase
        # 'task360_spolin_yesand_response_generation.json', # dialogue
        'task576_curiosity_dialogs_answer_generation.json', # dialogue
        'task073_commonsenseqa_answer_generation.json', # QA

        # 'task907_dialogre_identify_relationships.json', # extraction
        # 'task044_essential_terms_identifying_essential_words.json', # extraction
        'task1346_glue_cola_grammatical_correctness_classification.json', # classification
        # 'task296_storycloze_correct_end_classification.json', # matching
        # 'task1347_glue_sts-b_similarity_classification.json', # matching
        'task332_tellmewhy_answer_generation.json', # QA
    ]

    eval_task_list_left = [
        'task332_tellmewhy_answer_generation.json', # QA
        # 'task1713_convai3_sentence_generation.json', # dialogue 不在里面？
        'task1572_samsum_summary.json', # summary
        'task927_yelp_negative_to_positive_style_transfer.json', # paraphrase
        # 'task928_yelp_positive_to_negative_style_transfer.json', # paraphrase
        'task1502_hatexplain_classification.json', # classification
        # 'task906_dialogre_identify_names.json', # extraction
        # 'task907_dialogre_identify_relationships.json', # extraction
        # 'task039_qasc_find_overlapping_words.json', # extraction

    ]

    if args.eval_left:
        eval_task_list = eval_task_list_left

    if args.iid_test:
        eval_task_list = ''

    # 核心代码 (eval all permutations)
    all_results = {}
    # for shot in range(args.min_shot, args.max_shot + 1): 
    split = 'test' if not args.iid_test else 'iid_test'
    for shot in range(args.max_shot, args.min_shot - 1, -1): # 先暴露长文本的问题
        data_loader = create_niv2_dataloader(split, shot, shot, tokenizer, pnet_tokenizer, args.batch_size, max_seq_len=args.max_seq_len, max_demonstration_len=args.max_demonstration_len, task_size=args.test_task_size, shuffle=False, shuffle_demons=False, drop_last=False, left_padding=True, dataset_list=eval_task_list, use_instruction=args.use_instruction)
        # print ('shot, len(data_loader) = ', shot, len(data_loader)) # 10个数据集混合的
        all_results = eval_all_permutations(model, tokenizer, shot, all_results, data_loader, args)

    with open(args.metrics_save_path, "w") as fp:
        json.dump(all_results, fp, indent=2)

    # 算统计量 (avg/worst)
    res_dic, res_dic_details = calculate_statistics(all_results, args) 

    args.metrics_save_path = os.path.join(metrics_dir, f"eval_step{args.step}_shot{args.min_shot}-{args.max_shot}_res_dic.json")
    with open(args.metrics_save_path, "w") as fp:
        json.dump(res_dic, fp, indent=2)

    args.metrics_save_path = os.path.join(metrics_dir, f"eval_step{args.step}_shot{args.min_shot}-{args.max_shot}_res_dic_details.json")
    with open(args.metrics_save_path, "w") as fp:
        json.dump(res_dic_details, fp, indent=2)

    # 画图 ()
    thresholds = [0.3, 0.5, 0.7]
    save_path = os.path.join(args.model_path, 'figures' if not args.iid_test else 'figures_iid') if not args.accelerate else os.path.join(args.model_path, 'figures_accelerate')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for task_name, task_data in res_dic.items():
        plot_performance(task_data, task_name, save_path, args.step)
        plot_attack_success_rates(res_dic_details[task_name], task_name, thresholds, save_path, args.step)

    return


def generate_permutation_matrices(n_shot, max_num):
    '''生成一系列permutations，这些排列代表着不同的元素顺序'''
    random.seed(42) # 每次调用函数时执行一次
    unique_permutations = None
    # if n_shot > 5 or math.factorial(n_shot) > max_num: # 当 n_shot 大于 6 时，函数将生成不超过 120 个唯一的排列
    if math.factorial(n_shot) > max_num: # 当 n_shot 大于 6 时，函数将生成不超过 120 个唯一的排列
        max_permutations = max_num
        unique_permutations = set()
        original_list = list(range(n_shot)) # 创建一个从 0 到 n_shot-1 的数字列表
        while len(unique_permutations) < max_permutations:
            # 随机打乱 original_list 来获取一个新的排列
            random_permutation = random.sample(original_list, len(original_list))
            unique_permutations.add(tuple(random_permutation))
    else: # 当 n_shot 小于或等于 6 时，生成所有可能的排列组合
        unique_permutations = list(itertools.permutations(list(range(n_shot))))
    # 将集合或列表转换回列表，并确保恒等矩阵排在首位
    permutations = list(unique_permutations)
    identity_index = None
    for i, perm in enumerate(permutations):
        if perm == tuple(range(n_shot)):
            identity_index = i
            break
    if identity_index is not None:
        permutations[0], permutations[identity_index] = permutations[identity_index], permutations[0]

    # 转换排列为排列矩阵
    permutation_matrices = []
    for perm in permutations:
        matrix = np.zeros((n_shot, n_shot), dtype=int)
        for i, p in enumerate(perm):
            matrix[i, p] = 1
        permutation_matrices.append(matrix)

    return permutation_matrices


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
    parser.add_argument('--debug', type=str2bool, default=False)
    parser.add_argument('--eval_left', type=str2bool, default=False)
    parser.add_argument('--accelerate', type=str2bool, default=False)

    parser.add_argument('--iid_test', type=str2bool, default=False)
    parser.add_argument('--test_task_size', type=int, default=30)

    parser.add_argument('--model_path', type=str)
    parser.add_argument('--load_lora_model', type=str2bool, default=True)
    parser.add_argument('--step', type=int)

    parser.add_argument('--no_permutation', type=str2bool, default=False) # 不进行排列?
    parser.add_argument('--max_perm_num', type=int, default=150) # =120时，shot=5,6没区别

    parser.add_argument('--metrics_save_path', type=str)

    parser.add_argument('--use_instruction', type=str2bool, default=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--min_shot', type=int, default=3)
    parser.add_argument('--max_shot', type=int, default=10)
    
    parser.add_argument('--max_seq_len', type=int, default=1000)
    parser.add_argument('--max_demonstration_len', type=int, default=512)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(f"Running with: {args}")

    get_run_metrics(args)

