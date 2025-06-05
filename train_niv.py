import os
import random
from random import randint

import uuid
import numpy as np
from pprint import pprint

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import json
import argparse
import traceback
from rouge_score import rouge_scorer
from pprint import pformat

from torch.cuda.amp import autocast
from torch.nn.utils import clip_grad_norm_

from tasks import get_task_sampler, row_col_sum_loss, element_value_loss
from samplers import get_data_sampler
from lm_models import build_lm, build_pnet
from itertools import permutations

from niv2_dataloader import create_niv2_dataloader
from torch.nn import CrossEntropyLoss

import wandb
import math
import time

from scipy.optimize import linear_sum_assignment
import concurrent.futures

from transformers import AdamW, get_cosine_schedule_with_warmup

from eval_permutation_niv import eval_all_permutations, calculate_statistics, generate_permutation_matrices
from draw_ICLR import calculate_attack_success_rates, plot_attack_success_rates, plot_performance

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

torch.backends.cuda.matmul.allow_tf32 = True

# os.environ['WANDB_MODE'] = 'disabled'
os.environ['WANDB_MODE'] = 'offline'

torch.backends.cudnn.benchmark = True

random.seed(42)  
np.random.seed(42)  
torch.manual_seed(42)  

torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)  

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

test_task_dict = {
    "NLG": {
        "QA": {'task332_tellmewhy_answer_generation': [], 'task073_commonsenseqa_answer_generation': []},
        "Dialog": {'task360_spolin_yesand_response_generation': [], 'task576_curiosity_dialogs_answer_generation': []},
        "Summary": {'task1572_samsum_summary', 'task618_amazonreview_summary_text_generation'},
        "Paraphrase": {'task177_para-nmt_paraphrasing': [], 'task927_yelp_negative_to_positive_style_transfer': []},
    },
    "NLU": {
        "Classification": {'task1346_glue_cola_grammatical_correctness_classification': [], 'task1502_hatexplain_classification': []},
        "Matching": {'task1347_glue_sts-b_similarity_classification': [], 'task296_storycloze_correct_end_classification': []},
        # "NLI": {},
        "Extraction": {'task044_essential_terms_identifying_essential_words': [], 'task908_dialogre_identify_familial_relationships': []},
    }
}
best_dic = {'best_task_avg':[0,0.0]}
start_dic = {}

eval_task_list = [ 
    'task576_curiosity_dialogs_answer_generation.json', # dialogue
    'task073_commonsenseqa_answer_generation.json', # QA
    'task332_tellmewhy_answer_generation.json', # QA
    'task1346_glue_cola_grammatical_correctness_classification.json', # classification
]

train_task_list = None

class EMATracker:
    def __init__(self, beta=0.9):
        self.beta = beta
        self.ema = 0.0
        self.t = 0 

    def update(self, theta_t):
        self.t += 1
        self.ema = self.beta * self.ema + (1 - self.beta) * theta_t
        self.corrected_ema = self.ema / (1 - self.beta ** self.t)

    def get_ema(self):
        return self.corrected_ema if self.t > 0 else self.ema


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
    parser.add_argument('--config', type=str)
    parser.add_argument('--out_dir', type=str)

    parser.add_argument('--use_pnet', type=str2bool)

    parser.add_argument('--train_attacker', type=str2bool, default=False)

    parser.add_argument('--test_task', type=str, default='', nargs='?')
    parser.add_argument('--train_task', type=str, default='', nargs='?')
    parser.add_argument('--train_task_size', type=int, default=2000)
    parser.add_argument('--test_task_size', type=int, default=30)

    parser.add_argument('--shuffle_demons', type=str2bool, default=False) # 直观baseline
    parser.add_argument('--use_mixup', type=str2bool, default=False)
    parser.add_argument('--mix_method', type=str, default='mean')
    parser.add_argument('--n_candidate', type=int, default=3)

    parser.add_argument('--debug', type=str2bool, default=False)
    parser.add_argument('--do_eval', type=str2bool, default=True)
    parser.add_argument('--small_data', type=str2bool, default=False)
    parser.add_argument('--train_on_test', type=str2bool, default=False)
    parser.add_argument('--test_on_train', type=str2bool, default=False)

    parser.add_argument('--use_gradient_clip', type=str2bool, default=False)
    parser.add_argument('--max_perm_num', type=int, default=20) # =120时，shot=5,6没区别
    parser.add_argument('--use_instruction', type=str2bool, default=False)

    # Model configuration
    parser.add_argument('--model_name_or_path', type=str, default='bert-base-uncased')
    parser.add_argument('--pnet_name_or_path', type=str, default='bert-base-uncased')
    parser.add_argument('--checkpoint_dir', nargs='+', type=str, default=None, help='Directory where checkpoints are saved')
    
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--max_demonstration_len', type=int, default=128)

    # LoRA related arguments
    parser.add_argument('--lm_use_lora', type=str2bool, default=True, help='Whether to use LoRA adaptation for the model')
    parser.add_argument('--pnet_use_lora', type=str2bool, default=True, help='Whether to use LoRA adaptation for the model')
    parser.add_argument('--resume_lora_training', type=str2bool, default=False, help='Flag to resume LoRA training')

    parser.add_argument('--lm_lora_target', default=None)
    parser.add_argument('--pnet_lora_target', default=None)

    parser.add_argument('--lora_rank', type=int, default=8, help='The intrinsic dimension for LoRA fine-tuning')
    parser.add_argument('--lora_alpha', type=float, default=32.0, help='The scale factor for LoRA fine-tuning')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout rate')
    parser.add_argument('--additional_target', nargs='+', type=str, default=None, help='Additional modules to set as trainable and save')

    parser.add_argument('--no_permutation', type=str2bool, default=False)

    # Training related configuration
    parser.add_argument("--lm_gradient_accumulation_step", type=int, default=1,
                        help="to increase effective batch size and reduce synchronization")
    parser.add_argument("--pnet_gradient_accumulation_step", type=int, default=1,
                        help="to increase effective batch size and reduce synchronization")
    parser.add_argument('--lm_bz', type=int, default=32)
    parser.add_argument('--pnet_bz', type=int, default=32)
    parser.add_argument('--infer_bz', type=int, default=16)
    parser.add_argument('--train_hard_permutation', type=str2bool, default=False)  # Or bool/int if this should be a different type
    parser.add_argument('--hard_one', type=str2bool, default=False)  # Or bool/int if this should be a different type
    parser.add_argument('--use_hard_permutation', type=str2bool, default=True)  # Or bool/int if this should be a different type
    parser.add_argument('--data', type=str)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--min_shot', type=int, default=3)
    parser.add_argument('--max_shot', type=int, default=10)
    parser.add_argument('--log_every_step', type=int, default=1)
    parser.add_argument('--lambda_entropy_pen', type=float, default=0.0)
    parser.add_argument('--lambda_matrix_align', type=float, default=0.0)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--pnet_learning_rate', type=float)
    parser.add_argument('--lm_once_step', type=int)
    parser.add_argument('--n_iter_sinkhorn', type=int)
    parser.add_argument('--noise_factor', type=float)
    parser.add_argument('--num_tasks', type=int)
    parser.add_argument('--num_training_examples', type=int)
    parser.add_argument('--pnet_once_step', type=int)
    parser.add_argument('--resume_id', type=str)  # Or int if this should be a number
    parser.add_argument('--save_every_step', type=int)
    parser.add_argument('--eval_every_step', type=int)
    parser.add_argument('--start_train_pnet_in', type=int)
    parser.add_argument('--start_use_pnet_in', type=int)
    parser.add_argument('--temperature', type=float)
    parser.add_argument('--train_steps', type=int, default=100000000)

    parser.add_argument('--use_weight_decay', type=str2bool, default=False)
    parser.add_argument('--weight_decay_coefficient', type=float, default=0.1)
    parser.add_argument('--pnet_use_weight_decay', type=str2bool, default=True)
    parser.add_argument('--pnet_weight_decay_coefficient', type=float, default=0.1)

    # wandb configuration
    parser.add_argument('--wandb_entity', type=str)
    parser.add_argument('--wandb_log_every_step', type=int)
    parser.add_argument('--wandb_name', type=str)
    parser.add_argument('--wandb_notes', type=str)
    parser.add_argument('--wandb_project', type=str)

    args = parser.parse_args()
    return args


def find_closest_permutation(soft_perm_matrix):
    if np.isnan(soft_perm_matrix).any() or np.isinf(soft_perm_matrix).any():
        raise ValueError("Input matrix contains NaN or infinite values.")
    row_ind, col_ind = linear_sum_assignment(1 - soft_perm_matrix)
    perm_matrix = np.zeros_like(soft_perm_matrix)
    perm_matrix[row_ind, col_ind] = 1
    return perm_matrix


def find_closest_permutations_parallel(batch_soft_perm_matrices):
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = executor.map(find_closest_permutation, batch_soft_perm_matrices)
    return np.array(list(results))


def _get_special_ids(tokenizer):
    bos_ids = [tokenizer.bos_token_id]
    eos_ids = [tokenizer.eos_token_id]
    
    pad_ids = [tokenizer.pad_token_id]
    assert pad_ids != [None], pad_ids # 1. must have padding ids
    
    if bos_ids == [None] and eos_ids == [None]: # bert
        bos_ids = [tokenizer.cls_token_id]
        eos_ids = [tokenizer.sep_token_id]
        assert bos_ids != [None] and eos_ids != [None], (bos_ids, eos_ids)
        
    assert not (bos_ids == [None] and eos_ids == [None]), (bos_ids, eos_ids) # 2. bos, eos 
    return bos_ids, eos_ids, pad_ids


def gpt2_CE_loss(lm_logits, labels):
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
    assert shift_logits.shape[:2] == shift_labels.shape, (shift_logits.shape, shift_labels.shape)

    loss_fct = CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss


def permute_ICL_test(tokenizer, batch_lm_input_ids_list, batch_lm_attention_mask_list, batch_lm_input_ids_merged_list, batch_lm_attention_mask_merged_list, latent_permutations, use_instruction=False):

    bos_ids, eos_ids, pad_ids = _get_special_ids(tokenizer)
    xs = batch_lm_input_ids_list # [bz, shot + 1, shot_len]
    bz, shot = batch_lm_input_ids_list.shape[0], batch_lm_input_ids_list.shape[1] - 1
    if use_instruction:
        shot -= 1
    latent_permutations = latent_permutations.to(xs.device).to(xs.dtype)

    xs_pre = xs[:, :-1, :]  # [bz, shot, shot_len]
    xs_last = xs[:, -1:, :]  # [bz, 1, shot_len]
    if use_instruction: # to separate instruction
        xs_ins, xs_pre = xs_pre[:, :1, :], xs_pre[:, 1:, :] # [bz, 1, shot_len], [bz, shot, shot_len]
    xs_pre_for_permute = xs_pre.view(xs_pre.shape[0], xs_pre.shape[1], -1)
    # [bz, shot, shot] * [bz, shot, shot_len] --> [bz, shot, shot_len]
    xs_pre_permuted = torch.matmul(latent_permutations.to(xs.device), xs_pre_for_permute)
    xs_pre_permuted = xs_pre_permuted.view(xs_pre.shape)
    n_xs = torch.cat((xs_pre_permuted, xs_last), dim=1) # [bz, shot + 1, shot_len]
    if use_instruction:
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
    for i in range(bz): 
        split_elements[i] = torch.cat((torch.tensor(bos_ids), split_elements[i]), dim=0)
        split_elements[i] = torch.cat((padding_ids.repeat(target_seq_len - split_elements[i].shape[0]), split_elements[i]), dim=0)  # left padding
    res_n_xs = torch.stack(split_elements, dim=0)

    return res_n_xs


def permute_ICL_train(model, tokenizer, batch_lm_input_ids_list, batch_lm_attention_mask_list, batch_lm_input_ids_merged_list, batch_lm_labels_ids_merged_list, batch_lm_attention_mask_merged_list, latent_permutations, hard_permutations=None, use_instruction=False):
    bos_ids, eos_ids, pad_ids = _get_special_ids(tokenizer)
    embeddings = model.get_input_embeddings()
    latent_permutations = latent_permutations.to(model.device).to(model.dtype)
    if hard_permutations is not None:
        hard_permutations = hard_permutations.to(model.device).to(model.dtype)
    batch_lm_input_ids_list = batch_lm_input_ids_list.to(model.device)

    embedding_shape = embeddings.weight.shape
    bz, shot, demons_len, d = batch_lm_input_ids_list.shape[0], batch_lm_input_ids_list.shape[1] - 1, batch_lm_input_ids_list.shape[2], embedding_shape[1]
    if use_instruction:
        shot -= 1

    # segment embeddings
    xs = embeddings(batch_lm_input_ids_list) # [bz, shot + 1, shot_len, d]
    # assert xs.requires_grad, xs.requires_grad
    # print ('xs.requires_grad = ', xs.requires_grad)
    xs_pre = xs[:, :-1, :]  # [bz, shot, shot_len, d]
    xs_last = xs[:, -1:, :]  # [bz, 1, shot_len, d]
    if use_instruction: # to separate instruction
        xs_ins, xs_pre = xs_pre[:, :1, :], xs_pre[:, 1:, :] # [bz, 1, shot_len], [bz, shot, shot_len]
    xs_pre_for_permutate = xs_pre.view(xs_pre.shape[0], xs_pre.shape[1], -1) # [bz, shot, shot_len * d]
    assert xs_pre_for_permutate.shape == (bz, shot, demons_len * d), xs_pre_for_permutate.shape

    padding_vector = xs[batch_lm_input_ids_list == pad_ids[0]][-1] 
    bos_vector = xs[batch_lm_input_ids_list == bos_ids[0]][-1]
    eos_vector = xs[batch_lm_input_ids_list == eos_ids[0]][-1]
    
    # segment ids
    xs_pre_ids, xs_last_ids = batch_lm_input_ids_list[:, :-1, :], batch_lm_input_ids_list[:, -1:, :]
    if use_instruction: # to separate instruction
        xs_ins_ids, xs_pre_ids = xs_pre_ids[:, :1, :], xs_pre_ids[:, 1:, :] # [bz, 1, shot_len], [bz, shot, shot_len]
    xs_pre_for_permutate_ids = xs_pre_ids.view(xs_pre_ids.shape[0], xs_pre_ids.shape[1], -1) # [bz, shot, shot_len]
    assert xs_pre_for_permutate_ids.shape[:2] == xs_pre_for_permutate.shape[:2], (xs_pre_for_permutate_ids.shape, xs_pre_for_permutate.shape)

    xs_pre_permuted = torch.matmul(latent_permutations, xs_pre_for_permutate) # [bz, shot, shot_len * d] * [bz, shot, shot_len * d] --> [bz, shot, shot_len * d]
    cur_permutation = latent_permutations if hard_permutations is None else hard_permutations
    xs_pre_permuted_ids = torch.matmul(cur_permutation.float(), xs_pre_for_permutate_ids.float().to(latent_permutations.device)) # 这里必须是严格矩阵
    assert xs_pre_permuted_ids.shape[:2] == xs_pre_permuted.shape[:2], (xs_pre_permuted_ids.shape, xs_pre_permuted.shape)
    
    xs_pre_permuted = xs_pre_permuted.view(xs_pre.shape) # [bz, shot, shot_len, d]
    xs_pre_permuted_ids = xs_pre_permuted_ids.view(xs_pre_ids.shape) # [bz, shot, shot_len]
    n_xs = torch.cat((xs_pre_permuted, xs_last), dim=1) # [bz, shot + 1, shot_len, d]
    n_xs_ids = torch.cat((xs_pre_permuted_ids.long().to(xs_last_ids.device), xs_last_ids), dim=1) # [bz, shot + 1, shot_len]
    if use_instruction:
        n_xs = torch.cat((xs_ins, n_xs), dim=1)
        n_xs_ids = torch.cat((xs_ins_ids, n_xs_ids), dim=1)
    
    batch_mask = (n_xs_ids != pad_ids[0]) & (n_xs_ids != bos_ids[0]) 
    valid_elements_per_sample = batch_mask.sum(dim=(1, 2)) # [bz] 
    extracted_elements = n_xs[batch_mask] # 1D tensor
    split_elements = list(torch.split(extracted_elements, valid_elements_per_sample.tolist(), dim=0))

    assert len(split_elements) == bz and split_elements[0].shape[-1] == extracted_elements.shape[-1], (len(split_elements), bz, split_elements[0].shape, extracted_elements.shape[-1])

    target_seq_len = batch_lm_input_ids_merged_list.shape[-1]
    for i in range(bz): 
        split_elements[i] = torch.cat((bos_vector.view(1, -1), split_elements[i]), dim=0) 
        split_elements[i] = torch.cat((split_elements[i], padding_vector.repeat(target_seq_len - split_elements[i].shape[0], 1)), dim=0) # right padding
    n_xs = torch.stack(split_elements, dim=0) # n_xs is [bz, T, d], not [bz, (shot + 1) * shot_len, d]
    assert n_xs.shape[:2] == batch_lm_attention_mask_merged_list.shape == batch_lm_input_ids_merged_list.shape, (n_xs.shape, batch_lm_attention_mask_merged_list.shape, batch_lm_input_ids_merged_list.shape)
    assert n_xs.shape[-1] == xs.shape[-1], (n_xs.shape == xs.shape)

    # ==================== check n_xs ====================

    padding_vector_ = padding_vector.view(1, 1, -1)  # [1, 1, d]
    padding_idx = torch.all(torch.eq(n_xs, padding_vector_), dim=-1).to(batch_lm_attention_mask_merged_list.device)  # [bz, shot_len]
    assert torch.equal(padding_idx, batch_lm_attention_mask_merged_list == 0), (padding_idx, batch_lm_attention_mask_merged_list)

    label_vectors = n_xs[batch_lm_labels_ids_merged_list != -100]
    label_vectors_source = embeddings(batch_lm_input_ids_merged_list.to(model.device))[batch_lm_labels_ids_merged_list != -100]
    assert label_vectors.shape == label_vectors_source.shape, (label_vectors.shape, label_vectors_source.shape)
    assert torch.equal(label_vectors, label_vectors_source), (label_vectors[0], label_vectors_source[0])

    # assert n_xs.requires_grad == True, n_xs.requires_grad # lm用lora时，embedding是不参加训练的，但是仍然可以作为固定函数BP回pnet；
    # print ('n_xs.requires_grad = ', n_xs.requires_grad)
    return n_xs


def pnet_train_step(model, p_net, lm_tokenizer, batch, args):
    '''
    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
    '''
    model.train() # as LM also use pnet during training
    p_net.train()

    batch_task_name_list, batch_input_text_list, \
    batch_lm_input_ids_list, batch_lm_labels_ids_list, batch_lm_attention_mask_list, \
    batch_lm_input_ids_merged_list, batch_lm_labels_ids_merged_list, batch_lm_attention_mask_merged_list, \
    batch_pnet_input_ids_list, batch_pnet_labels_ids_list, batch_pnet_attention_mask_list, \
    batch_pnet_input_ids_merged_list, batch_pnet_labels_ids_merged_list, batch_pnet_attention_mask_merged_list = batch

    bz, shot = len(batch_lm_input_ids_list), len(batch_lm_input_ids_list[0]) - 1
    if args.use_instruction:
        shot -= 1
    assert 2 <= shot <= 15, shot

    # [bz, shot , shot]
    latent_permutations = p_net(bz, shot, batch_pnet_input_ids_merged_list.to(p_net.device), batch_pnet_attention_mask_merged_list.to(p_net.device), args.temperature, args.n_iter_sinkhorn, args.noise_factor, args)

    closest_matrices, latent_permutations_ = None, None
    matrix_align_loss = 0.0 
    reg_loss_val = 0.0 # entropy regularization
    if args.lambda_entropy_pen > 0:
        reg_loss_val = element_value_loss(latent_permutations)

    # if args.train_hard_permutation or args.use_alignment_loss or args.lambda_entropy_pen > 0:
    latent_permutations_cpu = latent_permutations.detach().cpu()
    try:
        closest_matrices = find_closest_permutations_parallel(latent_permutations_cpu)
        assert closest_matrices.shape == latent_permutations.shape, (closest_matrices.shape, latent_permutations.shape)
    except:
        print('lm_train_step: does not find closest permutations!')
        traceback.print_exc()
        return None

    closest_matrices = torch.tensor(closest_matrices, dtype=latent_permutations_cpu.dtype, device=latent_permutations_cpu.device)
    assert closest_matrices.shape == latent_permutations.shape, (closest_matrices.shape, latent_permutations.shape)

    if args.lambda_matrix_align > 0:
        matrix_align_loss = F.mse_loss(latent_permutations.cpu(), closest_matrices)

    if args.train_hard_permutation:
        closest_matrices_mask = closest_matrices.to(dtype=latent_permutations.dtype).to(device=latent_permutations.device)
        one_position = closest_matrices_mask == 1
        closest_matrices_mask[one_position] = 1/(latent_permutations[one_position].detach() + 1e-8)
        assert closest_matrices_mask.shape == latent_permutations.shape, (closest_matrices_mask.shape, latent_permutations.shape)
        latent_permutations_ = (latent_permutations + 1e-8) * closest_matrices_mask # 相乘后结果严格是 permutation 矩阵

    p = random.random()
    threshold = 0.01 if not args.debug else 0.5
    if p < threshold:
        print ('=='*10 + ' Training P-Net' + '=='*10)
        print ('Soft permutation matrix: #  \n', latent_permutations[0])
        if latent_permutations_ is not None:
            print ('Hard permutation matrix: #  \n', latent_permutations_[0])

    if args.train_hard_permutation:
        latent_permutations = latent_permutations_

    # check require_grad
    assert latent_permutations.requires_grad, latent_permutations.requires_grad
    n_xs = permute_ICL_train(model, lm_tokenizer, batch_lm_input_ids_list, batch_lm_attention_mask_list, batch_lm_input_ids_merged_list, batch_lm_labels_ids_merged_list, batch_lm_attention_mask_merged_list, latent_permutations, hard_permutations=closest_matrices, use_instruction=args.use_instruction)
    assert n_xs.requires_grad, n_xs.requires_grad

    logits = model(
        inputs_embeds=n_xs.to(model.device).to(model.dtype),
        attention_mask=batch_lm_attention_mask_merged_list.to(model.device),
        return_dict=True
    ).logits

    loss = gpt2_CE_loss(logits, batch_lm_labels_ids_merged_list)

    if torch.isnan(loss).any():
        print(f'pnet_train_step: Loss is NaN during epoch!!!!!!!')
        return None

    reg_loss = args.lambda_entropy_pen * reg_loss_val +  args.lambda_matrix_align * matrix_align_loss

    p_net_loss = loss + reg_loss.to(loss.device) if reg_loss != 0 else loss

    res = {
        'pnet_loss': p_net_loss,
        'adv_loss': loss,
        'reg_loss': reg_loss,
        'pnorm': -1
    }

    return res


def lm_train_step_mixup(model, lm_tokenizer, batch, start_use_pnet, args):
    model.train()
    if args.use_pnet:
        p_net.eval()
    bos_ids, eos_ids, pad_ids = _get_special_ids(lm_tokenizer)

    batch_task_name_list, batch_input_text_list, \
    batch_lm_input_ids_list, batch_lm_labels_ids_list, batch_lm_attention_mask_list, \
    batch_lm_input_ids_merged_list, batch_lm_labels_ids_merged_list, batch_lm_attention_mask_merged_list, \
    batch_pnet_input_ids_list, batch_pnet_labels_ids_list, batch_pnet_attention_mask_list, \
    batch_pnet_input_ids_merged_list, batch_pnet_labels_ids_merged_list, batch_pnet_attention_mask_merged_list = batch

    bz, shot = len(batch_lm_input_ids_list), len(batch_lm_input_ids_list[0]) - 1
    if args.use_instruction:
        shot -= 1
    assert 2 <= shot <= 10, shot

    embeddings = model.get_input_embeddings()
    batch_lm_input_ids_merged_list_embeds = embeddings(batch_lm_input_ids_merged_list.to(model.device))

    permutations = generate_permutation_matrices(shot, args.n_candidate + 1)[: args.n_candidate + 1]
    assert len(permutations) >= 2, (shot, len(permutations))
    
    loss = None
    permuted_loss_list = []
    for i in range(len(permutations)):
        latent_permutations = torch.from_numpy(permutations[i])
        n_xs = permute_ICL_train(model, lm_tokenizer, batch_lm_input_ids_list, batch_lm_attention_mask_list, batch_lm_input_ids_merged_list, batch_lm_labels_ids_merged_list, batch_lm_attention_mask_merged_list, latent_permutations, use_instruction=args.use_instruction)
        assert n_xs.shape[:2] == batch_lm_input_ids_merged_list.shape, (n_xs.shape, batch_lm_input_ids_merged_list.shape)
        logits = model(
            inputs_embeds=n_xs.to(model.device).to(model.dtype),
            attention_mask=batch_lm_attention_mask_merged_list.to(model.device),
            return_dict=True
        ).logits

        loss = gpt2_CE_loss(logits, batch_lm_labels_ids_merged_list)
        if torch.isnan(loss).any():
            print(f'Loss is NaN during epoch!')
            return None

        permuted_loss_list.append(loss)
    
    if args.mix_method == 'max':
        loss = torch.max(torch.stack(permuted_loss_list))
    elif args.mix_method == 'mean':
        loss = torch.mean(torch.stack(permuted_loss_list))
    else:
        raise ValueError(f'Unknown mix_method: {args.mix_method}')

    if torch.isnan(loss).any():
        print(f'Loss is NaN during epoch!')
        return None

    res = {
        'lm_loss': loss,
        'lm_norm': -1,
    }

    return res


def lm_train_step(model, p_net, lm_tokenizer, batch, start_use_pnet, args):
    model.train()
    if args.use_pnet:
        p_net.eval()
    bos_ids, eos_ids, pad_ids = _get_special_ids(lm_tokenizer)

    batch_task_name_list, batch_input_text_list, \
    batch_lm_input_ids_list, batch_lm_labels_ids_list, batch_lm_attention_mask_list, \
    batch_lm_input_ids_merged_list, batch_lm_labels_ids_merged_list, batch_lm_attention_mask_merged_list, \
    batch_pnet_input_ids_list, batch_pnet_labels_ids_list, batch_pnet_attention_mask_list, \
    batch_pnet_input_ids_merged_list, batch_pnet_labels_ids_merged_list, batch_pnet_attention_mask_merged_list = batch

    bz, shot = len(batch_lm_input_ids_list), len(batch_lm_input_ids_list[0]) - 1
    if args.use_instruction:
        shot -= 1
    assert 2 <= shot <= 10, shot

    embeddings = model.get_input_embeddings()
    batch_lm_input_ids_merged_list_embeds = embeddings(batch_lm_input_ids_merged_list.to(model.device))

    loss = -1
    if not start_use_pnet: 
        unpermutated_logits = model( 
            inputs_embeds=batch_lm_input_ids_merged_list_embeds,
            attention_mask=batch_lm_attention_mask_merged_list.to(model.device),
            return_dict=True
        ).logits
        unpermutated_loss = gpt2_CE_loss(unpermutated_logits, batch_lm_labels_ids_merged_list)
        loss = unpermutated_loss
    else:
        # 1. pnet generates latent permutations
        with torch.inference_mode():
            '''
            pnet不需要计算梯度，因此它不会为中间变量保留计算图，减少了GPU显存的使用;
            '''
            # [bz, shot , shot]
            latent_permutations = p_net(bz=bz, shot=shot, xs=batch_pnet_input_ids_merged_list.to(p_net.device), \
                        attention_mask=batch_pnet_attention_mask_merged_list.to(p_net.device), temperature=args.temperature, \
                        n_iter_sinkhorn=args.n_iter_sinkhorn, noise_factor=args.noise_factor, args=args).detach()

        # 2. harden soft permutation matrix (linear assignment)
        closest_matrices = None
        # if args.use_hard_permutation: 
        latent_permutations_cpu = latent_permutations.cpu()
        try:
            closest_matrices = find_closest_permutations_parallel(latent_permutations_cpu)
            closest_matrices = torch.from_numpy(closest_matrices).to(latent_permutations.device).to(latent_permutations.dtype)
            assert closest_matrices.shape == latent_permutations.shape, (closest_matrices.shape, latent_permutations.shape)
        except:
            traceback.print_exc()
            closest_matrices = None

        # check latent_permutations and closest matrices
        p = random.random()
        threshold = 0.01 if not args.debug else 1
        if p <= threshold:
            print ('=='*10 + ' Using P-Net' + '=='*10)
            print ('latent_permutations[0] = \n', latent_permutations[0])
            print ('closest_matrices[0] = \n', closest_matrices[0])

        if args.use_hard_permutation and closest_matrices is not None:
            latent_permutations = closest_matrices

        # 4. permute ICL for LM during training (core operation)
        assert not latent_permutations.requires_grad, latent_permutations.requires_grad
        n_xs = permute_ICL_train(model, lm_tokenizer, batch_lm_input_ids_list, batch_lm_attention_mask_list, batch_lm_input_ids_merged_list, batch_lm_labels_ids_merged_list, batch_lm_attention_mask_merged_list, latent_permutations, hard_permutations=closest_matrices, use_instruction=args.use_instruction)
        # assert n_xs.requires_grad, n_xs.requires_grad # lora不更新embedding

        # 5. LM loss
        logits = model(
            inputs_embeds=n_xs.to(model.device).to(model.dtype),
            attention_mask=batch_lm_attention_mask_merged_list.to(model.device),
            return_dict=True
        ).logits

        loss = gpt2_CE_loss(logits, batch_lm_labels_ids_merged_list)

        if torch.isnan(loss).any():
            print(f'Loss is NaN during epoch!')
            return None

        if args.debug:
            with torch.inference_mode():
                unpermutated_logits = model(
                    inputs_embeds=batch_lm_input_ids_merged_list_embeds,
                    attention_mask=batch_lm_attention_mask_merged_list.to(model.device),
                    return_dict=True
                ).logits
                # unpermutated_loss = CE_loss(unpermutated_logits, batch_lm_labels_ids_merged_list, loss_func)
                unpermutated_loss = gpt2_CE_loss(unpermutated_logits, batch_lm_labels_ids_merged_list)
            print ('=='*10 + ' loss comparision ' + '=='*10)
            print ('permuted loss = ', loss)
            print ('unpermuted loss = ', unpermutated_loss)


    res = {
        'lm_loss': loss,
        'lm_norm': -1,
    }

    return res


def pnet_once_train(i, model, p_net, batch, lm_tokenizer, optimizer_pnet, optimizer, scheduler_pnet, args):
    optimizer_pnet.zero_grad()
    optimizer.zero_grad()

    batch_task_name_list, batch_input_text_list, \
    batch_lm_input_ids_list, batch_lm_labels_ids_list, batch_lm_attention_mask_list, \
    batch_lm_input_ids_merged_list, batch_lm_labels_ids_merged_list, batch_lm_attention_mask_merged_list, \
    batch_pnet_input_ids_list, batch_pnet_labels_ids_list, batch_pnet_attention_mask_list, \
    batch_pnet_input_ids_merged_list, batch_pnet_labels_ids_merged_list, batch_pnet_attention_mask_merged_list = batch

    cur_pnet_bz = args.pnet_bz
    cur_inner_steps = len(batch_pnet_input_ids_merged_list) // cur_pnet_bz 
    cur_pnet_gradient_accumulation_step = args.pnet_gradient_accumulation_step
    assert len(batch_pnet_input_ids_merged_list) % cur_pnet_bz == 0, (len(batch_pnet_input_ids_merged_list), cur_pnet_bz)
    assert cur_inner_steps % cur_pnet_gradient_accumulation_step == 0, (cur_inner_steps, cur_pnet_gradient_accumulation_step)

    pnet_res_dict = {}
    for j in range(args.pnet_once_step): 
        for cur in range(cur_inner_steps): 
            cur_batch = [
                batch_task_name_list[cur*cur_pnet_bz: (cur + 1)*cur_pnet_bz], batch_input_text_list[cur*cur_pnet_bz: (cur + 1)*cur_pnet_bz], \
                batch_lm_input_ids_list[cur*cur_pnet_bz: (cur + 1)*cur_pnet_bz], batch_lm_labels_ids_list[cur*cur_pnet_bz: (cur + 1)*cur_pnet_bz], batch_lm_attention_mask_list[cur*cur_pnet_bz: (cur + 1)*cur_pnet_bz], \
                batch_lm_input_ids_merged_list[cur*cur_pnet_bz: (cur + 1)*cur_pnet_bz], batch_lm_labels_ids_merged_list[cur*cur_pnet_bz: (cur + 1)*cur_pnet_bz], batch_lm_attention_mask_merged_list[cur*cur_pnet_bz: (cur + 1)*cur_pnet_bz], \
                batch_pnet_input_ids_list[cur*cur_pnet_bz: (cur + 1)*cur_pnet_bz], batch_pnet_labels_ids_list[cur*cur_pnet_bz: (cur + 1)*cur_pnet_bz], batch_pnet_attention_mask_list[cur*cur_pnet_bz: (cur + 1)*cur_pnet_bz], \
                batch_pnet_input_ids_merged_list[cur*cur_pnet_bz: (cur + 1)*cur_pnet_bz], batch_pnet_labels_ids_merged_list[cur*cur_pnet_bz: (cur + 1)*cur_pnet_bz], batch_pnet_attention_mask_merged_list[cur*cur_pnet_bz: (cur + 1)*cur_pnet_bz]
            ]
            try:
                cur_res_dict = pnet_train_step(model, p_net, lm_tokenizer, cur_batch, args)
            except:
                traceback.print_exc()
                cur_res_dict = None
            # print('cur_res_dict = ', cur_res_dict)

            if not cur_res_dict or cur_res_dict == {}: # 跳过了一批数据
                print (f'Empty resutls! Skip {cur}-th batch!')
                optimizer_pnet.zero_grad()
                optimizer.zero_grad()
                continue
            pnet_res_dict = cur_res_dict

            p_net_loss = pnet_res_dict['pnet_loss'] / cur_pnet_gradient_accumulation_step # 如果不出错，应该可以整除的
            p_net_loss.backward()

            if (cur + 1) % cur_pnet_gradient_accumulation_step == 0 or cur == cur_inner_steps - 1:

                if args.use_gradient_clip:
                    torch.nn.utils.clip_grad_norm_(p_net.parameters(), max_norm=5.0)

                if j == args.pnet_once_step - 1:
                    pnet_res_dict['pnorm'] = sum(p.grad.data.norm(2).item() ** 2 for p in p_net.parameters() if p.grad is not None) ** 0.5

                optimizer_pnet.step()
                scheduler_pnet.step()
                optimizer_pnet.zero_grad()
                optimizer.zero_grad() 
                
        torch.cuda.empty_cache() 

    return pnet_res_dict


def lm_once_train(model, p_net, batch, lm_tokenizer, start_use_pnet, optimizer, scheduler, pbar, args):
    optimizer.zero_grad()

    batch_task_name_list, batch_input_text_list, \
    batch_lm_input_ids_list, batch_lm_labels_ids_list, batch_lm_attention_mask_list, \
    batch_lm_input_ids_merged_list, batch_lm_labels_ids_merged_list, batch_lm_attention_mask_merged_list, \
    batch_pnet_input_ids_list, batch_pnet_labels_ids_list, batch_pnet_attention_mask_list, \
    batch_pnet_input_ids_merged_list, batch_pnet_labels_ids_merged_list, batch_pnet_attention_mask_merged_list = batch

    cur_lm_bz = args.lm_bz
    cur_lm_gradient_accumulation_step = args.lm_gradient_accumulation_step
    cur_inner_steps = len(batch_lm_input_ids_merged_list) // cur_lm_bz
    assert len(batch_lm_input_ids_merged_list) % cur_lm_bz == 0, (len(batch_lm_input_ids_merged_list), cur_lm_bz)
    assert cur_lm_gradient_accumulation_step % cur_lm_gradient_accumulation_step == 0, (cur_lm_gradient_accumulation_step, cur_lm_gradient_accumulation_step)

    lm_res_dict = None
    for j in range(cur_inner_steps):
        cur_batch = [
            batch_task_name_list, batch_input_text_list[j*cur_lm_bz: (j + 1)*cur_lm_bz], \
            batch_lm_input_ids_list[j*cur_lm_bz: (j + 1)*cur_lm_bz], batch_lm_labels_ids_list[j*cur_lm_bz: (j + 1)*cur_lm_bz], batch_lm_attention_mask_list[j*cur_lm_bz: (j + 1)*cur_lm_bz], \
            batch_lm_input_ids_merged_list[j*cur_lm_bz: (j + 1)*cur_lm_bz], batch_lm_labels_ids_merged_list[j*cur_lm_bz: (j + 1)*cur_lm_bz], batch_lm_attention_mask_merged_list[j*cur_lm_bz: (j + 1)*cur_lm_bz], \
            batch_pnet_input_ids_list[j*cur_lm_bz: (j + 1)*cur_lm_bz], batch_pnet_labels_ids_list[j*cur_lm_bz: (j + 1)*cur_lm_bz], batch_pnet_attention_mask_list[j*cur_lm_bz: (j + 1)*cur_lm_bz], \
            batch_pnet_input_ids_merged_list[j*cur_lm_bz: (j + 1)*cur_lm_bz], batch_pnet_labels_ids_merged_list[j*cur_lm_bz: (j + 1)*cur_lm_bz], batch_pnet_attention_mask_merged_list[j*cur_lm_bz: (j + 1)*cur_lm_bz]
        ]

        try:
            if args.use_mixup:
                cur_res_dict = lm_train_step_mixup(model, lm_tokenizer, cur_batch, start_use_pnet, args)
            else:
                cur_res_dict = lm_train_step(model, p_net, lm_tokenizer, cur_batch, start_use_pnet, args)
        except:
            traceback.print_exc()
            cur_res_dict = None

        if not cur_res_dict or cur_res_dict == {}: 
            print (f'Empty resutls! Skip {j}-th batch!')
            optimizer.zero_grad()
            continue
        lm_res_dict = cur_res_dict
    
        loss = lm_res_dict['lm_loss'] / cur_lm_gradient_accumulation_step
        loss.backward()

        if (j + 1) % cur_lm_gradient_accumulation_step == 0 or j == args.lm_once_step - 1:
            pbar.update(1)

            if args.use_gradient_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            if j == args.lm_once_step - 1:
                lm_res_dict['lm_norm'] = sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    torch.cuda.empty_cache()

    return lm_res_dict


def evaluate_permutations(split, cur_step, model, tokenizer, pnet_tokenizer, args):
    all_results = {}
    for shot in range(args.max_shot - 1, args.min_shot - 1, -1): 
        max_seq_len = args.max_seq_len 
        data_loader = create_niv2_dataloader('test', shot, shot, tokenizer, pnet_tokenizer, args.infer_bz, max_seq_len=max_seq_len, max_demonstration_len=args.max_demonstration_len, task_size=args.test_task_size, shuffle=False, shuffle_demons=False, drop_last=False, left_padding=True, dataset_list=eval_task_list, use_instruction=args.use_instruction)
        all_results = eval_all_permutations(model, tokenizer, shot, all_results, data_loader, args)

    metrics_dir = os.path.join(args.out_dir, 'metrics')
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir, exist_ok=True)

    args.metrics_save_path = os.path.join(metrics_dir, f"eval_step{cur_step}_shot{args.min_shot}-{args.max_shot}.json")
    with open(args.metrics_save_path, "w") as fp:
        json.dump(all_results, fp, indent=2)

    res_dic, res_dic_details = calculate_statistics(all_results, args) 

    args.metrics_save_path = os.path.join(metrics_dir, f"eval_step{cur_step}_shot{args.min_shot}-{args.max_shot}_res_dic.json")
    with open(args.metrics_save_path, "w") as fp:
        json.dump(res_dic, fp, indent=2)

    args.metrics_save_path = os.path.join(metrics_dir, f"eval_step{cur_step}_shot{args.min_shot}-{args.max_shot}_res_dic_details.json")
    with open(args.metrics_save_path, "w") as fp:
        json.dump(res_dic_details, fp, indent=2)

    thresholds = [0.2, 0.4, 0.6]
    save_path = os.path.join(args.out_dir, 'figures') 
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    try:
        for task_name, task_data in res_dic.items():
            plot_performance(task_data, task_name, save_path, cur_step)
            plot_attack_success_rates(res_dic_details[task_name], task_name, thresholds, save_path, cur_step)
    except:
        traceback.print_exc()

    avg, worst = 0.0, 0.0
    for task_name, task_data in res_dic.items():
        avg += sum([results['Average'] for shot, results in task_data.items()]) / len(task_data)
        worst += sum([results['Worst'] for shot, results in task_data.items()]) / len(task_data)
    avg /= len(res_dic)
    worst /= len(res_dic)

    return avg, worst


def evaluate(split, step, shot, all_results, model, lm_tokenizer, pnet_tokenizer, infer_batch_size, cur_step, args):
    tokenizer = lm_tokenizer
    # data_loader = create_niv2_dataloader(split, args.min_shot, args.max_shot, lm_tokenizer, pnet_tokenizer, infer_batch_size, args.max_seq_len, args.max_demonstration_len, task_size=args.test_task_size, drop_last=False, shuffle=False, left_padding=True)
    data_loader = create_niv2_dataloader(split, shot, shot, lm_tokenizer, pnet_tokenizer, infer_batch_size, args.max_seq_len + 1000, args.max_demonstration_len, task_size=args.test_task_size, drop_last=False, shuffle=False, left_padding=False, dataset_list=eval_task_list, use_instruction=args.use_instruction)
    model.eval()
    human_results = [] # save for human review
    task_avg_metrics = {} # save for task average metrics
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    # with torch.no_grad():
    with torch.inference_mode():
        for i, batch in tqdm(enumerate(data_loader)):
            if i >= args.test_task_size:
                break

            batch_task_name_list, batch_input_text_list, \
                batch_lm_input_ids_list, batch_lm_labels_ids_list, batch_lm_attention_mask_list, \
                batch_lm_input_ids_merged_list, batch_lm_labels_ids_merged_list, batch_lm_attention_mask_merged_list, \
                batch_pnet_input_ids_list, batch_pnet_labels_ids_list, batch_pnet_attention_mask_list, \
                batch_pnet_input_ids_merged_list, batch_pnet_labels_ids_merged_list, batch_pnet_attention_mask_merged_list = batch

            bz, _, batch_max_len = batch_lm_input_ids_list.shape[0], batch_lm_input_ids_list.shape[1], torch.max(torch.sum(batch_lm_labels_ids_merged_list != -100, -1)).item()
            if not bz:
                continue

            # prepare for decoding input_ids
            input_ids, attention_mask = [], []
            for k in range(bz):
                padding_mask = batch_lm_input_ids_merged_list[k] == tokenizer.pad_token_id # mask
                label_mask =  batch_lm_input_ids_merged_list[k] == batch_lm_labels_ids_merged_list[k]
                combined_mask = padding_mask | label_mask

                pad_token_tensor = torch.full((combined_mask.sum(),), tokenizer.pad_token_id, dtype=torch.long)
                non_masked_input_ids = batch_lm_input_ids_merged_list[k][~combined_mask]

                cur_input_ids = torch.cat((pad_token_tensor, non_masked_input_ids), dim=-1)
                cur_attention_mask = torch.cat((torch.zeros_like(pad_token_tensor), torch.ones_like(non_masked_input_ids)), dim=-1)
                assert cur_input_ids.shape == batch_lm_input_ids_merged_list[k].shape, (cur_input_ids.shape, batch_lm_input_ids_merged_list[k].shape)
                assert cur_attention_mask.shape == batch_lm_attention_mask_merged_list[k].shape, (cur_attention_mask.shape, batch_lm_attention_mask_merged_list[k].shape)

                input_ids.append(cur_input_ids)
                attention_mask.append(cur_attention_mask)
            
            input_ids = torch.stack(input_ids, dim=0)
            attention_mask = torch.stack(attention_mask, dim=0)
            assert input_ids.shape == batch_lm_input_ids_merged_list.shape, (input_ids.shape, batch_lm_input_ids_merged_list.shape)
            assert attention_mask.shape == batch_lm_attention_mask_merged_list.shape, (attention_mask.shape, batch_lm_attention_mask_merged_list.shape)

            output = None
            try:
                output = model.generate(
                    input_ids=input_ids.to(model.device),
                    attention_mask=attention_mask.to(model.device),
                    max_new_tokens=batch_max_len + 3
                )
            except:
                traceback.print_exc()

                split_point = input_ids.shape[0] // 2

                output1 = model.generate(
                    input_ids=input_ids[:split_point].to(model.device),
                    attention_mask=attention_mask[:split_point].to(model.device),
                    max_new_tokens=batch_max_len + 3
                )
                output2 = model.generate(
                    input_ids=input_ids[split_point:].to(model.device),
                    attention_mask=attention_mask[split_point:].to(model.device),
                    max_new_tokens=batch_max_len + 3
                )
                output = torch.cat([output1, output2], dim=0)

            output = output[:,input_ids.shape[-1]:] # for decoder-only model (we only test decoder-only models)

            if type(batch_lm_labels_ids_list) == torch.Tensor:
                batch_lm_labels_ids_list = batch_lm_labels_ids_list.tolist()
            
            # compute metrics
            # queries = tokenizer.batch_decode(batch_lm_input_ids_merged_list, skip_special_tokens=True) 
            queries = tokenizer.batch_decode(input_ids, skip_special_tokens=True) 
            responses = tokenizer.batch_decode(output, skip_special_tokens=True) 
            references = [example[-1][-1].strip() for example in batch_input_text_list] 

            for j in range(len(responses)):

                task_name = batch_task_name_list[j]
                if task_name not in all_results[step]:
                    all_results[step][task_name] = {}
                if shot not in all_results[step][task_name]:
                    all_results[step][task_name][shot] = []

                reference = references[j].strip()
                response_extract = responses[j].strip().split("Input:")[0].strip()
                scores = scorer.score(reference, response_extract) # 不考虑eos等特殊符号
                rougel = scores['rougeL'].fmeasure

                all_results[step][task_name][shot].append(rougel)

                task_avg_metrics.setdefault('avg_' + batch_task_name_list[j], []).append(rougel)
                human_results.append({'rougel':rougel, 'response':responses[j], 'response_extract':response_extract, 'reference': reference, 'query': queries[j].strip()})

                if args.debug:
                    p = random.random()
                    if p < 0.0005:
                        print('-' * 50)
                        print ('batch_max_len = ', batch_max_len)
                        print ()
                        print ('query = \n', queries[j].strip())
                        print()
                        print ('response_extract = \n', response_extract)
                        print()
                        print('reference = \n', reference.strip())
                        print()
                        print('rougel = ', rougel)
                        print('-' * 50)

    rougel = sum([result['rougel'] for result in human_results]) / len(human_results)
    dir_path = f'{args.out_dir}/metrics'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(f'{dir_path}/step{cur_step}_{split}set_humanview.json', 'w') as w:
        json.dump(human_results, w, indent=4)

    cur_task_avg = []
    print ('\n' + '=='*20)
    for task_name in task_avg_metrics: 
        avg_rougel = round(sum(task_avg_metrics[task_name]) / len(task_avg_metrics[task_name]) * 100, 3) # 保留小数点后3位
        task_avg_metrics[task_name] = avg_rougel
        if cur_step == 0:
            start_dic[task_name] = avg_rougel
            best_dic[task_name] = (0, 0.0)

        cur_change = avg_rougel - start_dic[task_name]
        if cur_change > best_dic[task_name][1]:
            best_dic[task_name] = (cur_step, cur_change)

        cur_task_avg.append(avg_rougel)
        # print (f'{task_name}, rougl, change = ', avg_rougel, change)
        # print (f'{task_name}, step, cur_change = ', cur_step, cur_change)
        # print (f'{task_name}, step, best_change = ', best_dic[task_name][0], best_dic[task_name][1])
        # print ()

    cur_task_avg = sum(cur_task_avg) / len(cur_task_avg)
    if cur_task_avg > best_dic['best_task_avg'][1]:
        best_dic['best_task_avg'] = (cur_step, cur_task_avg)

    start_task_avg = sum(start_dic.values()) / len(start_dic)
    print ('step, start_task_avg = ', 0, start_task_avg)
    print ('step,   cur_task_avg = ', cur_step, cur_task_avg)
    print ('step,  best_task_avg = ', best_dic['best_task_avg'][0], best_dic['best_task_avg'][1])
    print ()

    with open(f'{dir_path}/step{cur_step}_{split}set_agv_metrics.json', 'w') as w:
        json.dump(best_dic, w, indent=4)

    return all_results, task_avg_metrics, rougel


def train(model, p_net, lm_tokenizer, pnet_tokenizer, args):
    # load training set
    lm_once_samples = args.lm_bz * args.lm_once_step  
    split, shuffle, train_task, task_size = 'train', True, args.train_task, args.train_task_size
    if args.train_on_test or args.train_attacker: 
        split, shuffle, train_task, task_size = 'test', False, '', 50
    # data_loader = create_niv2_dataloader(split=split, min_shot=args.min_shot, max_shot=args.max_shot, lm_tokenizer=lm_tokenizer, pnet_tokenizer=pnet_tokenizer, batch_size=lm_once_samples, max_seq_len=args.max_seq_len, max_demonstration_len=args.max_demonstration_len, drop_last=False, shuffle=shuffle, task_size=task_size, use_instruction=args.use_instruction)
    data_loader = create_niv2_dataloader(split=split, min_shot=args.min_shot, max_shot=args.max_shot, lm_tokenizer=lm_tokenizer, pnet_tokenizer=pnet_tokenizer, batch_size=lm_once_samples, max_seq_len=args.max_seq_len, max_demonstration_len=args.max_demonstration_len, drop_last=False, shuffle=shuffle, dataset_list=train_task_list, task_size=task_size, use_instruction=args.use_instruction)
    iterator = iter(data_loader)

    args.train_steps = min(args.train_steps, len(data_loader) * args.lm_once_step * args.epoch)  
    # pbar = tqdm(total=args.train_steps, desc='Progress Bar for LM')
    pbar = tqdm(total=args.train_steps)

    print ('--'*20)
    print (f'training data have batches: {len(data_loader) * args.lm_once_step}')
    print (f'training data have sample: {len(data_loader) * args.lm_once_step * args.lm_bz}')
    print (f'lm_bz, epoch: {args.lm_bz, args.epoch}')
    print (f'actual training steps: {args.train_steps}')
    print ('args.train_steps, len(data_loader), len(data_loader) * args.lm_once_step * args.epoch = \n', args.train_steps, len(data_loader), len(data_loader), args.lm_once_step, args.epoch)

    lm_once_step_source = args.lm_once_step

    assert args.pnet_bz % args.pnet_gradient_accumulation_step == 0
    assert args.lm_bz % args.lm_gradient_accumulation_step == 0

    args.pnet_bz = int(args.pnet_bz / args.pnet_gradient_accumulation_step) # bz for p_net
    args.lm_bz = int(args.lm_bz / args.lm_gradient_accumulation_step) # bz for lm

    # args.pnet_once_step = int(args.pnet_once_step * args.pnet_gradient_accumulation_step) # 走几遍 lm 的数据, 是固定设置
    args.lm_once_step = int(args.lm_once_step * args.lm_gradient_accumulation_step)

    # lm optimizer
    optimizer = None
    if args.use_weight_decay:
        print (f'Use weight decay {args.weight_decay_coefficient} for LM.')
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay_coefficient)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.lm_warmup_steps, num_training_steps=args.train_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.train_steps * 0.1),
        num_training_steps=args.train_steps,
        num_cycles=0.5, 
        last_epoch=-1 
    )

    # pnet optimizer
    optimizer_pnet = None
    if args.use_pnet:
        if args.pnet_use_weight_decay:
            print (f'Use weight decay {args.pnet_weight_decay_coefficient} for P-Net.')
            optimizer_pnet = torch.optim.AdamW(p_net.parameters(), lr=args.pnet_learning_rate, weight_decay=args.pnet_weight_decay_coefficient)
        else:
            optimizer_pnet = torch.optim.AdamW(p_net.parameters(), lr=args.pnet_learning_rate)
        # scheduler_pnet = get_linear_schedule_with_warmup(optimizer_pnet, num_warmup_steps=args.pnet_warmup_steps, num_training_steps=args.train_steps)
        scheduler_pnet = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(args.train_steps * 0.1),
            num_training_steps=args.train_steps,
            num_cycles=0.5, 
            last_epoch=-1 
        )

    starting_step, cur_step = 0, 0
    is_saved, is_eval, is_log = set(), set(), set()
    max_save_step, pointer = args.train_steps // args.save_every_step + 1, 1
    max_eval_step, pointer_eval = args.train_steps // args.eval_every_step + 1, 1
    max_log_step, pointer_log = args.train_steps // args.log_every_step + 1, 1
    best_avg, best_worst = 0.0, 0.0
    best_avg_step, best_worst_step = 0, 0
    all_results = {} 
    do_eval, do_save = False, False
    print ('starting_step, args.train_steps, lm_once_step_source = ', starting_step, args.train_steps, lm_once_step_source)
    for i in range(starting_step, args.train_steps, lm_once_step_source):

        # do_eval = not i or i + lm_once_step_source >= args.train_steps or (i < pointer_eval * args.eval_every_step and i + lm_once_step_source >= pointer_eval * args.eval_every_step and i not in is_eval)
        do_eval = i + lm_once_step_source >= args.train_steps or (i < pointer_eval * args.eval_every_step and i + lm_once_step_source >= pointer_eval * args.eval_every_step and i not in is_eval)
        do_save = i + lm_once_step_source >= args.train_steps or (i < pointer * args.save_every_step and i + lm_once_step_source >= pointer * args.save_every_step and i not in is_saved)

        # 1. load data & reset optimizer
        if i >= args.train_steps: 
            print ('#'*20)
            print ('Training is done!')
            return

        batch = None
        try:
            batch = next(iterator)
            while len(batch[2]) < lm_once_samples:
                batch = next(iterator)
                print (f'Skip batch size {len(batch[2])} < lm_once_samples {lm_once_samples}.')
        except StopIteration: 
            print ('#'*20)
            print ('One epoch is done!')
            do_eval, do_save = True, True
            if i < args.train_steps:
                data_loader.reset() # baseline: shuffle batch
                if args.shuffle_demons: # baseline: shuffle demonstrations
                    data_loader = create_niv2_dataloader(split=split, min_shot=args.min_shot, max_shot=args.max_shot, lm_tokenizer=lm_tokenizer, pnet_tokenizer=pnet_tokenizer, batch_size=lm_once_samples, max_seq_len=args.max_seq_len, max_demonstration_len=args.max_demonstration_len, drop_last=False, shuffle=shuffle, shuffle_demons=args.shuffle_demons, task_size=task_size, use_instruction=args.use_instruction)
                    
                iterator = iter(data_loader)
                batch = next(iterator)
            else:
                print ('#'*20)
                print ('Training is done!')
                return
        
        # check data
        bz, shot = len(batch[2]), len(batch[2][0]) - 1
        if args.use_instruction:
            shot -= 1
        if not bz and i:
            continue

        optimizer.zero_grad()
        if args.use_pnet:
            optimizer_pnet.zero_grad()

        # evaluate average and worst case
        avg_dict, worst_dict = {}, {}
        cur_step = i
        if do_eval and args.do_eval:
        # if not i or i + lm_once_step_source >= args.train_steps or (i < pointer_eval * args.eval_every_step and i + lm_once_step_source >= pointer_eval * args.eval_every_step and i not in is_eval):
        # if i + lm_once_step_source >= args.train_steps or (i < pointer_eval * args.eval_every_step and i + lm_once_step_source >= pointer_eval * args.eval_every_step and i not in is_eval):
            print(f'evaluating at {i}...')
            do_eval = False
            is_eval.add(i)
            pointer_eval += 1
            pointer_eval = min(pointer_eval, max_eval_step)
            split = 'test' if not args.test_on_train else 'train'

            ## evaluate: (1) average case; (2) worst case;
            with torch.inference_mode():
                cur_avg, cur_worst = evaluate_permutations(split, i, model, lm_tokenizer, pnet_tokenizer, args=args)
                if cur_avg > best_avg:
                    best_avg, best_avg_step = cur_avg, i
                if cur_worst > best_worst:
                    best_worst, best_worst_step = cur_worst, i

                print ('=='*20 + f' Evaluate Step {i} ' + '=='*20)
                print ('cur_avg, cur_step = ', cur_avg, i)
                print ('best_avg, best_avg_step = ', best_avg, best_avg_step)
                print ()
                print ('cur_worst, cur_step = ', cur_worst, i)
                print ('best_worst, best_worst_step = ', best_worst, best_worst_step)

    
        # 2. train p-net
        pnet_res_dict = {
            'pnet_loss': -1, # pnet_loss = adv_loss + reg_loss
            'adv_loss': -1,
            'reg_loss': -1,
            'pnorm': -1, 
        }
        
        start_train_pnet = (args.use_pnet and (i >= args.start_train_pnet_in)) or args.train_attacker
        if start_train_pnet: 
            res_dict_ = pnet_once_train(i, model, p_net, batch, lm_tokenizer, optimizer_pnet, optimizer, scheduler_pnet, args)
            if res_dict_ and res_dict_ != {}:
                pnet_res_dict = res_dict_
            else:
                print ('Empty results for P-Net!')
                continue
                    
        # 3. train lm
        lm_res_dict = {
            'lm_loss': -1,
            'lm_norm': -1,
        }

        if i != pbar.n: 
            assert i > pbar.n, (i, pbar.n)
            print (f'pbar.n = {pbar.n}, i = {i}')
            pbar.update(i - pbar.n)

        start_use_pnet = args.use_pnet and (i >= args.start_use_pnet_in) 
        if not args.train_attacker:
            lm_res_dict = lm_once_train(model, p_net, batch, lm_tokenizer, start_use_pnet, optimizer, scheduler, pbar, args)

        # 5. save models
        if do_save:
        # if not i or i + lm_once_step_source >= args.train_steps or (i < pointer * args.save_every_step and i + lm_once_step_source >= pointer * args.save_every_step and i not in is_saved):
            print(f'i: {i}, is_saved: {is_saved}')
            do_eval = False
            is_saved.add(i)
            pointer += 1
            pointer = min(pointer, max_save_step)
            if not args.train_attacker:
                model.save_pretrained(f'{args.out_dir}/model/{i}_step')
            if args.use_pnet:
                p_net._backbone.save_pretrained(f'{args.out_dir}/p_net/{i}_step')
                torch.save(p_net.read_out.state_dict(), f'{args.out_dir}/p_net/{i}_step_readout.pt')

        res_dict = {**pnet_res_dict, **lm_res_dict}
        wandb_dic = {
                "lm_loss": res_dict['lm_loss'],
                "lm_norm": res_dict['lm_norm'],

                "pnet_loss": res_dict['pnet_loss'], 
                'adv_loss': res_dict['adv_loss'],
                "reg_loss": res_dict['reg_loss'],
                'pnorm': res_dict['pnorm'],
        }

        for k, v in worst_dict.items():
            wandb_dic['worst_' + k] = v

        for k, v in avg_dict.items():
            wandb_dic['avg_' + k] = v

        wandb.log(
            wandb_dic,
            step=i + lm_once_step_source,
        ) 

        if args.use_pnet:
            print(f"lloss: {res_dict['lm_loss']:.2f}, lnorm: {res_dict['lm_norm']:.2f}, ploss: {res_dict['pnet_loss']:.2f}, adv: {res_dict['adv_loss']:.2f}, reg: {res_dict['reg_loss']:.2f}, pnorm: {res_dict['pnorm']:.2f}, shot: {shot}")
        else:
            print(f"lloss: {res_dict['lm_loss']:.2f}, lnorm: {res_dict['lm_norm']:.2f}, shot: {shot}")


def main(args):
    wandb.init(
        dir=args.out_dir,
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=args.__dict__,
        notes=args.wandb_notes,
        name=args.wandb_name,
        resume=True,
    )

    p_net, pnet_tokenizer = build_pnet(args)
    if args.use_pnet:
        p_net.train()

        p_net_trainable_params = sum(p.numel() for p in p_net.parameters() if p.requires_grad) / 1e9
        print(f"P-Net has {p_net_trainable_params * 1000:.2f}M trainable parameters")

    model, lm_tokenizer = build_lm(args)
    model.train()
    lm_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9
    print(f"LM has {lm_trainable_params * 1000:.2f}M trainable parameters")
    
    train(model, p_net, lm_tokenizer, pnet_tokenizer, args)


if __name__ == "__main__":
    args = parse_args()
    print(f"Running with: {args}")

    run_id = args.resume_id
    if run_id is None: 
        run_id = str(uuid.uuid4())

    out_dir = os.path.join(args.out_dir, run_id)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    args.out_dir = out_dir

    with open(os.path.join(out_dir, "config.json"), "w") as json_file:
        json.dump(args.__dict__, json_file, indent=4)

    main(args)
    
