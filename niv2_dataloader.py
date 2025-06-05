import json
import random
import torch
import os
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, LogitsProcessorList
from transformers import T5Tokenizer, T5ForConditionalGeneration, LlamaTokenizer, LlamaForCausalLM
from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from transformers import BertTokenizer
from tokenizers import AddedToken

random.seed(42) 
np.random.seed(42)  
torch.manual_seed(42)  
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)  

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class NIV2Dataset(Dataset):
    def __init__(self, split, min_shot, max_shot, lm_tokenizer, pnet_tokenizer, batch_size=32, max_seq_len=512, max_demonstration_len=284, shuffle=True, shuffle_demons=False, task_size=150, use_instruction=False, dataset_list=None, left_padding=False):
        random.seed(42) 
        self.lm_tokenizer = lm_tokenizer
        self.pnet_tokenizer = pnet_tokenizer
        
        self.max_seq_len = max_seq_len
        self.max_demonstration_len = max_demonstration_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shuffle_demons = shuffle_demons

        self.min_shot = min_shot
        self.max_shot = max_shot

        self.use_instruction = use_instruction
        self.left_padding = left_padding

        lm_name = lm_tokenizer.name_or_path.split('/')[-1]
        pnet_name = pnet_tokenizer.name_or_path.split('/')[-1]

        raw_data_path = f"niv2_data/{split}_data.json"
        if split == 'large_test':
            raw_data_path = '/home/aiscuser/natural-instructions/pearl_dataset/scaled_128ICL_test_data_unshuffled.json'
        processed_batches_path = f"niv2_data/{split}_{lm_name}_{pnet_name}_bz{batch_size}_shot{min_shot}-{max_shot}_seqlen{max_seq_len}_demolen{max_demonstration_len}_size{task_size}.json"

        if self.use_instruction:
            processed_batches_path = processed_batches_path.replace('.json', '_instruction.json')

        if self.left_padding:
            processed_batches_path = processed_batches_path.replace('.json', '_left_padding.json')

        print (processed_batches_path)

        if os.path.exists(processed_batches_path) and dataset_list == []:
            print (f"Loading processed batches from {processed_batches_path}")
            self.batches = torch.load(processed_batches_path)
        else:
            print (f"Processing batches from {raw_data_path}")
            with open(raw_data_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)

            self.dataset_size = len(self.data)

            # get special tokens of each tokenizer; generally applicable
            lm_bos_ids, lm_eos_ids, lm_pad_ids = self._get_special_ids(lm_tokenizer)
            pnet_bos_ids, pnet_eos_ids, pnet_pad_ids = self._get_special_ids(pnet_tokenizer)

            # store raw text
            input_text_list = []
            # store tokenized ids
            task_name_list = []
            lm_input_ids_list, lm_labels_ids_list, lm_input_ids_merged_list, lm_labels_ids_merged_list = [], [], [], []
            pnet_input_ids_list, pnet_labels_ids_list, pnet_input_ids_merged_list, pnet_labels_ids_merged_list = [], [], [], []

            # for task_name, task_dic in self.data.items():
            for task_name in sorted(self.data.keys()): 
                task_dic = self.data[task_name]
                task_instruction = task_dic['Definition'][0].strip()

                if dataset_list and task_name not in dataset_list:
                    continue

                cur_task_size = task_size
                for instance_dic in task_dic['Instances']:
                    
                    cur_shot = random.randint(self.min_shot, self.max_shot) if split == 'train' else self.min_shot
                    try:
                        demons_pairs = [(instance_dic['demonstrations'][i]['input'].strip(), instance_dic['demonstrations'][i]['output'].strip()) for i in range(cur_shot)]
                    except:
                        print (task_name, len(instance_dic['demonstrations']), cur_shot)
                        continue
                    predict_pairs = [(instance_dic['input'].strip(), instance_dic['output'].strip())]
                    if self.shuffle_demons: 
                        random.shuffle(demons_pairs)
                    input_text_pairs = demons_pairs + predict_pairs
                                     
                    assert len(input_text_pairs) - 1 == cur_shot, (len(input_text_pairs) - 1, cur_shot)

                    if self.use_instruction:
                        input_text_pairs = [(task_instruction, None)] + input_text_pairs

                    truncated_input_text_pairs, lm_input_ids, lm_labels_ids, lm_input_ids_merged, lm_labels_ids_merged = self.process_example(input_text_pairs, lm_tokenizer, lm_name, lm_bos_ids, lm_eos_ids, IGNORE_INDEX=-100)        
                    _, pnet_input_ids, pnet_labels_ids, pnet_input_ids_merged, pnet_labels_ids_merged = self.process_example(truncated_input_text_pairs, pnet_tokenizer, pnet_name, pnet_bos_ids, pnet_eos_ids, IGNORE_INDEX=-100)

                    if len(lm_input_ids_merged) > max_seq_len:
                        print (f'{task_name}: Skip a {cur_shot}-shot sample of length {len(lm_input_ids_merged)} > {max_seq_len}.\n')
                        continue

                    cur_task_size -= 1 
                    if cur_task_size < 0:
                        break
                    
                    task_name_list.append(task_name.strip('.json'))
                    input_text_list.append(truncated_input_text_pairs)

                    lm_input_ids_list.append(lm_input_ids)
                    lm_labels_ids_list.append(lm_labels_ids)
                    lm_input_ids_merged_list.append(lm_input_ids_merged)
                    lm_labels_ids_merged_list.append(lm_labels_ids_merged)

                    pnet_input_ids_list.append(pnet_input_ids)
                    pnet_labels_ids_list.append(pnet_labels_ids)
                    pnet_input_ids_merged_list.append(pnet_input_ids_merged)
                    pnet_labels_ids_merged_list.append(pnet_labels_ids_merged)

            if dataset_list:
                print('dataset_list = ', dataset_list)
            assert len(input_text_list) == len(lm_input_ids_list) == len(lm_labels_ids_list) == len(lm_input_ids_merged_list) == len(lm_labels_ids_merged_list) == len(pnet_input_ids_list) == len(pnet_labels_ids_list) == len(pnet_input_ids_merged_list) == len(pnet_labels_ids_merged_list), "All lists should have the same length"
            self.dataset = [
                task_name_list, input_text_list,
                lm_input_ids_list, lm_labels_ids_list, lm_input_ids_merged_list, lm_labels_ids_merged_list,
                pnet_input_ids_list, pnet_labels_ids_list, pnet_input_ids_merged_list, pnet_labels_ids_merged_list
            ]

            # Group the data by shot
            self.group_dataset_by_shot = {}
            print ('len(input_text_list) = ', len(input_text_list))
            for i in range(len(lm_input_ids_list)):
                shot = len(lm_input_ids_list[i]) - 1 
                if use_instruction:
                    shot -= 1
                if shot not in self.group_dataset_by_shot:
                    self.group_dataset_by_shot[shot] = []
                self.group_dataset_by_shot[shot].append(
                        [
                        task_name_list[i], input_text_list[i],
                        lm_input_ids_list[i], lm_labels_ids_list[i], lm_input_ids_merged_list[i], lm_labels_ids_merged_list[i],
                        pnet_input_ids_list[i], pnet_labels_ids_list[i], pnet_input_ids_merged_list[i], pnet_labels_ids_merged_list[i]
                    ]
                )
            print("每个shot的样本个数：")
            for key in self.group_dataset_by_shot.keys():
                print (key, len(self.group_dataset_by_shot[key]))

            # sort each shot by example length
            for shot, shot_list in self.group_dataset_by_shot.items():
                res_list = list(zip(*shot_list))
                task_name_list, input_text_list, lm_input_ids_list, lm_labels_ids_list, lm_input_ids_merged_list, lm_labels_ids_merged_list, pnet_input_ids_list, pnet_labels_ids_list, pnet_input_ids_merged_list, pnet_labels_ids_merged_list = res_list

                zipped = zip(task_name_list, input_text_list, lm_input_ids_list, lm_labels_ids_list, lm_input_ids_merged_list, lm_labels_ids_merged_list,
                             pnet_input_ids_list, pnet_labels_ids_list, pnet_input_ids_merged_list, pnet_labels_ids_merged_list)

                # sorted_zipped = sorted(zipped, key=lambda x: len(x[3]))
                sorted_zipped = zipped

                sorted_res_list = list(zip(*sorted_zipped)) # len = 9
                # sorted_res_list = list(sorted_zipped) # len(sorted_res_list) = len(input_text_list)
                self.group_dataset_by_shot[shot] = sorted_res_list

            self.batches = self._create_batches()
            torch.save(self.batches, processed_batches_path)
            
            
    def _create_batches(self):
        random.seed(42) 
        batches = [] 
        for shot, items in self.group_dataset_by_shot.items():
            task_name_list, input_text_list, lm_input_ids_list, lm_labels_ids_list, lm_input_ids_merged_list, lm_labels_ids_merged_list, pnet_input_ids_list, pnet_labels_ids_list, pnet_input_ids_merged_list, pnet_labels_ids_merged_list = self.group_dataset_by_shot[shot]
            # Create batches
            for i in range(0, len(input_text_list), self.batch_size):
                batch_input_text_list, batch_lm_input_ids_list, batch_lm_labels_ids_list, batch_lm_input_ids_merged_list, batch_lm_labels_ids_merged_list, batch_pnet_input_ids_list, batch_pnet_labels_ids_list, batch_pnet_input_ids_merged_list, batch_pnet_labels_ids_merged_list = input_text_list[i:i+self.batch_size], lm_input_ids_list[i:i+self.batch_size], lm_labels_ids_list[i:i+self.batch_size], lm_input_ids_merged_list[i:i+self.batch_size], lm_labels_ids_merged_list[i:i+self.batch_size], pnet_input_ids_list[i:i+self.batch_size], pnet_labels_ids_list[i:i+self.batch_size], pnet_input_ids_merged_list[i:i+self.batch_size], pnet_labels_ids_merged_list[i:i+self.batch_size]
                batch_task_name_list = task_name_list[i:i+self.batch_size]
                
                # [bz, batch_max_length], [bz, points, batch_demonstration_max_length]
                batch_lm_input_ids_list, batch_lm_labels_ids_list, batch_lm_attention_mask_list, \
                batch_lm_input_ids_merged_list, batch_lm_labels_ids_merged_list, batch_lm_attention_mask_merged_list = self.process_batch_examples(
                    input_ids_batch=list(batch_lm_input_ids_list), 
                    labels_batch=list(batch_lm_labels_ids_list), 
                    input_ids_batch_merged=list(batch_lm_input_ids_merged_list), 
                    labels_batch_merged=list(batch_lm_labels_ids_merged_list), 
                    tokenizer=self.lm_tokenizer, 
                    is_pnet=False, 
                    max_seq_len=self.max_seq_len)
                
                batch_pnet_input_ids_list, batch_pnet_labels_ids_list, batch_pnet_attention_mask_list, \
                batch_pnet_input_ids_merged_list, batch_pnet_labels_ids_merged_list, batch_pnet_attention_mask_merged_list = self.process_batch_examples(
                    input_ids_batch=list(batch_pnet_input_ids_list), 
                    labels_batch=list(batch_pnet_labels_ids_list), 
                    input_ids_batch_merged=list(batch_pnet_input_ids_merged_list), 
                    labels_batch_merged=list(batch_pnet_labels_ids_merged_list), 
                    tokenizer=self.pnet_tokenizer, 
                    is_pnet=True, 
                    max_seq_len=self.max_seq_len)

                batch = [
                         batch_task_name_list, batch_input_text_list, \
                         batch_lm_input_ids_list, batch_lm_labels_ids_list, batch_lm_attention_mask_list, \
                         batch_lm_input_ids_merged_list, batch_lm_labels_ids_merged_list, batch_lm_attention_mask_merged_list, \
                         batch_pnet_input_ids_list, batch_pnet_labels_ids_list, batch_pnet_attention_mask_list, \
                         batch_pnet_input_ids_merged_list, batch_pnet_labels_ids_merged_list, batch_pnet_attention_mask_merged_list]
                if len(batch[2]) <= self.batch_size//2:
                    continue
                batches.append(batch)
        if self.shuffle:
            random.shuffle(batches) 
        return batches
    
    
    def process_batch_examples(self, input_ids_batch, labels_batch, input_ids_batch_merged, labels_batch_merged, tokenizer, is_pnet=False, max_seq_len=1024):
        bos_ids, eos_ids, pad_ids = self._get_special_ids(tokenizer)
        IGNORE_INDEX=-100
        batch_max_length = 0
        demonstration_max_length = 0
        
        for i in range(len(input_ids_batch)):  
            batch_max_length = max(batch_max_length, len(input_ids_batch_merged[i]))
            assert len(input_ids_batch_merged) == len(labels_batch_merged)
            demonstration_max_length = max(demonstration_max_length, max([len(cur_input_ids) for cur_input_ids in input_ids_batch[i]]))
            
        
        attention_mask_batch, attention_mask_batch_merged = [], []
        for i in range(len(input_ids_batch)): # batch padding
            seq_pad_len = batch_max_length - len(input_ids_batch_merged[i]) 
            if not self.left_padding:
                # merged padding
                attention_mask_batch_merged.append([1] * len(input_ids_batch_merged[i]) + [0] * seq_pad_len)
                input_ids_batch_merged[i] += pad_ids * seq_pad_len
                labels_batch_merged[i] += [IGNORE_INDEX] * seq_pad_len
                assert len(input_ids_batch_merged[i]) == len(labels_batch_merged[i]) == batch_max_length, (len(input_ids_batch_merged[i]), len(labels_batch_merged[i]), batch_max_length)
                assert len(input_ids_batch_merged[i]) == len(labels_batch_merged[i]) == batch_max_length, (input_ids_batch_merged[i], labels_batch_merged[i], batch_max_length)
                
                # separate padding
                attention_mask_batch.append([[1] * len(cur_input_ids) + [0] * (demonstration_max_length - len(cur_input_ids)) for cur_input_ids in input_ids_batch[i]])
                input_ids_batch[i] = [cur_input_ids + pad_ids * (demonstration_max_length - len(cur_input_ids)) for cur_input_ids in input_ids_batch[i]]
                labels_batch[i] = [cur_label + [IGNORE_INDEX] * (demonstration_max_length - len(cur_label)) for cur_label in labels_batch[i]]
            else:
                # merged padding
                attention_mask_batch_merged.append([0] * seq_pad_len + [1] * len(input_ids_batch_merged[i]))
                input_ids_batch_merged[i] = pad_ids * seq_pad_len + input_ids_batch_merged[i]
                labels_batch_merged[i] = [IGNORE_INDEX] * seq_pad_len + labels_batch_merged[i]
                assert len(input_ids_batch_merged[i]) == len(labels_batch_merged[i]) == batch_max_length, (len(input_ids_batch_merged[i]), len(labels_batch_merged[i]), batch_max_length)
                
                # separate padding
                attention_mask_batch.append([[0] * (demonstration_max_length - len(cur_input_ids)) + [1] * len(cur_input_ids) for cur_input_ids in input_ids_batch[i]])
                input_ids_batch[i] = [pad_ids * (demonstration_max_length - len(cur_input_ids)) + cur_input_ids for cur_input_ids in input_ids_batch[i]]
                labels_batch[i] = [[IGNORE_INDEX] * (demonstration_max_length - len(cur_label)) + cur_label for cur_label in labels_batch[i]]    
            
            assert len(input_ids_batch[i][0]) == len(labels_batch[i][0]) == demonstration_max_length, (len(input_ids_batch[i][0]), len(labels_batch[i][0]), demonstration_max_length)
            
        # [bz, batch_max_length]
        input_ids_batch_merged = torch.tensor(input_ids_batch_merged)
        labels_batch_merged = torch.tensor(labels_batch_merged)
        attention_mask_batch_merged = torch.tensor(attention_mask_batch_merged)
        assert input_ids_batch_merged.shape == labels_batch_merged.shape == attention_mask_batch_merged.shape, (input_ids_batch_merged.shape, labels_batch_merged.shape, attention_mask_batch_merged.shape)
        # print (attention_mask_batch_merged.shape)
        
        # [bz, points, batch_demonstration_max_length]
        input_ids_batch = torch.tensor(input_ids_batch)
        labels_batch = torch.tensor(labels_batch)
        attention_mask_batch = torch.tensor(attention_mask_batch)
        # print(attention_mask_batch.shape)
        assert input_ids_batch.shape == labels_batch.shape == attention_mask_batch.shape, (input_ids_batch.shape, labels_batch.shape, attention_mask_batch.shape)
        
        return input_ids_batch, labels_batch, attention_mask_batch, input_ids_batch_merged, labels_batch_merged, attention_mask_batch_merged
    
           
    def process_example(self, input_text_pairs, tokenizer, tokenizer_name, bos_ids, eos_ids, IGNORE_INDEX=-100):
        bos_ids, eos_ids, pad_ids = self._get_special_ids(tokenizer)

        # convert to tokenized pairs
        encoded_pairs = []
        for turn_idx, (query, resp) in enumerate(input_text_pairs): # bz
            if not turn_idx and not resp and self.use_instruction: # 第一个是instruction
                query = "Instruction:\n" + query.strip()
            else:
                query = "\n\nInput:\n" + query.strip() + '\n' + "Output:\n" # IMPORTANT:【所有的 demons 格式必须一模一样、完全对称，否则会出问题（permute之后缺空行）】
                resp = resp.strip().strip()
                
            # 单轮: x, y需要分开tokenize，否则难以区分
            query_ids = tokenizer.encode(query) # 用query填充Prompt模版， tokenizer.encode(用query填充Prompt，)
            resp_ids = tokenizer.encode(resp) if resp else None # tokenizer.encode(resp)

            # 希望每个example只有一个bos or eos，而不是每个demonstration都有一个
            if 'llama' in tokenizer_name.lower() or 'gemma' in tokenizer_name.lower(): # 去掉开头的bos
                if resp_ids is not None:
                    assert query_ids[0] == bos_ids[0] and resp_ids[0] == bos_ids[0], (query_ids[0], resp_ids[0])
                    resp_ids = resp_ids[1:]
                query_ids = query_ids[1:]
            elif 'gpt' in tokenizer_name.lower(): # 什么都不做
                if resp_ids is not None:
                    assert query_ids[0] != bos_ids[0] and query_ids[-1] != eos_ids[0], (query_ids[0], query_ids[-1]) # eos ids
                    assert resp_ids[0] != bos_ids[0] and resp_ids[-1] != eos_ids[0], (resp_ids[0], resp_ids[-1]) # eos ids
            elif 'flan' in tokenizer_name.lower(): # 去掉结尾的eos --> 【每个pair后面跟一个eos，用于pnet的edge prediction】 
                # if resp_ids is not None:
                    # assert query_ids[-1] == eos_ids[0] and resp_ids[-1] == eos_ids[0], (query_ids[-1], resp_ids[0]) # eos ids
                    # resp_ids = resp_ids[:-1]
                # query_ids = query_ids[:-1]
                if resp_ids is not None:
                    assert query_ids[-1] == eos_ids[0] and resp_ids[-1] == eos_ids[0], (query_ids[-1], resp_ids[0]) # eos ids
                    query_ids = query_ids[:-1] # 如果没有respones, 那么是instruction，后面也要eos
            elif 'bert' in tokenizer_name.lower(): # 两边都去掉 --》【两边都保留】
                if resp_ids is not None:
                    assert query_ids[0] == bos_ids[0] and query_ids[-1] == eos_ids[0], (query_ids[0], query_ids[-1]) # cls ids
                    assert resp_ids[0] == bos_ids[0] and resp_ids[-1] == eos_ids[0], (resp_ids[0], resp_ids[-1]) # cls ids
                    resp_ids = resp_ids[1:]
                    # resp_ids = resp_ids[:-1] # 保留
                # query_ids = query_ids[1:] # 保留
                query_ids = query_ids[:-1]
                

            pair = (query_ids, resp_ids)
            encoded_pairs.append(pair) # [shot+1, 2]
        
        assert len(encoded_pairs) == len(input_text_pairs), (len(encoded_pairs), len(input_text_pairs))

        truncated_input_text_pairs = []
        input_ids_list, labels_list = [], [] 
        input_ids_list_merged, labels_list_merged = [], [] 
        for turn_idx, ((source_ids, target_ids), text_pairs) in enumerate(zip(encoded_pairs, input_text_pairs)):

            # llama: 在第一个demonstration前面加bos; bert也需要加;
            if not turn_idx and ('llama' in tokenizer_name.lower() or 'gemma' in tokenizer_name.lower()) or 'bert' in tokenizer_name.lower():
                source_ids = bos_ids + source_ids
            
            input_ids = source_ids + target_ids if target_ids is not None else source_ids # else: instruction
            labels = [IGNORE_INDEX]*len(input_ids)

            if turn_idx == len(encoded_pairs) - 1: # testing sample
                # if 'flan' in tokenizer_name.lower() or 'bert' in tokenizer_name.lower() or 'gpt' in tokenizer_name.lower():
                # for LM, we always teach model when to stop
                # Note: LM把所有shot看成一个样本，所以在最后加一个eos; pnet是全部pair后面都有eos了(在前面实现了), 不用再加；
                target_ids = target_ids + eos_ids if not ('flan' in tokenizer_name.lower() or 'bert' in tokenizer_name.lower()) else target_ids 
                # Note: LM在training能看到resp, inference只能看到query; 但是，pnet在inference时也应该可以看resp.
                input_ids = source_ids + target_ids if not self.left_padding or ('flan' in tokenizer_name.lower() or 'bert' in tokenizer_name.lower()) else source_ids # else: inference
                labels = [IGNORE_INDEX] * len(source_ids) + target_ids if not self.left_padding or ('flan' in tokenizer_name.lower() or 'bert' in tokenizer_name.lower()) else [IGNORE_INDEX] * len(source_ids) # 算loss时再错位; else: inference.
            assert len(input_ids) == len(labels), (len(input_ids), len(labels))

            input_ids_list.append(input_ids)
            labels_list.append(labels)
            truncated_input_text_pairs.append(text_pairs)

            input_ids_list_merged = input_ids_list_merged + input_ids
            labels_list_merged = labels_list_merged + labels

            assert len(input_ids_list_merged) == len(labels_list_merged), (len(input_ids_list_merged), len(labels_list_merged))
            assert len(truncated_input_text_pairs) == len(input_ids_list) == len(labels_list), (len(truncated_input_text_pairs), len(input_ids_list), len(labels_list))

        return truncated_input_text_pairs, input_ids_list, labels_list, input_ids_list_merged, labels_list_merged

        
    def _get_special_ids(self, tokenizer):
        # if tokenizer.bos_token_id is not None and getattr(tokenizer, "add_bos_token", True):
            # bos_ids = [tokenizer.bos_token_id]
        # else: # baichuan, qwen and gpt2 models have no bos token
            # bos_ids = []
        
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


    def __len__(self):
        # Return the number of batches, not the number of items
        return len(self.batches)

    
    def __getitem__(self, idx):
        return self.batches[idx]


    def reset(self):
        random.seed(42) # 每次调用函数时执行一次
        random.shuffle(self.batches)

        
def create_niv2_dataloader(split, min_shot, max_shot, lm_tokenizer, pnet_tokenizer, batch_size=32, max_seq_len=512, max_demonstration_len=256, drop_last=False, shuffle=True, shuffle_demons=False, dataset_list='', task_size=2000, left_padding=False, use_instruction=False):
    dataset = NIV2Dataset(split, min_shot, max_shot, lm_tokenizer, pnet_tokenizer, batch_size, max_seq_len, max_demonstration_len, shuffle=shuffle, shuffle_demons=shuffle_demons, dataset_list=dataset_list, task_size=task_size, left_padding=left_padding, use_instruction=use_instruction)
    return dataset

path=''
# pnet_tokenizer = T5Tokenizer.from_pretrained(f"{path}/cache/flan-t5-base")
# pnet_tokenizer.add_special_tokens({"additional_special_tokens": [AddedToken("\n")]})

# lm_tokenizer = LlamaTokenizer.from_pretrained(f"{path}/cache/TinyLlama_v1.1")
# # lm_tokenizer = LlamaTokenizer.from_pretrained(f"{path}/cache/TinyLlama-1.1B-Chat-v1.0")
# lm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# split, left_padding, task_size = 'train', False, 150  # 假设 split 可以是 'train', 'test', 或 'valid'
split, left_padding, task_size = 'test', True, 50

# use_instruction = False
use_instruction = True

# batch_size = 32
batch_size = 4
# max_seq_len, max_demonstration_len = 1000, 512
max_seq_len, max_demonstration_len = 4000, 1000
# min_shot, max_shot = 2, 6

# min_shot, max_shot = 2, 3
# data_loader = create_niv2_dataloader(split, min_shot, max_shot, lm_tokenizer, pnet_tokenizer, batch_size, max_seq_len, max_demonstration_len, drop_last=False, task_size=task_size, shuffle=True, left_padding=left_padding, use_instruction=use_instruction)

# min_shot, max_shot = 3, 3
# data_loader1 = create_niv2_dataloader(split, min_shot, max_shot, lm_tokenizer, pnet_tokenizer, batch_size, max_seq_len, max_demonstration_len, drop_last=False, task_size=task_size, shuffle=True, left_padding=left_padding, use_instruction=use_instruction)

def check_reproducibility(data_loader, data_loader1, lm_tokenizer):
    for i, (batch, batch1) in enumerate(zip(data_loader, data_loader1)):
        # [bz, batch_max_length], [bz, points, batch_demonstration_max_length]
        batch_task_name_list, batch_input_text_list, \
        batch_lm_input_ids_list, batch_lm_labels_ids_list, batch_lm_attention_mask_list, \
        batch_lm_input_ids_merged_list, batch_lm_labels_ids_merged_list, batch_lm_attention_mask_merged_list, \
        batch_pnet_input_ids_list, batch_pnet_labels_ids_list, batch_pnet_attention_mask_list, \
        batch_pnet_input_ids_merged_list, batch_pnet_labels_ids_merged_list, batch_pnet_attention_mask_merged_list = batch

        batch_task_name_list1, batch_input_text_list1, \
        batch_lm_input_ids_list1, batch_lm_labels_ids_list1, batch_lm_attention_mask_list1, \
        batch_lm_input_ids_merged_list1, batch_lm_labels_ids_merged_list1, batch_lm_attention_mask_merged_list1, \
        batch_pnet_input_ids_list1, batch_pnet_labels_ids_list1, batch_pnet_attention_mask_list1, \
        batch_pnet_input_ids_merged_list1, batch_pnet_labels_ids_merged_list1, batch_pnet_attention_mask_merged_list1 = batch1

        bz, bz1 = len(batch_lm_input_ids_list), len(batch_lm_input_ids_list1)
        assert bz == bz1, (bz, bz1)
        shot, shot1 = len(batch_lm_input_ids_list[0]) - 1, len(batch_lm_input_ids_list1[0]) - 1
        min_shot = min(shot, shot1)

        for j in range(bz):
            batch_input_text_list[j][:min_shot+1] == batch_input_text_list1[j][:min_shot+1]
            batch_input_text_list[j][-1] == batch_input_text_list1[j][-1]

            # print (batch_lm_input_ids_list.shape, batch_lm_input_ids_list1.shape)
            # assert torch.all(torch.equal(batch_lm_input_ids_list[j, :min_shot+1, :], batch_lm_input_ids_list1[j, :min_shot+1, :]))
            # assert torch.all(torch.equal(batch_lm_input_ids_list[j][-1], batch_lm_input_ids_list1[j][-1]))

            print ('=='*20)
            print (lm_tokenizer.decode(batch_lm_input_ids_merged_list[j], skip_special_tokens=True))
            print ('--'*20)
            print (lm_tokenizer.decode(batch_lm_input_ids_merged_list1[j], skip_special_tokens=True))

        break

# check_reproducibility(data_loader, data_loader1, lm_tokenizer)

def check_data(data_loader):
    num_epochs, toprint=5, False
    # num_epochs, toprint=1, True
    for epoch in range(num_epochs):
        print ('epoch = ', epoch)
        print ('len(data_loader) = ', len(data_loader))
        for i, batch in enumerate(data_loader):
            # [bz, batch_max_length], [bz, points, batch_demonstration_max_length]
            # if i % 32 == 0 or i % 49 ==0:
            batch_task_name_list, batch_input_text_list, \
            batch_lm_input_ids_list, batch_lm_labels_ids_list, batch_lm_attention_mask_list, \
            batch_lm_input_ids_merged_list, batch_lm_labels_ids_merged_list, batch_lm_attention_mask_merged_list, \
            batch_pnet_input_ids_list, batch_pnet_labels_ids_list, batch_pnet_attention_mask_list, \
            batch_pnet_input_ids_merged_list, batch_pnet_labels_ids_merged_list, batch_pnet_attention_mask_merged_list = batch

            bz, shot = len(batch_lm_input_ids_list), len(batch_lm_input_ids_list[0]) - 1
            if bz != batch_size:
                print ('epoch, step, bz, batch_size = ', epoch, i, bz, batch_size)
            
            if toprint:
                if len(batch_lm_input_ids_list[0]) > 4:
                    continue
                print (batch_lm_input_ids_list.shape, batch_lm_labels_ids_list.shape, batch_lm_attention_mask_list.shape)
                print (batch_lm_input_ids_merged_list.shape, batch_lm_labels_ids_merged_list.shape, batch_lm_attention_mask_merged_list.shape)
                print (batch_pnet_input_ids_list.shape, batch_pnet_labels_ids_list.shape, batch_pnet_attention_mask_list.shape,)
                print (batch_pnet_input_ids_merged_list.shape, batch_pnet_labels_ids_merged_list.shape, batch_pnet_attention_mask_merged_list.shape)
                
                print ('=='*20)
                for item in batch_input_text_list[0]:
                    print (item)
                    print()

                print ('--'*20)
                print ('LM separate')
                for item in batch_lm_input_ids_list[0]:
                    # print (lm_tokenizer.batch_decode(item, skip_special_tokens=True))
                    # print(item.shape) # torch.Size([91])
                    print (lm_tokenizer.decode(item))
                    print ()

                print (batch_lm_input_ids_list[0])            
                print (batch_lm_labels_ids_list[0]) 
                print (batch_lm_attention_mask_list[0]) 
                print (batch_lm_input_ids_list[0][batch_lm_labels_ids_list[0] != -100]) # check label
                assert (torch.any(batch_lm_input_ids_list[0][batch_lm_attention_mask_list[0] == 1] != 32000)) # check attention mask
                assert (torch.any(batch_lm_input_ids_list[0][batch_lm_attention_mask_list[0] == 0] == 32000))
                
                print ('--'*20)
                print ('LM merge')
                print (lm_tokenizer.decode(batch_lm_input_ids_merged_list[0]))
                print (batch_lm_input_ids_merged_list[0])
                print (batch_lm_labels_ids_merged_list[0])
                print (batch_lm_attention_mask_merged_list[0])

                print (batch_lm_input_ids_merged_list[0][batch_lm_labels_ids_merged_list[0] != -100]) # check label
                assert (torch.any(batch_lm_input_ids_merged_list[0][batch_lm_attention_mask_merged_list[0] == 1] != 32000)) # check attention mask
                assert (torch.sum(batch_lm_attention_mask_merged_list[0] == 0) == 0 or torch.any(batch_lm_input_ids_merged_list[0][batch_lm_attention_mask_merged_list[0] == 0] == 32000)) # llama的padding是32000

                print ('--'*20)
                print ('P-net separate')
                for item in batch_pnet_input_ids_list[0]:
                    # print (pnet_tokenizer.batch_decode(item, skip_special_tokens=True))
                    print (pnet_tokenizer.decode(item)) 
                print (batch_pnet_input_ids_list[0])
                print (batch_pnet_labels_ids_list[0])
                print (batch_pnet_attention_mask_list[0])

                print (batch_pnet_input_ids_list[0][batch_pnet_labels_ids_list[0] != -100]) # check label
                assert (torch.any(batch_pnet_input_ids_list[0][batch_pnet_attention_mask_list[0] == 1] != 0)) # check attention mask
                assert (torch.sum(batch_pnet_attention_mask_merged_list[0] == 0) == 0 or torch.any(batch_pnet_input_ids_list[0][batch_pnet_attention_mask_list[0] == 0] == 0))

                print ('--'*20)
                print ('P-net merge')
                print (pnet_tokenizer.decode(batch_pnet_input_ids_merged_list[0])) # 为啥没有空行？
                print (batch_pnet_input_ids_merged_list[0])
                print (batch_pnet_labels_ids_merged_list[0])
                print (batch_pnet_attention_mask_merged_list[0])

                print (batch_pnet_input_ids_merged_list[0][batch_pnet_labels_ids_merged_list[0] != -100]) # check label
                assert (torch.any(batch_pnet_input_ids_merged_list[0][batch_pnet_attention_mask_merged_list[0] == 1] != 0)) # check attention mask
                assert (torch.sum(batch_pnet_attention_mask_merged_list[0] == 0) == 0 or torch.any(batch_pnet_input_ids_merged_list[0][batch_pnet_attention_mask_merged_list[0] == 0] == 0)) # flan的padding是0

                break

        data_loader.reset()
    return 


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

def check_permute_ICL(data_loader, lm_tokenizer):
    random.seed(42) 
    bos_ids, eos_ids, pad_ids = _get_special_ids(lm_tokenizer)
    for i, batch in enumerate(data_loader):
        # [bz, batch_max_length], [bz, points, batch_demonstration_max_length]
        # if i % 32 == 0 or i % 49 ==0:
        batch_task_name_list, batch_input_text_list, \
        batch_lm_input_ids_list, batch_lm_labels_ids_list, batch_lm_attention_mask_list, \
        batch_lm_input_ids_merged_list, batch_lm_labels_ids_merged_list, batch_lm_attention_mask_merged_list, \
        batch_pnet_input_ids_list, batch_pnet_labels_ids_list, batch_pnet_attention_mask_list, \
        batch_pnet_input_ids_merged_list, batch_pnet_labels_ids_merged_list, batch_pnet_attention_mask_merged_list = batch

        bz, shot = batch_lm_input_ids_list.shape[0], batch_lm_input_ids_list.shape[1] - 1
        if use_instruction: 
            shot -= 1

        batch_lm_labels_ids_list = torch.squeeze(batch_lm_labels_ids_list).view(bz, -1)
        assert batch_lm_labels_ids_list.dim() == 2, batch_lm_labels_ids_list.dim()

        identity_matrix = torch.eye(shot)
        permutation = np.eye(shot)[np.random.permutation(shot)]
        
        while np.allclose(permutation, identity_matrix): 
            permutation = np.eye(shot)[np.random.permutation(shot)]
        permutation_tensor = torch.tensor(permutation, dtype=torch.long)
        identity_matrix_tensor = identity_matrix.to(torch.long)

        batch_permutation_tensor = permutation_tensor.unsqueeze(0).repeat(bz, 1, 1)

        # if left_padding or batch_lm_input_ids_list[0][0][0] == pad_ids[0] or batch_lm_input_ids_list[-1][0][0] == pad_ids[0]: # left padding (decoding)
        if left_padding: # left padding (decoding)
            print ('left padding')
            permuted_batch_lm_input_ids_merged_list = permute_ICL_test(lm_tokenizer, batch_lm_input_ids_list, batch_lm_attention_mask_list, batch_lm_input_ids_merged_list, batch_lm_attention_mask_merged_list, identity_matrix_tensor)

            # 1. identity matrix
            assert torch.equal(batch_lm_input_ids_merged_list, permuted_batch_lm_input_ids_merged_list), (batch_lm_input_ids_merged_list[0], permuted_batch_lm_input_ids_merged_list[0])

            permuted_batch_lm_input_ids_merged_list = permute_ICL_test(lm_tokenizer, batch_lm_input_ids_list, batch_lm_attention_mask_list, batch_lm_input_ids_merged_list, batch_lm_attention_mask_merged_list, batch_permutation_tensor)
            # 2. permutation matrix
            assert torch.equal(batch_lm_input_ids_merged_list[batch_lm_input_ids_merged_list == pad_ids[0]], permuted_batch_lm_input_ids_merged_list[batch_lm_input_ids_merged_list == pad_ids[0]]), (batch_lm_input_ids_merged_list[0], permuted_batch_lm_input_ids_merged_list[0])
            assert batch_lm_input_ids_merged_list.shape == permuted_batch_lm_input_ids_merged_list.shape, (batch_lm_input_ids_merged_list.shape, permuted_batch_lm_input_ids_merged_list.shape)

            print('batch_lm_input_ids_merged_list[0] = \n', lm_tokenizer.decode(batch_lm_input_ids_merged_list[0]))
            print('--'*20)
            print('permuted_batch_lm_input_ids_merged_list[0] = \n', lm_tokenizer.decode(permuted_batch_lm_input_ids_merged_list[0]))

        else: # right padding (decoding)
            print('right padding')
            model = LlamaForCausalLM.from_pretrained(f"{path}/cache/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16)
            model.resize_token_embeddings(len(lm_tokenizer))
            model = model.to('cuda:0')
            model.train()

            # 1. identity matrix
            permuted_batch_lm_input_ids_merged_list = permute_ICL_train(model, lm_tokenizer, batch_lm_input_ids_list, batch_lm_attention_mask_list, batch_lm_input_ids_merged_list, batch_lm_labels_ids_merged_list, batch_lm_attention_mask_merged_list, identity_matrix_tensor)
            embeddings = model.get_input_embeddings()
            batch_lm_input_ids_merged_list = batch_lm_input_ids_merged_list.to(model.device)
            xs_source = embeddings(batch_lm_input_ids_merged_list)
            assert xs_source.shape == permuted_batch_lm_input_ids_merged_list.shape, (xs_source.shape, permuted_batch_lm_input_ids_merged_list.shape)
            assert torch.equal(xs_source, permuted_batch_lm_input_ids_merged_list), (xs_source[0], permuted_batch_lm_input_ids_merged_list[0])

            # 2. permutation matrix
            permuted_batch_lm_input_ids_merged_list = permute_ICL_train(model, lm_tokenizer, batch_lm_input_ids_list, batch_lm_attention_mask_list, batch_lm_input_ids_merged_list, batch_lm_labels_ids_merged_list, batch_lm_attention_mask_merged_list, batch_permutation_tensor)
            assert xs_source.shape == permuted_batch_lm_input_ids_merged_list.shape, (xs_source.shape, permuted_batch_lm_input_ids_merged_list.shape)

        # if i > 1:
        break

def permute_ICL_test(tokenizer, batch_lm_input_ids_list, batch_lm_attention_mask_list, batch_lm_input_ids_merged_list, batch_lm_attention_mask_merged_list, latent_permutations):

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

def permute_ICL_train(model, tokenizer, batch_lm_input_ids_list, batch_lm_attention_mask_list, batch_lm_input_ids_merged_list, batch_lm_labels_ids_merged_list, batch_lm_attention_mask_merged_list, latent_permutations):
    bos_ids, eos_ids, pad_ids = _get_special_ids(tokenizer)
    embeddings = model.get_input_embeddings()
    latent_permutations = latent_permutations.to(model.device).to(model.dtype)
    batch_lm_input_ids_list = batch_lm_input_ids_list.to(model.device)

    embedding_shape = embeddings.weight.shape
    bz, shot, demons_len, d = batch_lm_input_ids_list.shape[0], batch_lm_input_ids_list.shape[1] - 1, batch_lm_input_ids_list.shape[2], embedding_shape[1]
    if use_instruction:
        shot -= 1

    # segment embeddings
    xs = embeddings(batch_lm_input_ids_list) # [bz, shot + 1, shot_len, d]
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

    # 先用 ramdom permutation 测试 (bz, shot, shot)
    xs_pre_permuted = torch.matmul(latent_permutations, xs_pre_for_permutate) # [bz, shot, shot_len * d] * [bz, shot, shot_len * d] --> [bz, shot, shot_len * d]
    xs_pre_permuted_ids = torch.matmul(latent_permutations.float(), xs_pre_for_permutate_ids.float().to(latent_permutations.device)) # 这里必须是严格矩阵
    assert xs_pre_permuted_ids.shape[:2] == xs_pre_permuted.shape[:2], (xs_pre_permuted_ids.shape, xs_pre_permuted.shape)
    
    xs_pre_permuted = xs_pre_permuted.view(xs_pre.shape) # [bz, shot, shot_len, d]
    xs_pre_permuted_ids = xs_pre_permuted_ids.view(xs_pre_ids.shape) # [bz, shot, shot_len]
    n_xs = torch.cat((xs_pre_permuted, xs_last), dim=1) # [bz, shot + 1, shot_len, d]
    n_xs_ids = torch.cat((xs_pre_permuted_ids.long().to(xs_last_ids.device), xs_last_ids), dim=1) # [bz, shot + 1, shot_len]
    if use_instruction:
        n_xs = torch.cat((xs_ins, n_xs), dim=1)
        n_xs_ids = torch.cat((xs_ins_ids, n_xs_ids), dim=1)
    
    # batch_lm_attention_mask_list 不能直接复用，因为shot顺序变了
    batch_mask = (n_xs_ids != pad_ids[0]) & (n_xs_ids != bos_ids[0]) # 有效位置
    valid_elements_per_sample = batch_mask.sum(dim=(1, 2)) # [bz]
    extracted_elements = n_xs[batch_mask] # 1D tensor
    split_elements = list(torch.split(extracted_elements, valid_elements_per_sample.tolist(), dim=0))

    assert len(split_elements) == bz and split_elements[0].shape[-1] == extracted_elements.shape[-1], (len(split_elements), bz, split_elements[0].shape, extracted_elements.shape[-1])

    target_seq_len = batch_lm_input_ids_merged_list.shape[-1]
    for i in range(bz): # example
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

    assert n_xs.requires_grad == True, n_xs.requires_grad

    return n_xs
