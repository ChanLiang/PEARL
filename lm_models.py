import torch
torch.backends.cuda.matmul.allow_tf32 = True

import copy
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, LogitsProcessorList
from transformers import T5Config, T5Tokenizer, T5Model, T5ForConditionalGeneration, T5ForSequenceClassification, LlamaTokenizer, LlamaForCausalLM, T5PreTrainedModel, T5EncoderModel, PreTrainedModel
from transformers.models.t5.modeling_t5 import T5Stack
from tqdm import tqdm
import os
import warnings
from transformers import BertModel, BertConfig

import my_sinkhorn_ops
import random

from typing import TYPE_CHECKING
from typing import Optional, Union, Tuple

from peft import (
    PeftModel,
    TaskType,
    LoraConfig,
    get_peft_model
)
from tokenizers import AddedToken
if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel

random.seed(42)  
torch.manual_seed(42)  

torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)  

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

path='/home/chenliang' # sandbox


def load_model(model_name, use_pnet=False):
    if 'flan-t5-base' in model_name:
        model_name = f"{path}/cache/flan-t5-base"
        tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        tokenizer.add_special_tokens({"additional_special_tokens": [AddedToken("\n")]})
        model = None
        if use_pnet:
            model = T5EncoderModel_cl.from_pretrained(model_name, torch_dtype=torch.bfloat16).to('cuda:0')
            if len(tokenizer) != model.config.vocab_size:
                model.resize_token_embeddings(len(tokenizer))
    elif 'flan-t5-large' in model_name:
        model_name = f"{path}/cache/flan-t5-large"
        tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        tokenizer.add_special_tokens({"additional_special_tokens": [AddedToken("\n")]})
        model = None
        if use_pnet:
            model = T5EncoderModel_cl.from_pretrained(model_name, torch_dtype=torch.bfloat16).to('cuda:0')
            # model = T5EncoderModel_cl.from_pretrained(model_name, torch_dtype=torch.bfloat16).to('cuda:1')
            if len(tokenizer) != model.config.vocab_size:
                model.resize_token_embeddings(len(tokenizer))
    elif 'flan-t5-xxl' in model_name:
        model_name = f"{path}/cache/flan-t5-xxl"
        tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        tokenizer.add_special_tokens({"additional_special_tokens": [AddedToken("\n")]})
        model = None
        if use_pnet:
            model = T5EncoderModel_cl.from_pretrained(model_name, torch_dtype=torch.bfloat16).to('cuda:1')
            if len(tokenizer) != model.config.vocab_size:
                model.resize_token_embeddings(len(tokenizer))
    elif 'flan-t5-xl' in model_name:
        model_name = f"{path}/cache/flan-t5-xl"
        tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        tokenizer.add_special_tokens({"additional_special_tokens": [AddedToken("\n")]})
        model = None
        if use_pnet:
            model = T5EncoderModel_cl.from_pretrained(model_name, torch_dtype=torch.bfloat16).to('cuda:2')
            if len(tokenizer) != model.config.vocab_size:
                model.resize_token_embeddings(len(tokenizer))
    elif 'gpt' in model_name:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(model_name).to('cuda')
    elif 'bert' in model_name:
        tokenizer = BertTokenizer.from_pretrained(model_name, legacy=False)
        model = None
        if use_pnet:
            model = BertModel.from_pretrained(model_name).to('cuda')
    elif 'vicuna' in model_name:
        print ("loading vicuna-7b-v1.5-16k")
        # tokenizer = LlamaTokenizer.from_pretrained(f"{path}/cache/vicuna-7b-v1.5-16k/", padding_side='left') # for inference
        tokenizer = LlamaTokenizer.from_pretrained(f"{path}/cache/vicuna-7b-v1.5-16k/")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model = LlamaForCausalLM.from_pretrained(f"{path}/cache/vicuna-7b-v1.5-16k/", device_map="auto", torch_dtype=torch.bfloat16)
        if len(tokenizer) != model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))
    elif 'llama2-chat-70b' in model_name:
        print ("loading llama2-chat-70b")
        tokenizer = LlamaTokenizer.from_pretrained(f"{path}/cache/Llama-2-70b-chat-hf")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # model = LlamaForCausalLM.from_pretrained(f"{path}/cache/Llama-2-70b-chat-hf", device_map="auto", torch_dtype=torch.bfloat16)
        model = LlamaForCausalLM.from_pretrained(f"{path}/cache/Llama-2-70b-hf", device_map="auto", load_in_8bit=True)
        if len(tokenizer) != model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))
    elif 'llama2-70b' in model_name:
        print ("loading llama2-70b")
        tokenizer = LlamaTokenizer.from_pretrained(f"{path}/cache/Llama-2-70b-hf")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model = LlamaForCausalLM.from_pretrained(f"{path}/cache/Llama-2-70b-hf", device_map="auto", load_in_8bit=True)
        if len(tokenizer) != model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))
    elif 'llama2-chat-13b' in model_name:
        print ("loading llama2")
        tokenizer = AutoTokenizer.from_pretrained(f"{path}/cache/Llama-2-13b-chat-hf")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model = LlamaForCausalLM.from_pretrained(f"{path}/cache/Llama-2-13b-chat-hf", use_auth_token=True, device_map="auto", torch_dtype=torch.bfloat16)
        if len(tokenizer) != model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))
    elif 'llama2-13b' in model_name:
        print ("loading llama2")
        tokenizer = AutoTokenizer.from_pretrained(f"{path}/cache/Llama-2-13b-hf")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model = LlamaForCausalLM.from_pretrained(f"{path}/cache/Llama-2-13b-hf", use_auth_token=True, device_map="auto", torch_dtype=torch.bfloat16)
        if len(tokenizer) != model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))
    elif 'llama2-7b' in model_name:
        print ("loading llama2")
        tokenizer = LlamaTokenizer.from_pretrained(f"{path}/cache/Llama-2-7b-hf")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        model = LlamaForCausalLM.from_pretrained(f"{path}/cache/Llama-2-7b-hf", device_map="auto", torch_dtype=torch.bfloat16) 
        
        if len(tokenizer) != model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))
    elif 'llama2-7b-chat' in model_name:
        print ("loading llama2")

        tokenizer = LlamaTokenizer.from_pretrained(f"{path}/cache/Llama-2-7b-chat-hf")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        model = LlamaForCausalLM.from_pretrained(f"{path}/cache/Llama-2-7b-chat-hf", torch_dtype=torch.bfloat16).to('cuda:0') # OK!
        if len(tokenizer) != model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))
    elif 'tinyllama-chat' in model_name:
        print ("loading TinyLlama-1.1B-Chat-v1.0")
        tokenizer = LlamaTokenizer.from_pretrained(f"{path}/cache/TinyLlama-1.1B-Chat-v1.0")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model = LlamaForCausalLM.from_pretrained(f"{path}/cache/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16).to('cuda:0') # OK!
        if len(tokenizer) != model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))
    elif 'tinyllama-base' in model_name:
        print ("loading TinyLlama-base")
        tokenizer = LlamaTokenizer.from_pretrained(f"{path}/cache/TinyLlama_v1.1")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model = LlamaForCausalLM.from_pretrained(f"{path}/cache/TinyLlama_v1.1", torch_dtype=torch.bfloat16).to('cuda:0') # OK!
        if len(tokenizer) != model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))
    elif 'llama3-8b' in model_name:
        print ("loading llama3-8b")
        tokenizer = AutoTokenizer.from_pretrained(f"{path}/cache/Llama-3-8B")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model = LlamaForCausalLM.from_pretrained(f"{path}/cache/Llama-3-8B", torch_dtype=torch.bfloat16).to('cuda:0') # OK!
        if len(tokenizer) != model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))
    elif 'gemma-7b' in model_name:
        print ("loading gemma-7b")
        tokenizer = AutoTokenizer.from_pretrained(f"{path}/cache/gemma-7b")
        model = AutoModelForCausalLM.from_pretrained(f"{path}/cache/gemma-7b", torch_dtype=torch.bfloat16).to('cuda:0') # OK!
    elif 'llama1-65b' in model_name:
        print ("loading llama")
        tokenizer = LlamaTokenizer.from_pretrained(f"{path}/cache/65B/")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model = LlamaForCausalLM.from_pretrained(f"{path}/cache/65B/", device_map="auto", torch_dtype=torch.bfloat16)
        if len(tokenizer) != model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True, device_map="auto", torch_dtype=torch.bfloat16)

    if model != None and len(tokenizer) != model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def find_all_linear_modules(
    model: "PreTrainedModel",
    quantization_bit=None,
    output_layer_name="lm_head"
):
    if quantization_bit is not None:
        import bitsandbytes as bnb
        linear_cls = bnb.nn.Linear4bit if quantization_bit == 4 else bnb.nn.Linear8bitLt
    else:
        linear_cls = torch.nn.Linear

    module_names = set()
    for name, module in model.named_modules():
        if output_layer_name not in name and isinstance(module, linear_cls):
            module_names.add(name.split(".")[-1])

    if output_layer_name in module_names:
        module_names.pop(output_layer_name)

    return list(module_names)


def init_adapter(model, args, is_trainable, is_mergeable, is_pnet=False):
    print("Fine-tuning method: LoRA")
    latest_checkpoint = None

    if args.checkpoint_dir is not None:
        print("Loaded fine-tuned model from checkpoint(s): {}".format(",".join(args.checkpoint_dir)))

        if (is_trainable and args.resume_lora_training) or (not is_mergeable): # continually fine-tuning
            checkpoints_to_merge, latest_checkpoint = args.checkpoint_dir[:-1], args.checkpoint_dir[-1]
        else:
            checkpoints_to_merge = args.checkpoint_dir

        for checkpoint in checkpoints_to_merge:
            model = PeftModel.from_pretrained(model, checkpoint)
            model = model.merge_and_unload()

        if len(checkpoints_to_merge) > 0:
            print ("Merged {} model checkpoint(s).".format(len(checkpoints_to_merge)))

        if latest_checkpoint is not None: # resume lora training or quantized inference
            model = PeftModel.from_pretrained(model, latest_checkpoint, is_trainable=is_trainable)

    # training
    if is_trainable and latest_checkpoint is None: # create new lora weights while training
        if len(args.lm_lora_target) == 1 and args.lm_lora_target[0] == "all":
            target_modules = find_all_linear_modules(model, args.quantization_bit)
        else: # llama2: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            target_modules = args.lm_lora_target

        if is_pnet:
            target_modules = args.pnet_lora_target

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_rank, # default=8, The intrinsic dimension for LoRA fine-tuning.
            lora_alpha=args.lora_alpha, # default=32.0, The scale factor for LoRA fine-tuning (similar with the learning rate).
            lora_dropout=args.lora_dropout, # default=0.1
            target_modules=target_modules,
            modules_to_save=args.additional_target # default=None, modules apart from LoRA layers to be set as trainable and saved in the final checkpoint.
        )
        model = get_peft_model(model, lora_config)
        if id(model.peft_config) != id(model.base_model.peft_config): # https://github.com/huggingface/peft/issues/923
            model.base_model.peft_config = model.peft_config
        model.print_trainable_parameters()

    return model


def build_lm(args):
    model, tokenizer = load_model(args.model_name_or_path, use_pnet=args.use_pnet)
    if args.lm_use_lora:
        args.lm_lora_target = args.lm_lora_target.split(',')
        print("lm_lora_target: ", args.lm_lora_target)
        model = init_adapter(model, args, is_trainable=True, is_mergeable=False)
    return model, tokenizer


def build_pnet(args):
    model, tokenizer = load_model(args.pnet_name_or_path, args.use_pnet)
    # print(model)
    if args.use_pnet:
        if args.pnet_use_lora:
            args.pnet_lora_target = args.pnet_lora_target.split(',')
            print("pnet_lora_target: ", args.pnet_lora_target)
            model = init_adapter(model, args, is_trainable=True, is_mergeable=False, is_pnet=True)
        model = FLANEncoder(model, tokenizer)
        is_trainable = model.read_out.weight.requires_grad # Linear layer
        print(f"The 'read_out' parameter is {'trainable' if is_trainable else 'not trainable'}.")
    return model, tokenizer


def get_special_ids(tokenizer):
    bos_ids = [tokenizer.bos_token_id]
    eos_ids = [tokenizer.eos_token_id]
    
    pad_ids = [tokenizer.pad_token_id]
    assert pad_ids != [None], pad_ids
    
    if bos_ids == [None] and eos_ids == [None]: # bert
        bos_ids = [tokenizer.cls_token_id]
        eos_ids = [tokenizer.sep_token_id]
        assert bos_ids != [None] and eos_ids != [None], (bos_ids, eos_ids)
        
    assert not (bos_ids == [None] and eos_ids == [None]), (bos_ids, eos_ids) 

    return bos_ids, eos_ids, pad_ids


class T5EncoderModel_cl(T5EncoderModel):
    _tied_weights_keys = ["encoder.embed_tokens.weight"]
    _keys_to_ignore_on_load_unexpected = [r"decoder"]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def get_input_embeddings(self):
        return self.shared

    def get_encoder(self):
        return self.encoder

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, T5EncoderModel

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-small")
        >>> model = T5EncoderModel.from_pretrained("t5-small")
        >>> input_ids = tokenizer(
        ...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model(input_ids=input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return encoder_outputs


class FLANEncoder(nn.Module):
# class FLANEncoder(PreTrainedModel):
    def __init__(self, backbone, tokenizer):
        super(FLANEncoder, self).__init__()  # 在这里调用基类的构造函数

        self._backbone = backbone
        self.tokenizer = tokenizer

        bos_ids, eos_ids, pad_ids = get_special_ids(tokenizer)

        h_size = backbone.config.hidden_size

        self.cls_ids = eos_ids[0] # flan
        if 'bert' in tokenizer.name_or_path:
            self.cls_ids = bos_ids[0]
        
        self.read_out = nn.Linear(h_size, h_size).to(backbone.device).to(backbone.dtype)
        
        self.device = self._backbone.device


    def forward(self, bz, shot, xs, attention_mask, temperature=0.05, n_iter_sinkhorn=50, noise_factor=1.0, args=None):
        '''
        xs: [bz, (shot + 1) * avg_len]
        attention_mask: [bz, (shot + 1) * avg_len]
        '''
        # feature extraction
        # last_hidden_state：[bz, (num_cls + 1) * avg_len, h_size]
        last_hidden_state = self._backbone(input_ids=xs, attention_mask=attention_mask).last_hidden_state

        cls_indices = xs == self.cls_ids
        assert cls_indices.shape == xs.shape, (cls_indices.shape, xs.shape)
        assert cls_indices.shape == last_hidden_state.shape[:2], (cls_indices.shape, last_hidden_state.shape)

        if args.use_instruction:
            assert torch.all(cls_indices.sum(dim=1) == shot + 2), (cls_indices.sum(dim=1), shot)
        else:
            assert torch.all(cls_indices.sum(dim=1) == shot + 1), (cls_indices.sum(dim=1), shot)

        # [bz, (num_cls + 1) * avg_len, h_size] --> [bz * num_cls, h_size] --> [bz, num_cls, h_size]
        # [bz, T, h_size] --> [bz * num_cls, h_size] --> [bz, num_cls, h_size]
        cls_representations = last_hidden_state[cls_indices.to(last_hidden_state.device)]
        # print ('cls_representations.shape =', cls_representations.shape) # torch.Size([93, 768])
        H = cls_representations.reshape(len(xs), -1, cls_representations.shape[-1]) 
        H = H[:,:-1,:]
        if args.use_instruction:
            H = H[:,1:,:]
        
        # [bz, num_cls, h_size] --> [bz, num_cls, h_size]
        transformed_H = self.read_out(H) # RuntimeError: mat1 and mat2 must have the same dtype

        # [bz, num_cls, h_size] x [bz, h_size, num_cls] --> [bz, num_cls, num_cls]
        permutation_matrix = torch.nn.functional.relu(torch.matmul(transformed_H, H.transpose(1, 2)))
        # print ('permutation_matrix.shape = ', permutation_matrix.shape) 

        log_alpha = permutation_matrix
        samples_per_num = 1
        soft_perms_inf, log_alpha_w_noise = my_sinkhorn_ops.my_gumbel_sinkhorn(log_alpha, temperature, samples_per_num, noise_factor,  n_iter_sinkhorn, squeeze=False)

        # [bz, 1, num_cls, num_cls] -> [bz, num_cls, num_cls]
        soft_perms_inf = soft_perms_inf.view(bz, shot, shot)
        # print ('soft_perms_inf.shape = ', soft_perms_inf.shape)

        return soft_perms_inf

