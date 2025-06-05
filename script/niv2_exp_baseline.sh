#!/bin/bash

date=TODAY

# Debug mode
debug=False

# Output directory
out_dir_base="../niv2/${date}_exp"

# Model configuration
model_name_or_path="llama3-8b"
pnet_name_or_path="flan-t5-large"

# Data configuration
max_seq_len=512
max_demonstration_len=512
min_shot=2
max_shot=5

train_task="" # Default: use all tasks
train_task_size=150
test_task_size=64

lr=3e-5
pnet_lr=1e-4

gradient_accumulation_step=4 
pnet_gradient_accumulation_step=4 
pnet_bz=16 
lm_bz=16

infer_bz=16

# Training parameters
epoch=2
train_steps=1000000
save_every_step=40
eval_every_step=40 
log_every_step=1

use_instruction=True

# LM regularization 
use_weight_decay=True
weight_decay_coefficient=0.1 # llama2 sft, we may need larger one as we have less data

# pnet regularization
pnet_use_weight_decay=True
pnet_weight_decay_coefficient=0.1
use_gradient_clip=False

lm_once_step=1
pnet_once_step=2
start_train_pnet_in=0
start_use_pnet_in=0

# OT 
noise_factor=0.3 # 1.0
temperature=0.1
n_iter_sinkhorn=80 # small temp needs large #interation
train_hard_permutation=False # harden output in training # 那么不用做 entropy penalty or matrix alignment 了，因为p-net无法通过改变语义来shotcut学习
use_hard_permutation=False # # harden output in inference

lambda_matrix_align=0
lambda_entropy_pen=1.0

# LoRA configuration
lm_lora_target="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
pnet_lora_target="q,v"
lora_rank=8
lora_alpha=32.0
lora_dropout=0.1

lm_use_lora=True
pnet_use_lora=True

use_pnet=False # Baseline

export CUDA_VISIBLE_DEVICES=0


shuffle_demons=True

# use_mixup=True
use_mixup=False
gradient_accumulation_step=16

mix_method='mean' # max mean
# n_candidate=3
n_candidate=2

# do_eval=True
do_eval=False


# EXP ID
data_params="T${train_task}${train_task_size}_instr${use_instruction}_shot${min_shot}-${max_shot}_len${max_seq_len}_demon${max_demonstration_len}_SD${shuffle_demons}_mix${use_mixup}${mix_method}${n_candidate}"
lm_params="L${model_name_or_path}_lr${lr}_bz${lm_bz}_once${lm_once_step}_Lora${lm_use_lora}_wd${use_weight_decay}${weight_decay_coefficient}"
pnet_params="P${pnet_name_or_path}_lr${pnet_lr}_bz${pnet_bz}_once${pnet_once_step}_trainin${start_train_pnet_in}_usein${start_use_pnet_in}_pLora${pnet_use_lora}_pwd${pnet_use_weight_decay}${pnet_weight_decay_coefficient}"
ot_params="OTtemp${temperature}_iter${n_iter_sinkhorn}_nois${noise_factor}_Ent${lambda_entropy_pen}"
train_params="step${train_steps}_epo${epoch}_clip${use_gradient_clip}"

if [ "$use_pnet" = "True" ]; then
    run_id="${date}_dro_${data_params}_${lm_params}_${pnet_params}_${ot_params}_${train_params}"
else
    run_id="${date}_bl_${data_params}_${lm_params}_${train_params}"
fi

echo $run_id
rm -rf "${out_dir_base}/${run_id}"
mkdir -p "${out_dir_base}/${run_id}"


python3 -u train_niv.py \
  --debug $debug \
  --do_eval $do_eval \
  --shuffle_demons $shuffle_demons \
  --use_mixup $use_mixup \
  --mix_method $mix_method \
  --n_candidate $n_candidate \
  --use_gradient_clip $use_gradient_clip \
  --use_instruction $use_instruction \
  --train_task $train_task \
  --train_task_size $train_task_size \
  --test_task_size $test_task_size \
  --test_task $test_task \
  --min_shot ${min_shot} \
  --max_shot ${max_shot} \
  --model_name_or_path $model_name_or_path \
  --pnet_name_or_path $pnet_name_or_path \
  --train_hard_permutation ${train_hard_permutation} \
  --use_hard_permutation ${use_hard_permutation} \
  --max_seq_len $max_seq_len \
  --max_demonstration_len $max_demonstration_len \
  --lm_use_lora $lm_use_lora \
  --pnet_use_lora $pnet_use_lora \
  --lm_lora_target $lm_lora_target \
  --pnet_lora_target $pnet_lora_target \
  --lora_rank $lora_rank \
  --lora_alpha $lora_alpha \
  --lora_dropout $lora_dropout \
  --temperature $temperature \
  --n_iter_sinkhorn $n_iter_sinkhorn \
  --noise_factor $noise_factor \
  --lambda_entropy_pen ${lambda_entropy_pen} \
  --lambda_matrix_align ${lambda_matrix_align} \
  --use_pnet ${use_pnet} \
  --use_weight_decay ${use_weight_decay} \
  --weight_decay_coefficient ${weight_decay_coefficient} \
  --pnet_use_weight_decay ${pnet_use_weight_decay} \
  --pnet_weight_decay_coefficient ${pnet_weight_decay_coefficient} \
  --lm_gradient_accumulation_step ${gradient_accumulation_step} \
  --pnet_gradient_accumulation_step ${pnet_gradient_accumulation_step} \
  --pnet_once_step ${pnet_once_step} \
  --lm_once_step ${lm_once_step} \
  --start_train_pnet_in ${start_train_pnet_in} \
  --start_use_pnet_in ${start_use_pnet_in} \
  --epoch ${epoch} \
  --resume_id $run_id \
  --out_dir $out_dir_base \
  --learning_rate $lr \
  --pnet_learning_rate $pnet_lr \
  --pnet_bz $pnet_bz \
  --lm_bz $lm_bz \
  --infer_bz $infer_bz \
  --train_steps $train_steps \
  --log_every_step $log_every_step \
  --save_every_step $save_every_step \
  --eval_every_step $eval_every_step 1>log/${run_id} 2>&1
