date=TODAY

accelerate=False

max_shot=5
min_shot=3

debug=True
# debug=False

use_instruction=True
test_task_size=50 
# test_task_size=64
batch_size=32 

max_perm_num=150 
# max_seq_len=1500
max_seq_len=512

eval_left=False

# baseline model
# step=246 # last point
# run_id=926_bl_TASK150_instructTrue_shot2-6_maxlen1000_demon512_LMtinyllama-base_lr1e-4_bz16_once2_LoraTrue_step1000000_epoch2
# model_path=${run_id}/model/${step}_step/
# load_lora_model=True

# dro model
# step=246 # last point
# run_id=926_dro_TASK150_instructTrue_shot2-6_maxlen1000_demon512_LMtinyllama-base_lr1e-4_bz16_once2_LoraTrue_PNETflan-t5-base_lr1e-4_bz16_once2_trainin0_usein100_LoraTrue_OTtemp0.1_noise0.3_Ent0.0_HardTrue_step1000000_epoch2
# model_path=${run_id}/model/${step}_step/
# load_lora_model=True

# base model: llama3-8b-instruct
# step=0
# run_id=Meta-Llama-3-8B-Instruct
# model_path=${run_id}
# load_lora_model=False

# base model: llama3-8b-base
step=0
run_id=Llama-3-8B
model_path=${run_id}
load_lora_model=False
eval_left=False

# # best dro model: 8b + DRO
# step=119
# run_id="dro_TASK150_instructTrue_shot2-5_maxlen512_demon512_LMllama3-8b_lr3e-5_bz16_once1_LoraTrue_wdTrue0.1_PNETflan-t5-large_lr1e-4_bz16_once2_trainin0_usein0_LoraTrue_OTtemp0.1_noise0.3_Ent1.0_HardFalseFalse_step1000000_epoch2_clipFalse"
# model_path=${run_id}/model/${step}_step/
# load_lora_model=True

export CUDA_VISIBLE_DEVICES=0

python3 -u eval_permutation_niv.py \
--load_lora_model $load_lora_model \
--model_path $model_path \
--accelerate $accelerate \
--eval_left $eval_left \
--use_instruction $use_instruction \
--test_task_size $test_task_size \
--batch_size $batch_size \
--max_perm_num $max_perm_num \
--max_seq_len $max_seq_len \
--max_shot $max_shot \
--min_shot $min_shot \
--step $step \
--debug $debug 1>log/${date}e_${run_id}_s${step}ms${max_seq_len}mp${max_perm_num} 2>&1

# wait

# # baseline model: 8b + ERM
# step=118
# run_id="bl_TASK150_instructTrue_shot2-5_maxlen512_demon512_LMllama3-8b_lr3e-5_bz16_once2_LoraTrue_wdTrue0.1_step1000000_epoch2_clipFalse"
# model_path=${run_id}/model/${step}_step/
# load_lora_model=True

# python3 -u eval_permutation_niv.py \
# --load_lora_model $load_lora_model \
# --model_path $model_path \
# --accelerate $accelerate \
# --eval_left $eval_left \
# --use_instruction $use_instruction \
# --test_task_size $test_task_size \
# --batch_size $batch_size \
# --max_perm_num $max_perm_num \
# --max_seq_len $max_seq_len \
# --max_shot $max_shot \
# --min_shot $min_shot \
# --step $step \
# --debug $debug 1>log/${date}e_${run_id}_s${step}ms${max_seq_len}mp${max_perm_num} 2>&1

# wait

# export CUDA_VISIBLE_DEVICES=0

# baseline model: 8b + ERM + DS
# step=111
# run_id="bl_T150_instrTrue_shot2-5_len512_demon512_SDTrue_mixFalsemean2_Lllama3-8b_lr3e-5_bz16_once1_LoraTrue_wdTrue0.1_step1000000_epo2_clipFalse"
# model_path=prlm/niv2/101_exp/${run_id}/model/${step}_step/
# load_lora_model=True

# python3 -u eval_permutation_niv.py \
# --load_lora_model $load_lora_model \
# --model_path $model_path \
# --accelerate $accelerate \
# --eval_left $eval_left \
# --use_instruction $use_instruction \
# --test_task_size $test_task_size \
# --batch_size $batch_size \
# --max_perm_num $max_perm_num \
# --max_seq_len $max_seq_len \
# --max_shot $max_shot \
# --min_shot $min_shot \
# --step $step \
# --debug $debug 1>log/${date}e_${run_id}_s${step}ms${max_seq_len}mp${max_perm_num} 2>&1

# wait

# baseline model: 8b + ERM + mixup mean
# step=111
# run_id="bl_T150_instrTrue_shot2-5_len512_demon512_SDFalse_mixTruemean2_Lllama3-8b_lr3e-5_bz16_once1_LoraTrue_wdTrue0.1_step1000000_epo2_clipFalse"
# model_path=${run_id}/model/${step}_step/
# load_lora_model=True

# export CUDA_VISIBLE_DEVICES=1

# python3 -u eval_permutation_niv.py \
# --load_lora_model $load_lora_model \
# --model_path $model_path \
# --accelerate $accelerate \
# --eval_left $eval_left \
# --use_instruction $use_instruction \
# --test_task_size $test_task_size \
# --batch_size $batch_size \
# --max_perm_num $max_perm_num \
# --max_seq_len $max_seq_len \
# --max_shot $max_shot \
# --min_shot $min_shot \
# --step $step \
# --debug $debug 1>log/${date}e_${run_id}_s${step}ms${max_seq_len}mp${max_perm_num} 2>&1

# # wait

# # baseline model: 8b + ERM + mixup max
# step=111
# run_id="101_bl_T150_instrTrue_shot2-5_len512_demon512_SDFalse_mixTruemax2_Lllama3-8b_lr3e-5_bz16_once1_LoraTrue_wdTrue0.1_step1000000_epo2_clipFalse"
# model_path=prlm/niv2/101_exp/${run_id}/model/${step}_step/
# load_lora_model=True

# export CUDA_VISIBLE_DEVICES=3

# python3 -u eval_permutation_niv.py \
# --load_lora_model $load_lora_model \
# --model_path $model_path \
# --accelerate $accelerate \
# --eval_left $eval_left \
# --use_instruction $use_instruction \
# --test_task_size $test_task_size \
# --batch_size $batch_size \
# --max_perm_num $max_perm_num \
# --max_seq_len $max_seq_len \
# --max_shot $max_shot \
# --min_shot $min_shot \
# --step $step \
# --debug $debug 1>log/${date}e_${run_id}_s${step}ms${max_seq_len}mp${max_perm_num} 2>&1


# # 1>log/${date}ev_${run_id}_s${step}_lft${eval_left}_max_perm_num
# # 1>log/${date}_neweval_${run_id}_step${step}_size${test_task_size}_instr${use_instruction}_accelerate${accelerate} 2>&1 

