date=TODAY

accelerate=False

debug=True
# debug=False

use_instruction=True
test_task_size=50 # 每个task的大小，需要考虑分到每个shot上有多少

batch_size=8

# max_perm_num=150 # > 120
max_perm_num=100 # > 120

max_seq_len=4000
# max_seq_len=512

eval_left=False

# best dro model: 8b + DRO
step=119
run_id="9dro_TASK150_instructTrue_shot2-5_maxlen512_demon512_LMllama3-8b_lr3e-5_bz16_once1_LoraTrue_wdTrue0.1_PNETflan-t5-large_lr1e-4_bz16_once2_trainin0_usein0_LoraTrue_OTtemp0.1_noise0.3_Ent1.0_HardFalseFalse_step1000000_epoch2_clipFalse"
model_path=prlm/iclr_models/${run_id}/model/${step}_step/
load_lora_model=True

export CUDA_VISIBLE_DEVICES=0

for min_shot in 64 32 16 8; do

max_shot=$min_shot

python3 -u eval_large_permutation_niv.py \
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
--debug $debug 
# 1>log/${date}_bestdro_shot${max_shot}_s${max_seq_len}p${max_perm_num}

wait

done

# baseline model: 8b + ERM
# step=118
# run_id="bl_TASK150_instructTrue_shot2-5_maxlen512_demon512_LMllama3-8b_lr3e-5_bz16_once2_LoraTrue_wdTrue0.1_step1000000_epoch2_clipFalse"
# # model_path=prlm/niv2/929_exp/${run_id}/model/${step}_step/
# model_path=/mnt/teamdrive/projects/backup_chenlian/prlm/iclr_models/${run_id}/model/${step}_step/
# load_lora_model=True

# python3 -u eval_large_permutation_niv.py \
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
# --debug $debug 1>log/${date}_bl_shot${max_shot}_s${max_seq_len}p${max_perm_num}
