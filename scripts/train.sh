model="meta-llama/Llama-2-7b-chat-hf"
task_name="gsm8k"
template="zero-shot-target-aware"
pos_path="data/dataset/gsm8k/gsm8k_gpt-3.5_positive.json"
neg_path="data/dataset/gsm8k/gsm8k_gpt-3.5_negative.json"
pos_num="5000"
neg_num="10000"
prompt_path="prompts/gsm8k/gsm8k_conversation.json"
question_path="data/dataset/gsm8k/train.json"
output_dir="data/checkpoints/NAT-7b-math"

epochs="2"
lr="5e-5"
batch_size="2"
gradient_accumulation_steps="8"
max_length="4096"

cmd="torchrun --nproc_per_node=4 --nnodes=1 --master_port=29522 train/train.py \
    --model_name_or_path  ${model} \
    --task_name ${task_name} \
    --template ${template} \
    --pos_path ${pos_path} \
    --neg_path ${neg_path} \
    --pos_num ${pos_num} \
    --neg_num ${neg_num} \
    --prompt_path ${prompt_path} \
    --question_path ${question_path} \
    --deepspeed configs/deepspeed_z3_config.json \
    --output_dir ${output_dir} \
    --num_train_epochs ${epochs} \
    --per_device_train_batch_size ${batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --save_strategy "no" \
    --learning_rate ${lr} \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --logging_steps 4 \
    --model_max_length ${max_length} \
    --bf16 True"
echo $cmd
eval $cmd