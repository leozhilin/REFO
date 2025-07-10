export DATA_PATH=data/UrbanVideo-Bench
export CKPT_PATH=Qwen/Qwen2.5-VL-3B-Instruct
export SAVE_PATH=models/ckpt/qwen2_5_3b_urbanvideo

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_qwen2_5_3b_urbanvideo.txt"

# Disable P2P and IB communication for RTX 4000 series
# export NCCL_P2P_DISABLE="1"
# export NCCL_IB_DISABLE="1"
# export CUDA_VISIBLE_DEVICES=0,1

# Set Hugging Face endpoint to use mirror
export HF_ENDPOINT=https://hf-mirror.com
# export WANDB_BASE_URL=https://api.bandw.top
# export TORCH_HOME=/share/leozhilin/.torch
# export CUDA_LAUNCH_BLOCKING=1

torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    train/src/open_r1/mygrpo.py \
    --output_dir ${SAVE_PATH}  \
    --model_name_or_path ${CKPT_PATH} \
    --dataset_name ${DATA_PATH} \
    --deepspeed train/src/local_scripts/zero3.json \
    --max_prompt_length 10240 \
    --max_completion_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 2 \
    --run_name Qwen2_5-VL-3B_GRPO_urbanvideo \
    --save_steps 100 \
    --save_only_model true \
    --num_generations 4 \
    --gradient_checkpointing True \
    --use_peft true \
    --lora_dropout 0.1 \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    # --use_vllm \
    # --vllm_device cuda:5