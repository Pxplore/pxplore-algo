# pip install math_verify # reward function
# pip install -U trl
# GPU memory: 80GiB
# register customized plugin in external_plugins file

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
swift rlhf \
    --rlhf_type grpo \
    --model /data/ljy/model/Llama-3.1-8B-Instruct/sft\
    --model_type llama3 \
    --external_plugins model/training/ms-swift/examples/train/grpo/plugin/plugin.py \
    --reward_funcs external_pxplore_reward \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset model/training/ms-swift/examples/train/grpo/qwen3/pxplore.json \
    --max_completion_length 1024 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 1 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 8192 \
    --output_dir /data/ljy/model/Llama-3.1-8B-Instruct/grpo \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 2 \
    --dataset_num_proc 2 \
    --num_generations 2 \
    --temperature 0.5 \
    --system model/training/ms-swift/examples/train/grpo/qwen3/system_prompt.txt \
    --log_completions true
