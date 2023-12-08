#
# We include a simple lora finetuning & inference script here. The goal is so that you can finetune both
# both skill and slot model on 7B model on consumer grade hardware like x090.
#
python3 opencui/finetune/generation.py \
    --model_name_or_path TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T \
    --output_dir ./output/lr1e-5_ep5_top1_2023-11-21 \
    --logging_steps 50 \
    --save_strategy epoch \
    --data_seed 42 \
    --save_total_limit 6 \
    --evaluation_strategy epoch \
    --eval_dataset_size 512 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 32 \
    --dataloader_num_workers 3 \
    --group_by_length False \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --warmup_ratio 0.05 \
    --lr_scheduler_type constant \
    --source_max_len 512 \
    --target_max_len 128 \
    --per_device_train_batch_size 16 \
    --max_steps 0 \
    --num_train_epochs 4 \
    --learning_rate 2e-5 \
    --adam_beta2 0.999 \
    --max_grad_norm 1.0 \
    --weight_decay 0.0 \
    --seed 0 \
    --debug_dataset False \
    --training_mode nli \
    --trust_remote_code \
    --report_to wandb