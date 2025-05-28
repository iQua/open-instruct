export CUDA_VISIBLE_DEVICES=0,1,2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch \
--mixed_precision bf16 \
--dynamo_backend inductor \
--num_machines 1 \
--num_processes 1 \
open_instruct/finetune.py \
--model_name_or_path allenai/OLMoE-1B-7B-0924 \
--tokenizer_name allenai/OLMoE-1B-7B-0924 \
--use_flash_attn \
--use_lora \
--lora_rank 32 \
--lora_alpha 16 \
--lora_dropout 0.1 \
--max_seq_length 1024 \
--preprocessing_num_workers 4 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 16 \
--learning_rate 2e-05 \
--lr_scheduler_type linear \
--warmup_ratio 0.03 \
--weight_decay 0.0 \
--num_train_epochs 2 \
--output_dir output/ \
--logging_steps 1 \
--reduce_loss sum \
--model_revision main \
--dataset_mixer_list allenai/tulu-v3.1-mix-preview-4096-OLMoE 0.1 \
--checkpointing_steps epoch \
--add_bos \
--gradient_checkpointing