# SAVE_DIR="model_zoo/loftq/"
# python quantize_save_load.py \
#     --model_name_or_path meta-llama/Llama-2-7b-hf \
#     --token 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 \
#     --bits 4 \
#     --iter 5 \
#     --rank 8 \
#     --save_dir $SAVE_DIR


python finetune.py \
--base_model 'model_zoo/loftq/Llama-2-7b-hf-4bit-8rank' \
--data_path 'yahma/alpaca-cleaned' \
--output_dir './lora-alpaca-loftq' \
--batch_size 32 \
--micro_batch_size 16 \
--num_epochs 3 \
--learning_rate 1e-4 \
--cutoff_len 512 \
--val_set_size 2000 \
--lora_r 8 \
--lora_alpha 16 \
--lora_dropout 0.05 \
--lora_target_modules '[q_proj,v_proj]' \
--train_on_inputs \
--group_by_length \
--load_in_4bit True \
--use_loftq True
