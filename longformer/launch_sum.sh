python script_summarization.py \
  --model_path sshleifer/tiny-mbart \
  --batch_size 6 \
  --num_workers 8 \
  --epochs 2 \
  --attention_window 256 \
  --disable_checkpointing \
  --save_dir ./trained_model \
  --max_input_len 1024 \
  --max_output_len 256