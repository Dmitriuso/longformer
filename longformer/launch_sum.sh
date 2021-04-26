python script_summarization.py \
  --model_path sshleifer/tiny-mbart \
  --batch_size 4 \
  --num_workers 8 \
  --epochs 1 \
  --attention_window 512 \
  --save_dir ./trained_model