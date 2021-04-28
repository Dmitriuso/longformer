#!/usr/bin/env bash
python3 script_mbart_to_led.py \
  --base_model sshleifer/tiny-mbart \
  --tokenizer_name_or_path facebook/mbart-large-cc25 \
  --save_model_to models/LongTinyMBART \
  --max_pos 16384 \
  --attention_window 512