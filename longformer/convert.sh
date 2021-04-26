python3 script_mbart_to_led.py \
  --base_model facebook/mbart-large-cc25\
  --tokenizer_name_or_path facebook/mbart-large-cc25 \
  --save_model_to LongMBART-mbart_tokenizer/ \
  --max_pos 16384 \
  --attention_window 512