#!/usr/bin/env bash
python conditional_sum-simple-mBART-extended.py \
	--model_path ../LongMBART-25-8K-wiki-es-15epochs \
	--input_file /data/summarization/wikipedia_data/clean_es/test.source \
	--output_file ./evaluation/LongmBART-25-8K-wiki-15-extended.txt \
	--eval_file ./evaluation/metrics-LongmBART-25-8K-wiki-15-extended.json \
	--input_max_length 8192 \
	--sum_max_length 128 \
	--num_beams 2 \
  	--length_penalty 0.7