#!/usr/bin/env bash
python -m torch.distributed.launch --nproc_per_node=2 fine_tuning-LED.py