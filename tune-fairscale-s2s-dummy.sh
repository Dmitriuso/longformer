#!/usr/bin/env bash
python -m torch.distributed.launch --nproc_per_node=8 s2s-dummy.py