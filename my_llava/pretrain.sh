#!/bin/bash
set -x

# 自定义参数

# 默认参数
PARTITION=${PARTITION:-"llm_razor"}
GPUS=${GPUS:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
QUOTA_TYPE=${QUOTA_TYPE:-"reserved"}
NODES=$((GPUS / GPUS_PER_NODE))
CPUS_PER_TASK=${CPUS_PER_TASK:-1}
MIRCO_BATCH_SIZE=${MIRCO_BATCH_SIZE:-2}
# SRUN_ARGS=${SRUN_ARGS:-""}

export PYTHONPATH="/data1/exs_data/miniconda3/envs/lite/bin/python"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3

OUTPUT_DIR='work_dirs/llava_pretrain_internlm2_7b'
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# 运行训练脚本
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 torchrun --standalone --nnodes=1 --nproc-per-node=1 llava_train.py \
  --llm /data1/exs_data/ckpt/hf_models/internlm/internlm2_5-1_8b \
  --vit /data1/exs_data/ckpt/hf_models/openai/clip-vit-base-patch32 \
  --chat-template 'internlm2' \
  --freeze-llm \
  --freeze-vit \
  --datasets data/llava_pretrain.json \
  --max-length 2048 \
  --num-workers 4 \
  --mirco-batch-size $MIRCO_BATCH_SIZE \
  --global-batch-size $((MIRCO_BATCH_SIZE*GPUS)) \
  --lr 1e-4 \
  --wd 0.0 \
  --warmup-ratio 0.03 \
  --work-dir ${OUTPUT_DIR} \
  --log-interval 10 \
  --seed 42 \
  --checkpoint-interval 50 \
  --shard-strategy 'zero2' \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
# --checkpoint-drop-optimizer \