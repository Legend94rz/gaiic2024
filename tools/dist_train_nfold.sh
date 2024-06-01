#!/usr/bin/env bash

# CONFIG=$1
# GPUS=$2
# # NNODES=${NNODES:-1}
# # NODE_RANK=${NODE_RANK:-0}
# PORT=${PORT:-29500}
# MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# torchrun \
#     # --nnodes=$NNODES \
#     # --node_rank=$NODE_RANK \
#     --master_addr=$MASTER_ADDR \
#     --nproc_per_node=$GPUS \
#     --master_port=$PORT \
#     $(dirname "$0")/train.py \
#     $CONFIG \
#     --launcher pytorch ${@:3}


export OMP_NUM_THREADS=1
export PYTHONPATH=.
nnode=2
# cfg="projects/gaiic2014/configs/mean_fuse.py"
# cfg="projects/gaiic2014/configs/debug.py"
cfg="projects/gaiic2014/configs/codetr_all_in_one.py"
# cfg="projects/gaiic2014/configs/glip_all_in_one.py"

time=`date +"%Y%m%d_%H%M%S"`
opt2="train_cfg={'max_epochs': 12}"

for i in {0..4}
do
    opt="train_dataloader={'dataset': {'ann_file': 'annotations/train_0527_${i}_train.json'}}"
    # opt="train_dataloader={'dataset': {'ann_file': 'annotations/fold_${i}_train.json'}}"
    torchrun --nproc-per-node=${nnode} tools/train.py ${cfg} --launcher pytorch --work-dir ./work_dirs/`basename ${cfg} .py`/_${time}_fold_${i} --cfg-options "${opt}" "${opt2}"
done

