#!/bin/bash
GPU_NUM=${1:-2}
CPU_NUM=${2:-$((GPU_NUM * 8))}
MEMORY=${3:-$((GPU_NUM * 64000))}
echo "GPU_NUM: ${GPU_NUM}, CPU_NUM: ${CPU_NUM}, MEMORY: ${MEMORY}"

rlaunch --gpu=${GPU_NUM} \
    --cpu=${CPU_NUM} \
    --memory=${MEMORY} \
    --private-machine=yes \
    --charged-group=stu \
    --mount=gpfs://gpfs1/ailab-sys/guojihu:/mnt/shared-storage-user/ailab-sys/guojihu \
    --mount=gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public \
    --image=registry.h.pjlab.org.cn/ailab-sys-sys_gpu/megatron:25.12-py3-nvshmem-tmux \
    --workdir=/mnt/shared-storage-user/ailab-sys/guojihu/vllm \
    --entrypoint /bin/bash

