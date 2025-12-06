export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=1


export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


wandb online

torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=101 --rdzv_endpoint=localhost:29502 main.py
