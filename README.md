# BenchmarksDistribuitedTraining

## Deepspeed
- Run on multiple gpus:
> deepspeed --hostfile=hostfile train.py --deepspeed_config ds_config.json --epochs=1 --batch_size 64

- Run on single gpu:
> deepspeed train.py --deepspeed_config ds_config.json --epochs=1 --batch_size 64

## Colossal AI
- Run on multiple gpus:
> colossalai run --nproc_per_node 1 --hostfile ./hostfile --master_addr [ip_host_1] train.py

- Run on single gpu:
> colossalai run --nproc_per_node 1 train.py

