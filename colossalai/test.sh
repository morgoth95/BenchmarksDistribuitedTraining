colossalai run --nproc_per_node 1 --hostfile ./hostfile --master_addr 172.31.4.100 train.py --models="resnet152" --limit_data=6000 --find_bs --name resnet152
