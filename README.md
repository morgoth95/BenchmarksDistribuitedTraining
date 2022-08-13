# BenchmarksDistribuitedTraining

## Setup
- Configure password-less ssh between the machines (https://linuxize.com/post/how-to-setup-passwordless-ssh-login/).
- If using aws, the machines must be in the same security group, and the traffic between the machines must be enabled in the inbounding/outbounding rules sections.
- The code must be manually moved in all the machines, and also the required python modules must be pre-installed.

## Deepspeed
- Run on multiple gpus:
> deepspeed --hostfile=hostfile train.py --deepspeed_config ds_config.json --epochs=1

- Run on single gpu:
> deepspeed train.py --deepspeed_config ds_config.json --epochs=1

## Colossal AI
- Run on multiple gpus:
> colossalai run --nproc_per_node 1 --hostfile ./hostfile --master_addr [ip_host_1] train.py

- Run on single gpu:
> colossalai run --nproc_per_node 1 train.py

