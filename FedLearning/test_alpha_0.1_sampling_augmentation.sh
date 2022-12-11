#!/bin/bash

######################################################
# alpha 0.1  withour sampling method
######################################################

# smote r_under r_over augment

# Fix Setting
is_mp='False';num_clients=100;fraction=0.1
rounds=100;seed=42;alpha=3;sampling_type=smote

# CIFAR-10
dataset_name='cifar10';tm_local_bs=10

device=7
tm_criterion=CrossEntropyLoss;method=fedavg;mu=None
filename="${seed}_${dataset_name}_${alpha}_${method}(mu:${mu})_${tm_criterion}_${sampling_type}"
python main.py --mp=${is_mp} --method=${method} --sampling_type=${sampling_type} --tm_mu=${mu} --tm_criterion=${tm_criterion} --tm_local_bs=${tm_local_bs} --device=${device} --alpha=${alpha} --dataset_name=${dataset_name} --num_clients=${num_clients} --fraction=${fraction} --rounds=${rounds} > ./output/${filename}.out

wait;echo "done"
