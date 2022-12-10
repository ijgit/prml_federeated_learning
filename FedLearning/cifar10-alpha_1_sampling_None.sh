#!/bin/bash

######################################################
# alpha 0.1  withour sampling method
######################################################

# Fix Setting
is_mp='False';num_clients=100;fraction=0.1
rounds=5;seed=42;alpha=1

# CIFAR-10
dataset_name='cifar10';tm_local_bs=10
div1=3;div2=4;div3=5

#################### FedProx ########################
device=${div1}
tm_criterion=CrossEntropyLoss;method=fedavg;mu=None
filename="${seed}_${dataset_name}_${alpha}_${method}(mu:${mu})_${tm_criterion}"  #_${sampling}
python main.py --mp=${is_mp} --method=${method} --tm_mu=${mu} --tm_criterion=${tm_criterion} --tm_local_bs=${tm_local_bs} --device=${device} --alpha=${alpha} --dataset_name=${dataset_name} --num_clients=${num_clients} --fraction=${fraction} --rounds=${rounds} > ./output/${filename}.out 2>&1 &

tm_criterion=FocalLoss;method=fedavg;mu=None
filename="${seed}_${dataset_name}_${alpha}_${method}(mu:${mu})_${tm_criterion}"  #_${sampling}
python main.py --mp=${is_mp} --method=${method} --tm_mu=${mu} --tm_criterion=${tm_criterion} --tm_local_bs=${tm_local_bs} --device=${device} --alpha=${alpha} --dataset_name=${dataset_name} --num_clients=${num_clients} --fraction=${fraction} --rounds=${rounds} > ./output/${filename}.out 2>&1 &

tm_criterion=Ratio_Cross_Entropy;method=fedavg;mu=None
filename="${seed}_${dataset_name}_${alpha}_${method}(mu:${mu})_${tm_criterion}"  #_${sampling}
python main.py --mp=${is_mp} --method=${method} --tm_mu=${mu} --tm_criterion=${tm_criterion} --tm_local_bs=${tm_local_bs} --device=${device} --alpha=${alpha} --dataset_name=${dataset_name} --num_clients=${num_clients} --fraction=${fraction} --rounds=${rounds} > ./output/${filename}.out 2>&1 &

#################### FedProx ########################
device=${div2}
tm_criterion=CrossEntropyLoss;method=fedprox;mu=0.0001
filename="${seed}_${dataset_name}_${alpha}_${method}(mu:${mu})_${tm_criterion}"  #_${sampling}
python main.py --mp=${is_mp} --method=${method} --tm_mu=${mu} --tm_criterion=${tm_criterion} --tm_local_bs=${tm_local_bs} --device=${device} --alpha=${alpha} --dataset_name=${dataset_name} --num_clients=${num_clients} --fraction=${fraction} --rounds=${rounds} > ./output/${filename}.out 2>&1 &

tm_criterion=FocalLoss;method=fedprox;mu=0.0001
filename="${seed}_${dataset_name}_${alpha}_${method}(mu:${mu})_${tm_criterion}"  #_${sampling}
python main.py --mp=${is_mp} --method=${method} --tm_mu=${mu} --tm_criterion=${tm_criterion} --tm_local_bs=${tm_local_bs} --device=${device} --alpha=${alpha} --dataset_name=${dataset_name} --num_clients=${num_clients} --fraction=${fraction} --rounds=${rounds} > ./output/${filename}.out 2>&1 &

tm_criterion=Ratio_Cross_Entropy;method=fedprox;mu=0.0001
filename="${seed}_${dataset_name}_${alpha}_${method}(mu:${mu})_${tm_criterion}"  #_${sampling}
python main.py --mp=${is_mp} --method=${method} --tm_mu=${mu} --tm_criterion=${tm_criterion} --tm_local_bs=${tm_local_bs} --device=${device} --alpha=${alpha} --dataset_name=${dataset_name} --num_clients=${num_clients} --fraction=${fraction} --rounds=${rounds} > ./output/${filename}.out 2>&1 &

#################### FedDyn ########################
device=${div3}
tm_criterion=CrossEntropyLoss;method=feddyn;mu=None
filename="${seed}_${dataset_name}_${alpha}_${method}(mu:${mu})_${tm_criterion}"  #_${sampling}
python main.py --mp=${is_mp} --method=${method} --tm_mu=${mu} --tm_criterion=${tm_criterion} --tm_local_bs=${tm_local_bs} --device=${device} --alpha=${alpha} --dataset_name=${dataset_name} --num_clients=${num_clients} --fraction=${fraction} --rounds=${rounds} > ./output/${filename}.out 2>&1 &

tm_criterion=FocalLoss;method=feddyn;mu=None
filename="${seed}_${dataset_name}_${alpha}_${method}(mu:${mu})_${tm_criterion}"  #_${sampling}
python main.py --mp=${is_mp} --method=${method} --tm_mu=${mu} --tm_criterion=${tm_criterion} --tm_local_bs=${tm_local_bs} --device=${device} --alpha=${alpha} --dataset_name=${dataset_name} --num_clients=${num_clients} --fraction=${fraction} --rounds=${rounds} > ./output/${filename}.out 2>&1 &

tm_criterion=Ratio_Cross_Entropy;method=feddyn;mu=None
filename="${seed}_${dataset_name}_${alpha}_${method}(mu:${mu})_${tm_criterion}"  #_${sampling}
python main.py --mp=${is_mp} --method=${method} --tm_mu=${mu} --tm_criterion=${tm_criterion} --tm_local_bs=${tm_local_bs} --device=${device} --alpha=${alpha} --dataset_name=${dataset_name} --num_clients=${num_clients} --fraction=${fraction} --rounds=${rounds} > ./output/${filename}.out 2>&1 &

wait;echo "done"