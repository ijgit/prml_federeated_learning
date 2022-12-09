#!/bin/bash

######################################################
# alpha 0.1  withour sampling method
######################################################

# smote r_under r_over

# Fix Setting
is_mp='False';num_clients=100;fraction=0.1
rounds=2;seed=42;alpha=10;sampling_type=r_under

# CIFAR-10
dataset_name='cifar10';tm_local_bs=10

#################### FedProx ########################
device=0
tm_criterion=CrossEntropyLoss;method=fedavg;mu=None
filename="${seed}_${dataset_name}_${alpha}_${method}(mu:${mu})_${tm_criterion}_${sampling_type}"
python main.py --mp=${is_mp} --method=${method} --sampling_type=${sampling_type} --tm_mu=${mu} --tm_criterion=${tm_criterion} --tm_local_bs=${tm_local_bs} --device=${device} --alpha=${alpha} --dataset_name=${dataset_name} --num_clients=${num_clients} --fraction=${fraction} --rounds=${rounds} > ./output/${filename}.out 2>&1 &

tm_criterion=FocalLoss;method=fedavg;mu=None
filename="${seed}_${dataset_name}_${alpha}_${method}(mu:${mu})_${tm_criterion}_${sampling_type}"
python main.py --mp=${is_mp} --method=${method} --sampling_type=${sampling_type} --tm_mu=${mu} --tm_criterion=${tm_criterion} --tm_local_bs=${tm_local_bs} --device=${device} --alpha=${alpha} --dataset_name=${dataset_name} --num_clients=${num_clients} --fraction=${fraction} --rounds=${rounds} > ./output/${filename}.out 2>&1 &

tm_criterion=Ratio_Cross_Entropy;method=fedavg;mu=None
filename="${seed}_${dataset_name}_${alpha}_${method}(mu:${mu})_${tm_criterion}_${sampling_type}"
python main.py --mp=${is_mp} --method=${method} --sampling_type=${sampling_type} --tm_mu=${mu} --tm_criterion=${tm_criterion} --tm_local_bs=${tm_local_bs} --device=${device} --alpha=${alpha} --dataset_name=${dataset_name} --num_clients=${num_clients} --fraction=${fraction} --rounds=${rounds} > ./output/${filename}.out 2>&1 &

#################### FedProx ########################
device=1
tm_criterion=CrossEntropyLoss;method=fedprox;mu=0.1
filename="${seed}_${dataset_name}_${alpha}_${method}(mu:${mu})_${tm_criterion}_${sampling_type}"
python main.py --mp=${is_mp} --method=${method} --sampling_type=${sampling_type} --tm_mu=${mu} --tm_criterion=${tm_criterion} --tm_local_bs=${tm_local_bs} --device=${device} --alpha=${alpha} --dataset_name=${dataset_name} --num_clients=${num_clients} --fraction=${fraction} --rounds=${rounds} > ./output/${filename}.out 2>&1 &

tm_criterion=FocalLoss;method=fedprox;mu=0.1
filename="${seed}_${dataset_name}_${alpha}_${method}(mu:${mu})_${tm_criterion}_${sampling_type}"
python main.py --mp=${is_mp} --method=${method} --sampling_type=${sampling_type} --tm_mu=${mu} --tm_criterion=${tm_criterion} --tm_local_bs=${tm_local_bs} --device=${device} --alpha=${alpha} --dataset_name=${dataset_name} --num_clients=${num_clients} --fraction=${fraction} --rounds=${rounds} > ./output/${filename}.out 2>&1 &

tm_criterion=Ratio_Cross_Entropy;method=fedprox;mu=0.1
filename="${seed}_${dataset_name}_${alpha}_${method}(mu:${mu})_${tm_criterion}_${sampling_type}"
python main.py --mp=${is_mp} --method=${method} --sampling_type=${sampling_type} --tm_mu=${mu} --tm_criterion=${tm_criterion} --tm_local_bs=${tm_local_bs} --device=${device} --alpha=${alpha} --dataset_name=${dataset_name} --num_clients=${num_clients} --fraction=${fraction} --rounds=${rounds} > ./output/${filename}.out 2>&1 &

device=2
tm_criterion=CrossEntropyLoss;method=fedprox;mu=1
filename="${seed}_${dataset_name}_${alpha}_${method}(mu:${mu})_${tm_criterion}_${sampling_type}"
python main.py --mp=${is_mp} --method=${method} --sampling_type=${sampling_type} --tm_mu=${mu} --tm_criterion=${tm_criterion} --tm_local_bs=${tm_local_bs} --device=${device} --alpha=${alpha} --dataset_name=${dataset_name} --num_clients=${num_clients} --fraction=${fraction} --rounds=${rounds} > ./output/${filename}.out 2>&1 &

tm_criterion=FocalLoss;method=fedprox;mu=1
filename="${seed}_${dataset_name}_${alpha}_${method}(mu:${mu})_${tm_criterion}_${sampling_type}"
python main.py --mp=${is_mp} --method=${method} --sampling_type=${sampling_type} --tm_mu=${mu} --tm_criterion=${tm_criterion} --tm_local_bs=${tm_local_bs} --device=${device} --alpha=${alpha} --dataset_name=${dataset_name} --num_clients=${num_clients} --fraction=${fraction} --rounds=${rounds} > ./output/${filename}.out 2>&1 &

tm_criterion=Ratio_Cross_Entropy;method=fedprox;mu=1
filename="${seed}_${dataset_name}_${alpha}_${method}(mu:${mu})_${tm_criterion}_${sampling_type}"
python main.py --mp=${is_mp} --method=${method} --sampling_type=${sampling_type} --tm_mu=${mu} --tm_criterion=${tm_criterion} --tm_local_bs=${tm_local_bs} --device=${device} --alpha=${alpha} --dataset_name=${dataset_name} --num_clients=${num_clients} --fraction=${fraction} --rounds=${rounds} > ./output/${filename}.out 2>&1 &

#################### FedDyn ########################
device=3
tm_criterion=CrossEntropyLoss;method=feddyn;mu=None
filename="${seed}_${dataset_name}_${alpha}_${method}(mu:${mu})_${tm_criterion}_${sampling_type}"
python main.py --mp=${is_mp} --method=${method} --sampling_type=${sampling_type} --tm_mu=${mu} --tm_criterion=${tm_criterion} --tm_local_bs=${tm_local_bs} --device=${device} --alpha=${alpha} --dataset_name=${dataset_name} --num_clients=${num_clients} --fraction=${fraction} --rounds=${rounds} > ./output/${filename}.out 2>&1 &

tm_criterion=FocalLoss;method=feddyn;mu=None
filename="${seed}_${dataset_name}_${alpha}_${method}(mu:${mu})_${tm_criterion}_${sampling_type}"
python main.py --mp=${is_mp} --method=${method} --sampling_type=${sampling_type} --tm_mu=${mu} --tm_criterion=${tm_criterion} --tm_local_bs=${tm_local_bs} --device=${device} --alpha=${alpha} --dataset_name=${dataset_name} --num_clients=${num_clients} --fraction=${fraction} --rounds=${rounds} > ./output/${filename}.out 2>&1 &

tm_criterion=Ratio_Cross_Entropy;method=feddyn;mu=None
filename="${seed}_${dataset_name}_${alpha}_${method}(mu:${mu})_${tm_criterion}_${sampling_type}"
python main.py --mp=${is_mp} --method=${method} --sampling_type=${sampling_type} --tm_mu=${mu} --tm_criterion=${tm_criterion} --tm_local_bs=${tm_local_bs} --device=${device} --alpha=${alpha} --dataset_name=${dataset_name} --num_clients=${num_clients} --fraction=${fraction} --rounds=${rounds} > ./output/${filename}.out 2>&1 &


###########################################################

# EMNIST
dataset_name='emnist';tm_local_bs=100

#################### FedProx ########################
device=4
tm_criterion=CrossEntropyLoss;method=fedavg;mu=None
filename="${seed}_${dataset_name}_${alpha}_${method}(mu:${mu})_${tm_criterion}_${sampling_type}"
python main.py --mp=${is_mp} --method=${method} --sampling_type=${sampling_type} --tm_mu=${mu} --tm_criterion=${tm_criterion} --tm_local_bs=${tm_local_bs} --device=${device} --alpha=${alpha} --dataset_name=${dataset_name} --num_clients=${num_clients} --fraction=${fraction} --rounds=${rounds} > ./output/${filename}.out 2>&1 &

tm_criterion=FocalLoss;method=fedavg;mu=None
filename="${seed}_${dataset_name}_${alpha}_${method}(mu:${mu})_${tm_criterion}_${sampling_type}"
python main.py --mp=${is_mp} --method=${method} --sampling_type=${sampling_type} --tm_mu=${mu} --tm_criterion=${tm_criterion} --tm_local_bs=${tm_local_bs} --device=${device} --alpha=${alpha} --dataset_name=${dataset_name} --num_clients=${num_clients} --fraction=${fraction} --rounds=${rounds} > ./output/${filename}.out 2>&1 &

tm_criterion=Ratio_Cross_Entropy;method=fedavg;mu=None
filename="${seed}_${dataset_name}_${alpha}_${method}(mu:${mu})_${tm_criterion}_${sampling_type}"
python main.py --mp=${is_mp} --method=${method} --sampling_type=${sampling_type} --tm_mu=${mu} --tm_criterion=${tm_criterion} --tm_local_bs=${tm_local_bs} --device=${device} --alpha=${alpha} --dataset_name=${dataset_name} --num_clients=${num_clients} --fraction=${fraction} --rounds=${rounds} > ./output/${filename}.out 2>&1 &

#################### FedProx ########################
device=5
tm_criterion=CrossEntropyLoss;method=fedprox;mu=0.1
filename="${seed}_${dataset_name}_${alpha}_${method}(mu:${mu})_${tm_criterion}_${sampling_type}"
python main.py --mp=${is_mp} --method=${method} --sampling_type=${sampling_type} --tm_mu=${mu} --tm_criterion=${tm_criterion} --tm_local_bs=${tm_local_bs} --device=${device} --alpha=${alpha} --dataset_name=${dataset_name} --num_clients=${num_clients} --fraction=${fraction} --rounds=${rounds} > ./output/${filename}.out 2>&1 &

tm_criterion=FocalLoss;method=fedprox;mu=0.1
filename="${seed}_${dataset_name}_${alpha}_${method}(mu:${mu})_${tm_criterion}_${sampling_type}"
python main.py --mp=${is_mp} --method=${method} --sampling_type=${sampling_type} --tm_mu=${mu} --tm_criterion=${tm_criterion} --tm_local_bs=${tm_local_bs} --device=${device} --alpha=${alpha} --dataset_name=${dataset_name} --num_clients=${num_clients} --fraction=${fraction} --rounds=${rounds} > ./output/${filename}.out 2>&1 &

tm_criterion=Ratio_Cross_Entropy;method=fedprox;mu=0.1
filename="${seed}_${dataset_name}_${alpha}_${method}(mu:${mu})_${tm_criterion}_${sampling_type}"
python main.py --mp=${is_mp} --method=${method} --sampling_type=${sampling_type} --tm_mu=${mu} --tm_criterion=${tm_criterion} --tm_local_bs=${tm_local_bs} --device=${device} --alpha=${alpha} --dataset_name=${dataset_name} --num_clients=${num_clients} --fraction=${fraction} --rounds=${rounds} > ./output/${filename}.out 2>&1 &

device=6
# tm_criterion=CrossEntropyLoss;method=fedprox;mu=1
# filename="${seed}_${dataset_name}_${alpha}_${method}(mu:${mu})_${tm_criterion}_${sampling_type}"
# python main.py --mp=${is_mp} --method=${method} --sampling_type=${sampling_type} --tm_mu=${mu} --tm_criterion=${tm_criterion} --tm_local_bs=${tm_local_bs} --device=${device} --alpha=${alpha} --dataset_name=${dataset_name} --num_clients=${num_clients} --fraction=${fraction} --rounds=${rounds} > ./output/${filename}.out 2>&1 &

# tm_criterion=FocalLoss;method=fedprox;mu=1
# filename="${seed}_${dataset_name}_${alpha}_${method}(mu:${mu})_${tm_criterion}_${sampling_type}"
# python main.py --mp=${is_mp} --method=${method} --sampling_type=${sampling_type} --tm_mu=${mu} --tm_criterion=${tm_criterion} --tm_local_bs=${tm_local_bs} --device=${device} --alpha=${alpha} --dataset_name=${dataset_name} --num_clients=${num_clients} --fraction=${fraction} --rounds=${rounds} > ./output/${filename}.out 2>&1 &

# tm_criterion=Ratio_Cross_Entropy;method=fedprox;mu=1
# filename="${seed}_${dataset_name}_${alpha}_${method}(mu:${mu})_${tm_criterion}_${sampling_type}"
# python main.py --mp=${is_mp} --method=${method} --sampling_type=${sampling_type} --tm_mu=${mu} --tm_criterion=${tm_criterion} --tm_local_bs=${tm_local_bs} --device=${device} --alpha=${alpha} --dataset_name=${dataset_name} --num_clients=${num_clients} --fraction=${fraction} --rounds=${rounds} > ./output/${filename}.out 2>&1 &

#################### FedDyn ########################
device=6
# device=7
tm_criterion=CrossEntropyLoss;method=feddyn;mu=None
filename="${seed}_${dataset_name}_${alpha}_${method}(mu:${mu})_${tm_criterion}_${sampling_type}"
python main.py --mp=${is_mp} --method=${method} --sampling_type=${sampling_type} --tm_mu=${mu} --tm_criterion=${tm_criterion} --tm_local_bs=${tm_local_bs} --device=${device} --alpha=${alpha} --dataset_name=${dataset_name} --num_clients=${num_clients} --fraction=${fraction} --rounds=${rounds} > ./output/${filename}.out 2>&1 &

tm_criterion=FocalLoss;method=feddyn;mu=None
filename="${seed}_${dataset_name}_${alpha}_${method}(mu:${mu})_${tm_criterion}_${sampling_type}"
python main.py --mp=${is_mp} --method=${method} --sampling_type=${sampling_type} --tm_mu=${mu} --tm_criterion=${tm_criterion} --tm_local_bs=${tm_local_bs} --device=${device} --alpha=${alpha} --dataset_name=${dataset_name} --num_clients=${num_clients} --fraction=${fraction} --rounds=${rounds} > ./output/${filename}.out 2>&1 &

tm_criterion=Ratio_Cross_Entropy;method=feddyn;mu=None
filename="${seed}_${dataset_name}_${alpha}_${method}(mu:${mu})_${tm_criterion}_${sampling_type}"
python main.py --mp=${is_mp} --method=${method} --sampling_type=${sampling_type} --tm_mu=${mu} --tm_criterion=${tm_criterion} --tm_local_bs=${tm_local_bs} --device=${device} --alpha=${alpha} --dataset_name=${dataset_name} --num_clients=${num_clients} --fraction=${fraction} --rounds=${rounds} > ./output/${filename}.out 2>&1 &

