
#!/bin/bash

######################################################
# alpha 0.1  withour sampling method
######################################################

# Fix Setting
is_mp='False';num_clients=100;fraction=0.1
rounds=300;seed=42;sampling_type=r_under

# CIFAR-10
dataset_name='cifar10';tm_local_bs=10
div1=0;div2=1;div3=2;div4=3

alpha=0.1; device=${div1}
tm_criterion=Ratio_Cross_Entropy;method=fedavg;mu=None
filename="${seed}_${dataset_name}_${alpha}_${method}(mu:${mu})_${tm_criterion}_${sampling_type}"
python main.py --mp=${is_mp} --method=${method} --sampling_type=${sampling_type} --tm_mu=${mu} --tm_criterion=${tm_criterion} --tm_local_bs=${tm_local_bs} --device=${device} --alpha=${alpha} --dataset_name=${dataset_name} --num_clients=${num_clients} --fraction=${fraction} --rounds=${rounds} > ./output/${filename}.out 2>&1 &

alpha=1; device=${div2}
tm_criterion=Ratio_Cross_Entropy;method=fedprox;mu=0.0001
filename="${seed}_${dataset_name}_${alpha}_${method}(mu:${mu})_${tm_criterion}_${sampling_type}"
python main.py --mp=${is_mp} --method=${method} --sampling_type=${sampling_type} --tm_mu=${mu} --tm_criterion=${tm_criterion} --tm_local_bs=${tm_local_bs} --device=${device} --alpha=${alpha} --dataset_name=${dataset_name} --num_clients=${num_clients} --fraction=${fraction} --rounds=${rounds} > ./output/${filename}.out 2>&1 &

alpha=5; device=${div3}
tm_criterion=Ratio_Cross_Entropy;method=feddyn;mu=None
filename="${seed}_${dataset_name}_${alpha}_${method}(mu:${mu})_${tm_criterion}_${sampling_type}"
python main.py --mp=${is_mp} --method=${method} --sampling_type=${sampling_type} --tm_mu=${mu} --tm_criterion=${tm_criterion} --tm_local_bs=${tm_local_bs} --device=${device} --alpha=${alpha} --dataset_name=${dataset_name} --num_clients=${num_clients} --fraction=${fraction} --rounds=${rounds} > ./output/${filename}.out 2>&1 &

alpha=10; device=${div4}
tm_criterion=Ratio_Cross_Entropy;method=feddyn;mu=None
filename="${seed}_${dataset_name}_${alpha}_${method}(mu:${mu})_${tm_criterion}_${sampling_type}"
python main.py --mp=${is_mp} --method=${method} --sampling_type=${sampling_type} --tm_mu=${mu} --tm_criterion=${tm_criterion} --tm_local_bs=${tm_local_bs} --device=${device} --alpha=${alpha} --dataset_name=${dataset_name} --num_clients=${num_clients} --fraction=${fraction} --rounds=${rounds} > ./output/${filename}.out 2>&1 &

# wait;echo "done"