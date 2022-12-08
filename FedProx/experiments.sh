device=1
num_clients=100
fraction=0.1
is_mp='True'


# CIFAR-10
# dataset_name='cifar10'
# tm_local_bs=10
# rounds=100 
# alpha=1
# python3 main.py --mp=True --tm_local_bs=${tm_local_bs} --device=${device} --alpha=${alpha} --dataset_name=${dataset_name} --num_clients=${num_clients} --fraction=${fraction} --rounds=${rounds}


# EMNIST
dataset_name='emnist'
tm_local_bs=100
rounds=300
alpha=1

for mu in 0 0.1 0.5
do
    python3 main.py --mp=True --tm_local_bs=${tm_local_bs} --device=${device} --alpha=${alpha} --dataset_name=${dataset_name} --num_clients=${num_clients} --fraction=${fraction} --rounds=${rounds}
done


# CIFAR-100
# dataset_name='cifar100'
# rounds=300
# tm_lr=0.03
# alpha=1
# python3 main.py --mp=True --tm_local_bs=${tm_local_bs} --device=${device} --alpha=${alpha} --dataset_name=${dataset_name} --num_clients=${num_clients} --fraction=${fraction} --rounds=${rounds}


