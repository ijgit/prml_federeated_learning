device=2
num_clients=100
fraction=0.1
is_mp='True'


# CIFAR-10
dataset_name='cifar10'
tm_local_bs=10
rounds=300 
alpha=10
tm_criterion=Ratio_Cross_Entropy

python3 main.py --mp=True --tm_criterion=${tm_criterion} --tm_local_bs=${tm_local_bs} --device=${device} --alpha=${alpha} --dataset_name=${dataset_name} --num_clients=${num_clients} --fraction=${fraction} --rounds=${rounds}


# EMNIST
# dataset_name='emnist'
# tm_local_bs=100
# rounds=300
# alpha=1

# for tm_criterion in FocalLoss #'CrossEntropyLoss' #'FocalLoss' 
# do
#   python3 main.py --device=${device} --tm_criterion=${tm_criterion} --alpha=${alpha} --tm_local_bs=${tm_local_bs} --dataset_name=${dataset_name} --num_clients=${num_clients} --fraction=${fraction} --rounds=${rounds}
# done

# for alpha in 0.1 0.5 1
# do
#    
# done


# # CIFAR-100
# dataset_name='cifar100'
# rounds=300
# tm_lr=0.03

# for alpha in 0.1 0.5 1
# do
#     python3 main.py --device=${device} --alpha=${alpha} --tm_lr=${tm_lr} --dataset_name=${dataset_name} --num_clients=${num_clients} --fraction=${fraction} --rounds=${rounds}
# done

