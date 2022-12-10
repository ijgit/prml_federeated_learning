from random import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

from tqdm import tqdm

import cv2 as cv
import imageio
import random
import math
import pandas as pd

import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, TensorDataset, DataLoader
import os

   
# split_noniid(train_dataset., train_dataset.targets)
def split_noniid(train_idcs, train_labels, alpha, n_clients, seed):
    '''
    Splits a list of data indices with corresponding labels
    into subsets according to a dirichlet distribution with parameter
    alpha
    '''
    # print(train_idcs)
    train_idcs = np.array(train_idcs)
    train_labels = np.array(train_labels)

    n_classes = max(train_labels)+1
    label_distribution = np.random.default_rng(seed=seed).dirichlet(alpha=np.repeat(alpha,n_clients), size=n_classes)
    # label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)

    class_idcs = [np.argwhere(train_labels[train_idcs]==y).flatten() 
           for y in range(n_classes)]

    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [train_idcs[np.concatenate(idcs)] for idcs in client_idcs]
  

    dist = np.zeros((n_clients, n_classes))
    for i in range(0, n_clients):
        for j in client_idcs[i]:
            dist[i, train_labels[j]] += 1
    return client_idcs, dist




def partition_dataset(targets, class_to_idx, num_client, alpha, seed):
    num_class = len(class_to_idx.values())

    num_of_data = len(targets)
    sorted_indecies = torch.argsort(torch.Tensor(targets))
    sorted_labels = torch.Tensor(targets)[sorted_indecies]

    # counts the number of labels corresponding to the index.
    num_of_class_item = {}
    for i in class_to_idx.values():
        num_of_class_item[i] = len(sorted_labels[sorted_labels==i])

    # store the index corresponding to each label
    sorted_class_indecies = {}

    init_idx = 0
    for idx in num_of_class_item.keys():
        sorted_class_indecies[idx] = sorted_indecies[init_idx:init_idx+num_of_class_item[idx]]
        init_idx += num_of_class_item[idx]
        
        
    ''' split dataset into clients via label'''
    # init client_data_idx
    client_data_indecies  = {}
    count_client_data = np.zeros((num_client, num_class))
    for i in range(0, num_client):
        client_data_indecies[i] = None

    dist = np.random.default_rng(seed=seed).dirichlet(alpha=np.repeat(alpha,num_client), size=num_class)

    for class_i, class_dist in zip(range(0, num_class), dist):
        indecies = sorted_class_indecies[class_i]
        num_each_class = len(indecies)
        init_idx = 0


        for client_i in range(0, num_client):
            client_data_idx = init_idx + round(class_dist[client_i]*len(indecies))
            
            if init_idx >= num_each_class:
                break

            if client_data_idx > num_each_class:
                client_data_idx = num_each_class
                
            # if client_i == num_client-1 and client_data_idx > num_each_class:
            #    client_data_idx = num_each_class

            count_client_data[client_i][class_i] = client_data_idx-init_idx

            if client_data_indecies[client_i] == None:
                client_data_indecies[client_i] = indecies[init_idx: client_data_idx]
            else:
                client_data_indecies[client_i] = torch.cat([client_data_indecies[client_i], indecies[init_idx: client_data_idx]], dim=0)
            init_idx = client_data_idx

    return client_data_indecies, count_client_data, dist.T


class ClientDataset(Dataset):
    """TensorDataset with support of transforms."""
    def __init__(self, data, targets, class_to_idx, sampling_type, seed, transform=None):
        self.s_types = ["smote", "r_over", "r_under", "augment"]
        self.seed = seed

        # assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.data = data
        self.targets = targets
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.sampling_type = sampling_type

        print(f"sampling_type: {self.sampling_type}")
        if self.sampling_type in self.s_types:
            self.data, self.targets = self.sampling()
        else :
            print(f"sampling types : {self.s_types}")

    def __getitem__(self, index):
        if type(self.data[index]) == str:
            data = torch.Tensor(imageio.imread(self.data[index], as_gray=False, pilmode="RGB"))
        else:
            data = self.data[index]
        targets = self.targets[index]
        if self.transform:
            data = self.transform(data)
        return data, targets
    
    def __data__(self):
        return self.data
    
    def __targets__(self):
        return self.targets

    def __class_to_idx__(self):
        return self.class_to_idx

    def __len__(self):
        return self.data.shape[0]
        
        #len(self.data) if type(self.data) == list else self.data.size(0)

    def sampling(self):
        _X = self.data
        _y = self.targets.numpy().reshape(-1,1)

        # set resampler for each sampling type
        if self.sampling_type == "smote":
            from imblearn.over_sampling import SMOTE
            reshaped_X_train = _X.reshape(_X.shape[0], -1)
            resampler = SMOTE(random_state=self.seed)
        elif self.sampling_type == "r_under":
            from imblearn.under_sampling import RandomUnderSampler
            reshaped_X_train = _X.reshape(_X.shape[0], -1)
            resampler = RandomUnderSampler(random_state=self.seed)

        elif self.sampling_type == "r_over":
            from imblearn.over_sampling import RandomOverSampler
            reshaped_X_train = _X.reshape(_X.shape[0], -1)
            resampler = RandomOverSampler(random_state=self.seed)

        elif self.sampling_type == "augment":
            reshaped_X_train = _X
            resampler = MyAugmentator(random_state=self.seed)

        else :
            return self.data, self.targets # do nothing

        print(f">>>>> Resampling Started : {self.sampling_type}")
        if len(set(_y.flatten().tolist())) == 1:
            return self.data, self.targets # do nothing

        print(f"  orig.shape; {_X.shape}, {_y.shape}")
        X_resampled, y_resampled = resampler.fit_resample(reshaped_X_train, _y)
        origin_shape = list(self.data.shape)
        origin_shape[0] = -1
        _X = X_resampled.reshape(origin_shape)
        _y = torch.Tensor(y_resampled)
        print(f"  resampled.shape; {_X.shape}, {_y.shape}")
        print(f">>>>> Resampling Completed : {self.sampling_type}")
        return _X, _y

class MyAugmentator():
    def __init__(self, random_state):
        self.random_state = random_state
        np.random.seed(self.random_state)

        degrees=[-15,15]
        scale=(0.9, 1.1)
        translate=(0.01, 0.01)
        interpolation=transforms.InterpolationMode.NEAREST

        MyRandomAffine = transforms.RandomAffine(degrees=degrees, 
                                                    scale = scale,
                                                    translate = translate,
                                                    interpolation = interpolation)

        self.my_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    MyRandomAffine,
                    transforms.ToTensor(),
        ])

    def fit_resample(self, reshaped_X_train, _y):

        K, V = np.unique(_y, return_counts=True)
        zip_orig = zip(K, V)
        
        # count each class count
        if len(set(_y.flatten().tolist())) == 1:
            return reshaped_X_train, _y

        max_value = max(V)
        dic_gen_count = dict()

        count_new_gen = 0
        for k, v in zip_orig:
            dic_gen_count[k] = max_value - v
            count_new_gen = count_new_gen + dic_gen_count[k]
        
        print(f"max_value: {max_value}, #{count_new_gen} will be generated.")

        x_origin_shape = list(reshaped_X_train.shape)
        x_new_shape = x_origin_shape.copy()
        x_new_shape[0] = x_origin_shape[0] + count_new_gen
        resampled_x = np.zeros((tuple(x_new_shape)))
        resampled_x[:x_origin_shape[0]] = reshaped_X_train

        y_origin_shape = list(_y.shape)
        y_new_shape = y_origin_shape.copy()
        y_new_shape[0] = y_new_shape[0] + count_new_gen
        resampled_y = np.zeros((tuple(y_new_shape)))
        resampled_y[:y_origin_shape[0]] = _y

        created_count = 0
        for k, v in dic_gen_count.items():
            _class_no = k
            _left_count = v

            # find index where _y == _class_no
            index_list = list(np.where(_y == _class_no)[0])

            has_channel = False
            if len(list(reshaped_X_train[index_list[0]].shape)) > 2:
                has_channel = True

            while(_left_count > 0) :
                rand_index = index_list[np.random.randint(0, len(index_list))]

                will_be_augmentated = reshaped_X_train[rand_index]
                # print(f"will_be_augmentated.shape: {will_be_augmentated.shape}")

                if has_channel:
                    augmentated = self.my_transform(will_be_augmentated).permute(1, 2, 0).numpy() # change C, H, W to H, W, C 
                else:
                    augmentated = self.my_transform(will_be_augmentated).numpy().reshape(will_be_augmentated.shape) # in EMNIST case
                
                # print(f"augmentated.shape: {augmentated.shape}")
                resampled_x[x_origin_shape[0] + created_count] = augmentated
                resampled_y[x_origin_shape[0] + created_count] = _y[rand_index]

                created_count = created_count + 1
                _left_count = _left_count -1

        return resampled_x, resampled_y.flatten()


class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms."""
    def __init__(self, data, targets, class_to_idx, transform=None):

        # assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.data = data
        self.targets = targets
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __getitem__(self, index):
        if type(self.data[index]) == str:
            data = torch.Tensor(imageio.imread(self.data[index], as_gray=False, pilmode="RGB"))
        else:
            data = self.data[index]
        targets = self.targets[index]

        if self.transform:
            data = self.transform(data)
        return data, targets

    def __data__(self):
        return self.data
    
    def __targets__(self):
        return self.targets
   
    def __class_to_idx__(self):
        return self.class_to_idx

    def __len__(self):
        return self.data.shape[0] #len(self.data)# if type(self.data) == list else self.data.size(0)



def partition_with_dirichlet_distribution(dataset_name, data, targets, class_to_idx, num_client, alpha, transform, seed, sampling_type):

    idcs = [i for i in range(0, len(data.data))]
    client_data_indecies, client_dist = split_noniid(idcs, targets, alpha, num_client, seed)

    def softmax(arr):
        return torch.Tensor(arr/np.sum(arr))

    splited_client_dataset = [
        ClientDataset(
            data = data[client_data_indecies[idx]], # client_datas[idx] if type(data[0]) == str else torch.Tensor(client_datas[idx]),
            targets = torch.Tensor(targets)[client_data_indecies[idx]],
            class_to_idx = class_to_idx, sampling_type=sampling_type, seed=seed,
            transform=transform
        )
        for idx in range(0, num_client)
    ]
    return splited_client_dataset

    
def prepare_dataset(seed, dataset_name, num_client, alpha, sampling_type = None):
    dataset_name = dataset_name.upper()

    if hasattr(torchvision.datasets, dataset_name):
        if dataset_name == "MNIST" or dataset_name == "EMNIST":
            transform = transforms.Compose(
                [
                    transforms.ToTensor()
                ]
            )
        
        elif dataset_name == "CIFAR10" or dataset_name == "CIFAR100":
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
            )

        if dataset_name in ["EMNIST"]:
            train_dataset = torchvision.datasets.__dict__[dataset_name](root='./data', train=True, split="byclass",
                                                download=True, transform=transform)
            test_dataset = torchvision.datasets.__dict__[dataset_name](root='./data', train=False, split="byclass",
                                                download=True, transform=transform)
        else:
            train_dataset = torchvision.datasets.__dict__[dataset_name](root='./data', train=True,
                                                download=True, transform=transform)
            test_dataset = torchvision.datasets.__dict__[dataset_name](root='./data', train=False,
                                                download=True, transform=transform)

        if "ndarray" not in str(type(train_dataset.data)):
            train_dataset.data = np.asarray(train_dataset.data)
            test_dataset.data = np.asarray(test_dataset.data)
        if "list" not in str(type(train_dataset.targets)):
            train_dataset.targets = train_dataset.targets.tolist()
            test_dataset.targets = test_dataset.targets.tolist()
        

        # split test dataset into validation set and test set
        # test_dataset_indecies = gen_target_distribution_data(test_dataset.targets, test_dataset.class_to_idx, target_dist)
        test_dataset = CustomTensorDataset(
            data = test_dataset.data,
            targets = torch.Tensor(test_dataset.targets),
            class_to_idx = test_dataset.class_to_idx,
            transform=transform
        )

        partitioned_train_set = partition_with_dirichlet_distribution(dataset_name, train_dataset.data, train_dataset.targets, train_dataset.class_to_idx, num_client, alpha, transform, seed, sampling_type)
        
    return partitioned_train_set, test_dataset