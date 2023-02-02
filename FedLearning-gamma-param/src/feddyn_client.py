import gc
from multiprocessing import reduction
import os
import pickle
import logging
from threading import local
import copy 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torch.utils.data import DataLoader
from .optimizer import *
from .criterion import *

from .sampling import ClientDataset

logger = logging.getLogger(__name__)

def model_parameter_vector(model):
    param = [p.view(-1) for p in model.parameters()]
    return torch.cat(param, dim=0)

class Client(object):
    """Class for client object having its own (private) data and resources to train a model.

    Participating client has its own dataset which are usually non-IID compared to other clients.
    Each client only communicates with the center server with its trained parameters or globally aggregated parameters.

    Attributes:
        id: Integer indicating client's id.
        data: torch.utils.data.Dataset instance containing local data.
        device: Training machine indicator (e.g. "cpu", "cuda").
        __model: torch.nn instance as a local model.
    """
    def __init__(self, client_id, local_data, device, log_path):
        """Client object is initiated by the center server."""
        self.alpha=1e-2
        self.id = client_id
        self.data = local_data
        self.origClientDataset = local_data
        self.num_class = len(self.data.class_to_idx.values())
        self.log_path = log_path
        self.device = device
        self.round = 0
        self.__t_model = None

    @property
    def t_model(self):
        """Local model getter for parameter aggregation."""
        return self.__t_model

    @t_model.setter
    def t_model(self, t_model):
        """Local model setter for passing globally aggregated model parameters."""
        self.__t_model = t_model
        
        if self.round == 0:
            # init
            self.global_model_vector = None
            old_grad = copy.deepcopy(self.t_model)
            old_grad = model_parameter_vector(old_grad)
            self.old_grad = torch.zeros_like(old_grad)
        else:
            for new_param, old_param in zip(self.t_model.parameters(), self.t_model.parameters()):
                old_param.data = new_param.data.clone()
            self.global_model_vector = model_parameter_vector(self.t_model).detach().clone()

    def __len__(self):
        """Return a total size of the client's local data."""
        return len(self.data)

    def update_undersampling(self):
        print(f"run undersampling on client #{self.id}")
        new_seed = np.random.randint(np.iinfo(np.int32).max)

        newly_undersampling = ClientDataset(
            data = self.origClientDataset.data,
            targets = self.origClientDataset.targets,
            class_to_idx = self.origClientDataset.class_to_idx, 
            sampling_type="r_under_client", 
            seed=new_seed,
            transform=self.origClientDataset.transform
        )

        self.data = newly_undersampling
        self.dataloader = DataLoader(self.data, batch_size=self.tm_local_bs, shuffle=True)

    def setup(self, tm_config):
        """Set up common configuration of each client; called by center server."""
        self.tm_local_bs = tm_config['local_bs']
        self.tm_local_ep = tm_config['local_ep']
        self.tm_criterion = tm_config['criterion']
        self.tm_optimizer = tm_config['optimizer']
        self.tm_optim_config = {'lr': tm_config['lr'], 'momentum': tm_config['momentum'], 'mu': tm_config['mu']}

        self.dataloader = DataLoader(self.data, batch_size=self.tm_local_bs, shuffle=True)

    def task_model_update(self):
        '''update target local model using local dataset'''
        self.t_model.train()
        self.t_model.to(self.device)

        if self.origClientDataset.sampling_type == "r_under":
            self.update_undersampling()

        if self.tm_optimizer == "SGD":
            optimizer = torch.optim.__dict__[self.tm_optimizer](
                self.t_model.parameters(), lr=self.tm_optim_config['lr'], momentum=self.tm_optim_config['momentum']
            )
        else:
            pass
        # optimizer = PerturbedGradientDescent(
            # self.t_model.parameters(), lr=self.tm_optim_config['lr'], mu=self.tm_optim_config['mu'])
        
        for e in range(self.tm_local_ep):
            for data, labels in self.dataloader:
                data, labels = data.float().to(self.device), labels.long().to(self.device)
  
                optimizer.zero_grad()
                preds = self.t_model(data)

                if self.tm_criterion == 'CrossEntropyLoss':
                    loss = torch.nn.__dict__[self.tm_criterion]()(preds, labels)
                    
                elif self.tm_criterion == 'FocalLoss':
                    loss_func = FocalLoss(size_average=True)
                    loss = loss_func(preds, labels)
                    
                elif self.tm_criterion == 'Ratio_Cross_Entropy':
                    loss_func = Ratio_Cross_Entropy(device=self.device, class_num=self.num_class, size_average=True)
                    loss = loss_func(preds, labels)


                if self.global_model_vector != None:
                    self.old_grad = self.old_grad.to(self.device)
                    self.global_model_vector = self.global_model_vector.to(self.device)
                    
                    v1 = model_parameter_vector(self.t_model)
                    loss += self.alpha/2 * torch.norm(v1 - self.global_model_vector, 2)
                    loss -= torch.dot(v1, self.old_grad)
                
                loss.backward()
                optimizer.step() 

            if self.device != 'cpu': torch.cuda.empty_cache()
        
        if self.global_model_vector != None:
            v1 = model_parameter_vector(self.t_model).detach()
            self.old_grad = self.old_grad - self.alpha * (v1 - self.global_model_vector)

        self.t_model.to("cpu")

    def client_update(self, round):
        self.round = round
        self.task_model_update()

    def task_model_evaluate(self):
        self.t_model.eval()
        self.t_model.to(self.device)

        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, labels in self.dataloader:
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                preds = self.t_model(data)

                if self.tm_criterion == 'CrossEntropyLoss':
                    loss = torch.nn.__dict__[self.tm_criterion]()(preds, labels)
                    
                elif self.tm_criterion == 'FocalLoss':
                    loss_func = FocalLoss(size_average=True)
                    loss = loss_func(preds, labels)
                    
                elif self.tm_criterion == 'Ratio_Cross_Entropy':
                    loss_func = Ratio_Cross_Entropy(device=self.device, class_num=self.num_class, size_average=True)
                    loss = loss_func(preds, labels)
                    
                test_loss += loss.item()
                
                predicted = preds.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()

                if self.device != 'cpu': torch.cuda.empty_cache()
        self.t_model.to("cpu")

        test_loss = test_loss / len(self.dataloader)
        test_accuracy = correct / len(self.data)

        message = f"\t[Client {str(self.id).zfill(4)}] ...finished evaluation!\
            \n\t=> Test loss: {test_loss:.4f}\
            \n\t=> Test accuracy: {100. * test_accuracy:.2f}%\n"
        print(message, flush=True); logging.info(message)
        del message; gc.collect()

        return test_loss, test_accuracy


    def client_evaluate(self):
        """Evaluate local model using local dataset (same as training set for convenience)."""
        task_loss, task_acc = self.task_model_evaluate()

        return task_loss, task_acc
