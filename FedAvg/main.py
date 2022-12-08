from ast import arg
from inspect import stack
import os
import sys
import time
import datetime
import pickle
import threading
import logging
import numpy as np

from src.set_seed import set_seed
from src.options import args_parser

from torch.utils.tensorboard import SummaryWriter

from src.server import Server
from src.utils import launch_tensor_board
from src.sampling import prepare_dataset

def softmax(arr):
    return arr/np.sum(arr)


if __name__ == "__main__":
    time_config = str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
    
    args = args_parser()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    set_seed(args.seed)

    if args.dataset_name.upper() == 'MNIST':
        num_classes = 10
        tm_name = 'mnist_fc'
    elif args.dataset_name.upper() == 'CIFAR10':
        num_classes = 10
        tm_name = 'cifar10_cnn'
    elif args.dataset_name.upper() == 'CIFAR100':
        num_classes = 100
        tm_name = 'cifar100_cnn'
    elif args.dataset_name.upper() == "EMNIST":
        num_classes = 62
        tm_name = 'emnist_cnn'
    elif args.dataset_name.upper() == 'FEMNIST':
        num_classes = 62
        tm_name = 'femnist_cnn'
    elif args.dataset_name.upper() == 'TINYIMGNET':
        num_classes = 200
        tm_name = 'tinyimgnet_cnn'

    log_dir = f'{args.log_dir}/{args.dataset_name}/{str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))}'
    log_path, log_file = log_dir, 'FL_Log.log'
    # log_path = os.path.join(log_path, f'a_{args.alpha}')

    fed_config = {
        'init_round': args.init_round, 'num_rounds': args.rounds, 'num_clients': args.num_clients, 'fraction': args.fraction, 'alpha': args.alpha
    }
    data_config = {
        'name': args.dataset_name, 'num_classes': num_classes, 'alpha': args.alpha, 
    }
    tm_config = {
        'lr': args.tm_lr, 'momentum': args.tm_momentum, 'name': tm_name,
        'criterion': args.tm_criterion, 'optimizer': args.tm_optimizer,
        'local_ep': args.tm_local_ep, 'local_bs': args.tm_local_bs
    }
    system_config = {
        'is_mp': args.mp, 'device': 'cuda', 'seed': args.seed, 'log_dir': log_path, 'time_config': time_config
    }


    # initiate TensorBaord for tracking losses and metrics
    writer = SummaryWriter(log_dir=log_path, filename_suffix="FL")
    # tb_thread = threading.Thread(
    #     target=launch_tensor_board,
    #     args=([log_path, 5252, "0.0.0.0"])
    #     ).start()
    # time.sleep(3.0)

    # set the configuration of global logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(log_path, log_file),
        level=logging.INFO,
        format="[%(levelname)s](%(asctime)s) %(message)s",
        datefmt="%Y/%m/%d/ %I:%M:%S %p")

    logging.info(f"fed_config: {fed_config}")
    logging.info(f"tm_config: {tm_config}")
    logging.info(f"system_config: {system_config}")

    # federated learning
    partitioned_train_set, test_dataset = prepare_dataset(seed=args.seed, dataset_name=args.dataset_name, num_client=args.num_clients,alpha=args.alpha)
    message = "\n[WELCOME] Unfolding configurations...!"
    print(message); logging.info(message)

    central_server = Server(writer, partitioned_train_set, test_dataset,
                            fed_config, data_config, tm_config, system_config)
    central_server.setup()
    central_server.fit()

    with open(os.path.join(log_path, "result.pkl"), "wb") as f:
        pickle.dump(central_server.results, f)

    message = "...done all learning process!\n...exit program!"
    print(message); logging.info(message)
    time.sleep(3); os._exit(0)
  

    # # initialize federated learning 
    # central_server = Server(writer, model_config, global_config, data_config, init_config, fed_config, optim_config)
    # central_server.setup()

    # # do federated learning
    # central_server.fit()

    # # save resulting losses and metrics
    # with open(os.path.join(log_config["log_path"], "result.pkl"), "wb") as f:
    #     pickle.dump(central_server.results, f)
    
    # # bye!
    # message = "...done all learning process!\n...exit program!"
    # print(message); logging.info(message)
    # time.sleep(3); exit()

