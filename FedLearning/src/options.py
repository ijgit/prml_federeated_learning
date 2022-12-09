import argparse
import sys

def parse_none(param):
    print(param)
    if param == "None":
        return None
    else:
        return param

def str_to_bool(param):
    if isinstance(param, bool):
        return param
    if param.lower() in ('true', '1'): 
        return True
    elif param.lower() in ('false', '0'):
        return False
    else:
        raise argparse.argparse.ArgumentTypeError('boolean value expected')

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--method', default='fedavg')

    # federated learning arguments
    parser.add_argument('--rounds', type=int, default=100, help="number of round of training")
    parser.add_argument('--num_clients', type=int, default=100, help='number of client (K)')
    parser.add_argument('--fraction', type=float, default=0.1, help='fraction of client (C)')
    parser.add_argument('--tm_local_ep', type=int, default=1, help='the number of local epochs of task model: E_t')
    parser.add_argument('--tm_local_bs', type=int, default=10, help='batch size of local target model: B_t')

    # target model arguments
    parser.add_argument('--tm_lr', type=float, default=0.01, help='learning rate of task model')
    parser.add_argument('--tm_criterion', default='CrossEntropyLoss', help='criterion of task model')
    parser.add_argument('--tm_optimizer', default='SGD', help='optimizer of task model')
    parser.add_argument('--tm_momentum', type=float, default=0.9, help='momentum of optimizer of task model')
    parser.add_argument('--tm_mu', type=parse_none, default=None, help='mu for FedProx')
    # parser.add_argument('--tm_name', help='task model name')
    # parser.add_argument('--tm_num_classes', type=int, help='number of classes of dataset')

    # distribution model arguments
    # parser.add_argument('--dm_lr', type=float, default=0.001, help='learning rate of task model')
    # parser.add_argument('--dm_criterion', default='MultiLabelSoftMarginLoss', help='criterion of distribution model')
    # parser.add_argument('--dm_optimizer', default='Adam', help='optimizer of distribution model')
    # parser.add_argument('--dm_name', default='distribution_fcn', help='task model name')
    # # parser.add_argument('--dm_input_size', type=int, help='size of input of distribution model')
    # parser.add_argument('--dm_preprocessing', default='none', help='model parameter preprocessing option: none/norm_normalization/min_max')
    # parser.add_argument('--dm_input_type', default='gradient', help='gradient/weight/y_derivative/weighted_average')

    # dataset setting
    parser.add_argument('--dataset_name', help='dataset name')
    parser.add_argument('--alpha',  type=float, default=1, help='alpha of dirichlet distribution')
    parser.add_argument('--target_dist_op', type=int, default=4, help='target distribution select option')

    # system setting
    parser.add_argument('--mp',  type=str_to_bool, default=False, help='multi processing')
    parser.add_argument('--device', default='cuda', help='set specific GPU number of CPU')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--log_dir', default='log', help='Log directory')


    # save directory
    parser.add_argument('--init_round', type=int, default=None, help='init round')

    args = parser.parse_args()
    return args