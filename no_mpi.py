import os
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import datasets, models
import torch.optim as optim
import logging
from training_utils import *
from torch.utils.tensorboard import SummaryWriter


# init parameters
parser = argparse.ArgumentParser(description='Distributed Client')
parser.add_argument('--dataset_type', type=str, default='CIFAR10')
parser.add_argument('--model_type', type=str, default='AlexNet')
parser.add_argument('--worker_num', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--data_pattern', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--decay_rate', type=float, default=0.993)
parser.add_argument('--min_lr', type=float, default=0.005)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--momentum', type=float, default=-1)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--data_path', type=str, default='./data')
parser.add_argument('--device', type=str, default='cpu', help='The device to run the program')
parser.add_argument('--expname', type=str, default='MergeSFL')

args = parser.parse_args()
device = torch.device(args.device)

def non_iid_partition(ratio, train_class_num, worker_num):
    partition_sizes = np.ones((train_class_num, worker_num)) * ((1 - ratio) / (worker_num - 1))

    for i in range(train_class_num):
        partition_sizes[i][i % worker_num] = ratio

    return partition_sizes

def dirichlet_partition(dataset_type: str, alpha: float, worker_num: int, nclasses: int):
    partition_sizes = []
    filepath = './data_partition/%s-part_dir%.1f.npy' % (dataset_type, alpha)
    if os.path.exists(filepath):
        partition_sizes = np.load(filepath)
    else:
        for _ in range(nclasses):
            partition_sizes.append(np.random.dirichlet([alpha] * worker_num))
        partition_sizes = np.array(partition_sizes)

    return partition_sizes

def partition_data(dataset_type, data_pattern, worker_num=10):
    train_dataset, test_dataset = datasets.load_datasets(dataset_type)
    labels = None
    if dataset_type == "CIFAR10" or dataset_type == "FashionMNIST":
        train_class_num = 10

    if data_pattern == 0:
        partition_sizes = np.ones((train_class_num, worker_num)) * (1.0 / worker_num)
    elif data_pattern == 1:
        non_iid_ratio = 0.2
        partition_sizes = non_iid_partition(non_iid_ratio, train_class_num, worker_num)
    elif data_pattern == 2:
        non_iid_ratio = 0.4
        partition_sizes = non_iid_partition(non_iid_ratio, train_class_num, worker_num)
    elif data_pattern == 3:
        non_iid_ratio = 0.6
        partition_sizes = non_iid_partition(non_iid_ratio, train_class_num, worker_num)
    elif data_pattern == 4:
        non_iid_ratio = 0.8
        partition_sizes = non_iid_partition(non_iid_ratio, train_class_num, worker_num)
    elif data_pattern == 5:
        partition_sizes = dirichlet_partition(dataset_type, 1.0, worker_num, train_class_num)
    elif data_pattern == 6:
        partition_sizes = dirichlet_partition(dataset_type, 0.5, worker_num, train_class_num)
    elif data_pattern == 7:
        partition_sizes = dirichlet_partition(dataset_type, 0.1, worker_num, train_class_num)
    elif data_pattern == 8:
        partition_sizes = dirichlet_partition(dataset_type, 0.01, worker_num, train_class_num)

    train_data_partition = datasets.LabelwisePartitioner(train_dataset, partition_sizes=partition_sizes, class_num=train_class_num, labels=labels)
    return train_dataset, test_dataset, train_data_partition, labels

def main():
    worker_num = args.worker_num

    client_global_model, server_global_model = models.create_model_instance_SL(args.dataset_type, args.model_type, 1)
    nets, _ = models.create_model_instance_SL(args.dataset_type, args.model_type, worker_num)
    
    client_global_model_first = client_global_model[0][0]
    client_global_model_last = client_global_model[0][1]

    # Load the initial state
    global_model_par_first = client_global_model_first.state_dict()
    global_model_par_last = client_global_model_last.state_dict()

    for net_id, net in nets.items():
        net[0].load_state_dict(global_model_par_first)
        net[0].load_state_dict(global_model_par_first)

    # Create model instance
    train_dataset, test_dataset, train_data_partition, labels = partition_data(args.dataset_type, args.data_pattern, worker_num)

    if labels:
        test_loader = datasets.create_dataloaders(test_dataset, batch_size=64, shuffle=False, collate_fn=lambda x: datasets.collate_fn(x, labels))
    else:
        test_loader = datasets.create_dataloaders(test_dataset, batch_size=64, shuffle=False)

    # Clients data loaders
    bsz_list = np.ones(worker_num, dtype=int) * args.batch_size
    client_train_loader = []
    for worker_idx in range(worker_num):
        if labels:
            client_train_loader.append(datasets.create_dataloaders(train_dataset, batch_size=int(bsz_list[worker_idx]), selected_idxs=train_data_partition.use(worker_idx), pin_memory=False, drop_last=True, collate_fn=lambda x: datasets.collate_fn(x, labels)))
        else:
            print(f'for worker {worker_idx}')
            client_train_loader.append(datasets.create_dataloaders(train_dataset, batch_size=int(bsz_list[worker_idx]), selected_idxs=train_data_partition.use(worker_idx), pin_memory=False, drop_last=True))

    epoch_lr = args.lr
    print('Start training')
    for epoch_idx in range(1, 1 + args.epoch):
        print(f'In epoch:{epoch_idx}')
        start_time = time.time()
        # Learning rate adjustment
        if epoch_idx > 1:
            epoch_lr = max((args.decay_rate * epoch_lr, args.min_lr))

        # Define optimizers    
        if args.momentum < 0:
            global_optim = optim.SGD(server_global_model.parameters(), lr=epoch_lr, weight_decay=args.weight_decay)
        else:
            global_optim = optim.SGD(server_global_model.parameters(), lr=epoch_lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)

        client_optimizers_first = []
        client_optimizers_last = []
        for worker_idx in range(worker_num):
            if args.momentum < 0:
                client_optimizers_first.append(optim.SGD(nets[worker_idx][0].parameters(), lr=epoch_lr, weight_decay=args.weight_decay))
                client_optimizers_last.append(optim.SGD(nets[worker_idx][1].parameters(), lr=epoch_lr, weight_decay=args.weight_decay))
            else:
                client_optimizers_first.append(optim.SGD(nets[worker_idx][0].parameters(), lr=epoch_lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay))
                client_optimizers_last.append(optim.SGD(nets[worker_idx][1].parameters(), lr=epoch_lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay))

        server_global_model.train()
        local_steps = 42
        server_global_model.to(device)

        # Training loop - client side
        for iter_idx in range(local_steps):
            sum_bsz = sum([bsz_list[i] for i in range(worker_num)])
            global_optim.zero_grad()
            for worker_idx in range(worker_num):
                inputs, targets = next(client_train_loader[worker_idx])

                inputs, targets = inputs.to(device), targets.to(device) 

                nets[worker_idx][0].to(device)
                nets[worker_idx][1].to(device)

                # forward prop
                clients_smash_data = nets[worker_idx][0](inputs)

                clients_send_data = clients_smash_data.detach()
                clients_send_data.requires_grad_()

                # server side forward propagation
                server_smash_data = server_global_model(clients_send_data)
                server_send_data = server_smash_data.detach()

                server_send_data.requires_grad_()
                output = nets[worker_idx][1](server_send_data)

                # backward propagation
                loss = F.cross_entropy(output, targets.long())

                client_optimizers_last[worker_idx].zero_grad()
                loss.backward()
                client_optimizers_last[worker_idx].step()

                # server backprop
                server_grad = server_send_data.grad
                server_smash_data.backward(server_grad.to(device))

                client_grads = clients_send_data.grad

                client_optimizers_first[worker_idx].zero_grad()
                clients_smash_data.backward(client_grads.to(device))
                client_optimizers_first[worker_idx].step()
            global_optim.step()

        # AGGREGATION
        with torch.no_grad():
            for worker_idx in range(worker_num):
                nets[worker_idx][0].to('cpu')
                nets[worker_idx][1].to('cpu')
                net_para_first = nets[worker_idx][0].cpu().state_dict()
                net_para_last = nets[worker_idx][1].cpu().state_dict()
                
                if worker_idx == 0:
                    for key in net_para_first:
                        global_model_par_first[key] = net_para_first[key]/worker_num
                    for key in net_para_last:
                        global_model_par_last[key] = net_para_last[key]/worker_num
                else:
                    for key in net_para_first:
                        global_model_par_first[key] += net_para_first[key]/worker_num
                    for key in net_para_last:
                        global_model_par_last[key] += net_para_last[key]/worker_num
                
                client_global_model_first.load_state_dict(global_model_par_first)
                client_global_model_last.load_state_dict(global_model_par_last)
        
        server_global_model.to('cpu')
        test_loss, acc = test((client_global_model_first, client_global_model_last), server_global_model, test_loader)
        print("Epoch: {}, accuracy: {}, test_loss: {}".format(epoch_idx, acc, test_loss))

        global_model_par_first = client_global_model_first.state_dict()
        global_model_par_last = client_global_model_last.state_dict()
        for net_id, net in nets.items():
            net[0].load_state_dict(global_model_par_first)
            net[1].load_state_dict(global_model_par_last)



if __name__ == '__main__':
    main()

