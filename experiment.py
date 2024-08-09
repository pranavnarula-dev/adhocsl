import os
import argparse
import time
import numpy as np
import torch
import copy
import torch.nn.functional as F
import datasets, models
import torch.optim as optim
import logging
from training_utils import *
from torch.utils.tensorboard import SummaryWriter
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


#init parameters
parser = argparse.ArgumentParser(description='Distributed Client')
parser.add_argument('--dataset_type', type=str, default='CIFAR10')
parser.add_argument('--model_type', type=str, default='AlexNet')
parser.add_argument('--initial_workers', type=int, default=10)
parser.add_argument('--chosen_worker_num', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=10) 
parser.add_argument('--data_pattern', type=int, default=0)
parser.add_argument('--client_lr', type=float, default=0.1)
parser.add_argument('--server_lr', type=float, default=0.1)
parser.add_argument('--decay_rate', type=float, default=0.993)
parser.add_argument('--min_lr', type=float, default=0.005)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--momentum', type=float, default=-1)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--data_path', type=str, default='./data')
parser.add_argument('--device', type=str, default='cpu', help='The device to run the program') #cpu
parser.add_argument('--expname', type=str, default='MergeSFL')
parser.add_argument('--two_splits', action="store_true", help='do U-Shape')
parser.add_argument('--type_noniid', type=str, default='default')
parser.add_argument('--level', type=int, default=10)
parser.add_argument('--num_servers', type=int, default=1, choices=[1, 2, 3, 4], help='Number of intermediate servers')
parser.add_argument('--selection_strategy', type=str, default='first', choices=['first', 'random'], help='Client selection strategy')

args = parser.parse_args()
device = torch.device(args.device)

def select_clients(strategy, initial_workers, chosen_worker_num):
    if strategy == 'first':
        return list(range(chosen_worker_num))
    elif strategy == 'random':
        return random.sample(range(initial_workers), chosen_worker_num)

def non_iid_partition(ratio, train_class_num, initial_workers):
    partition_sizes = np.ones((train_class_num, initial_workers)) * ((1 - ratio) / (initial_workers-1))

    for i in range(train_class_num):
        partition_sizes[i][i%initial_workers]=ratio

    return partition_sizes


def non_iid_partition_strict(ratio, level, train_class_num, initial_workers):
    partition_sizes = np.zeros((train_class_num, initial_workers))

    for i in range(train_class_num):
        for j in range(level):
            partition_sizes[i][(i+j)%initial_workers]=ratio

    return partition_sizes

def dirichlet_partition(dataset_type: str, alpha: float, initial_workers: int, nclasses: int):
    partition_sizes = []
    filepath = './data_partition/%s-part_dir%.1f.npy' % (dataset_type, alpha)
    if os.path.exists(filepath):
        partition_sizes = np.load(filepath)
    else:
        for _ in range(nclasses):
            partition_sizes.append(np.random.dirichlet([alpha] * initial_workers))
        partition_sizes = np.array(partition_sizes)

    return partition_sizes


def partition_data(dataset_type, data_pattern, initial_workers=10):
    train_dataset, test_dataset = datasets.load_datasets(dataset_type)
    print(f"Total training dataset size: {len(train_dataset)}")
    labels = None
    if dataset_type == "CIFAR10" or dataset_type == "FashionMNIST":
        train_class_num = 10
    elif dataset_type=='UCIHAR':
        train_class_num = 6
    if data_pattern == 0:
        partition_sizes = np.ones((train_class_num, initial_workers)) * (1.0 / initial_workers)
    elif data_pattern == 1:
        non_iid_ratio = 0.2
        partition_sizes = non_iid_partition(non_iid_ratio, train_class_num, initial_workers)
    elif data_pattern == 2:
        non_iid_ratio = 0.4
        partition_sizes = non_iid_partition(non_iid_ratio, train_class_num, initial_workers)
    elif data_pattern == 3:
        non_iid_ratio = 0.6
        partition_sizes = non_iid_partition(non_iid_ratio, train_class_num, initial_workers)
    elif data_pattern == 4:
        non_iid_ratio = 0.8
        partition_sizes = non_iid_partition(non_iid_ratio, train_class_num, initial_workers)
    elif data_pattern in [5, 6, 7, 8]:
        alpha = {5: 1.0, 6: 0.5, 7: 0.1, 8: 0.05}[data_pattern]
        print(f'Dirichlet partition {alpha}')
        partition_sizes = dirichlet_partition(dataset_type, alpha, initial_workers, train_class_num)
        
        # Ensure each client gets at least some data
        min_samples = max(1, len(train_dataset) // (initial_workers * 10))  # At least 1/10 of average
        for class_idx in range(train_class_num):
            total_samples = int(sum(partition_sizes[class_idx]) * len(train_dataset) / train_class_num)
            for worker_idx in range(initial_workers):
                if partition_sizes[class_idx][worker_idx] * total_samples < min_samples:
                    partition_sizes[class_idx][worker_idx] = min_samples / total_samples
            
            # Normalize to ensure the sum is still 1
            partition_sizes[class_idx] /= sum(partition_sizes[class_idx])
    print(partition_sizes)
    train_data_partition = datasets.LabelwisePartitioner(train_dataset, partition_sizes=partition_sizes, class_num=train_class_num, labels=labels)
    # Add this line to create the plot
    #datasets.plot_data_distribution(train_dataset, train_data_partition, dataset_type, data_pattern, initial_workers)
    return train_dataset, test_dataset, train_data_partition, labels

def partition_data_non_iid_strict(dataset_type, data_pattern, initial_workers=10):
    train_dataset, test_dataset = datasets.load_datasets(dataset_type)
    labels = None
    if dataset_type == "CIFAR10" or dataset_type == "FashionMNIST":
        train_class_num = 10
    elif dataset_type=='UCIHAR':
        train_class_num = 6

    
    if data_pattern == 10:
        partition_sizes = np.ones((train_class_num, initial_workers)) * (1.0 / initial_workers)
    else:
        non_iid_ratio = 1/data_pattern 
        partition_sizes = non_iid_partition_strict(non_iid_ratio, data_pattern, train_class_num, initial_workers)
    
    print(partition_sizes)
    
    train_data_partition = datasets.LabelwisePartitioner(train_dataset, partition_sizes=partition_sizes, class_num=train_class_num, labels=labels)
    # Add this line to create the plot
    datasets.plot_data_distribution(train_dataset, train_data_partition, dataset_type, data_pattern, initial_workers)
    return train_dataset, test_dataset, train_data_partition, labels

def print_worker_data_counts(train_data_partition, initial_workers):
    total_count = 0
    for worker_idx in range(initial_workers):
        worker_data_count = len(train_data_partition.use(worker_idx))
        print(f"Worker {worker_idx} has {worker_data_count} training samples")
        total_count += worker_data_count
    
    print(f"Total samples across all workers: {total_count}")
    return total_count

def main():
    torch.manual_seed(42)
    initial_workers = args.initial_workers
    chosen_worker_num = args.chosen_worker_num
    print(args.__dict__)

    if args.two_splits:
        client_global_model, server_global_model = models.create_model_instance_SL_two_splits(args.dataset_type, args.model_type, worker_num=1, num_servers=1)
        nets, _ = models.create_model_instance_SL_two_splits(args.dataset_type, args.model_type, initial_workers)
        
        if args.num_servers > 1:
            _, intermediate_server_models = models.create_model_instance_SL_two_splits(args.dataset_type, args.model_type, 1, num_servers=args.num_servers)

        client_global_model_first = client_global_model[0][0]
        client_global_model_last = client_global_model[0][1]

        # Load the initial state
        global_model_par_first = client_global_model_first.state_dict()
        global_model_par_last = client_global_model_last.state_dict()
        
        for net_id, net in nets.items():
            net[0].load_state_dict(global_model_par_first)
            net[1].load_state_dict(global_model_par_last)
    else:
        client_global_model, server_global_model = models.create_model_instance_SL(args.dataset_type, args.model_type, 1)
        nets, _ = models.create_model_instance_SL(args.dataset_type, args.model_type, initial_workers)

        client_global_model = client_global_model[0]
        global_model_par = client_global_model.state_dict()
        for net_id, net in nets.items():
            net.load_state_dict(global_model_par)

    # Create model instance
    if args.type_noniid == 'default':
        train_dataset, test_dataset, train_data_partition, labels = partition_data(args.dataset_type, args.data_pattern, initial_workers)
    else:
        train_dataset, test_dataset, train_data_partition, labels = partition_data_non_iid_strict(args.dataset_type, args.data_pattern, initial_workers)

    if labels:
        test_loader = datasets.create_dataloaders(test_dataset, batch_size=64, shuffle=False, collate_fn=lambda x: datasets.collate_fn(x, labels))
    else:
        test_loader = datasets.create_dataloaders(test_dataset, batch_size=64, shuffle=False)
    
    print("\nData distribution across workers:")
    total_samples = print_worker_data_counts(train_data_partition, initial_workers)
    print(f"Verification: Total dataset size is {len(train_dataset)}, distributed samples: {total_samples}")

    # Clients data loaders
    bsz_list = np.ones(initial_workers, dtype=int) * args.batch_size
    client_train_loader = []
    for worker_idx in range(initial_workers):
        if labels:
            client_train_loader.append(datasets.create_dataloaders(train_dataset, batch_size=int(bsz_list[worker_idx]), selected_idxs=train_data_partition.use(worker_idx), pin_memory=False, drop_last=True, collate_fn=lambda x: datasets.collate_fn(x, labels)))
        else:
            print(f'for worker {worker_idx}')
            client_train_loader.append(datasets.create_dataloaders(train_dataset, batch_size=int(bsz_list[worker_idx]), selected_idxs=train_data_partition.use(worker_idx), pin_memory=False, drop_last=True))

    
    # Client selection
    selected_clients = select_clients(args.selection_strategy, initial_workers, chosen_worker_num)
    print(f"Selected clients for training: {selected_clients}")
    
    epoch_client_lr = args.client_lr
    epoch_server_lr = args.server_lr
    
    print('Start training') 
    for epoch_idx in range(1, 1 + args.epoch):
        print(f'In epoch:{epoch_idx}')
        start_time = time.time()
        # Learning rate adjustment
        if epoch_idx > 1:
            epoch_client_lr = max((args.decay_rate * epoch_client_lr, args.min_lr))
            epoch_server_lr = max((args.decay_rate * epoch_server_lr, args.min_lr))

        # Define optimizers    
        if args.momentum < 0:
            global_optim = optim.SGD(server_global_model.parameters(), lr=epoch_server_lr, weight_decay=args.weight_decay)
        else:
            global_optim = optim.SGD(server_global_model.parameters(), lr=epoch_server_lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)

        if args.two_splits:
            client_optimizers_first = {}
            client_optimizers_last = {}
            if args.num_servers > 1:
                if args.momentum < 0:
                    intermediate_optimizers = [optim.SGD(intermediate_server_models[i].parameters(), lr=epoch_server_lr, weight_decay=args.weight_decay) for i in range(args.num_servers)] 
                else:
                    intermediate_optimizers = [optim.SGD(intermediate_server_models[i].parameters(), lr=epoch_server_lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay) for i in range(args.num_servers)] 

        else:
            clients_optimizers = {}
        for worker_idx in selected_clients:
            if args.momentum < 0:
                if args.two_splits:
                    client_optimizers_first[worker_idx] = optim.SGD(nets[worker_idx][0].parameters(), lr=epoch_client_lr, weight_decay=args.weight_decay)
                    client_optimizers_last[worker_idx] = optim.SGD(nets[worker_idx][1].parameters(), lr=epoch_client_lr, weight_decay=args.weight_decay)
                else:
                    clients_optimizers[worker_idx] = optim.SGD(nets[worker_idx].parameters(), lr=epoch_client_lr, weight_decay=args.weight_decay)
            else:
                if args.two_splits:
                    client_optimizers_first[worker_idx] = optim.SGD(nets[worker_idx][0].parameters(), lr=epoch_client_lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)
                    client_optimizers_last[worker_idx] = optim.SGD(nets[worker_idx][1].parameters(), lr=epoch_client_lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)
                else:
                    clients_optimizers[worker_idx] = optim.SGD(nets[worker_idx].parameters(), lr=epoch_client_lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)

        server_global_model.train()
        if args.two_splits and args.num_servers > 1:
            for server in intermediate_server_models:
                server.train()
        local_steps = 42
        server_global_model.to(device)
        if args.two_splits and args.num_servers > 1:
            for server in intermediate_server_models:
                server.to(device)

        # Training loop - client side
        for iter_idx in range(local_steps):
            if args.two_splits:
                if args.num_servers == 1:
                    sum_bsz = sum([bsz_list[i] for i in selected_clients])
                    global_optim.zero_grad()
                    det_out_as = []
                    my_outas = []
                    clients_part1_send_targets = []
                    for i, worker_idx in enumerate(selected_clients):
                        inputs, targets = next(client_train_loader[worker_idx])
                        inputs, targets = inputs.to(device), targets.to(device) 
                        clients_part1_send_targets.append(targets)
                        nets[worker_idx][0].to(device)

                        # forward prop
                        out_a = nets[worker_idx][0](inputs)
                        my_outas.append(out_a.requires_grad_(True))
                        det_out_a = out_a.clone().detach().requires_grad_(True)
                        det_out_a.to(device)
                        det_out_as.append(det_out_a)

                    # clients send to server det_out_as
                    det_out_a_all = torch.cat(det_out_as)
                    out_b = server_global_model(det_out_a_all)
                    det_out_b = out_b.clone().detach().requires_grad_(True)
                    det_out_b.to(device)
                    # forward to helpers model part c
                    
                    #server sends det_out_b[bsz_s: bsz_s + bsz_list[worker_idx]].clone().detach() to each client
                    grad_bs = []
                    bsz_s = 0 
                    for i, worker_idx in enumerate(selected_clients):
                        det_out_b_ = det_out_b[bsz_s: bsz_s + bsz_list[worker_idx]].clone().detach().requires_grad_(True) # "transmission"
                        bsz_s += bsz_list[worker_idx]
                    
                        out = nets[worker_idx][1](det_out_b_)
                        out.to(device)

                        client_optimizers_last[worker_idx].zero_grad()
                        loss = F.cross_entropy(out, clients_part1_send_targets[i].long())  #* args.batch_size) / (args.batch_size*(worker_num))
                        loss.backward()
                        client_optimizers_last[worker_idx].step()
                        grad_b = (det_out_b_.grad.clone().detach()  * args.batch_size) / (args.batch_size*chosen_worker_num)
                        grad_b.to(device)
                        grad_bs.append(grad_b)
                    
                    # clients send to server grad_bs
                    global_optim.zero_grad()
                    grad_b_all = torch.cat(grad_bs) 
                    grad_b_all.to(device)
                    out_b.to(device)
                    out_b.backward(grad_b_all)
                    global_optim.step()

                    bsz_s = 0 
                    for i, worker_idx in enumerate(selected_clients):
                        client_optimizers_first[worker_idx].zero_grad()
                        outa_a = det_out_as[i] 
                        bsz_s += bsz_list[worker_idx]

                        grad_a = outa_a.grad.clone().detach()  #* args.batch_size) / (args.batch_size*(worker_num))
                        # server sends grad_a to individual workers
                        my_outas[i].backward(grad_a)
                        client_optimizers_first[worker_idx].step()
                elif args.num_servers == 2:
                    sum_bsz = sum([bsz_list[i] for i in selected_clients])
                    global_optim.zero_grad()
                    intermediate_optimizers[0].zero_grad()
                    intermediate_optimizers[1].zero_grad()
                    for optimizer in intermediate_optimizers:
                        optimizer.zero_grad()

                    det_out_as = []
                    my_outas = []
                    clients_part1_send_targets = []

                    for i, worker_idx in enumerate(selected_clients):
                        inputs, targets = next(client_train_loader[worker_idx])
                        inputs, targets = inputs.to(device), targets.to(device)
                        clients_part1_send_targets.append(targets)

                        nets[worker_idx][0].to(device)

                        out_a = nets[worker_idx][0](inputs)
                        my_outas.append(out_a.requires_grad_(True))
                        det_out_a = out_a.clone().detach().requires_grad_(True)
                        det_out_a.to(device)
                        det_out_as.append(det_out_a)

                    # Split data for intermediate servers
                    det_out_a_all = torch.cat(det_out_as)
                    split_point = det_out_a_all.shape[0] // 2
                    det_out_a_all_1 = det_out_a_all[:split_point]
                    det_out_a_all_2 = det_out_a_all[split_point:]

                    # Process through intermediate servers
                    out_b_1 = intermediate_server_models[0](det_out_a_all_1)
                    out_b_2 = intermediate_server_models[1](det_out_a_all_2)

                    # Recombine outputs
                    out_b = torch.cat([out_b_1, out_b_2], dim=0)
                    det_out_b = out_b.clone().detach().requires_grad_(True)
                    det_out_b.to(device)

                    # Forward to helpers model part c
                    grad_bs = []
                    bsz_s = 0 
                    for i, worker_idx in enumerate(selected_clients):
                        det_out_b_ = det_out_b[bsz_s: bsz_s + bsz_list[worker_idx]].clone().detach().requires_grad_(True)
                        bsz_s += bsz_list[worker_idx]
                    
                        out = nets[worker_idx][1](det_out_b_)
                        out.to(device)

                        client_optimizers_last[worker_idx].zero_grad()
                        loss = F.cross_entropy(out, clients_part1_send_targets[i].long())
                        loss.backward()
                        client_optimizers_last[worker_idx].step()
                        grad_b = (det_out_b_.grad.clone().detach() * args.batch_size) / (args.batch_size * chosen_worker_num)
                        grad_b.to(device)
                        grad_bs.append(grad_b)

                    # Backward pass through intermediate servers
                    global_optim.zero_grad()
                    intermediate_optimizers[0].zero_grad()
                    intermediate_optimizers[1].zero_grad()
                    grad_b_all = torch.cat(grad_bs) 
                    grad_b_all.to(device)
                    # Split gradients for intermediate servers
                    split_point = grad_b_all.shape[0] // 2
                    grad_b_1 = grad_b_all[:split_point]
                    grad_b_2 = grad_b_all[split_point:]

                    # Backward pass for each intermediate server
                    out_b_1.backward(grad_b_1)
                    out_b_2.backward(grad_b_2)

                    # Update intermediate servers
                    intermediate_optimizers[0].step()
                    intermediate_optimizers[1].step()

                    # Client backward pass (first part)
                    bsz_s = 0 
                    for i, worker_idx in enumerate(selected_clients):
                        client_optimizers_first[worker_idx].zero_grad()
                        outa_a = det_out_as[i] 
                        bsz_s += bsz_list[worker_idx]

                        grad_a = outa_a.grad.clone().detach()
                        my_outas[i].backward(grad_a)
                        client_optimizers_first[worker_idx].step()
                elif args.num_servers == 4:
                    sum_bsz = sum([bsz_list[i] for i in selected_clients])
                    global_optim.zero_grad()
                    for i in range(args.num_servers):
                        intermediate_optimizers[i].zero_grad()
                    for optimizer in intermediate_optimizers:
                        optimizer.zero_grad()

                    det_out_as = []
                    my_outas = []
                    clients_part1_send_targets = []

                    for i, worker_idx in enumerate(selected_clients):
                        inputs, targets = next(client_train_loader[worker_idx])
                        inputs, targets = inputs.to(device), targets.to(device)
                        clients_part1_send_targets.append(targets)

                        nets[worker_idx][0].to(device)

                        out_a = nets[worker_idx][0](inputs)
                        my_outas.append(out_a.requires_grad_(True))
                        det_out_a = out_a.clone().detach().requires_grad_(True)
                        det_out_a.to(device)
                        det_out_as.append(det_out_a) ## here

                    # Split data for intermediate servers
                    det_out_a_all = torch.cat(det_out_as)
                    split_points = [det_out_a_all.shape[0] // 4, det_out_a_all.shape[0] // 2, 3 * det_out_a_all.shape[0] // 4]
                    det_out_a_all_1 = det_out_a_all[:split_points[0]]
                    det_out_a_all_2 = det_out_a_all[split_points[0]:split_points[1]]
                    det_out_a_all_3 = det_out_a_all[split_points[1]:split_points[2]]
                    det_out_a_all_4 = det_out_a_all[split_points[2]:]

                    # Process through intermediate servers
                    out_b_1 = intermediate_server_models[0](det_out_a_all_1)
                    out_b_2 = intermediate_server_models[1](det_out_a_all_2)
                    out_b_3 = intermediate_server_models[2](det_out_a_all_3)
                    out_b_4 = intermediate_server_models[3](det_out_a_all_4)

                    # Recombine outputs
                    out_b = torch.cat([out_b_1, out_b_2, out_b_3, out_b_4], dim=0)
                    det_out_b = out_b.clone().detach().requires_grad_(True)
                    det_out_b.to(device)

                    # Forward to helpers model part c
                    grad_bs = []
                    bsz_s = 0 
                    for i, worker_idx in enumerate(selected_clients):
                        det_out_b_ = det_out_b[bsz_s: bsz_s + bsz_list[worker_idx]].clone().detach().requires_grad_(True)
                        bsz_s += bsz_list[worker_idx]
                    
                        out = nets[worker_idx][1](det_out_b_)
                        out.to(device)

                        client_optimizers_last[worker_idx].zero_grad()
                        loss = F.cross_entropy(out, clients_part1_send_targets[i].long())
                        loss.backward()
                        client_optimizers_last[worker_idx].step()
                        grad_b = (det_out_b_.grad.clone().detach() * args.batch_size) / (args.batch_size * chosen_worker_num)
                        grad_b.to(device)
                        grad_bs.append(grad_b)

                    # Backward pass through intermediate servers
                    global_optim.zero_grad()
                    for i in range(args.num_servers):
                        intermediate_optimizers[i].zero_grad()
                    grad_b_all = torch.cat(grad_bs) 
                    grad_b_all.to(device)
                    # Split gradients for intermediate servers
                    grad_b_1 = grad_b_all[:split_points[0]]
                    grad_b_2 = grad_b_all[split_points[0]:split_points[1]]
                    grad_b_3 = grad_b_all[split_points[1]:split_points[2]]
                    grad_b_4 = grad_b_all[split_points[2]:]

                    # Backward pass for each intermediate server
                    out_b_1.backward(grad_b_1)
                    out_b_2.backward(grad_b_2)
                    out_b_3.backward(grad_b_3)
                    out_b_4.backward(grad_b_4)

                    # Update intermediate servers
                    for i in range(args.num_servers):
                        intermediate_optimizers[i].step()

                    # Client backward pass (first part)
                    bsz_s = 0 
                    for i, worker_idx in enumerate(selected_clients):
                        client_optimizers_first[worker_idx].zero_grad()
                        outa_a = det_out_as[i] 
                        bsz_s += bsz_list[worker_idx]

                        grad_a = outa_a.grad.clone().detach()
                        my_outas[i].backward(grad_a)
                        client_optimizers_first[worker_idx].step()
                
                elif args.num_servers == 3:
                    sum_bsz = sum([bsz_list[i] for i in selected_clients])
                    global_optim.zero_grad()
                    for i in range(args.num_servers):
                        intermediate_optimizers[i].zero_grad()
                    for optimizer in intermediate_optimizers:
                        optimizer.zero_grad()

                    det_out_as = []
                    my_outas = []
                    clients_part1_send_targets = []

                    for i, worker_idx in enumerate(selected_clients):
                        inputs, targets = next(client_train_loader[worker_idx])
                        inputs, targets = inputs.to(device), targets.to(device)
                        clients_part1_send_targets.append(targets)

                        nets[worker_idx][0].to(device)

                        out_a = nets[worker_idx][0](inputs)
                        my_outas.append(out_a.requires_grad_(True))
                        det_out_a = out_a.clone().detach().requires_grad_(True)
                        det_out_a.to(device)
                        det_out_as.append(det_out_a) ## here

                    # Split data for intermediate servers
                    det_out_a_all = torch.cat(det_out_as)
                    split_points = [det_out_a_all.shape[0] // 3, det_out_a_all.shape[0] * 2// 3]
                    det_out_a_all_1 = det_out_a_all[:split_points[0]]
                    det_out_a_all_2 = det_out_a_all[split_points[0]:split_points[1]]
                    det_out_a_all_3 = det_out_a_all[split_points[1]:]

                    # Process through intermediate servers
                    out_b_1 = intermediate_server_models[0](det_out_a_all_1)
                    out_b_2 = intermediate_server_models[1](det_out_a_all_2)
                    out_b_3 = intermediate_server_models[2](det_out_a_all_3)

                    # Recombine outputs
                    out_b = torch.cat([out_b_1, out_b_2, out_b_3], dim=0)
                    det_out_b = out_b.clone().detach().requires_grad_(True)
                    det_out_b.to(device)

                    # Forward to helpers model part c
                    grad_bs = []
                    bsz_s = 0 
                    for i, worker_idx in enumerate(selected_clients):
                        det_out_b_ = det_out_b[bsz_s: bsz_s + bsz_list[worker_idx]].clone().detach().requires_grad_(True)
                        bsz_s += bsz_list[worker_idx]
                    
                        out = nets[worker_idx][1](det_out_b_)
                        out.to(device)

                        client_optimizers_last[worker_idx].zero_grad()
                        loss = F.cross_entropy(out, clients_part1_send_targets[i].long())
                        loss.backward()
                        client_optimizers_last[worker_idx].step()
                        grad_b = (det_out_b_.grad.clone().detach() * args.batch_size) / (args.batch_size * chosen_worker_num)
                        grad_b.to(device)
                        grad_bs.append(grad_b)

                    # Backward pass through intermediate servers
                    global_optim.zero_grad()
                    for i in range(args.num_servers):
                        intermediate_optimizers[i].zero_grad()
                    grad_b_all = torch.cat(grad_bs) 
                    grad_b_all.to(device)
                    # Split gradients for intermediate servers
                    grad_b_1 = grad_b_all[:split_points[0]]
                    grad_b_2 = grad_b_all[split_points[0]:split_points[1]]
                    grad_b_3 = grad_b_all[split_points[1]:]

                    # Backward pass for each intermediate server
                    out_b_1.backward(grad_b_1)
                    out_b_2.backward(grad_b_2)
                    out_b_3.backward(grad_b_3)

                    # Update intermediate servers
                    for i in range(args.num_servers):
                        intermediate_optimizers[i].step()

                    # Client backward pass (first part)
                    bsz_s = 0 
                    for i, worker_idx in enumerate(selected_clients):
                        client_optimizers_first[worker_idx].zero_grad()
                        outa_a = det_out_as[i] 
                        bsz_s += bsz_list[worker_idx]

                        grad_a = outa_a.grad.clone().detach()
                        my_outas[i].backward(grad_a)
                        client_optimizers_first[worker_idx].step()
                elif args.num_servers == 4:
                    sum_bsz = sum([bsz_list[i] for i in selected_clients])
                    global_optim.zero_grad()
                    for i in range(args.num_servers):
                        intermediate_optimizers[i].zero_grad()
                    for optimizer in intermediate_optimizers:
                        optimizer.zero_grad()

                    det_out_as = []
                    my_outas = []
                    clients_part1_send_targets = []

                    for i, worker_idx in enumerate(selected_clients):
                        inputs, targets = next(client_train_loader[worker_idx])
                        inputs, targets = inputs.to(device), targets.to(device)
                        clients_part1_send_targets.append(targets)

                        nets[worker_idx][0].to(device)

                        out_a = nets[worker_idx][0](inputs)
                        my_outas.append(out_a.requires_grad_(True))
                        det_out_a = out_a.clone().detach().requires_grad_(True)
                        det_out_a.to(device)
                        det_out_as.append(det_out_a) ## here

                    # Split data for intermediate servers
                    det_out_a_all = torch.cat(det_out_as)
                    split_points = [det_out_a_all.shape[0] // 4, det_out_a_all.shape[0] // 2, 3 * det_out_a_all.shape[0] // 4]
                    det_out_a_all_1 = det_out_a_all[:split_points[0]]
                    det_out_a_all_2 = det_out_a_all[split_points[0]:split_points[1]]
                    det_out_a_all_3 = det_out_a_all[split_points[1]:split_points[2]]
                    det_out_a_all_4 = det_out_a_all[split_points[2]:]

                    # Process through intermediate servers
                    out_b_1 = intermediate_server_models[0](det_out_a_all_1)
                    out_b_2 = intermediate_server_models[1](det_out_a_all_2)
                    out_b_3 = intermediate_server_models[2](det_out_a_all_3)
                    out_b_4 = intermediate_server_models[3](det_out_a_all_4)

                    # Recombine outputs
                    out_b = torch.cat([out_b_1, out_b_2, out_b_3, out_b_4], dim=0)
                    det_out_b = out_b.clone().detach().requires_grad_(True)
                    det_out_b.to(device)

                    # Forward to helpers model part c
                    grad_bs = []
                    bsz_s = 0 
                    for i, worker_idx in enumerate(selected_clients):
                        det_out_b_ = det_out_b[bsz_s: bsz_s + bsz_list[worker_idx]].clone().detach().requires_grad_(True)
                        bsz_s += bsz_list[worker_idx]
                    
                        out = nets[worker_idx][1](det_out_b_)
                        out.to(device)

                        client_optimizers_last[worker_idx].zero_grad()
                        loss = F.cross_entropy(out, clients_part1_send_targets[i].long())
                        loss.backward()
                        client_optimizers_last[worker_idx].step()
                        grad_b = (det_out_b_.grad.clone().detach() * args.batch_size) / (args.batch_size * chosen_worker_num)
                        grad_b.to(device)
                        grad_bs.append(grad_b)

                    # Backward pass through intermediate servers
                    global_optim.zero_grad()
                    for i in range(args.num_servers):
                        intermediate_optimizers[i].zero_grad()
                    grad_b_all = torch.cat(grad_bs) 
                    grad_b_all.to(device)
                    # Split gradients for intermediate servers
                    grad_b_1 = grad_b_all[:split_points[0]]
                    grad_b_2 = grad_b_all[split_points[0]:split_points[1]]
                    grad_b_3 = grad_b_all[split_points[1]:split_points[2]]
                    grad_b_4 = grad_b_all[split_points[2]:]

                    # Backward pass for each intermediate server
                    out_b_1.backward(grad_b_1)
                    out_b_2.backward(grad_b_2)
                    out_b_3.backward(grad_b_3)
                    out_b_4.backward(grad_b_4)

                    # Update intermediate servers
                    for i in range(args.num_servers):
                        intermediate_optimizers[i].step()

                    # Client backward pass (first part)
                    bsz_s = 0 
                    for i, worker_idx in enumerate(selected_clients):
                        client_optimizers_first[worker_idx].zero_grad()
                        outa_a = det_out_as[i] 
                        bsz_s += bsz_list[worker_idx]

                        grad_a = outa_a.grad.clone().detach()
                        my_outas[i].backward(grad_a)
                        client_optimizers_first[worker_idx].step()

            else:
                # Ensure chosen_worker_num is a multiple of 10
                assert chosen_worker_num % 10 == 0, "Number of chosen workers must be a multiple of 10"

                # Group selected clients into batches of 10
                client_batches = [selected_clients[i:i+10] for i in range(0, len(selected_clients), 10)]

                # Cycle through client batches
                batch_idx = iter_idx % len(client_batches)
                current_batch = client_batches[batch_idx]

                clients_smash_data = []
                clients_send_data = []
                clients_send_targets = []

                sum_bsz = sum([bsz_list[i] for i in selected_clients])
                for worker_idx in current_batch:
                    inputs, targets = next(client_train_loader[worker_idx])

                    inputs, targets = inputs.to(device), targets.to(device)

                    nets[worker_idx].to(device)

                    clients_smash_data.append(nets[worker_idx](inputs))

                    send_smash = clients_smash_data[-1].detach()
                    clients_send_data.append(send_smash)
                    clients_send_targets.append(targets)
                
                m_data = torch.cat(clients_send_data, dim=0)
                m_target = torch.cat(clients_send_targets, dim=0)

                m_data.requires_grad_()

                # server side fp
                outputs = server_global_model(m_data)
                
                loss = F.cross_entropy(outputs, m_target.long())

                # server side bp
                global_optim.zero_grad()
                loss.backward()
                global_optim.step()

                # gradient dispatch
                bsz_s = 0
                for i, worker_idx in enumerate(current_batch):
                    clients_grad = m_data.grad[bsz_s: bsz_s + bsz_list[worker_idx]] * sum_bsz / bsz_list[worker_idx]
                    bsz_s += bsz_list[worker_idx]

                    clients_optimizers[worker_idx].zero_grad()
                    clients_smash_data[i].backward(clients_grad.to(device))
                    clients_optimizers[worker_idx].step()
        
        # AGGREGATION
        with torch.no_grad():
            # Aggregate client models
            for i, worker_idx in enumerate(selected_clients):
                if args.two_splits:
                    nets[worker_idx][0].to('cpu')
                    nets[worker_idx][1].to('cpu')
                    net_para_first = nets[worker_idx][0].cpu().state_dict()
                    net_para_last = nets[worker_idx][1].cpu().state_dict()
                else:
                    nets[worker_idx].to('cpu')
                    net_para = nets[worker_idx].cpu().state_dict()
                
                if i == 0:
                    if args.two_splits:
                        for key in net_para_first:
                            global_model_par_first[key] = net_para_first[key] / chosen_worker_num
                        for key in net_para_last:
                            global_model_par_last[key] = net_para_last[key] / chosen_worker_num
                    else: 
                        for key in net_para:
                            global_model_par[key] = net_para[key] / chosen_worker_num
                else:
                    if args.two_splits:
                        for key in net_para_first:
                            global_model_par_first[key] += net_para_first[key] / chosen_worker_num
                        for key in net_para_last:
                            global_model_par_last[key] += net_para_last[key] / chosen_worker_num
                    else:
                        for key in net_para:
                            global_model_par[key] += net_para[key] / chosen_worker_num
            
            # Aggregate and synchronize intermediate server models if num_servers == 2
            if args.num_servers == 2:
                for server_idx in range(2):
                    intermediate_server_models[server_idx].to('cpu')
                
                server_model_0 = intermediate_server_models[0].cpu().state_dict()
                server_model_1 = intermediate_server_models[1].cpu().state_dict()
                
                # Average the parameters of the two intermediate models
                for key in server_model_0:
                    server_model_0[key] = (server_model_0[key] + server_model_1[key]) / 2
                
                # Update both intermediate server models with the averaged parameters
                for server_idx in range(2):
                    intermediate_server_models[server_idx].load_state_dict(server_model_0)
                
                # Update the server_global_model with the synchronized parameters
                server_global_model.load_state_dict(server_model_0)
            
            # Aggregate and synchronize intermediate server models if num_servers == 3
            elif args.num_servers == 3:
                for server_idx in range(3):
                    intermediate_server_models[server_idx].to('cpu')
                
                server_model_0 = intermediate_server_models[0].cpu().state_dict()
                server_model_1 = intermediate_server_models[1].cpu().state_dict()
                server_model_2 = intermediate_server_models[2].cpu().state_dict()
                
                # Average the parameters of the two intermediate models
                for key in server_model_0:
                    server_model_0[key] = (server_model_0[key] + server_model_1[key] + server_model_2[key]) / 3
                
                # Update both intermediate server models with the averaged parameters
                for server_idx in range(3):
                    intermediate_server_models[server_idx].load_state_dict(server_model_0)
                
                # Update the server_global_model with the synchronized parameters
                server_global_model.load_state_dict(server_model_0)
            # Aggregate and synchronize intermediate server models if num_servers == 4
            elif args.num_servers == 4:
                for server_idx in range(4):
                    intermediate_server_models[server_idx].to('cpu')
                
                server_model_0 = intermediate_server_models[0].cpu().state_dict()
                server_model_1 = intermediate_server_models[1].cpu().state_dict()
                server_model_2 = intermediate_server_models[2].cpu().state_dict()
                server_model_3 = intermediate_server_models[3].cpu().state_dict()
                
                # Average the parameters of the two intermediate models
                for key in server_model_0:
                    server_model_0[key] = (server_model_0[key] + server_model_1[key] + server_model_2[key] + server_model_3[key]) / 4
                
                # Update both intermediate server models with the averaged parameters
                for server_idx in range(4):
                    intermediate_server_models[server_idx].load_state_dict(server_model_0)
                
                # Update the server_global_model with the synchronized parameters
                server_global_model.load_state_dict(server_model_0)
            
            # Load aggregated parameters into global models
            if args.two_splits:
                client_global_model_first.load_state_dict(global_model_par_first)
                client_global_model_last.load_state_dict(global_model_par_last)
            else:
                client_global_model.load_state_dict(global_model_par)

            # Distribute the aggregated model back to all clients
            if args.two_splits:
                global_model_par_first = client_global_model_first.state_dict()
                global_model_par_last = client_global_model_last.state_dict()
                for net_id, net in nets.items():
                    net[0].load_state_dict(global_model_par_first)
                    net[1].load_state_dict(global_model_par_last)
            else:
                global_model_par = client_global_model.state_dict()
                for net_id, net in nets.items():
                    net.load_state_dict(global_model_par)

        # Move server model to CPU for testing
        server_global_model.to('cpu')

        # Perform testing
        if args.two_splits:
            test_loss, acc = test((client_global_model_first, client_global_model_last), server_global_model, test_loader, two_split=args.two_splits)
        else:
            test_loss, acc = test(client_global_model, server_global_model, test_loader, two_split=args.two_splits)

        # Print epoch results
        print("Epoch: {}, accuracy: {}, test_loss: {}".format(epoch_idx, acc, test_loss))

if __name__ == '__main__':
    main()
