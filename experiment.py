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


#init parameters
parser = argparse.ArgumentParser(description='Distributed Client')
parser.add_argument('--dataset_type', type=str, default='CIFAR10')
parser.add_argument('--model_type', type=str, default='AlexNet')
parser.add_argument('--worker_num', type=int, default=10) #
parser.add_argument('--batch_size', type=int, default=10) 
parser.add_argument('--data_pattern', type=int, default=0)
parser.add_argument('--client_lr', type=float, default=0.01)
parser.add_argument('--server_lr', type=float, default=0.1)
parser.add_argument('--decay_rate', type=float, default=0.993)
parser.add_argument('--min_lr', type=float, default=0.005)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--momentum', type=float, default=-1)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--data_path', type=str, default='./data')
parser.add_argument('--device', type=str, default='cpu', help='The device to run the program') #cpu
parser.add_argument('--expname', type=str, default='MergeSFL')
parser.add_argument('--two_splits', action="store_true", help='do U-Shape')
parser.add_argument('--type_noniid', type=str, default='default')
parser.add_argument('--level', type=int, default=10)
parser.add_argument('--num_servers', type=int, default=1, choices=[1, 2], help='Number of intermediate servers')

args = parser.parse_args()
device = torch.device(args.device)

def non_iid_partition(ratio, train_class_num, worker_num):
    partition_sizes = np.ones((train_class_num, worker_num)) * ((1 - ratio) / (worker_num-1))

    for i in range(train_class_num):
        partition_sizes[i][i%worker_num]=ratio

    return partition_sizes

def non_iid_partition_strict(ratio, level, train_class_num, worker_num):
    partition_sizes = np.zeros((train_class_num, worker_num))

    for i in range(train_class_num):
        for j in range(level):
            partition_sizes[i][(i+j)%worker_num]=ratio

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
    elif dataset_type=='UCIHAR':
        train_class_num = 6
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

    elif data_pattern == 5:  # dir-1.0
        print('Dirichlet partition 1.0')
        partition_sizes = dirichlet_partition(dataset_type, 1.0, worker_num, train_class_num)

    elif data_pattern == 6:  # dir-0.5
        print('Dirichlet partition 0.5')
        partition_sizes = dirichlet_partition(dataset_type, 0.5, worker_num, train_class_num)

    elif data_pattern == 7:  # dir-0.1
        print('Dirichlet partition 0.1')
        partition_sizes = dirichlet_partition(dataset_type, 0.1, worker_num, train_class_num)

    elif data_pattern == 8:  # dir-0.05
        print('Dirichlet partition 0.05')
        partition_sizes = dirichlet_partition(dataset_type, 0.05, worker_num, train_class_num)
    print(partition_sizes)
    train_data_partition = datasets.LabelwisePartitioner(train_dataset, partition_sizes=partition_sizes, class_num=train_class_num, labels=labels)
    return train_dataset, test_dataset, train_data_partition, labels

def partition_data_non_iid_strict(dataset_type, data_pattern, worker_num=10):
    train_dataset, test_dataset = datasets.load_datasets(dataset_type)
    labels = None
    if dataset_type == "CIFAR10" or dataset_type == "FashionMNIST":
        train_class_num = 10
    elif dataset_type=='UCIHAR':
        train_class_num = 6

    
    if data_pattern == 10:
        partition_sizes = np.ones((train_class_num, worker_num)) * (1.0 / worker_num)
    else:
        non_iid_ratio = 1/data_pattern 
        partition_sizes = non_iid_partition_strict(non_iid_ratio, data_pattern, train_class_num, worker_num)
    
    print(partition_sizes)
    
    train_data_partition = datasets.LabelwisePartitioner(train_dataset, partition_sizes=partition_sizes, class_num=train_class_num, labels=labels)
    return train_dataset, test_dataset, train_data_partition, labels

def main():
    torch.manual_seed(42)
    worker_num = args.worker_num
    print(args.__dict__)

    if args.two_splits:
        client_global_model, server_global_model = models.create_model_instance_SL_two_splits(args.dataset_type, args.model_type, worker_num=1, num_servers=1)
        nets, _ = models.create_model_instance_SL_two_splits(args.dataset_type, args.model_type, worker_num)
        
        if args.num_servers == 2:
            _, intermediate_server_models = models.create_model_instance_SL_two_splits(args.dataset_type, args.model_type, 1, num_servers=2)

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
        nets, _ = models.create_model_instance_SL(args.dataset_type, args.model_type, worker_num)

        client_global_model = client_global_model[0]
        global_model_par = client_global_model.state_dict()
        for net_id, net in nets.items():
            net.load_state_dict(global_model_par)

    # Create model instance
    if args.type_noniid == 'default':
        train_dataset, test_dataset, train_data_partition, labels = partition_data(args.dataset_type, args.data_pattern, worker_num)
    else:
        train_dataset, test_dataset, train_data_partition, labels = partition_data_non_iid_strict(args.dataset_type, args.data_pattern, worker_num)

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
            client_optimizers_first = []
            client_optimizers_last = []
            if args.num_servers == 2:
                if args.momentum < 0:
                    intermediate_optimizers = [optim.SGD(server_global_model.parameters(), lr=epoch_server_lr, weight_decay=args.weight_decay) for server in intermediate_server_models] 
                else:
                    intermediate_optimizers = [optim.SGD(server_global_model.parameters(), lr=epoch_server_lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay) for server in intermediate_server_models] 

        else:
            clients_optimizers = []
        for worker_idx in range(worker_num):
            if args.momentum < 0:
                if args.two_splits:
                    client_optimizers_first.append(optim.SGD(nets[worker_idx][0].parameters(), lr=epoch_client_lr, weight_decay=args.weight_decay))
                    client_optimizers_last.append(optim.SGD(nets[worker_idx][1].parameters(), lr=epoch_client_lr, weight_decay=args.weight_decay))
                else:
                    clients_optimizers.append(optim.SGD(nets[worker_idx].parameters(), lr=epoch_client_lr, weight_decay=args.weight_decay))
            else:
                if args.two_splits:
                    client_optimizers_first.append(optim.SGD(nets[worker_idx][0].parameters(), lr=epoch_client_lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay))
                    client_optimizers_last.append(optim.SGD(nets[worker_idx][1].parameters(), lr=epoch_client_lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay))
                else:
                    clients_optimizers.append(optim.SGD(nets[worker_idx].parameters(), lr=epoch_client_lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay))

        server_global_model.train()
        if args.two_splits and args.num_servers == 2:
            for server in intermediate_server_models:
                server.train()
        local_steps = 42
        server_global_model.to(device)
        if args.two_splits and args.num_servers == 2:
            for server in intermediate_server_models:
                server.to(device)

        # Training loop - client side
        for iter_idx in range(local_steps):
            if args.two_splits:
                if args.num_servers == 1:
                    sum_bsz = sum([bsz_list[i] for i in range(worker_num)])
                    global_optim.zero_grad()
                    det_out_as = []
                    my_outas = []
                    clients_part1_send_targets = []
                    for worker_idx in range(worker_num):
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
                    for worker_idx in range(worker_num):
                        det_out_b_ = det_out_b[bsz_s: bsz_s + bsz_list[worker_idx]].clone().detach().requires_grad_(True) # "transmission"
                        bsz_s += bsz_list[worker_idx]
                    
                        out = nets[worker_idx][1](det_out_b_)
                        out.to(device)

                        client_optimizers_last[worker_idx].zero_grad()
                        loss = F.cross_entropy(out, clients_part1_send_targets[worker_idx].long())  #* args.batch_size) / (args.batch_size*(worker_num))
                        loss.backward()
                        client_optimizers_last[worker_idx].step()
                        grad_b = (det_out_b_.grad.clone().detach()  * args.batch_size) / (args.batch_size*(worker_num))
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
                    for worker_idx in range(worker_num):
                        client_optimizers_first[worker_idx].zero_grad()
                        outa_a = det_out_as[worker_idx] 
                        bsz_s += bsz_list[worker_idx]

                        grad_a = outa_a.grad.clone().detach()  #* args.batch_size) / (args.batch_size*(worker_num))
                        # server sends grad_a to individual workers
                        my_outas[worker_idx].backward(grad_a)
                        client_optimizers_first[worker_idx].step()
                if args.num_servers == 2:
                    sum_bsz = sum([bsz_list[i] for i in range(worker_num)])
                    global_optim.zero_grad()
                    intermediate_optimizers[0].zero_grad()
                    intermediate_optimizers[1].zero_grad()
                    for optimizer in intermediate_optimizers:
                        optimizer.zero_grad()

                    det_out_as = []
                    my_outas = []
                    clients_part1_send_targets = []

                    for worker_idx in range(worker_num):
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
                    for worker_idx in range(worker_num):
                        det_out_b_ = det_out_b[bsz_s: bsz_s + bsz_list[worker_idx]].clone().detach().requires_grad_(True)
                        bsz_s += bsz_list[worker_idx]
                    
                        out = nets[worker_idx][1](det_out_b_)
                        out.to(device)

                        client_optimizers_last[worker_idx].zero_grad()
                        loss = F.cross_entropy(out, clients_part1_send_targets[worker_idx].long())
                        loss.backward()
                        client_optimizers_last[worker_idx].step()
                        grad_b = (det_out_b_.grad.clone().detach() * args.batch_size) / (args.batch_size * worker_num)
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
                    for worker_idx in range(worker_num):
                        client_optimizers_first[worker_idx].zero_grad()
                        outa_a = det_out_as[worker_idx] 
                        bsz_s += bsz_list[worker_idx]

                        grad_a = outa_a.grad.clone().detach()
                        my_outas[worker_idx].backward(grad_a)
                        client_optimizers_first[worker_idx].step()


            else:
                clients_smash_data = []
                clients_send_data = []
                clients_send_targets = []

                sum_bsz = sum([bsz_list[i] for i in range(worker_num)])
                for worker_idx in range(worker_num):
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
                for worker_idx in range(worker_num):
                    clients_grad = m_data.grad[bsz_s: bsz_s + bsz_list[worker_idx]] * sum_bsz / bsz_list[worker_idx]
                    bsz_s += bsz_list[worker_idx]

                    clients_optimizers[worker_idx].zero_grad()
                    clients_smash_data[worker_idx].backward(clients_grad.to(device))
                    clients_optimizers[worker_idx].step()
        
        # AGGREGATION
        with torch.no_grad():
            # Aggregate client models
            for worker_idx in range(worker_num):
                if args.two_splits:
                    nets[worker_idx][0].to('cpu')
                    nets[worker_idx][1].to('cpu')
                    net_para_first = nets[worker_idx][0].cpu().state_dict()
                    net_para_last = nets[worker_idx][1].cpu().state_dict()
                else:
                    nets[worker_idx].to('cpu')
                    net_para = nets[worker_idx].cpu().state_dict()
                
                if worker_idx == 0:
                    if args.two_splits:
                        for key in net_para_first:
                            global_model_par_first[key] = net_para_first[key] / worker_num
                        for key in net_para_last:
                            global_model_par_last[key] = net_para_last[key] / worker_num
                    else: 
                        for key in net_para:
                            global_model_par[key] = net_para[key] / worker_num
                else:
                    if args.two_splits:
                        for key in net_para_first:
                            global_model_par_first[key] += net_para_first[key] / worker_num
                        for key in net_para_last:
                            global_model_par_last[key] += net_para_last[key] / worker_num
                    else:
                        for key in net_para:
                            global_model_par[key] += net_para[key] / worker_num
            
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
