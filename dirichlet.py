import torch
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
import argparse
import flwr_datasets
from flwr_datasets.partitioner import DirichletPartitioner
from flwr_datasets.visualization import plot_label_distributions
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner, SizePartitioner, ShardPartitioner
from flwr_datasets.visualization import plot_comparison_label_distribution

#init parameters
parser = argparse.ArgumentParser(description='Distributed Client')
parser.add_argument('--dataset_type', type=str, default='cifar10')
parser.add_argument('--model_name', type=str, default='simple-cnn')
parser.add_argument('--num_clients', type=int, default=10) 
parser.add_argument('--cut', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=10) 
parser.add_argument('--training_lr', type=float, default=0.1)
parser.add_argument('--decay_rate', type=float, default=0.993)
parser.add_argument('--min_lr', type=float, default=0.005)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--local_epoch', type=int, default=1)
parser.add_argument('--momentum', type=float, default=-1)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--data_path', type=str, default='./data')
parser.add_argument('--device', type=str, default='cpu', help='The device to run the program')
parser.add_argument('--partition_method', type=str, default='iid', help='This can be iid/dirichlet/quantity/archetype')
parser.add_argument('--partition_target', type=str, default='label', help="The target according to which the data will be partioned, e.g., age', 'household_position', 'household_size', etc.")
parser.add_argument('--alpha', type=float, default=1, help='for dirichlet')
parser.add_argument('--num_shards_per_partition', type=int, default=2, help='Number of shrads each client will have in archetype')
parser.add_argument('--shard_size', type=int, default=500, help='Shrads size in archetype')

from math import sqrt

args = parser.parse_args()
device = torch.device(args.device)
torch.manual_seed(42)


def generate_partitioner(partion_args):
    partitioner = DirichletPartitioner(
                num_partitions=partion_args['num_clients'],
                partition_by=partion_args['partition_target'],
                alpha=partion_args['alpha'],
                seed=partion_args.get('dataset_seed', 42),
                min_partition_size=0)


def execute_partition_and_plot(partion_args):
    fds = FederatedDataset(
        dataset=partion_args['dataset_name'],
        partitioners={
            "train": generate_partitioner(partion_args)
        },
    )

    fig, axes, df_list = plot_label_distributions(
        partitioner=fds.partitioners["train"],
        label_name=partion_args['partition_target'],
        title=f"{partion_args['dataset_name']} - {partion_args['partition_method']} - {partion_args['partition_target']}",
        legend=True,
        verbose_labels=True,
    )
    #plt.show()
    plt.savefig('distribution.png')
    print(df_list)
    # TODO: store also the df_list?

    def apply_transforms_train_cifar10(batch):
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
        
        transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

        key_ = 'img'
        batch[key_] = [transform(img) for img in batch[key_]]
        return batch
    
    def apply_transforms_train_mnist(batch):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
            ]
        )

        key_ = 'image'
        batch[key_] = [transform(img) for img in batch[key_]]
        return batch
    
    def apply_transforms_test_cifar10(batch):
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
        
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize
            ]
        )

        key_ = 'img'
        batch[key_] = [transform(img) for img in batch[key_]]
        return batch
    
    def apply_transforms_test_mnist(batch):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        )

        key_ = 'image'
        batch[key_] = [transform(img) for img in batch[key_]]
        return batch

    # Create train/val for each partition and wrap it into DataLoader
    trainloaders = []
    valloaders = []
    for partition_id in range(partion_args['num_clients']):
        partition = fds.load_partition(partition_id, "train")

        partition = partition.train_test_split(train_size=0.8, seed=42)
        
        if partion_args['dataset_name'] == 'cifar10':
            partition["train"] = partition["train"].with_transform(apply_transforms_train_cifar10)
            partition["test"] = partition["test"].with_transform(apply_transforms_test_cifar10)
        
        if partion_args['dataset_name'] == 'mnist':
            partition["train"] = partition["train"].with_transform(apply_transforms_train_mnist)
            partition["test"] = partition["test"].with_transform(apply_transforms_test_mnist)
        
        trainloaders.append(DataLoader(partition["train"], batch_size=args.batch_size))
        valloaders.append(DataLoader(partition["test"], batch_size=32))
    if partion_args['dataset_name'] == 'cifar10':
        testset = fds.load_split("test").with_transform(apply_transforms_test_cifar10)
    if partion_args['dataset_name'] == 'mnist':
        testset = fds.load_split("test").with_transform(apply_transforms_test_mnist)
    testloader = DataLoader(testset, batch_size=32)
    return trainloaders, valloaders, testloader


if __name__ == '__main__':
    partition_args = {
            'dataset_name': args.dataset_type, ## Change this datasets for any dataset name available in https://huggingface.co/datasets.
            'num_clients': args.num_clients,
            'partition_method': args.partition_method,
            'partition_target': args.partition_target,
            'alpha': args.alpha,    # for dirichlet
            'shard_size' : args.shard_size,
            'num_shards_per_partition' : args.num_shards_per_partition
        }

    trainloaders, valloaders, testloader = execute_partition_and_plot(partition_args) ## there is a single testloader