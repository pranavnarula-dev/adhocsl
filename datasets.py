import random
import numpy as np
import torch
import os
from torch.utils.data import DataLoader, TensorDataset, Sampler
from torchvision import datasets, transforms
#from torchaudio.datasets import SPEECHCOMMANDS

class SmallDatasetSampler(Sampler):
    def __init__(self, data_source, num_samples):
        self.data_source = data_source
        self.num_samples = num_samples

    def __iter__(self):
        n = len(self.data_source)
        if n == 0:
            return iter([])
        return (self.data_source[i % n] for i in torch.randperm(self.num_samples))

    def __len__(self):
        return self.num_samples
    
class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

class DataLoaderHelper(object):
    def __init__(self, dataloader):
        self.loader = dataloader
        self.dataiter = iter(self.loader)

    def __next__(self):
        try:
            data, target = next(self.dataiter)
        except StopIteration:
            self.dataiter = iter(self.loader)
            data, target = next(self.dataiter)
        
        return data, target

class RandomPartitioner(object):

    def __init__(self, data, partition_sizes, seed=2020):
        self.data = data
        self.partitions = []
        rng = random.Random()
        rng.seed(seed)

        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in partition_sizes:
            part_len = round(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        selected_idxs = self.partitions[partition]

        return selected_idxs
    
    def __len__(self):
        return len(self.data)


class LabelwisePartitioner(object):

    def __init__(self, data, partition_sizes, class_num=10, labels=None, seed=2020):
        # sizes is a class_num * vm_num matrix
        self.data = data
        self.partitions = [list() for _ in range(len(partition_sizes[0]))]
        rng = random.Random()
        rng.seed(seed)

        label_indexes = list()
        class_len = list()
        # label_indexes includes class_num lists. Each list is the set of indexs of a specific class
        if hasattr(data, 'classes'):
            # focus here
            for class_idx in range(len(data.classes)):
                label_indexes.append(list(np.where(np.array(data.targets) == class_idx)[0]))
                class_len.append(len(label_indexes[class_idx]))
                rng.shuffle(label_indexes[class_idx])
        elif hasattr(data, 'labels'):
            for class_idx in range(class_num): 
                label_indexes.append(list(np.where(np.array(data.labels) == class_idx)[0]))
                class_len.append(len(label_indexes[class_idx]))
                rng.shuffle(label_indexes[class_idx])
        else:
            label_indexes = [list() for _ in range(class_num)]
            class_len = [0] * class_num
            # label_indexes includes class_num lists. Each list is the set of indexs of a specific class
            for i, j in enumerate(data):
                if labels:
                    class_idx = labels.index(j[2])
                else:
                    class_idx = int(j[1])
                class_len[class_idx] += 1
                label_indexes[class_idx].append(i)
            # print(class_size)
            for i in range(class_num):
                rng.shuffle(label_indexes[i])
        
        # distribute class indexes to each vm according to sizes matrix
        try:
            for class_idx in range(len(data.classes)):
                begin_idx = 0
                for vm_idx, frac in enumerate(partition_sizes[class_idx]):
                    end_idx = begin_idx + round(frac * class_len[class_idx])
                    end_idx = int(end_idx)
                    self.partitions[vm_idx].extend(label_indexes[class_idx][begin_idx:end_idx])
                    begin_idx = end_idx
        except AttributeError:
            for class_idx in range(class_num):
                begin_idx = 0
                for vm_idx, frac in enumerate(partition_sizes[class_idx]):
                    end_idx = begin_idx + round(frac * class_len[class_idx])
                    end_idx = int(end_idx)
                    self.partitions[vm_idx].extend(label_indexes[class_idx][begin_idx:end_idx])
                    begin_idx = end_idx

    def use(self, partition):
        selected_idxs = self.partitions[partition]

        return selected_idxs
    
    def __len__(self):
        return len(self.data)


def create_dataloaders(dataset, batch_size, selected_idxs=None, shuffle=True, pin_memory=True, num_workers=4, drop_last=False, collate_fn=None):
    if selected_idxs is None:
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers, drop_last=drop_last, collate_fn=collate_fn)
    else:
        partition = Partition(dataset, selected_idxs)
        
        # Use custom sampler if the partition is small
        if len(partition) < batch_size * 10:  # Arbitrary threshold, adjust as needed
            sampler = SmallDatasetSampler(range(len(partition)), num_samples=batch_size * 42)  # 42 is the number of local steps
            dataloader = DataLoader(partition, batch_size=batch_size, sampler=sampler,
                                    pin_memory=pin_memory, num_workers=num_workers, drop_last=drop_last, collate_fn=collate_fn)
        else:
            dataloader = DataLoader(partition, batch_size=batch_size,
                                    shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers, drop_last=drop_last, collate_fn=collate_fn)
    
    return DataLoaderHelper(dataloader)

# def create_dataloaders(dataset, batch_size, selected_idxs=None, shuffle=True, pin_memory=True, num_workers=4, drop_last=False, collate_fn=None):
#     if selected_idxs == None:
#         dataloader = DataLoader(dataset, batch_size=batch_size,
#                                     shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers, drop_last=drop_last, collate_fn=collate_fn)
#     else:
#         partition = Partition(dataset, selected_idxs)
#         dataloader = DataLoader(partition, batch_size=batch_size,
#                                     shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers, drop_last=drop_last, collate_fn=collate_fn)
    
#     return DataLoaderHelper(dataloader)


def load_datasets(dataset_type, data_path="./data/"):
    
    train_transform = load_default_transform(dataset_type, train=True)
    test_transform = load_default_transform(dataset_type, train=False)

    if dataset_type == 'CIFAR10':
        train_dataset = datasets.CIFAR10(data_path, train = True, 
                                            download = True, transform=train_transform)
        test_dataset = datasets.CIFAR10(data_path, train = False, 
                                            download = True, transform=test_transform)

    elif dataset_type == 'CIFAR100':
        train_dataset = datasets.CIFAR100(data_path, train = True,
                                            download = True, transform=train_transform)
        test_dataset = datasets.CIFAR100(data_path, train = False, 
                                            download = True, transform=test_transform)

    elif dataset_type == 'FashionMNIST':
        train_dataset = datasets.FashionMNIST(data_path, train = True, 
                                            download = True, transform=train_transform)
        test_dataset = datasets.FashionMNIST(data_path, train = False, 
                                            download = True, transform=test_transform)

    elif dataset_type == 'MNIST':
        train_dataset = datasets.MNIST(data_path, train = True, 
                                            download = True, transform=train_transform)
        test_dataset = datasets.MNIST(data_path, train = False, 
                                            download = True, transform=train_transform)
    
    elif dataset_type == 'SVHN':
        train_dataset = datasets.SVHN(data_path+'/SVHN_data', split='train',
                                            download = True, transform=train_transform)
        test_dataset = datasets.SVHN(data_path+'/SVHN_data', split='test', 
                                            download = True, transform=train_transform)
    # elif dataset_type == 'EMNIST':
    #     train_dataset = datasets.ImageFolder('/data/ymliao/data/emnist/byclass_train', transform = train_transform)
    #     test_dataset = datasets.ImageFolder('/data/ymliao/data/emnist/byclass_test', transform = train_transform)
    elif dataset_type == 'EMNIST':
        train_dataset = datasets.EMNIST(data_path, split = 'byclass', train = True, download = True, transform=train_transform)
        test_dataset = datasets.EMNIST(data_path, split = 'byclass', train = False, transform=train_transform)

    elif dataset_type == 'tinyImageNet':
        train_dataset = datasets.ImageFolder('/data1/ymliao/data/tiny-imagenet-200/train', transform = train_transform)
        test_dataset = datasets.ImageFolder('/data1/ymliao/data/tiny-imagenet-200/val', transform = train_transform)

    elif dataset_type == 'image100':
        train_dataset = datasets.ImageFolder('/data/zpsun/data/IMAGE100/train', transform = train_transform)
        test_dataset = datasets.ImageFolder('/data/zpsun/data/IMAGE100/test', transform = train_transform)
    
    elif dataset_type == 'SPEECH':
        train_dataset = SubsetSC("training")
        test_dataset = SubsetSC("testing")

    elif dataset_type == 'UCIHAR':
        INPUT_SIGNAL_TYPES = ["body_acc_x_","body_acc_y_","body_acc_z_",
        "body_gyro_x_","body_gyro_y_","body_gyro_z_",
        "total_acc_x_","total_acc_y_","total_acc_z_"]

        # Output classes to learn how to classify
        LABELS = ["WALKING","WALKING_UPSTAIRS","WALKING_DOWNSTAIRS",
            "SITTING","STANDING","LAYING"]
        
        base_path = os.path.join(os.getcwd(), 'data', 'UCIHAR')

        X_train_signals_paths = [os.path.join(base_path, "train/Inertial_Signals", signal + "train.txt") for signal in INPUT_SIGNAL_TYPES]
        X_test_signals_paths = [os.path.join(base_path, "test/Inertial_Signals", signal + "test.txt") for signal in INPUT_SIGNAL_TYPES]
        Y_train_path = os.path.join(base_path, "train/y_train.txt")
        Y_test_path = os.path.join(base_path, "test/y_test.txt")

        def load_X(X_signals_paths):
            X_signals = []
            for signal_type_path in X_signals_paths:
                file = open(signal_type_path, 'r')
                X_signals.append([np.array(serie, dtype=np.float32) for serie in [
                        row.replace('  ', ' ').strip().split(' ') for row in file]])
                file.close()
            return np.transpose(np.array(X_signals), (1, 2, 0))
        
        def load_Y(y_path):
            file = open(y_path, 'r')
            y_ = np.array([elem for elem in [
                    row.replace('  ', ' ').strip().split(' ') for row in file
                ]],
                dtype=np.int32
            )
            file.close()
            return y_ - 1

        X_train = load_X(X_train_signals_paths)
        X_test = load_X(X_test_signals_paths)
        y_train = load_Y(Y_train_path)
        y_test = load_Y(Y_test_path)
        train_dataset = TensorDataset(torch.from_numpy(X_train.reshape(-1, 1, X_train.shape[1], X_train.shape[2])), torch.from_numpy(y_train.reshape(-1)))
        test_dataset = TensorDataset(torch.from_numpy(X_test.reshape(-1, 1, X_test.shape[1], X_test.shape[2])), torch.from_numpy(y_test.reshape(-1)))

    return train_dataset, test_dataset


def load_default_transform(dataset_type, train=False):
    if dataset_type == 'CIFAR10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
        if train:
            dataset_transform = transforms.Compose([
                           transforms.RandomHorizontalFlip(),
                           transforms.RandomCrop(32, 4),
                           transforms.ToTensor(),
                           normalize
                         ])
        else:
            dataset_transform = transforms.Compose([
                            transforms.ToTensor(),
                            normalize
                            ])

    elif dataset_type == 'CIFAR100':
        # reference:https://github.com/weiaicunzai/pytorch-cifar100/blob/master/utils.py
        normalize = transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), 
                                                    (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
        if train:
            dataset_transform = transforms.Compose([
                                transforms.RandomCrop(32, 4),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(15),
                                transforms.ToTensor(),
                                normalize
                            ])
        else:
            dataset_transform = transforms.Compose([
                            transforms.ToTensor(),
                            normalize
                            ])

    elif dataset_type == 'FashionMNIST':
          dataset_transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                         ])
    
    elif dataset_type == 'MNIST':
          dataset_transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                         ])

    elif dataset_type == 'SVHN':
        dataset_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])

    elif dataset_type == 'EMNIST':
        dataset_transform =  transforms.Compose([
                           transforms.Resize(28),
                           #transforms.CenterCrop(227),
                           transforms.Grayscale(1),
                           transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))  
                        ])
    
    elif dataset_type == 'tinyImageNet':
        dataset_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ])

    elif dataset_type == 'image100':
        dataset_transform = transforms.Compose([transforms.Resize((144,144)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                                ])
    else:
        dataset_transform=None

    return dataset_transform


def load_customized_transform(dataset_type):
    if dataset_type == 'CIFAR10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
        dataset_transform = transforms.Compose([
                           transforms.RandomHorizontalFlip(),
                           transforms.RandomCrop(32, 4),
                           transforms.ToTensor(),
                           normalize
                         ])

    elif dataset_type == 'CIFAR100':
          dataset_transform = transforms.Compose([
                           transforms.RandomHorizontalFlip(1.0),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                         ])

    elif dataset_type == 'FashionMNIST':
          dataset_transform = transforms.Compose([
                           transforms.RandomHorizontalFlip(1.0),
                           transforms.ToTensor()
                         ])
    
    elif dataset_type == 'MNIST':
          dataset_transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                         ])
                       
    return dataset_transform


class SubsetSC(): #SPEECHCOMMANDS
    def __init__(self, subset=None, partition=None):
        super().__init__("/data/ymliao/data/speech/", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            if partition is not None:
                tmp_walker = [w for w in self._walker if w not in excludes]
                self._walker = [w for idx, w in enumerate(tmp_walker) if idx in partition]
            else:
                self._walker = [w for w in self._walker if w not in excludes]


def collate_fn(batch, labels):

    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [torch.tensor(labels.index(label))]

    def pad_sequence(batch):
        # Make all tensor in a batch the same length by padding with zeros
        batch = [item.t() for item in batch]
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
        return batch.permute(0, 2, 1)

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets


def plot_data_distribution(train_dataset, train_data_partition, dataset_type, data_pattern, initial_workers):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # Prepare data
    class_distribution = []
    total_samples = len(train_dataset)
    max_samples_per_client = 0

    for worker_idx in range(initial_workers):
        worker_data = train_data_partition.use(worker_idx)
        if isinstance(worker_data, list):
            labels = [train_dataset.targets[idx] for idx in worker_data]
        else:
            labels = [sample[1] for sample in worker_data]
        
        unique, counts = np.unique(labels, return_counts=True)
        worker_samples = len(worker_data)
        max_samples_per_client = max(max_samples_per_client, worker_samples)

        for class_idx in range(len(train_dataset.classes)):
            count = counts[np.where(unique == class_idx)[0][0]] if class_idx in unique else 0
            class_distribution.append({
                'Client': f'Client {worker_idx}',  # Changed to start from 0
                'Class': train_dataset.classes[class_idx],
                'Count': count,
                'Total': worker_samples
            })

    # Create DataFrame
    df = pd.DataFrame(class_distribution)

    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f'{dataset_type} Distribution (Pattern {data_pattern})', fontsize=16)

    # Stacked bar plot
    df_pivot = df.pivot(index='Client', columns='Class', values='Count')
    df_pivot = df_pivot.reindex(sorted(df_pivot.index, key=lambda x: int(x.split()[-1])))  # Sort clients numerically
    df_pivot.plot(kind='bar', stacked=True, ax=ax1)
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Class Distribution per Client')
    ax1.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Total samples bar plot
    client_totals = df.groupby('Client')['Count'].sum().sort_index()
    client_totals.plot(kind='bar', ax=ax2, color='skyblue')
    ax2.set_ylabel('Total Samples')
    ax2.set_title('Total Samples per Client')
    
    # Percentage text on total samples bars
    for i, v in enumerate(client_totals):
        ax2.text(i, v, f'{v/total_samples:.2%}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f'{dataset_type}_distribution_pattern_{data_pattern}_workers_{initial_workers}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Print detailed distribution information
    print(f"\nDetailed class distribution for {dataset_type} (Pattern {data_pattern}):")
    for worker_idx in range(initial_workers):
        worker_dist = df[df['Client'] == f'Client {worker_idx}']
        worker_samples = worker_dist['Total'].iloc[0]
        print(f"\nClient {worker_idx}: {worker_samples} samples ({worker_samples/total_samples:.2%} of total data)")
        for _, row in worker_dist.iterrows():
            print(f"  {row['Class']}: {row['Count']} ({row['Count']/worker_samples:.2%})")
