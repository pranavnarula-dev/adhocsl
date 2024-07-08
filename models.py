import torch
import torch.nn as nn
import torch.nn.functional as F

def create_model_instance_SL(dataset_type, model_type, worker_num, class_num=10, m=4):
    client_nets = {net_i: None for net_i in range(worker_num)}

    if dataset_type == 'CIFAR10':
        server = AlexNet_U_Shape(3, -1, class_num, m=m)
        for net_i in range(worker_num):
            net = AlexNet_U_Shape(0, 3, class_num, m=m)
            client_nets[net_i] = net
        return client_nets, server

    elif dataset_type == 'image100':
        server = VGG16_U_Shape(3, -1)
        for net_i in range(worker_num):
            net = VGG16_U_Shape(0, 3)
            client_nets[net_i] = net
        return client_nets, server

    elif dataset_type == 'UCIHAR':
        server = CNN_HAR_U_Shape(2, -1)
        for net_i in range(worker_num):
            net = CNN_HAR_U_Shape(0, 2)
            client_nets[net_i] = net
        return client_nets, server

    elif dataset_type == 'SPEECH':
        server = M5_U_Shape(1, -1)
        for net_i in range(worker_num):
            net = M5_U_Shape(0, 1)
            client_nets[net_i] = net
        return client_nets, server

def create_model_instance_SL_two_splits(dataset_type, model_type, worker_num, class_num=10, m=4):
    client_nets = {net_i: None for net_i in range(worker_num)}

    if dataset_type == 'CIFAR10':
        server = AlexNet_U_Shape(1, 8, class_num, m=m)
        for net_i in range(worker_num):
            net = (AlexNet_U_Shape(0, 1, class_num, m=m), AlexNet_U_Shape(8, -1, class_num, m=m))
            client_nets[net_i] = net
        return client_nets, server

    elif dataset_type == 'image100':
        server = VGG16_U_Shape(3, 6)
        for net_i in range(worker_num):
            net = (VGG16_U_Shape(0, 3), VGG16_U_Shape(6, -1))
            client_nets[net_i] = net
        return client_nets, server

    elif dataset_type == 'UCIHAR':
        server = CNN_HAR_U_Shape(1, 6)
        for net_i in range(worker_num):
            net = (CNN_HAR_U_Shape(0, 1), CNN_HAR_U_Shape(6, -1))
            client_nets[net_i] = net
        return client_nets, server

    elif dataset_type == 'SPEECH':
        server = M5_U_Shape(1, 2)
        for net_i in range(worker_num):
            net = (M5_U_Shape(0, 1), M5_U_Shape(2, -1))
            client_nets[net_i] = net
        return client_nets, server

class AlexNet_U_Shape(nn.Module):
    def __init__(self, first_cut=-1, last_cut=-1, class_num=10, m=4):
        super(AlexNet_U_Shape, self).__init__()
        self.first_cut = first_cut
        self.last_cut = last_cut
        self.class_num = class_num

        self.conv_layers = [
            nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),
            nn.Sequential(
                nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),
            nn.Sequential(
                nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        ]

        self.fc_layers = [
            nn.Dropout(),
            nn.Linear(256 * m * m, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, class_num),
        ]

        self.all_layers = self.conv_layers + self.fc_layers
        self.model_parts = nn.ModuleList()
        self._initialize_layers()

    def _initialize_layers(self):
        if self.first_cut == -1:
            self.first_cut = 0
        if self.last_cut == -1:
            self.last_cut = len(self.all_layers)

        for i in range(self.first_cut, self.last_cut):
            self.model_parts.append(self.all_layers[i])

    def forward(self, x):
        if self.first_cut > len(self.conv_layers):
            x = x.view(x.size(0), -1)
        
        for layer in self.model_parts:
            if isinstance(layer, nn.Sequential):
                for sublayer in layer:
                    if isinstance(sublayer, nn.Linear):
                        if x.dim() > 2:
                            x = x.view(x.size(0), -1)
                    x = sublayer(x)
            else:
                if isinstance(layer, nn.Linear) and x.dim() > 2:
                    x = x.view(x.size(0), -1)
                x = layer(x)
                
        if self.last_cut == len(self.all_layers):
            x = F.log_softmax(x, dim=1)
        return x

class VGG16_U_Shape(nn.Module):
    def __init__(self, first_cut=-1, last_cut=-1, class_num=100):
        super(VGG16_U_Shape, self).__init__()
        self.first_cut = first_cut
        self.last_cut = last_cut

        self.conv_layers = [
            nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            ),
            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            ),
            nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2, padding=1),
            ),
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            )
        ]

        self.fc_layers = [
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, class_num),
        ]

        self.all_layers = self.conv_layers + self.fc_layers
        self.model_parts = nn.ModuleList()
        self._initialize_layers()

    def _initialize_layers(self):
        if self.first_cut == -1:
            self.first_cut = 0
        if self.last_cut == -1:
            self.last_cut = len(self.all_layers)

        for i in range(self.first_cut, self.last_cut):
            self.model_parts.append(self.all_layers[i])

    def forward(self, x):
        if self.first_cut > len(self.conv_layers):
            x = x.view(x.size(0), -1)
        
        for layer in self.model_parts:
            if isinstance(layer, nn.Sequential):
                for sublayer in layer:
                    if isinstance(sublayer, nn.Linear):
                        if x.dim() > 2:
                            x = x.view(x.size(0), -1)
                    x = sublayer(x)
            else:
                if isinstance(layer, nn.Linear) and x.dim() > 2:
                    x = x.view(x.size(0), -1)
                x = layer(x)
                
        if self.last_cut == len(self.all_layers):
            x = F.log_softmax(x, dim=1)
        return x

class CNN_HAR_U_Shape(nn.Module):
    def __init__(self, first_cut=-1, last_cut=-1):
        super(CNN_HAR_U_Shape, self).__init__()
        self.first_cut = first_cut
        self.last_cut = last_cut

        self.conv_layers = [
            nn.Sequential(
                nn.Conv2d(1, 12, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),
            nn.Sequential(
                nn.Conv2d(12, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),
            nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
            )
        ]

        self.fc_layers = [
            nn.Linear(64 * 32 * 2, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 6)
        ]

        self.all_layers = self.conv_layers + self.fc_layers
        self.model_parts = nn.ModuleList()
        self._initialize_layers()

    def _initialize_layers(self):
        if self.first_cut == -1:
            self.first_cut = 0
        if self.last_cut == -1:
            self.last_cut = len(self.all_layers)

        for i in range(self.first_cut, self.last_cut):
            self.model_parts.append(self.all_layers[i])

    def forward(self, x):
        if self.first_cut > len(self.conv_layers):
            x = x.view(x.size(0), -1)
        
        for layer in self.model_parts:
            if isinstance(layer, nn.Sequential):
                for sublayer in layer:
                    if isinstance(sublayer, nn.Linear):
                        if x.dim() > 2:
                            x = x.view(x.shape[0], -1)
                    x = sublayer(x)
            else:
                if isinstance(layer, nn.Linear) and x.dim() > 2:
                    x = x.view(x.shape[0], -1)
                x = layer(x)
                
        if self.last_cut == len(self.all_layers):
            x = F.log_softmax(x, dim=1)
        return x

class M5_U_Shape(nn.Module):
    def __init__(self, first_cut=-1, last_cut=-1, n_input=1, n_output=35, n_channel=32):
        super(M5_U_Shape, self).__init__()
        self.first_cut = first_cut
        self.last_cut = last_cut

        self.conv_layers = [
            nn.Sequential(
                nn.Conv1d(n_input, n_channel, kernel_size=80, stride=16),
                nn.BatchNorm1d(n_channel),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(4),
            ),
            nn.Sequential(
                nn.Conv1d(n_channel, n_channel, kernel_size=3),
                nn.BatchNorm1d(n_channel),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(4),
            ),
            nn.Sequential(
                nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3),
                nn.BatchNorm1d(2 * n_channel),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(4),
            )
        ]

        self.fc_layers = [
            nn.Linear(2 * n_channel, n_output)
        ]

        self.all_layers = self.conv_layers + self.fc_layers
        self.model_parts = nn.ModuleList()
        self._initialize_layers()

    def _initialize_layers(self):
        if self.first_cut == -1:
            self.first_cut = 0
        if self.last_cut == -1:
            self.last_cut = len(self.all_layers)

        for i in range(self.first_cut, self.last_cut):
            self.model_parts.append(self.all_layers[i])

    def forward(self, x):
        if self.first_cut > len(self.conv_layers):
            x = x.view(x.size(0), -1)
        
        for layer in self.model_parts:
            if isinstance(layer, nn.Sequential):
                for sublayer in layer:
                    if isinstance(sublayer, nn.Linear):
                        x = F.avg_pool1d(x, x.shape[-1])
                        x = x.permute(0, 2, 1)
                        x = sublayer(x)
                    else:
                        x = sublayer(x)
            else:
                if isinstance(layer, nn.Linear):
                    x = F.avg_pool1d(x, x.shape[-1])
                    x = x.permute(0, 2, 1)
                    x = layer(x)
                else:
                    x = layer(x)
                
        if self.last_cut == len(self.all_layers):
            x = F.log_softmax(x, dim=2).squeeze()
        return x
