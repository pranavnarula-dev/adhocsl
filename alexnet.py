import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet_U_Shape(nn.Module):
    def __init__(self, first_cut=-1, last_cut=-1, class_num=10):
        super(AlexNet_U_Shape, self).__init__()
        self.first_cut = first_cut
        self.last_cut = last_cut
        self.class_num = class_num

        # defining convolutional layers
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

        # defining fully connected layers separately
        self.fc_layers = [
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 4096),  # Adjust this according to the correct input dimensions
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, class_num),
        ]

        # combining all layers
        self.all_layers = self.conv_layers + self.fc_layers

        # initializing model parts as an empty ModuleList
        self.model_parts = nn.ModuleList()

        # calling method to initialize layers based on the cut points
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
            # If we start from fully connected layers, reshape the input
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

def get_AlexNet_split(first_cut, last_cut, class_num=10):
    model_part_a = AlexNet_U_Shape(-1, first_cut, class_num)
    model_part_b = AlexNet_U_Shape(first_cut, last_cut, class_num)
    model_part_c = AlexNet_U_Shape(last_cut, -1, class_num)

    return model_part_a, model_part_b, model_part_c

# Define cuts
first_cut = 2
last_cut = 4

# Instantiate model parts
model_part_a, model_part_b, model_part_c = get_AlexNet_split(first_cut, last_cut, class_num=10)
# print('model_part_a', model_part_a.model_parts)
# print('model_part_b', model_part_b.model_parts)
# print('model_part_c', model_part_c.model_parts)

# Example input
torch.manual_seed(seed=2)
x = torch.randn(1, 3, 32, 32)  # Example input tensor for AlexNet

# Forward pass through each part and print layer information
output_a = model_part_a(x)
print('Output A Shape:', output_a.shape)
output_b = model_part_b(output_a)
print('Output B Shape:', output_b.shape)
output_c = model_part_c(output_b)
print('Output C:', output_c)
