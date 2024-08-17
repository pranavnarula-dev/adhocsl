This code implements MergeSFL along with its extension in the form of the proposed algorithm, AdhocSL. The current state of the algorithm allows a U-shape configuration to the model with the first and last part residing on the clients
whereas the intermediate part resides on the server. The two split solution also allows for data parallelism with a maximum of 4 helpers. In this case the gradients from each client are split and sent to each of the helpers in the training process.

The code is currently tuned to work with the CIFAR-10 dataset with AlexNet and VGG-16 models although it is extendable to a couple of other data sources and corresponding models as taken from the [MergeSFL implementation](https://github.com/ymliao98/MergeSFL).

The models.py file has the model descriptions and performs appropriate splitting based on the two_split command-line argument.
The data_utils.py file holds the data partitionining classes and the dataloader functions.
The test accuracy and loss calculation occurs in the training_utils.py file.
The main code exectutes from the experiment.py file.

