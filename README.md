# AdhocSL

This repository contains the implementation of the AdhocSL algorithm, an extension of MergeSFL. AdhocSL enables a U-shaped configuration in which the first and last parts of the model reside on clients, while the intermediate part resides on the server. This setup is particularly effective for split learning scenarios, where data parallelism is achieved with a maximum of 4 helpers. During training, gradients from each client are distributed among the helpers, optimizing the learning process.

The current implementation is tailored for the CIFAR-10 dataset and supports AlexNet and VGG-16 models. However, it is flexible and can be extended to other datasets and models, as inherited from the [MergeSFL implementation](https://github.com/ymliao98/MergeSFL).

## Table of Contents
1. [File Descriptions](#file-descriptions)
2. [Dependencies](#dependencies)
3. [Usage](#usage)
4. [Extending to Other Datasets and Models](#extending-to-other-datasets-and-models)

## File Descriptions

- **`models.py`**: Defines the model architectures, including AlexNet and VGG-16. It also handles model splitting based on the `two_split` command-line argument, allowing flexible distribution of the model between clients and server.

- **`data_utils.py`**: Contains data partitioning classes and functions for loading datasets. This module is crucial for managing how data is distributed among clients and helpers in the split learning setup.

- **`training_utils.py`**: Implements functions for calculating test accuracy and loss during the training process. It also includes utility functions for monitoring and evaluating model performance.

- **`experiment.py`**: The main script for executing experiments. This file integrates the model, data, and training utilities to run the AdhocSL algorithm on the CIFAR-10 dataset.

- **`requirements.txt`**: Lists all the Python packages and dependencies needed to run the code. These can be installed using the command provided in the [Dependencies](#dependencies) section.

## Dependencies

To install the necessary packages, ensure you have Python installed and then run:

```bash
pip install -r requirements.txt
```

This command will install all the required packages, ensuring that the environment is set up correctly for running the AdhocSL algorithm.

## Usage

### Clone the repository:

```bash
git clone https://github.com/yourusername/AdhocSL.git
cd AdhocSL
```

### Install dependencies:
```bash
pip install -r requirements.txt
```
### Run experiments:
Execute the main script with appropriate arguments to run the AdhocSL algorithm:
```bash
python experiment.py --two_split --data_pattern 8 --use_flower --model AlexNet --dataset CIFAR10
```
Replace AlexNet and CIFAR10 with VGG16 or other supported models and datasets as needed.

## Extending to Other Datasets and Models

The current setup is optimized for the CIFAR-10 dataset with AlexNet and VGG-16 models. To extend this to other datasets and models:

- **Data:** Modify the `data_utils.py` file to include new data loaders and partitioning schemes.
- **Models:** Update `models.py` with the architecture of the new model and ensure proper splitting based on the U-shaped configuration.
- **Training:** Adjust the `training_utils.py` for any specific requirements related to the new data or model.
