import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def test(model_fe, model_p, data_loader, device=torch.device("cpu")):
    model_p.eval()
    model_fe[0].eval()
    model_fe[1].eval()
    data_loader = data_loader.loader
    test_loss = 0.0
    test_accuracy = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:

            data, target = data.to(device), target.to(device)

            output1 = model_fe[0](data)
            output = model_p(output1)
            output = model_fe[1](output)

            # sum up batch loss
            loss_func = nn.CrossEntropyLoss(reduction='sum') 
            test_loss += loss_func(output, target.long()).item()

            pred = output.argmax(1, keepdim=True)
            batch_correct = pred.eq(target.view_as(pred)).sum().item()

            correct += batch_correct
            
    test_loss /= len(data_loader.dataset)
    test_accuracy = np.float64(1.0 * correct / len(data_loader.dataset))
    return test_loss, test_accuracy

