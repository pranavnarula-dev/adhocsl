import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def test(model_fe, model_p, data_loader, device=torch.device("cpu"), two_split=False):
    if isinstance(model_p, list):
        for model in model_p:
            model.eval()
    else:
        model_p.eval()

    if two_split:
        model_fe[0].eval()
        model_fe[1].eval()
    else: 
        model_fe.eval()

    data_loader = data_loader.loader
    test_loss = 0.0
    test_accuracy = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)

            if two_split:
                output1 = model_fe[0](data)
            else:
                output1 = model_fe(data)
            
            if isinstance(model_p, list):
                # In case of multiple server models, use the first one for testing
                output = model_p[0](output1)
            else:
                output = model_p(output1)
            
            if two_split:
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
