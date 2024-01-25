from __future__ import print_function
import torch
import torch.nn.functional as F


def eval(model, device, data_loader, branch=None):
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            if branch is not None:
                output = model(data)[branch]
            else:
                output = model(data)
            loss += F.cross_entropy(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    loss /= len(data_loader.dataset)

    accuracy = correct / len(data_loader.dataset)
    print('Accuracy:',  accuracy)
    print('loss:', loss)

    return loss, accuracy, correct
