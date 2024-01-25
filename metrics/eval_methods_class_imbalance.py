import os
os.sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../"))
from glob import glob
import pandas as pd
import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from metrics.evaluate_accuracy import eval
import argparse
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser(description='Evaluate Knowledge Distilation Methods')
parser.add_argument('--test-batch-size', type=int, default=200)
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--data-dir', default='data')
parser.add_argument('--model-path', default='./checkpoints/model_cifar_wrn.pt')
parser.add_argument('--model-dir', default='/data/output/fahad.sarfraz/kd_methods_analysis',)
args = parser.parse_args()

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


def eval(model, device, data_loader, branch=None):

    lst_targets = []
    lst_pred = []

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

            lst_targets.append(target.detach().cpu().numpy())
            lst_pred.append(pred.detach().cpu().numpy())

    loss /= len(data_loader.dataset)

    accuracy = correct / len(data_loader.dataset)
    # print('Accuracy:',  accuracy)
    # print('loss:', loss)

    lst_pred = np.concatenate(lst_pred).reshape(-1).tolist()
    lst_targets = np.concatenate(lst_targets).reshape(-1).tolist()

    df = pd.DataFrame({
        'target': lst_targets,
        'pred': lst_pred
    })

    df['correct'] = df['target'] == df['pred']

    class_acc = df.groupby(by='target').mean()['correct'].to_dict()

    return loss, accuracy, correct, class_acc


# Load Datasets
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

cifar10_testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
cifar100_testset = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform_test)
cifar10_test_loader = torch.utils.data.DataLoader(cifar10_testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
cifar100_test_loader = torch.utils.data.DataLoader(cifar100_testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

# Evaluate Models
lst_teacher = []
lst_student = []
lst_dataset = []
lst_imbalance_gamma = []
lst_seed = []
lst_accuracy = []
lst_model_path = []
lst_kd_method = []
lst_class0 = []
lst_class1 = []
lst_class2 = []
lst_class3 = []
lst_class4 = []
lst_class5 = []
lst_class6 = []
lst_class7 = []
lst_class8 = []
lst_class9 = []

lst_paths = glob(args.model_dir + '/class_imbalance/*/*/*/*/*.pt')
lst_paths += glob(args.model_dir + '/class_imbalance/*/*/*/*/final_model1.pt')  # DML
lst_paths += glob(args.model_dir + '/class_imbalance/*/*/*/*.pt')  # zero shot

lst_methods = ['normal', 'hinton', 'ft', 'fitnet', 'at', 'fsp', 'sp', 'rkd', 'rkd_d', 'bss', 'one', 'dml']

count = 0
for path in lst_paths:

    if 'zeroshot' in path.lower():
        imb_gamma, model_path, seed_path, _ = path.split('/')[-4:]
    else:
        imb_gamma, model_path, seed_path, _, _ = path.split('/')[-5:]

    seed = int(seed_path.split('_')[-1][4:])
    imb_gamma = float(imb_gamma.split('_')[-1])

    kd_method = ''
    for method in lst_methods:
        if method in model_path.split('_'):
            kd_method = method.lower()

    if kd_method == 'ZEROSHOTKT':
        kd_method = 'zeroshot'

    if kd_method == 'rkd':
        if 'rkd_a_' in model_path:
            kd_method = 'rkd_a'
        if 'rkd_d_' in model_path:
            kd_method = 'rkd_d'
        if 'rkd_da_' in model_path:
            kd_method = 'rkd_da'

    if kd_method == 'ft':
        if 'v1' in model_path:
            kd_method += '-v1'
        else:
            kd_method += '-v2'
        if '200epochs' in model_path:
            kd_method += '-200epochs'
        else:
            kd_method += '-350epochs'

    if kd_method == 'normal':
        if 'dropout' in model_path:
            kd_method += '-dropout-0.4'

    if kd_method in ['one', 'dml']:
        teacher, student = 'wrn-16-2', 'wrn-16-2'
    else:
        teacher, student = 'wrn-40-2', 'wrn-16-2'

    # Load model
    print('=============================================================================')

    model = torch.load(path).to(device)
    dataset = 'cifar10'
    model_arch = 'wrn'

    data_loader = cifar10_test_loader

    if kd_method == 'dml':
        _, accuracy1, _, class_acc1 = eval(model, device, data_loader)

        path = path.replace('final_model1.pt', 'final_model2.pt')
        model = torch.load(path).to(device)
        _, accuracy2, _, class_acc2 = eval(model, device, data_loader)

        lst_acc = [accuracy1, accuracy2]
        lst_class_acc = [class_acc1, class_acc2]

        sel_model = np.argmax(lst_acc)
        accuracy = lst_acc[sel_model]
        class_acc = lst_class_acc[sel_model]

    elif kd_method == 'one':
        _, accuracy1, _, class_acc1 = eval(model, device, data_loader, branch=0)
        _, accuracy2, _, class_acc2 = eval(model, device, data_loader, branch=1)
        _, accuracy3, _, class_acc3 = eval(model, device, data_loader, branch=2)

        lst_acc = [accuracy1, accuracy2, accuracy3]
        lst_class_acc = [class_acc1, class_acc2, class_acc3]

        sel_model = np.argmax(lst_acc)
        accuracy = lst_acc[sel_model]
        class_acc = lst_class_acc[sel_model]

    else:

        _, accuracy, _, class_acc = eval(model, device, data_loader)

    print('Model Sucessfully Loaded')
    print('Accuracy:', accuracy)

    lst_teacher.append(teacher)
    lst_student.append(student)
    lst_dataset.append(dataset)
    lst_kd_method.append(kd_method)
    lst_accuracy.append(accuracy)
    lst_seed.append(seed)
    lst_model_path.append(path)
    lst_imbalance_gamma.append(imb_gamma)

    lst_class0.append(class_acc[0])
    lst_class1.append(class_acc[1])
    lst_class2.append(class_acc[2])
    lst_class3.append(class_acc[3])
    lst_class4.append(class_acc[4])
    lst_class5.append(class_acc[5])
    lst_class6.append(class_acc[6])
    lst_class7.append(class_acc[7])
    lst_class8.append(class_acc[8])
    lst_class9.append(class_acc[9])

    data_dict = {
        'dataset': lst_dataset,
        'teacher': lst_teacher,
        'student': lst_student,
        'imbalance_gamma': lst_imbalance_gamma,
        'kd_method': lst_kd_method,
        'seed': lst_seed,
        'accuracy': lst_accuracy,
        'class0': lst_class0,
        'class1': lst_class1,
        'class2': lst_class2,
        'class3': lst_class3,
        'class4': lst_class4,
        'class5': lst_class5,
        'class6': lst_class6,
        'class7': lst_class7,
        'class8': lst_class8,
        'class9': lst_class9,
        'path': lst_model_path,
    }

    df = pd.DataFrame(data_dict).to_csv('kd_class_imbalance.csv', index=False)
    print("'---------------------------------------------------------------")
    count += 1
    print("%s of %s evaluated" % (count, len(lst_paths)))
