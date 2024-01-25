import os
os.sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../"))
from glob import glob
import pandas as pd
import torch
import torchvision
from torchvision import transforms
from metrics.evaluate_accuracy import eval
import argparse
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
lst_label_corr = []
lst_seed = []
lst_accuracy = []
lst_model_path = []
lst_kd_method = []

lst_paths = glob(args.model_dir + '/label_noise/*/*/*/*/*.pt')
lst_paths += glob(args.model_dir + '/label_noise/*/*/*/*/final_model1.pt')  # DML
lst_paths += glob(args.model_dir + '/label_noise/*/*/*/*.pt')  # zero shot

lst_methods = ['normal', 'hinton', 'ft', 'fitnet', 'at', 'fsp', 'sp', 'rkd', 'rkd_d', 'bss', 'one', 'ZEROSHOTKT', 'dml']

count = 0
for path in lst_paths:
    print(path)
    if 'zeroshot' in path.lower():
        label_noise, model_path, seed_path, _ = path.split('/')[-4:]
    else:
        label_noise, model_path, seed_path, _, _ = path.split('/')[-5:]

    seed = int(seed_path.split('_')[-1][4:])
    label_noise = float(label_noise.split('_')[-1])

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
        elif 'v2' in model_path:
            kd_method += '-v2'
        else:
            kd_method += '-v3'
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

    dataset = 'cifar10'
    model_arch = 'wrn'

    # Load model
    print('=============================================================================')
    print('Path:', path)
    print('Method:', kd_method)
    print('Dataset:', dataset)

    model = torch.load(path).to(device)
    data_loader = cifar10_test_loader

    if kd_method == 'dml':
        _, accuracy1, _ = eval(model, device, data_loader)

        path = path.replace('final_model1.pt', 'final_model2.pt')
        model = torch.load(path).to(device)
        _, accuracy2, _ = eval(model, device, data_loader)

        accuracy = max(accuracy1, accuracy2)

    if kd_method == 'one':
        _, accuracy1, _ = eval(model, device, data_loader, branch=0)
        _, accuracy2, _ = eval(model, device, data_loader, branch=1)
        _, accuracy3, _ = eval(model, device, data_loader, branch=2)

        accuracy = max(accuracy1, accuracy2, accuracy3)
    else:

        _, accuracy, _ = eval(model, device, data_loader)

    print('Model Sucessfully Loaded')
    print('Accuracy:', accuracy)

    lst_teacher.append(teacher)
    lst_student.append(student)
    lst_dataset.append(dataset)
    lst_kd_method.append(kd_method)
    lst_accuracy.append(accuracy)
    lst_seed.append(seed)
    lst_model_path.append(path)
    lst_label_corr.append(label_noise/100)

    data_dict = {
        'dataset': lst_dataset,
        'teacher': lst_teacher,
        'student': lst_student,
        'label_noise': lst_label_corr,
        'kd_method': lst_kd_method,
        'seed': lst_seed,
        'accuracy': lst_accuracy,
        'path': lst_model_path
    }

    df = pd.DataFrame(data_dict).to_csv('kd_label_corruption.csv', index=False)
    print("'---------------------------------------------------------------")
    count += 1
    print("%s of %s evaluated" % (count, len(lst_paths)))
