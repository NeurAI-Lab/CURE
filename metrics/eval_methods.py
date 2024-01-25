import os
os.sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../"))
from glob import glob
import pandas as pd
import numpy as np
import torch
import torchvision
from torchvision import transforms
from metrics.evaluate_accuracy import eval
import argparse
from analysis.avg_entropy import eval_entropy
from analysis.calibration import eval_calibration
from analysis.logit_dist import eval_logits
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
svhn_testset = torchvision.datasets.SVHN(root=args.data_dir, train=False, download=True, transform=transform_test)
cifar10_test_loader = torch.utils.data.DataLoader(cifar10_testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
cifar100_test_loader = torch.utils.data.DataLoader(cifar100_testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

# Evaluate Models
lst_model_arch = []
lst_teacher = []
lst_student = []
lst_dataset = []
lst_seed = []
lst_accuracy = []
lst_ece = []
lst_avg_entropy = []
lst_avg_logit = []
lst_model_path = []
lst_identifier = []
lst_kd_method = []


lst_paths = glob(args.model_dir + '/clean_data/*/*/*/*/*/*/*.pt')
lst_paths += glob(args.model_dir + '/clean_data/*/*/*/*/*/*/final_model1.pt')  # for DML
lst_paths += glob(args.model_dir + '/clean_data/*/*/*/*/*/*.pt')  # For zeroshot

# lst_paths = glob(args.model_dir + '/clean_data/wrn/cifar10/teacher_wrn_40_2_student_wrn_16_2_cifar10/*/*/*/*.pt')
# lst_paths += glob(args.model_dir + '/clean_data/wrn/cifar10/teacher_wrn_40_2_student_wrn_16_2_cifar10/*/*/*/final_model1.pt')
# lst_paths += glob(args.model_dir + '/clean_data/wrn/cifar10/teacher_wrn_40_2_student_wrn_16_2_cifar10/*/*/*.pt')

lst_methods = ['normal', 'hinton', 'ft', 'fitnet', 'at', 'fsp', 'sp', 'rkd', 'rkd', 'rkd', 'bss', 'one', 'ZEROSHOTKT', 'dml']

count = 0

for path in lst_paths:

    if 'zeroshot' in path.lower():
        model_arch, dataset, identifier, model_path, seed_path, _ = path.split('/')[-6:]
    else:
        model_arch, dataset, identifier, model_path, seed_path, _, _ = path.split('/')[-7:]

    seed = int(seed_path.split('_')[-1][4:])

    kd_method = ''
    for method in lst_methods:
        if method in model_path.split('_'):
            kd_method = method.lower()

    if kd_method == 'ZEROSHOTKT':
        kd_method = 'zeroshot'

    if kd_method == 'rkd':
        kd_method = '_'.join(model_path.split('_')[:2])

    if kd_method == 'normal':
        if 'dropout' in model_path:
            kd_method += '-dropout-0.4'

    if 'wrn' in identifier:
        _, teacher_arch, teacher_width, teacher_depth, _, student_arch, student_width, student_depth, _ = identifier.split('_')
        teacher = '-'.join([teacher_arch, teacher_width, teacher_depth])
        student = '-'.join([student_arch, student_width, student_depth])
    else:
        _, teacher, _, student, _ = identifier.split('_')

    # Load model
    print('=============================================================================')
    print('Path:', path)
    print('Method:', kd_method)
    print('Dataset:', dataset)

    model = torch.load(path).to(device)
    if dataset == 'cifar10':
        data_loader = cifar10_test_loader
    elif dataset == 'cifar100':
        data_loader = cifar100_test_loader
    else:
        raise ValueError('Incorrect Dataset %s' % dataset)

    plot_path = os.path.join('calib_plots', seed_path + '.png')
    logit_path = os.path.join('logit_dist', seed_path + '.png')

    if kd_method == 'dml':
        _, accuracy1, _ = eval(model, device, data_loader)

        path = path.replace('final_model1.pt', 'final_model2.pt')
        model2 = torch.load(path).to(device)
        _, accuracy2, _ = eval(model2, device, data_loader)

        lst_acc = [accuracy1, accuracy2]
        sel_model = np.argmax(lst_acc)
        accuracy = lst_acc[sel_model]

        if sel_model == 1:
            model = model2

        loss, avg_entropy = eval_entropy(model, device, data_loader)
        ece = eval_calibration(model, device, data_loader, plot_path)
        avg_logit = eval_logits(model, device, data_loader, logit_path)

    elif kd_method == 'one':
        print(dataset)
        _, accuracy1, _ = eval(model, device, data_loader, branch=0)
        _, accuracy2, _ = eval(model, device, data_loader, branch=1)
        _, accuracy3, _ = eval(model, device, data_loader, branch=2)

        lst_acc = [accuracy1, accuracy2, accuracy3]

        sel_branch = np.argmax(lst_acc)
        accuracy = lst_acc[sel_branch]

        # eval avg entropy and ece
        loss, avg_entropy = eval_entropy(model, device, data_loader, branch=sel_branch)
        ece = eval_calibration(model, device, data_loader, plot_path, branch=sel_branch)
        avg_logit = eval_logits(model, device, data_loader, logit_path, branch=sel_branch)

    else:
        _, accuracy, _ = eval(model, device, data_loader)

        # eval avg entropy and ece
        loss, avg_entropy = eval_entropy(model, device, data_loader)
        ece = eval_calibration(model, device, data_loader, plot_path)
        avg_logit = eval_logits(model, device, data_loader, logit_path)

    print('Model Sucessfully Loaded')
    print('Accuracy:', accuracy)

    lst_model_arch.append(model_arch)
    lst_teacher.append(teacher)
    lst_student.append(student)
    lst_dataset.append(dataset)
    lst_kd_method.append(kd_method)
    lst_accuracy.append(accuracy)
    lst_seed.append(seed)
    lst_model_path.append(path)
    lst_identifier.append(identifier)
    lst_avg_logit.append(avg_logit)

    # ece and avg entropy
    lst_ece.append(ece)
    lst_avg_entropy.append(avg_entropy)

    data_dict = {
        'dataset': lst_dataset,
        'network_arch': lst_model_arch,
        'teacher': lst_teacher,
        'student': lst_student,
        'kd_method': lst_kd_method,
        'seed': lst_seed,
        'accuracy': lst_accuracy,
        'avg_entropy': lst_avg_entropy,
        'ece': lst_ece,
        'avg_logit': lst_avg_logit,
        'identifier': lst_identifier,
        'path': lst_model_path
    }

    df = pd.DataFrame(data_dict).to_csv('kd_results.csv', index=False)
    print("'---------------------------------------------------------------")
    count += 1
    print("%s of %s evaluated" % (count, len(lst_paths)))
