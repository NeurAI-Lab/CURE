from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
from models.wideresnet import WideResNet
from models.cifar.wideresnet import wrapper_model
import numpy as np
import pandas as pd
from kd_lib.utilities.utils import set_torch_seeds

parser = argparse.ArgumentParser(description='PyTorch CIFAR PGD Attack Evaluation')
parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps', default=20,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.003,
                    help='perturb step size')
parser.add_argument('--random',
                    default=True,
                    help='random initialization for PGD')
parser.add_argument('--model-path',
                    default='./checkpoints/model_cifar_wrn.pt',
                    help='model for white-box attack evaluation')
parser.add_argument('--source-model-path',
                    default='./checkpoints/model_cifar_wrn.pt',
                    help='source model for black-box attack evaluation')
parser.add_argument('--target-model-path',
                    default='./checkpoints/model_cifar_wrn.pt',
                    help='target model for black-box attack evaluation')
parser.add_argument('--white-box-attack', default=True,
                    help='whether perform white-box attack')

args = parser.parse_args([])

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
set_torch_seeds(0)

# set up data loader
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size):

    X_pgd = Variable(X.data, requires_grad=True)

    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            out_pgd = model(X_pgd)
            loss = nn.CrossEntropyLoss()(out_pgd, y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    return X_pgd.detach()


def create_adv_examples(model, device, test_loader, num_steps):
    """
    evaluate model by white-box attack
    """
    model.eval()
    lst_adv = []

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        X_adv = _pgd_whitebox(model, X, y, num_steps=num_steps)

        lst_adv.append(X_adv)

    return lst_adv


# ======================================================================================
# Evaluate Transferability
# ======================================================================================
def eval_transferability(lst_adv, model_source, model_target, device, test_loader, condition_on_correct=True):
    """
    evaluate model by black-box attack
    """
    model_target.eval()
    model_source.eval()

    attacks_evaluated_total = 0
    attacks_transferred_total = 0

    count = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        X_pgd = lst_adv[count]

        if condition_on_correct:
            out_src = model_source(data)
            out_target = model_target(data)

            src_correct = out_src.data.max(1)[1] == target.data
            target_correct = out_target.data.max(1)[1] == target.data
            successful_attack = model_source(X_pgd).data.max(1)[1] != target.data

            select_idx = src_correct * target_correct * successful_attack

            X_pgd = X_pgd[select_idx]
            target = target[select_idx]

        out = model_target(X_pgd)
        err = (out.data.max(1)[1] != target.data).float().sum()

        attacks_evaluated_total += len(X_pgd)
        attacks_transferred_total += err

        count += 1

    print('Attacks conducted: ', attacks_evaluated_total)
    print('Attacks Transferred: ', attacks_transferred_total.item())
    print('Transferability:', attacks_transferred_total.item() / attacks_evaluated_total)

    return attacks_transferred_total.item(), attacks_evaluated_total


def main():

    lst_steps = [20]

    lst_seeds = [0, 10, 20, 30, 40]

    selected_branches_one = {
        'ONE-WRN-40-2': {'seed_0': 2,
                         'seed_10': 0,
                         'seed_20': 1,
                         'seed_30': 0,
                         'seed_40': 1
                         },
        'ONE-WRN-16-2': {'seed_0': 2,
                         'seed_10': 2,
                         'seed_20': 0,
                         'seed_30': 2,
                         'seed_40': 0
                         },
    }


    lst_models = [
        # ('normal_WRN-40-2_normal_mode_CIFAR10_200epochs', 'WRN-40-2'),
        # ('normal_WRN-16-2_normal_mode_CIFAR10_200epochs', 'WRN-16-2'),
        # ('hinton_v1_WRN-40-2_teacher_WRN-16-2_student_kd_mode_CIFAR10_200epochs', 'Hinton'),
        # ('fitnet_v1_WRN-40-2_teacher_WRN-16-2_student_fitnet_mode_CIFAR10_200epochs', 'FitNet'),
        # ('at_v1_WRN-40-2_teacher_WRN-16-2_student_at_mode_CIFAR10_200epochs', 'AT'),
        # ('at_hinton_v1_WRN-40-2_teacher_WRN-16-2_student_at_mode_CIFAR10_200epochs', 'AT + Hinton'),
        # ('fsp_v1_WRN-40-2_teacher_WRN-16-2_student_fsp_mode_CIFAR10_200epochs', 'FSP'),
        # ('sp_v1_WRN-40-2_teacher_WRN-16-2_student_sp_mode_CIFAR10_200epochs', 'SP'),
        # ('sp_hinton_v1_WRN-40-2_teacher_WRN-16-2_student_sp_mode_CIFAR10_200epochs', 'SP + Hinton'),
        # ('bss_v3_WRN-16-2_bss_mode_CIFAR10_200epochs', 'BSS'),
        # ("ZEROSHOTKT_CIFAR10_WRN-40-2_WRN-16-2_gi1_si10_zd100_plr0.001_slr0.002_bs128_T1.0_beta250.0", 'ZeroShot'),
        # ("one_v12_one_wrn-40-2_cifar10_300epochs", 'ONE-WRN-40-2'),
        # ("one_v13_one_wrn-16-2_cifar10_300epochs", 'ONE-WRN-16-2'),
        # ("rkd_d_v1_WRN-40-2_teacher_WRN-16-2_student_rkd_mode_CIFAR10_200epochs", "RKD-D"),
        # ("rkd_d_v2_WRN-40-2_teacher_WRN-16-2_student_rkd_mode_CIFAR10_200epochs", "RKD-D + Hinton"),
        # ("rkd_a_v1_WRN-40-2_teacher_WRN-16-2_student_rkd_mode_CIFAR10_200epochs", 'RKD-A'),
        # ("rkd_a_v2_WRN-40-2_teacher_WRN-16-2_student_rkd_mode_CIFAR10_200epochs", 'RKD-A + Hinton'),
        # ("rkd_da_v1_WRN-40-2_teacher_WRN-16-2_student_rkd_mode_CIFAR10_200epochs", "RKD-DA"),
        # ("rkd_da_v2_WRN-40-2_teacher_WRN-16-2_student_rkd_mode_CIFAR10_200epochs", "RKD-DA"),
        # ("dropout_0.4_v1_WRN-40-2_normal_mode_CIFAR10_200epochs", "WRN-40-2-Dropout"),
        # ("dropout_0.4_v1_WRN-40-2_teacher_WRN-16-2_student_kd_mode_CIFAR10_350epochs", "Fickle Teacher"),
        ('hinton_WRN-40-2_teacher_WRN-16-2_student_kd_mode_CIFAR10_200epochs', 'Hinton w dropout'),
    ]

    # lst_kd_methods_w_dropout = [
    #     ('hinton_WRN-40-2_teacher_WRN-16-2_student_kd_mode_CIFAR10_200epochs', 'Hinton w dropout'),
    #     ('fitnet_WRN-40-2_teacher_WRN-16-2_student_fitnet_mode_CIFAR10_200epochs', 'FitNet'),
    #     ('at_WRN-40-2_teacher_WRN-16-2_student_at_mode_CIFAR10_200epochs', 'AT'),
    #     ('at_hinton_v1_WRN-40-2_teacher_WRN-16-2_student_at_mode_CIFAR10_200epochs', 'AT + Hinton'),
    #     ('fsp_WRN-40-2_teacher_WRN-16-2_student_fsp_mode_CIFAR10_200epochs', 'FSP'),
    #     ('sp_v1_WRN-40-2_teacher_WRN-16-2_student_sp_mode_CIFAR10_200epochs', 'SP'),
    #     ('sp_hinton_v1_WRN-40-2_teacher_WRN-16-2_student_sp_mode_CIFAR10_200epochs', 'SP + Hinton'),
    #     ('bss_WRN-16-2_bss_mode_CIFAR10_200epochs', 'BSS'),
    #     ("ZEROSHOTKT_CIFAR10_WRN-40-2_WRN-16-2_gi1_si10_zd100_plr0.001_slr0.002_bs128_T1.0_beta250.0", 'ZeroShot'),
    #     ("rkd_d_v1_WRN-40-2_teacher_WRN-16-2_student_rkd_mode_CIFAR10_200epochs", "RKD-D"),
    #     ("rkd_d_v2_WRN-40-2_teacher_WRN-16-2_student_rkd_mode_CIFAR10_200epochs", "RKD-D + Hinton"),
    #     ("rkd_a_v1_WRN-40-2_teacher_WRN-16-2_student_rkd_mode_CIFAR10_200epochs", 'RKD-A'),
    #     ("rkd_a_v2_WRN-40-2_teacher_WRN-16-2_student_rkd_mode_CIFAR10_200epochs", 'RKD-A + Hinton'),
    #     ("rkd_da_v1_WRN-40-2_teacher_WRN-16-2_student_rkd_mode_CIFAR10_200epochs", "RKD-DA"),
    #     ("rkd_da_v2_WRN-40-2_teacher_WRN-16-2_student_rkd_mode_CIFAR10_200epochs", "RKD-DA"),
    # ]
    # kd_methods_w_dropout_model_dir = r'/data/input/datasets/trained_models_unnormalized/kd_w_dropout'
    # kd_methods_w_dropout_output = r'/data/users/fahad.sarfraz/workspace/nie_knowledge_distillation/analysis/kd_methods/kd_methods_w_dropout'

    # src_model = r"/data/input/datasets/trained_models_unnormalized/WRN-40-2/normal_WRN-40-2_normal_mode_CIFAR10_200epochs/normal_WRN-40-2_normal_mode_CIFAR10_200epochs_seed40/checkpoints/final_model.pt"
    src_model = r"/data/input/datasets/noise_distillation/trained_models/dropout/dropout_0.4_v1_WRN-40-2_normal_mode_CIFAR10_200epochs/dropout_0.4_v1_WRN-40-2_normal_mode_CIFAR10_200epochs_seed40/checkpoints/final_model.pt"


    model_source = torch.load(src_model).to(device)
    model_source.eval()

    target_model_dir = r'/data/input/datasets/trained_models_unnormalized/kd_w_dropout'

    rob_analysis_dict = dict()
    rob_analysis_dict['method'] = []
    rob_analysis_dict['attack'] = []
    rob_analysis_dict['num_steps'] = []
    rob_analysis_dict['seed'] = []
    rob_analysis_dict['transferability'] = []

    count = 0

    for num_steps in lst_steps:
        print('*' * 60 + '\nAttack: %s Step PGD\n' % num_steps + '=' * 60)
        # Create adversarial Examples for the source model
        lst_adv = create_adv_examples(model_source, device, test_loader, num_steps)

        for target_model_basename, method in lst_models:

            print('=' * 60 + '\nModel Name: %s\n' % target_model_basename + '=' * 60)
            args.num_steps = num_steps

            for seed in lst_seeds:
                print('-' * 60 + '\nSeed %s\n' % seed + '-' * 60)

                taget_model_path = os.path.join(target_model_dir, target_model_basename, target_model_basename + '_seed%s/checkpoints/final_model.pt' % seed)
                model_target = torch.load(taget_model_path).to(device)

                if method.startswith('ONE'):
                    model_target = wrapper_model(model_target, selected_branches_one[method]['seed_%s' % seed])

                model_target.eval()

                print('Transferability:')
                attacks_transferred, attacks_conducted = eval_transferability(lst_adv, model_source, model_target, device, test_loader, condition_on_correct=True)
                transferability = attacks_transferred / attacks_conducted

                rob_analysis_dict['method'].append(method)
                rob_analysis_dict['attack'].append('PDG')
                rob_analysis_dict['num_steps'].append(num_steps)
                rob_analysis_dict['seed'].append(seed)
                rob_analysis_dict['transferability'].append(transferability)

        count += 1
        df = pd.DataFrame(rob_analysis_dict)
        df.to_csv('analysis/kd_methods/transferability_interim.csv', index=False)

    df = pd.DataFrame(rob_analysis_dict)
    df.to_csv('analysis/kd_methods/transferability.csv', index=False)


if __name__ == '__main__':
    main()
