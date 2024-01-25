from __future__ import print_function
import os
import argparse
import torch
import torchvision
from torchvision import transforms
import pandas as pd
from kd_lib.attacks.attack_BSS import AttackBSS
from kd_lib.utilities.utils import set_torch_seeds
from models.cifar.wideresnet import wrapper_model

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

args = parser.parse_args()

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
set_torch_seeds(0)

# set up data loader
transform_test = transforms.Compose([
    transforms.ToTensor(),

])

testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

attack = AttackBSS(targeted=True, num_steps=10, max_epsilon=16, step_alpha=0.3, cuda=True, norm=2)


def evaluate_decision_boundary_metrics(t_net, s_net, testloader):

    t_net.eval()
    s_net.eval()

    mag_sim = torch.tensor(0.0)
    ang_sim = torch.tensor(0.0)

    successful_attacks = 0
    for data, target in testloader:
        data, target = data.cuda(), target.cuda()

        for k in range(9):

            attack_target = k * (target.data != k).long() + 9 * (target.data == k).long()
            attack_target = attack_target.cuda()

            t_bss = attack.run(t_net, data, attack_target)
            t_out = t_net(t_bss)

            t_pert = t_bss - data

            s_bss = attack.run(s_net, data, attack_target)
            s_out = s_net(s_bss)

            s_pert = s_bss - data

            success = ((torch.max(t_out.data, 1)[1] != target.data) & (
                    torch.max(s_out.data, 1)[1] != target.data))

            num_samples = len(s_pert)
            for i in range(num_samples):
                if success[i]:
                    s_norm = s_pert[i].norm(2)
                    t_norm = t_pert[i].norm(2)

                    if not (torch.isnan(s_norm) or torch.isnan(t_norm)):
                        successful_attacks += 1
                        mag_sim += (min(s_norm, t_norm) / max(s_norm, t_norm, 1e-8))
                        ang_sim += torch.dot(s_pert[i].view(-1), t_pert[i].view(-1)) / (
                                s_norm * t_norm)

    mag_sim = mag_sim.item() / successful_attacks
    ang_sim = ang_sim.item() / successful_attacks

    return mag_sim, ang_sim


def main():

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
        ("rkd_a_v2_WRN-40-2_teacher_WRN-16-2_student_rkd_mode_CIFAR10_200epochs", 'RKD-A + Hinton'),
        # ("rkd_da_v1_WRN-40-2_teacher_WRN-16-2_student_rkd_mode_CIFAR10_200epochs", "RKD-DA"),
        # ("rkd_da_v2_WRN-40-2_teacher_WRN-16-2_student_rkd_mode_CIFAR10_200epochs", "RKD-DA + Hinton"),
        # ("dropout_0.4_v1_WRN-40-2_teacher_WRN-16-2_student_kd_mode_CIFAR10_350epochs", "Fickle Teacher")
    ]

    src_model = r"/data/input/datasets/trained_models_unnormalized/WRN-40-2/normal_WRN-40-2_normal_mode_CIFAR10_200epochs/normal_WRN-40-2_normal_mode_CIFAR10_200epochs_seed40/checkpoints/final_model.pt"
    # src_model = r"/data/input/datasets/noise_distillation/trained_models/dropout/dropout_0.4_v1_WRN-40-2_normal_mode_CIFAR10_200epochs/dropout_0.4_v1_WRN-40-2_normal_mode_CIFAR10_200epochs_seed40/checkpoints/final_model.pt"
    model_source = torch.load(src_model).to(device)
    model_source.eval()

    target_model_dir = r"/data/input/datasets/trained_models_unnormalized/WRN-16-2/"

    db_similarity_dict = dict()
    db_similarity_dict['method'] = []
    db_similarity_dict['seed'] = []
    db_similarity_dict['magsim'] = []
    db_similarity_dict['angsim'] = []

    count = 0
    for target_model_basename, method in lst_models:

        print('=' * 60 + '\nModel Name: %s\n' % target_model_basename + '=' * 60)

        for seed in lst_seeds:
            print('-' * 60 + '\nSeed %s\n' % seed + '-' * 60)

            taget_model_path = os.path.join(target_model_dir, target_model_basename, target_model_basename + '_seed%s/checkpoints/final_model.pt' % seed)
            model_target = torch.load(taget_model_path).to(device)

            if method.startswith('ONE'):
                model_target = wrapper_model(model_target, selected_branches_one[method]['seed_%s' % seed])

            print('Measuring Decision Boundary Similarity')
            mag_sim, ang_sim = evaluate_decision_boundary_metrics(model_source, model_target, test_loader)
            print('AngSim:', ang_sim)
            print('MagSim:', mag_sim)
            db_similarity_dict['method'].append(method)
            db_similarity_dict['seed'].append(seed)
            db_similarity_dict['magsim'].append(mag_sim)
            db_similarity_dict['angsim'].append(ang_sim)

        count += 1
        df = pd.DataFrame(db_similarity_dict)
        df.to_csv('analysis/kd_methods/decision_boundary_sim_interim.csv', index=False)

    df = pd.DataFrame(db_similarity_dict)
    df.to_csv('analysis/kd_methods/decision_boundary_sim.csv', index=False)


if __name__ == '__main__':
    main()
