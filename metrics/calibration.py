from __future__ import print_function
import os
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from models.cifar.wideresnet import wrapper_model
from matplotlib.offsetbox import AnchoredText
import numpy as np
from matplotlib import pyplot as plt

# settings
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# set up data loader
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

test_batch_size = 200
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, **kwargs)

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
    'ONE-label-corr-20': {'seed_0': 0,
                     'seed_10': 0,
                     'seed_20': 2,
                     'seed_30': 1,
                     'seed_40': 0
                     },
    'ONE-label-corr-40': {'seed_0': 1,
                     'seed_10': 0,
                     'seed_20': 0,
                     'seed_30': 2,
                     'seed_40': 1
                     },
    'ONE-label-corr-60': {'seed_0': 2,
                     'seed_10': 2,
                     'seed_20': 0,
                     'seed_30': 2,
                     'seed_40': 0
                     },

}


def eval_calibration(lst_models, model_dir, output):

    lst_seeds = [0, 10, 20, 30, 40]

    calibration_dict = dict()
    calibration_dict['method'] = []
    calibration_dict['seed'] = []
    calibration_dict['ece'] = []

    for model_basename, method in lst_models:
        print('=' * 60 + '\nModel Name: %s\n' % model_basename + '=' * 60)

        for seed in lst_seeds:
            print('-' * 60 + '\nSeed %s\n' % seed + '-' * 60)

            try:
                lst_logits = []
                lst_labels = []

                model_path = os.path.join(model_dir, model_basename, model_basename + '_seed%s/checkpoints/final_model.pt' % seed)
                model_target = torch.load(model_path).to(device)
                model_target.eval()

                if method.startswith('ONE'):
                    model_target = wrapper_model(model_target, selected_branches_one[method]['seed_%s' % seed])

                n_bins = 10
                bin_boundaries = torch.linspace(0, 1, n_bins + 1)
                bin_lowers = bin_boundaries[:-1]
                bin_uppers = bin_boundaries[1:]

                with torch.no_grad():
                    for input, label in test_loader:
                        input = input.cuda()
                        logits = model_target(input)
                        lst_logits.append(logits)
                        lst_labels.append(label)

                logits = torch.cat(lst_logits).cuda()
                labels = torch.cat(lst_labels).cuda()

                softmaxes = F.softmax(logits, dim=1)
                confidences, predictions = torch.max(softmaxes, 1)
                accuracies = predictions.eq(labels)

                lst_acc_in_bin = []
                lst_conf_in_bin = []

                ece = torch.zeros(1, device=logits.device)
                for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                    # Calculated |confidence - accuracy| in each bin
                    in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
                    prop_in_bin = in_bin.float().mean()
                    if prop_in_bin.item() > 0:
                        accuracy_in_bin = accuracies[in_bin].float().mean()
                        avg_confidence_in_bin = confidences[in_bin].mean()

                        lst_acc_in_bin.append(accuracy_in_bin.item())
                        lst_conf_in_bin.append(avg_confidence_in_bin.item())

                        ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                    else:
                        lst_acc_in_bin.append(0)
                        lst_conf_in_bin.append(0)

                ece *= 100
                print('ECE: %s' % ece.item())
                col = (237, 129, 121)
                col = np.array(col) / 255
                col = tuple(col.tolist())

                fig, ax = plt.subplots(figsize=(3, 3))
                x_axis = np.array(list(range(0, n_bins))) / n_bins
                plt.bar(x_axis, x_axis, align='edge', width=0.1, facecolor=(1, 0, 0, 0.3), edgecolor=col, hatch='//', label='Gap')
                plt.bar(x_axis, lst_acc_in_bin, align='edge', width=0.1, facecolor=(0, 0, 1, 0.5), edgecolor='blue', label='Outputs')
                x_axis = np.array(list(range(0, n_bins + 1))) / n_bins
                plt.plot(x_axis, x_axis, '--', color='k')
                plt.axis('equal')
                ax.legend(fontsize=10)
                plt.xlim(0, 1)
                plt.ylim(0, 1)
                plt.gca().set_aspect('equal', adjustable='box')
                ax.set_ylabel('Accuracy', labelpad=11, fontweight='bold', fontsize=11, color='k')
                ax.set_xlabel('Confidence', fontsize=11, fontweight='bold')

                anchored_text = AnchoredText('ECE=%.2f' % ece, loc='lower right', prop=dict(fontsize=11))
                ax.add_artist(anchored_text)

                plt.savefig('analysis/calibration_all_seeds/%s_seed_%s_calib.png' % (method, seed), quality=100, bbox_inches='tight')

                calibration_dict['method'].append(method)
                calibration_dict['seed'].append(seed)
                calibration_dict['ece'].append(ece.item())


            except Exception as e:
                print(e)
                pass

        df = pd.DataFrame(calibration_dict)
        print(output + '_interim.csv')
        df.to_csv(output + '_interim.csv', index=False)

    df = pd.DataFrame(calibration_dict)
    df.to_csv(output + '.csv', index=False)