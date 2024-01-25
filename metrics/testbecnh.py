import os
# os.sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../"))
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
args = parser.parse_args([])

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

model = torch.load(r'/data/output/fahad.sarfraz/kd_methods_analysis/clean_data/resnet/cifar10/teacher_resnet26_student_resnet8_cifar10/rkd_da_v1_ResNet26_teacher_ResNet8_student_rkd_mode_CIFAR10_200epochs/rkd_da_v1_ResNet26_teacher_ResNet8_student_rkd_mode_CIFAR10_200epochs_seed0/checkpoints/final_model.pt')
_, accuracy, _ = eval(model, device, cifar10_test_loader)

