import os
import math
import csv
import random
import os.path
import shutil
import torch
import numpy as np
from utilities import dist_utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def get_data_loader(dataset, batch_size, cuda=False, shuffle=True):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        **({'num_workers': 1, 'pin_memory': True} if cuda else {})
    )

def save_checkpoint(model, model_dir, epoch, precision, best=True, save_splits=False):
    path = os.path.join(model_dir, model.name)
    path_best = os.path.join(model_dir, '{}-best'.format(model.name))

    # save the checkpoint.
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    checkpoint_dict = {
        'state': model.state_dict(),
        'epoch': epoch,
        'precision': precision,

    }

    if save_splits:
        checkpoint_dict['split_level'] = model.split_level
        checkpoint_dict['in_perm'] = model.in_perm
        checkpoint_dict['out_perm'] = model.out_perm

    torch.save(checkpoint_dict, path)

    # override the best model if it's the best.
    if best:
        shutil.copy(path, path_best)
        print('=> updated the best model of {name} at {path}'.format(
            name=model.name, path=path_best
        ))

    # notify that we successfully saved the checkpoint.
    print('=> saved the model {name} to {path}'.format(
        name=model.name, path=path
    ))

def load_checkpoint(model, model_dir, best=True):
    path = os.path.join(model_dir, model.name)
    path_best = os.path.join(model_dir, '{}-best'.format(model.name))

    # load the checkpoint.
    checkpoint = torch.load(path_best if best else path)
    print('=> loaded checkpoint of {name} from {path}'.format(
        name=model.name, path=(path_best if best else path)
    ))

    # load parameters and return the checkpoint's epoch and precision.
    model.load_state_dict(checkpoint['state'])
    epoch = checkpoint['epoch']
    precision = checkpoint['precision']
    return epoch, precision

def load_checkpoint_from_path(model, checkpoint_path):
    # load the checkpoint.
    checkpoint = torch.load(checkpoint_path)
    print('=> loaded checkpoint of {name} from {path}'.format(
        name=model.name, path=(checkpoint_path)
    ))

    # load parameters and return the checkpoint's epoch and precision.
    model.load_state_dict(checkpoint['state'])
    epoch = checkpoint['epoch']
    precision = checkpoint['precision']
    return epoch, precision

class ModelSaver:

    def __init__(self, save_root, base_name='model',
                 best_name='best', latest_name='latest',
                 models_to_keep=1):
        self.save_root = save_root
        self.base_name = base_name
        self.best_name = best_name
        self.latest_name = latest_name
        self.models_to_keep = models_to_keep
        self.best = 0
        self.best_epoch = 0

    def read_models(self):
        task_to_models = []

        for file in os.listdir(self.save_root):
            if os.path.isdir(file):
                continue
            extension = file.split('.')[-1]
            if extension != 'pth':
                continue
            names = file.split('_')[1:-1]
            if names[0] == 'best':
                continue
            task = '_'.join(names)
            task_to_models.append(file)

        return task_to_models

    def delete_old(self, task_to_models):
        models = sorted(task_to_models)
        if len(models) > self.models_to_keep:
            for m in models[:-self.models_to_keep]:
                os.remove(os.path.join(self.save_root, m))

    def save(self, save_dict, epoch, name, accuracy):
        accuracy *= 100

        if name == 'best':
            for file in os.listdir(self.save_root):
                if os.path.isdir(file):
                    continue
                extension = file.split('.')[-1]
                if extension != 'pth':
                    continue
                names = file.split('_')[:]
                if names[0] == self.base_name and names[1] == 'best':
                    os.remove(os.path.join(self.save_root, file))

        file_name = f'{self.base_name}_{name}_epoch{epoch:03d}_acc{accuracy:02.2f}.pth'
        dist_utils.save_on_master(save_dict, os.path.join(self.save_root, file_name),
                                  _use_new_zipfile_serialization=False)

    def save_models(self, model, optimizer, epoch, accuracy):

        save_dict = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'precision': accuracy,

        }

        if accuracy > self.best:
            self.best = accuracy
            self.best_epoch = epoch
            self.save(save_dict, epoch, self.best_name, accuracy)
        else:
            self.save(save_dict, epoch, self.latest_name, accuracy)

        task_to_models = self.read_models()
        self.delete_old(task_to_models)

    def load_checkpoint(self, model, optimizer, model_dir, best=False):
        model_files = list(filter(lambda x: x.endswith('.pth'), os.listdir(model_dir)))
        best_files = []
        for f in model_files:
            if 'best' in f:
                best_files.append(f)
        files = []
        for f in best_files:
            if self.base_name in f:
                files.append(f)

        # best_files = [f if 'best' in f else None for f in model_files]
        # file = [f if self.base_name in f else None for f in best_files]

        for f in files:
            if f is not None:
                filename = os.path.join(model_dir, f)

        # load the checkpoint.
        if not os.path.isfile(filename):
            print("=> no checkpoint found at '{}'".format(filename))

        checkpoint = torch.load(filename)
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("loaded checkpoint '{}' (epoch {})".format(filename, checkpoint["epoch"]))

        return model, optimizer, start_epoch

def validate(args, model, dataset, cuda=False, verbose=True):
    # set model mode as test mode.
    model_mode = model.training
    model.train(mode=False)

    # prepare the data loader and the statistics.
    # data_loader = get_data_loader(dataset, 32, cuda=cuda)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                              shuffle=False)
    total_tested = 0
    total_correct = 0

    for data, labels in data_loader:
        # test the model.
        data = Variable(data).cuda() if cuda else Variable(data)
        labels = Variable(labels).cuda() if cuda else Variable(labels)
        scores = model(data)
        scores = scores[0] if isinstance(scores, tuple) else scores
        _, predicted = torch.max(scores, 1)
        # update statistics.
        total_correct += (predicted == labels).sum().item()
        total_tested += len(data)

    # recover the model mode.
    model.train(mode=model_mode)

    # return the precision.
    precision = total_correct / total_tested
    if verbose:
        print('=> precision: {:.3f}'.format(precision))
    return precision

def validate_ensemble(model1, model2, dataset1, dataset2, cuda=False, verbose=True):
    # set model mode as test mode.
    model_mode1 = model1.training
    model1.train(mode=False)
    model_mode2 = model2.training
    model2.train(mode=False)

    # prepare the data loader and the statistics.
    data_loader1 = get_data_loader(dataset1, 8, cuda=cuda, shuffle=False)
    data_loader2 = get_data_loader(dataset2, 8, cuda=cuda, shuffle=False)
    total_tested = 0
    total_correct = 0

    for (data, labels), (data2, labels2) in zip(data_loader1, data_loader2):
        # test the model.
        data = Variable(data).cuda() if cuda else Variable(data)
        data2 = Variable(data2).cuda() if cuda else Variable(data2)
        labels = Variable(labels).cuda() if cuda else Variable(labels)
        scores1 = model1(data)
        scores2 = model2(data2)

        # scores = (scores1+scores2)/2
        # scores = torch.max(scores1, scores2)
        scores1 = scores1[0] if isinstance(scores1, tuple) else scores1
        scores2 = scores2[0] if isinstance(scores2, tuple) else scores2

        softmaxes1 = F.softmax(scores1, dim=1)
        softmaxes2 = F.softmax(scores2, dim=1)
        scores = (softmaxes1+softmaxes2)/2
        # scores = torch.max(softmaxes1, softmaxes2)
        _, predicted = torch.max(scores, 1)

        # # scores = (scores1+scores2)/2
        # _, predicted = torch.max(scores, 1)

        # update statistics.
        total_correct += (predicted == labels).sum().item()
        total_tested += len(data)

    # recover the model mode.
    model1.train(mode=model_mode1)
    model2.train(mode=model_mode2)

    # return the precision.
    precision = total_correct / total_tested
    if verbose:
        print('=> precision: {:.3f}'.format(precision))
    return precision


class csvWriter():
    def __init__(self, args):
        self.names = [
        "exp_identifier",
        "model",
        "mode",
        "dataset",
        "epochs",
        "accuracy",
        "output_path"
        ]

        base_name = "%s_%s_%s_mode_%s_%sepochs" % (
        args.exp_identifier, args.model_architecture, args.mode, args.dataset, args.epochs)

        self.file = os.path.join(args.output_dir, base_name, "results.csv")

        np.savetxt(self.file, (self.names), delimiter=",", fmt="%s")

    def write(self, args, test_accuracy):
        values = [
            args.exp_identifier,
            args.model_architecture,
            args.mode,
            args.dataset,
            args.epochs,
            test_accuracy,
            os.path.dirname(self.file)
        ]
        with open(self.file, "a") as f:
            writer = csv.writer(f)
            writer.writerow(values)

def check_final(args):
    model_dir = os.path.join(args.experiment_name, 'checkpoints')
    if not os.path.exists(model_dir):
        return False
    else:
        model_files = list(filter(lambda x: x.endswith('.pth'), os.listdir(model_dir)))

        for file in model_files:
            if 'final_model' in file:
                print("Model fully trained")
                return True
        else:
            return False

def set_random_seed(seed: int) -> None:
    """
    Sets the seeds at a certain value.
    :param seed: the value to be set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_data_parallel_model(model_path):
    state_dict = torch.load(model_path)["state_dict"]
    remove_prefix = 'module.'
    state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in state_dict.items()}
    return state_dict


matplotlib.rcParams['interactive'] == False

def plot_grad_flow(named_parameters, iter, file_path):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.mean().item())
            max_grads.append(p.grad.abs().max().item())

    if iter%500 == 0:
        fig = plt.figure()
        # plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.2, lw=1, color="r")
        plt.bar(np.arange(len(max_grads)), ave_grads, lw=1, color="r")
        # plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom=min(ave_grads), top=max(ave_grads))  # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        # plt.ylabel("Average gradient")
        plt.grid(True)
        plt.tight_layout()
        # plt.legend([Line2D([0], [0], color="r", lw=4),
        #             Line2D([0], [0], color="b", lw=4),
        #             Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        file = os.path.join(file_path, "vis","{}_{}.png".format(iter, "orig"))
        plt.savefig(file)

        # plt.savefig("/volumes2/reinit_adv/art/classifier/out_grad_flow/vis/{}.png".format(i))
        plt.close(fig)


def plot_grad_flow_orig(named_parameters, iter, file_path):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.mean().item())
            max_grads.append(p.grad.abs().max().item())

    if iter%500 == 0:
        fig = plt.figure()
        # plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.2, lw=1, color="r")
        plt.bar(np.arange(len(max_grads)), ave_grads, lw=1, color="b")
        # plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom=min(ave_grads), top=max(ave_grads))  # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        # plt.ylabel("Average gradient")
        plt.grid(True)
        plt.tight_layout()
        # plt.legend([Line2D([0], [0], color="r", lw=4),
        #             Line2D([0], [0], color="b", lw=4),
        #             Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

        file = os.path.join(file_path, "vis","{}_{}.png".format(iter, "orig"))
        plt.savefig(file)
        # plt.savefig("/volumes2/reinit_adv/art/classifier/out_grad_flow/vis/{}.png".format(i))
        plt.close(fig)

def plot_grad_flow_perc(removed, layers, iter):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    # zero_grads = []
    # ave_grads = []
    # layers = []
    # for n, p in named_parameters:
    #     if (p.requires_grad) and ("bias" not in n):
    #         layers.append(n)
    #         grad = p.grad
    #         ave_grads.append(p.grad.abs().mean().item())
    #         zeros = grad.numel() - grad.nonzero().size(0) #torch.sum((p == 0).int()).data[0]
    #         retain = grad.nonzero().size(0) #torch.sum((p == 0).int()).data[0]
    #         num_params = 1
    #         # for i in p.grad.size():
    #         #     num_params *= i
    #
    #         perc = (retain/grad.numel())*100.0
    #         zero_grads.append(perc)

    if iter%500 == 0:
        plt.bar(np.arange(len(removed)), removed, lw=1, color="r")
        plt.xticks(range(0, len(removed), 1), layers, rotation="vertical")
        plt.tight_layout()

        plt.savefig("/volumes2/reinit_adv/art/classifier/out_grad_flow3/vis_perc/{}_{}.png".format(iter, "cure"))

    return


def plot_grad_epoch(method):
    fig, axs = plt.subplots(4, 2)

    grad_orig = method.grads_orig

    iter = []
    for k in grad_orig:
        iter.append(k)
    # heatmap
    lst1 = lst2 = lst3 = lst4 = lst5 = lst6 = lst7 = lst8 =[]
    for i in iter:
        if grad_orig[i]:
            lst1.append(grad_orig[i]['layer1.0.conv2.weight'])
            lst2.append(grad_orig[i]['layer1.1.conv2.weight'])
            lst3.append(grad_orig[i]['layer2.0.conv2.weight'])
            lst4.append(grad_orig[i]['layer2.1.conv2.weight'])
            lst5.append(grad_orig[i]['layer3.0.conv2.weight'])
            lst6.append(grad_orig[i]['layer3.1.conv2.weight'])
            lst7.append(grad_orig[i]['layer4.0.conv2.weight'])
            lst8.append(grad_orig[i]['layer4.1.conv2.weight'])
