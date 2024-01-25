from argparse import ArgumentParser
import os
import json
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from utilities import adv_utils as adv_utils
from kd_lib import utilities as utils
from utilities import dist_utils
from utilities.utils import ModelSaver, check_final, plot_grad_epoch
from data.dataset import DATASETS
from data.transforms import build_transforms
from utilities.train import Cure
from utilities.args import get_args
from utilities.results import Results
from utilities.logger import Logger
# ======================================================================================
# Training Function
# ======================================================================================
def solver(args):

    log_dir = os.path.join(args.experiment_name, 'logs')
    if args.reinit:
        args.model_dir = args.reinit_path
    save_model_dir = os.path.join(args.experiment_name, 'checkpoints') if args.adv_mode =='normal' else \
        os.path.join(args.experiment_name, 'checkpoints_adv')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_model_dir, exist_ok=True)

    logger = Logger(os.path.join(log_dir, 'log.txt'), title=args.experiment_name)
    logger.set_names(['Nat train loss', 'Nat train acc', 'Nat test acc',
                      'Adv train acc', 'Adv test acc',
                      'Nat test acc ema', 'Adv test acc ema',
                      ])

    log_path = os.path.join(log_dir, datetime.now().strftime('%Y%m%d_%H%M'))
    writer = SummaryWriter(log_path)

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('device: %s' % device)

    if use_cuda:
        torch.cuda.set_device(0)
        cudnn.benchmark = True

    if args.dataset =='imagenet' or args.dataset =='imagenet200':
        ds_class = DATASETS[args.dataset](args.data_path, cache_dataset=True)
    else:
        ds_class = DATASETS[args.dataset](args.data_path)

    #load transforms
    transform_train, transform_test = build_transforms(args, ds_class)
    # load dataset
    trainset = ds_class.get_dataset('train', transform_train, transform_test)
    testset = ds_class.get_dataset('test', transform_train, transform_test)

    #data
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.num_workers)

    # Load model Init
    args.num_classes = ds_class.NUM_CLASSES
    method = Cure(args, device)

    saver = ModelSaver(save_model_dir)
    start_epoch = 0
    if "deit" in args.model_architecture:
        optimizer = torch.optim.Adam(method.model.parameters(), lr=args.lr)
    else:
        optimizer = SGD(method.model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    #optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    scheduler = None
    if args.scheduler == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.epoch_step, gamma=0.1)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # if os.listdir(save_model_dir) and not args.reinit:
    #     method.model, optimizer, start_epoch = saver.load_checkpoint(method.model, optimizer, save_model_dir)

    print('*' * 60 + '\nTraining Mode: %s\n' % args.adv_mode + '*' * 60)
    for epoch in tqdm(range(1, args.epochs + 1), desc='training epochs'):
        if epoch <= start_epoch:
            continue
        # adjust learning rate for SGD
        if scheduler:
            scheduler.step()
        else:
            utils.adjust_learning_rate(epoch, args.epoch_step, args.lr_decay_ratio, optimizer)
        method.train(train_loader, optimizer, epoch, writer)

        # evaluation on natural examples ##
        if args.train_eval_freq == 0 or epoch % args.train_eval_freq == 0:
            print("================================================================")
            train_loss, train_accuracy, correct = utils.eval(method.model, device, train_loader)
            print("Training: Average loss: {:.4f}, Accuracy: {}/{} ({}%)".format(
                train_loss, correct, len(train_loader.dataset), train_accuracy * 100))
            print("================================================================")
            writer.add_scalar("Train/train_loss", train_loss, epoch)
            writer.add_scalar("Train/train_accuracy", train_accuracy, epoch)
            adv_train_accuracy = 0
            if args.adv_mode != 'normal':
                _, adv_train_accuracy = adv_utils.eval_adv_robustness(method.model, train_loader, device)

        if args.test_eval_freq == 0 or epoch % args.test_eval_freq == 0:
            print("================================================================")
            nat_test_loss, nat_test_accuracy, correct = utils.eval(method.model, device, test_loader)
            print("Test: Average loss: {:.4f}, Accuracy: {}/{} ({}%)".format(
                nat_test_loss, correct, len(test_loader.dataset), nat_test_accuracy * 100))
            print("================================================================")
            writer.add_scalar("Test/test_loss", nat_test_loss, epoch)
            writer.add_scalar("Test/test_accuracy", nat_test_accuracy, epoch)
            print("================================================================")
            nat_test_accuracy2 = adv_test_accuracy = 0
            if args.adv_mode != 'normal':
                nat_test_accuracy2, adv_test_accuracy = adv_utils.eval_adv_robustness(method.model, test_loader, device)
                writer.add_scalar("Test/nat_test_accuracy2", nat_test_accuracy2, epoch)
                writer.add_scalar("Test/adv_test_accuracy", adv_test_accuracy, epoch)
            nat_test_accuracy_ema = adv_test_accuracy_ema = 0
            if 'ema' in args.train_mode:
                nat_test_accuracy_ema, adv_test_accuracy_ema = adv_utils.eval_adv_robustness(method.model_ema,
                                                                                             test_loader, device)
            if epoch != args.epochs:
                saver.save_models(method.model, optimizer, epoch, nat_test_accuracy)
            logger.append([train_loss, train_accuracy, nat_test_accuracy, adv_train_accuracy, adv_test_accuracy, nat_test_accuracy_ema, adv_test_accuracy_ema])

    # get final test accuracy
    test_accuracy_ema = 0
    test_loss, test_accuracy, correct = utils.eval(method.model, device, test_loader)
    if 'ema' in args.train_mode:
        test_loss_ema, test_accuracy_ema, correct_ema = utils.eval(method.model_ema, device, test_loader)

    adv_test_accuracy = adv_test_accuracy_ema = 0
    if args.adv_mode != 'normal':
        nat_test_accuracy2, adv_test_accuracy = adv_utils.eval_adv_robustness(method.model, test_loader, device)
        if 'ema' in args.train_mode:
            nat_test_accuracy_ema, adv_test_accuracy_ema = adv_utils.eval_adv_robustness(method.model_ema, test_loader, device)

    writer.close()

    res = Results(args, test_loss, test_accuracy, test_accuracy_ema, adv_test_accuracy, adv_test_accuracy_ema)

    # save model
    torch.save(method.model, os.path.join(save_model_dir, 'final_model.pth'))
    if 'ema' in args.train_mode:
        torch.save(method.model_ema, os.path.join(save_model_dir, 'final_model_ema.pth'))

    return res    #, saver.best, saver.best_epoch

def main(args):

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    #base_name = "%s_%s_%s_mode_%s_%sepochs" % (args.exp_identifier, args.model_architecture, args.mode, args.dataset, args.epochs)
    base_name = "%s" % (args.exp_identifier)
    base_dir = os.path.join(args.output_dir, base_name)
    os.makedirs(base_dir, exist_ok=True)

    # save training arguments
    args_path = os.path.join(base_dir, 'args.txt')
    z = vars(args).copy()
    with open(args_path, 'w') as f:
        f.write('arguments: ' + json.dumps(z) + '\n')

    utils.set_torch_seeds(args.seeds[0])
    args.experiment_name = os.path.join(args.output_dir, base_name, base_name + '_seed' + str(args.seeds[0]))
    if check_final(args):
        exit()

    res = solver(args)
    res.save_results_normal(os.path.join(args.output_dir, base_name, 'results.csv'))


if __name__ == '__main__':
    parser = ArgumentParser(description='mammoth', allow_abbrev=False)
    get_args(parser)
    args = parser.parse_args()
    dist_utils.init_distributed_mode(args)

    main(args)
