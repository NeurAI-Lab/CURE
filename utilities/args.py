from argparse import ArgumentParser

def get_args(parser: ArgumentParser) -> None:

    # Model options
    parser.add_argument('--exp_identifier', type=str, default='')
    parser.add_argument('--model_architecture', type=str, default='ResNet18')
    parser.add_argument('--adv_mode', default='normal', choices=['normal', 'madry', 'trades', 'cure_dual'])
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--resnet_multiplier', default=1, type=int)
    parser.add_argument('--dtype', default='float', type=str)
    parser.add_argument('--nthread', default=4, type=int)
    # Dataset
    parser.add_argument("--data_path", type=str, default='data'), #choices=['data', '/data/input/datasets/tiny_imagenet/tiny-imagenet-200', '/data/input/datasets/ImageNet_2012', 'data/input/datasets/CelebA', '/data/input/datasets/Delaunay'])
    parser.add_argument("--data_path_style", type=str, default='/data/input-ai/datasets/tiny_imagenet/tiny-imagenet-200_stylized_train/alpha_1.0'),
    parser.add_argument("--cache-dataset", action="store_true", default=False)
    # Training options
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run for training the network')
    parser.add_argument('--epoch_step', nargs='*', type=int, default=[100, 150], help='json list with epochs to drop lr on')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=0.0005, type=float)
    parser.add_argument('--checkpoint', default='', type=str)
    parser.add_argument('--seeds', nargs='*', type=int, default=[0, 10])
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--scheduler', default='None', choices=['None','cosine', 'multistep'])

    # Device options
    parser.add_argument('--cuda', action='store_true')
    # evaluation options
    parser.add_argument("--save_freq", default=10, type=int, help="save frequency")
    parser.add_argument("--train_eval_freq",  default=50, type=int, help="evaluation frequency")
    parser.add_argument("--test_eval_freq",  default=50, type=int, help="evaluation frequency")
    # storage options
    parser.add_argument('--dataroot', default='data', type=str)
    parser.add_argument('--output_dir', default='experiments', type=str)
    parser.add_argument('--checkpoint_dir', default='', type=str)
    # Feature prior options
    parser.add_argument('--norm_std', type=str, default='False')
    # Adversarial training
    parser.add_argument('--step_size', type=float, default=0.007)
    parser.add_argument('--epsilon', type=float, default=0.031)
    parser.add_argument('--perturb_steps', type=int,  default=10)
    parser.add_argument('--distance', default='l_inf', choices=['l_2', 'l_inf'])
    parser.add_argument('--trades_beta', type=float, default=5)
    parser.add_argument('--mixup_alpha', type=float, default=1)
    parser.add_argument('--sev', type=int,  default=1)
    # ema
    parser.add_argument('--train_mode', default='normal', type=str, choices=['normal', 'cure'])
    parser.add_argument('--aux_loss_type', nargs='*', type=str, default=['kl'], help="--loss_type kl fitnet")
    parser.add_argument('--aux_loss_wt_kl1', type=float, default='1.0')
    parser.add_argument('--aux_loss_wt_l21', type=float, default='1.0')
    parser.add_argument('--aux_loss_wt_at1', type=float, default='0.0')
    parser.add_argument('--ema_mode', default='nat', type=str, choices=['nat', 'adv'])
    parser.add_argument('--ema_alpha', type=float, default=0.999, help='ema decay weight.')
    parser.add_argument('--ema_update_freq', type=float, default=0.2, help='frequency.')
    parser.add_argument("--ema_dynamic", action="store_true", default=False)

    ## freeze/cgi
    parser.add_argument('--reinit_path', default='', type=str)
    parser.add_argument('--freeze_mode', default='1', type=int)
    parser.add_argument('--reinit',  action="store_true")
    parser.add_argument('--reinit_mode', default='', type=str, choices=['','freeze', 'rgp', 'rgp_old', 'rgp_soft'])
    parser.add_argument('--reinit_bn',  action="store_true")
    parser.add_argument('--reinit_layer',  action="store_false")
    parser.add_argument('--grad_mode', default='', type=str, choices=['', 'all'])
    parser.add_argument('--percentile', default=0.01, type=float)
    parser.add_argument('--w_nat', default=1.0, type=float)
    parser.add_argument('--w_rob', default=1.0, type=float)

    #Noise
    parser.add_argument('--percent_label', default='1.0', type=float)
    parser.add_argument('--corrupt_label', default='0.0', type=float)

    #warmup
    parser.add_argument('--epochs_warmup', default=80, type=int, help='number of epochs for natural pre-training')

    #transformer
    parser.add_argument('--img_size', default=32, type=int, help='number of epochs for natural pre-training')
    parser.add_argument('--patch_size', default=4, type=int, help='number of epochs for natural pre-training')
