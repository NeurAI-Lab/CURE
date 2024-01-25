import torch
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from kd_lib import losses as kd_losses
from kd_lib.losses.kd_losses import dml_Loss, collate_loss
from models.selector import select_model
from train_util import adversarial_loss as adv_losses
from utilities.reinit_util import ReInit, RGP


# ======================================================================================
# Helper Functions
# ======================================================================================
class Cure():

    def __init__(self, args, device):
        self.args = args
        self.device = device
        cifar_resnet = True
        if args.dataset == 'imagenet200':
            cifar_resnet = False
        model = select_model(args.model_architecture, args.num_classes, cifar_resnet, args.img_size, args.patch_size).to(device)
        self.model = model
        self.layer_names = [name for name, _ in self.model.named_parameters()]
        # Reinit Init
        if self.args.reinit:
            self.reinit_init()
        #if 'ema' in args.train_mode:
        self.model_ema = deepcopy(self.model).to(device)
        self.global_step = 0


    def reinit_init(self):
        reinit_cls = ReInit(self.args.model_dir, self.args.freeze_mode, self.args.reinit_bn, self.args.reinit_layer)
        self.model = reinit_cls.load_model(self.model)
        if self.args.reinit_mode == 'freeze':
            # Reinit
            self.model = reinit_cls.freeze_layers(self.model)
        elif self.args.reinit_mode == 'rgp_soft':
            self.rgp = RGP(self.args.percentile, self.args.w_nat, self.args.w_rob, self.args.trades_beta, self.layer_names)

    def train(self, train_loader, optimizer, epoch, writer):
        if self.args.train_mode == 'cure':
            self.train_cure(train_loader, optimizer, epoch, writer)
        else:
            self.train_orig(train_loader, optimizer, epoch, writer)

    def loss(self, data, target, iteration, optimizer, writer):

        if self.args.adv_mode == 'normal':
            x_adv = None
            out = self.model(data)
            loss = kd_losses.cross_entropy(out, target)
            writer.add_scalar('train_util/loss', loss.item(), iteration)

        elif self.args.adv_mode == 'madry':
            loss, out, x_adv = adv_losses.madry_loss(self.model, data, target, optimizer)
            writer.add_scalar('train_util/loss', loss.item(), iteration)

        elif self.args.adv_mode == 'trades':
            loss, out, x_adv = adv_losses.trades_loss(self.model, data, target, optimizer, beta=self.args.trades_beta, args=self.args)
            writer.add_scalar('train_util/loss', loss.item(), iteration)

        elif self.args.adv_mode == 'cure_dual':
            if self.args.reinit_mode == 'rgp_soft':
                loss_natural, loss_robust_ce, loss_robust, loss_ema_nat, out_adv = adv_losses.cure_loss_dual(self.model, self.model_ema, data, target, optimizer, beta=self.args.trades_beta, args=self.args)
                return loss_natural, loss_robust_ce, loss_robust, loss_ema_nat, out_adv
            else:
                loss, loss_dist, out, x_adv = adv_losses.cure_loss_dual(self.model, self.model_ema, data, target, optimizer, beta=self.args.trades_beta, args=self.args)
                writer.add_scalar('train_util/loss', loss.item(), iteration)
                return loss, loss_dist, out, x_adv

        else:
            raise ValueError('Incorrect Method selected')

        return loss, out, x_adv

    def train_orig(self, train_loader, optimizer, epoch, writer):
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        num_batches = len(train_loader)
        for batch_idx, (data, target) in tqdm(enumerate(train_loader), desc='batch training', total=num_batches, position=0, leave=True):

            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            iteration = (epoch * num_batches) + batch_idx

            loss, out , _ = self.loss(data, target, iteration, optimizer, writer)
            # perform back propagation
            loss.backward()


            if self.args.reinit and self.args.reinit_mode == 'rgp_old' or self.args.reinit_mode == 'rgp':
                for name, wt in self.model.named_parameters():
                    if 'linear' in name:
                        continue
                    weight_temp = np.abs(wt.grad.cpu().detach().numpy())
                    percentile = self.args.percentile * 100
                    percentile = np.percentile(weight_temp, percentile)
                    under_threshold = weight_temp < percentile
                    wt.grad[under_threshold] = 0.0

            optimizer.step()
            train_loss += loss.data.item()
            _, predicted = torch.max(out.data, 1)
            total += target.size(0)
            correct += predicted.eq(target.data).cpu().float().sum()
            b_idx = batch_idx

        print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / (b_idx + 1), 100. * correct / total, correct, total))

    def train_cure(self, train_loader, optimizer, epoch, writer):

        self.model.train()
        self.model_ema.train()
        train_loss = 0
        correct = 0
        total = 0
        num_batches = len(train_loader)
        for batch_idx, (data, target) in tqdm(enumerate(train_loader), desc='batch training', total=num_batches, position=0, leave=True):

            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            iteration = (epoch * num_batches) + batch_idx

            if self.args.reinit:
                if self.args.reinit_mode == 'rgp_soft':
                    loss_nat_ce, loss_rob_ce, loss, loss_ema, out = self.loss(data, target, iteration, optimizer, writer)
                    loss_dict = collate_loss(self.args, loss, loss_dml=loss_ema.loss_dml, loss_nat_ce=loss_nat_ce, loss_rob_ce=loss_rob_ce,
                                             m1=True, epoch=epoch)

                    loss_m1 = loss_dict['loss']
                    writer.add_scalar('train_util/loss', loss_m1, iteration)

                    if self.args.grad_mode == 'all':
                        grads, shapes, has_grads = self.rgp.acc_grads(loss_dict, optimizer)
                        fuse_grads = self.rgp.fuse_grads_all(grads, shapes, has_grads)
                        self.rgp.set_grad(optimizer, fuse_grads)
                    else:
                        grads, shapes, has_grads = self.rgp.acc_grads(loss_dict, optimizer)
                        fuse_grads = self.rgp.fuse_grads(grads, shapes, has_grads)
                        self.rgp.set_grad(optimizer, fuse_grads)
                else:
                    loss_m1, loss_ema, out, x_adv = self.loss(data, target, iteration, optimizer, writer)
                    loss_dict = collate_loss(self.args, _loss_main=loss_m1, loss_dml=loss_ema.loss_dml, m1=True,
                                             epoch=epoch)
                    loss_m1 = loss_dict['loss']
                    # perform back propagation
                    loss_m1.backward()
            else:
                loss_m1, out, x_adv = self.loss(data, target, iteration, optimizer, writer)
                loss_dict = collate_loss(self.args, _loss_main=loss_m1, m1=True,
                                         epoch=epoch)
                loss_m1 = loss_dict['loss']
                # perform back propagation
                loss_m1.backward()

            if self.args.reinit_mode == 'rgp_old' or self.args.reinit_mode == 'rgp':

                for name, wt in self.model.named_parameters():
                    if 'linear' in name or 'fc'in name:
                        continue
                    weight_temp = np.abs(wt.grad.cpu().detach().numpy())
                    percentile = self.args.percentile * 100
                    percentile = np.percentile(weight_temp, percentile)
                    under_threshold = torch.from_numpy(weight_temp < percentile)
                    wt.grad[under_threshold] = 0.0

            optimizer.step()

            # Update the ema model
            self.global_step += 1
            if self.args.ema_dynamic and epoch > 100:
                self.args.ema_update_freq /= 2
            if torch.rand(1) < self.args.ema_update_freq:
                self.update_ema_model_variables()

            train_loss += loss_m1.data.item()
            _, predicted = torch.max(out.data, 1)
            total += target.size(0)
            correct += predicted.eq(target.data).cpu().float().sum()
            b_idx = batch_idx

        print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / (b_idx + 1), 100. * correct / total, correct, total))

    def update_ema_model_variables(self):
        alpha = min(1 - 1 / (self.global_step + 1), self.args.ema_alpha)
        for ema_param, param in zip(self.model_ema.parameters(), self.model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)




