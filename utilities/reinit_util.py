import math
import torch
import random
import numpy as np
import torch.nn as nn

def renint_usnig_method(mask, method='kaiming'):
    if method == 'kaiming':
        nn.init.kaiming_uniform_(mask, a=math.sqrt(5))
    else:
        nn.init.xavier_uniform_(mask)


class ReInit():
    def __init__(self, model_dir, freeze_mode, reinit_bn=False, reinit_layer=True):
        self.model_dir = model_dir
        self.freeze_mode = freeze_mode
        self.reinit_bn = reinit_bn
        self.reinit_layer = reinit_layer

    def load_model(self, model):
        # load the checkpoint.
        checkpoint = torch.load(self.model_dir)
        # load parameters and return the checkpoint's epoch and precision.
        if isinstance(checkpoint, dict):
            if "state_dict" not in checkpoint:
                model.load_state_dict(checkpoint["model"], strict=True)
            else:
                model.load_state_dict(checkpoint["state_dict"], strict=True)
        else:
            model =  torch.load(self.model_dir)
        return model

    def re_init_weights(self, layer):
        # mask = torch.empty(m, requires_grad=False)
        for m in layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if self.reinit_bn:
                    nn.init.constant_(m.weight.data, 1)
                    nn.init.constant_(m.bias.data, 0)

        return layer

    def freeze_layers(self, model):

        if self.freeze_mode == 1:
            if self.reinit_layer:
                # init the block1 layer
                self.re_init_weights(model.layer1)
            for param in model.layer2.parameters():
                param.requires_grad_(False)
            for param in model.layer3.parameters():
                param.requires_grad_(False)
            for param in model.layer4.parameters():
                param.requires_grad_(False)
            for param in model.linear.parameters():
                param.requires_grad_(False)

        elif self.freeze_mode == 2:
            for param in model.layer1.parameters():
                param.requires_grad_(False)
            if self.reinit_layer:
                # init the block2 layer
                self.re_init_weights(model.layer2)
            for param in model.layer3.parameters():
                param.requires_grad_(False)
            for param in model.layer4.parameters():
                param.requires_grad_(False)
            for param in model.linear.parameters():
                param.requires_grad_(False)

        elif self.freeze_mode == 3:
            for param in model.layer1.parameters():
                param.requires_grad_(False)
            for param in model.layer2.parameters():
                param.requires_grad_(False)
            if self.reinit_layer:
                # init the block3 layer
                self.re_init_weights(model.layer3)
            for param in model.layer4.parameters():
                param.requires_grad_(False)
            for param in model.linear.parameters():
                param.requires_grad_(False)

        elif self.freeze_mode == 4:
            for param in model.layer1.parameters():
                param.requires_grad_(False)
            for param in model.layer2.parameters():
                param.requires_grad_(False)
            for param in model.layer3.parameters():
                param.requires_grad_(False)
            if self.reinit_layer:
                # init the block4 layer
                self.re_init_weights(model.layer4)
            for param in model.linear.parameters():
                param.requires_grad_(False)

        elif self.freeze_mode == 5:
            for param in model.layer1.parameters():
                param.requires_grad_(False)
            for param in model.layer2.parameters():
                param.requires_grad_(False)
            for param in model.layer3.parameters():
                param.requires_grad_(False)
            for param in model.layer4.parameters():
                param.requires_grad_(False)
            if self.reinit_layer:
                # init the fc layer
                model.linear.weight.data.normal_(mean=0.0, std=0.01)
                model.linear.bias.data.zero_()

        elif self.freeze_mode == 12:
            if self.reinit_layer:
                # init B1,B2
                self.re_init_weights(model.layer1)
                self.re_init_weights(model.layer2)
            for param in model.layer3.parameters():
                param.requires_grad_(False)
            for param in model.layer4.parameters():
                param.requires_grad_(False)
            for param in model.linear.parameters():
                param.requires_grad_(False)

        elif self.freeze_mode == 23:
            for param in model.layer1.parameters():
                param.requires_grad_(False)
            if self.reinit_layer:
                # init B2,B3
                self.re_init_weights(model.layer2)
                self.re_init_weights(model.layer3)
            for param in model.layer4.parameters():
                param.requires_grad_(False)
            for param in model.linear.parameters():
                param.requires_grad_(False)

        elif self.freeze_mode == 34:
            for param in model.layer1.parameters():
                param.requires_grad_(False)
            for param in model.layer2.parameters():
                param.requires_grad_(False)
            if self.reinit_layer:
                # init B3,B4
                self.re_init_weights(model.layer3)
                self.re_init_weights(model.layer4)
            for param in model.linear.parameters():
                param.requires_grad_(False)

        elif self.freeze_mode == 45:
            for param in model.layer1.parameters():
                param.requires_grad_(False)
            for param in model.layer2.parameters():
                param.requires_grad_(False)
            for param in model.layer3.parameters():
                param.requires_grad_(False)
            if self.reinit_layer:
                self.re_init_weights(model.layer4)
                # init the fc layer and B4
                model.linear.weight.data.normal_(mean=0.0, std=0.01)
                model.linear.bias.data.zero_()

        elif self.freeze_mode == 123:
            if self.reinit_layer:
                self.re_init_weights(model.layer1)
                self.re_init_weights(model.layer2)
                self.re_init_weights(model.layer3)
            for param in model.layer4.parameters():
                param.requires_grad_(False)
            for param in model.linear.parameters():
                param.requires_grad_(False)

        elif self.freeze_mode == 234:
            for param in model.layer1.parameters():
                param.requires_grad_(False)
            if self.reinit_layer:
                self.re_init_weights(model.layer2)
                self.re_init_weights(model.layer3)
                self.re_init_weights(model.layer4)
            for param in model.linear.parameters():
                param.requires_grad_(False)

        elif self.freeze_mode == 345:
            for param in model.layer1.parameters():
                param.requires_grad_(False)
            for param in model.layer2.parameters():
                param.requires_grad_(False)
            if self.reinit_layer:
                self.re_init_weights(model.layer3)
                self.re_init_weights(model.layer4)
                # init the fc layer anD B3 , B4
                model.linear.weight.data.normal_(mean=0.0, std=0.01)
                model.linear.bias.data.zero_()

        elif self.freeze_mode == 2345:
            for param in model.layer1.parameters():
                param.requires_grad_(False)
            if self.reinit_layer:
                self.re_init_weights(model.layer2)
                self.re_init_weights(model.layer3)
                self.re_init_weights(model.layer4)
                # init the fc layer and B2, B3, B4, B5
                model.linear.weight.data.normal_(mean=0.0, std=0.01)
                model.linear.bias.data.zero_()

        elif self.freeze_mode == 12345:
            if self.reinit_layer:
                self.re_init_weights(model.layer1)
                self.re_init_weights(model.layer2)
                self.re_init_weights(model.layer3)
                self.re_init_weights(model.layer4)
                # init all layers
                model.linear.weight.data.normal_(mean=0.0, std=0.01)
                model.linear.bias.data.zero_()

        # for name, param in model.named_parameters():
        #     if param.requires_grad: print(name)

        elif self.freeze_mode == 0:
            n = random.randint(2, 4)

            for param in model.layer1.parameters():
                param.requires_grad_(False)
            if n == 2:# init B2,B3
                self.re_init_weights(model.layer2)
                for param in model.layer3.parameters():
                    param.requires_grad_(False)
                for param in model.layer4.parameters():
                    param.requires_grad_(False)
            elif n == 3:
                for param in model.layer2.parameters():
                    param.requires_grad_(False)
                self.re_init_weights(model.layer3)
                for param in model.layer4.parameters():
                    param.requires_grad_(False)
            elif n == 4:
                for param in model.layer2.parameters():
                    param.requires_grad_(False)
                for param in model.layer3.parameters():
                    param.requires_grad_(False)
                self.re_init_weights(model.layer4)

            for param in model.linear.parameters():
                param.requires_grad_(False)

        return model


class RGP():
    def __init__(self, percentile, w_nat, w_rob, beta, layer_names):
        self.percentile = percentile * 100
        self.w_nat = w_nat
        self.w_rob = w_rob
        self.beta = beta
        self.layer_names = layer_names

    def retrieve_grad(self,optim):
        '''
        get the gradient of the parameters of the network with specific
        objective

        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grad, shape, has_grad = [], [], []
        for group in optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad

    def acc_grads(self, losses, optim):
        grads, shapes, has_grads = {}, {}, {}

        keys = ['m1_loss_nat_ce', 'm1_loss_rob_ce', 'loss', 'loss_ema']
        n_obj = len(keys)
        i = 0
        for n, obj in enumerate(losses):
            # print(obj)
            if not obj in keys:
                n+=1
                continue
            else:
                loss = losses[obj]
                optim.zero_grad(set_to_none=True)
                if i == (n_obj - 1):
                    loss.backward()
                else:
                    loss.backward(retain_graph=True)
                grad, shape, has_grad = self.retrieve_grad(optim)
                grads[obj] = grad
                has_grads[obj] = has_grad
                shapes[obj] = shape
                i+=1
        return grads, shapes, has_grads

    def fuse_grads(self, grads, shapes, has_grads):

        grad_nat = [grads['m1_loss_nat_ce'][i] * self.w_nat for i in range(len(grads['m1_loss_nat_ce']))]
        grad_rob = [grads['m1_loss_rob_ce'][i] * self.w_rob for i in range(len(grads['m1_loss_rob_ce']))]

        grad_rgp = [grad_nat[i] + grad_rob[i] for i in range(len(grad_nat))]

        grad_loss = []
        for l in range(len(grad_rgp)):
            grad_layer = grads['loss'][l]

            if l >0 and 'conv1' in self.layer_names[l]:
                percentile = np.percentile(grad_rgp[l].cpu(), self.percentile)
                grad_layer[grad_rgp[l] < percentile] = 0.0
            grad_loss.append(grad_layer)

        return grad_loss

    def fuse_grads_all(self, grads, shapes, has_grads):

        grad_nat = [grads['m1_loss_nat_ce'][i] * self.w_nat for i in range(len(grads['m1_loss_nat_ce']))]
        grad_rob = [grads['m1_loss_rob_ce'][i] * self.w_rob for i in range(len(grads['m1_loss_rob_ce']))]

        grad_rgp = [grad_nat[i] + grad_rob[i] for i in range(len(grad_nat))]
        grad_rgp = [grad_rgp[i] + grads['loss_ema'][i] for i in range(len(grad_nat))]

        grad_loss = []
        for l in range(len(grad_rgp)):
            grad_layer = grad_rgp[l]

            if l >0 and 'conv1' in self.layer_names[l]:
                percentile = np.percentile(grad_rgp[l].cpu(), self.percentile)
                grad_layer[grad_rgp[l] < percentile] = 0.0
            grad_loss.append(grad_layer)

        return grad_loss

    def set_grad(self, optim, grads):
        '''
        set the modified gradients to the network
        '''

        idx = 0
        for group in optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                p.grad = grads[idx]
                idx += 1
        return


