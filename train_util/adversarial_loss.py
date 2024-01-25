import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from kd_lib.losses.kd_losses import dml_Loss

def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


# =============================================================================
# Adversarial Training
# =============================================================================
def trades_loss_v0(
        model,
        x_natural,
        y,
        optimizer,
        step_size=0.003,
        epsilon=0.031,
        perturb_steps=10,
        beta=1.0,
        distance='l_inf'
):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                           F.softmax(model(x_natural), dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(model(x_natural), dim=1))
    loss = loss_natural + beta * loss_robust
    return loss


def trades_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.007,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                args=None,
                distance='l_inf'):

    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(x_natural.device).detach()
    out_nat = model(x_natural)
    out_nat = out_nat[0] if isinstance(out_nat, tuple) else out_nat

    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                out_adv = model(x_adv)
                out_adv = out_adv[0] if isinstance(out_adv, tuple) else out_adv
                loss_kl = criterion_kl(F.log_softmax(out_adv, dim=1),
                                       F.softmax(out_nat, dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                out_adv = model(x_adv)
                out_adv = out_adv[0] if isinstance(out_adv, tuple) else out_adv

                loss = (-1) * criterion_kl(F.log_softmax(out_adv, dim=1),
                                           F.softmax(out_nat, dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    # calculate robust loss
    out_nat = model(x_natural)
    out_nat = out_nat[0] if isinstance(out_nat, tuple) else out_nat

    loss_natural = F.cross_entropy(out_nat, y)

    out_adv = model(x_adv)
    out_adv = out_adv[0] if isinstance(out_adv, tuple) else out_adv
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(out_adv, dim=1),
                                                    F.softmax(out_nat, dim=1))

    loss = loss_natural + beta * loss_robust

    return loss, out_adv, x_adv

def cure_loss_dual(model,
                model_ema,
                x_natural,
                y,
                optimizer,
                step_size=0.007,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                args=None,
                distance='l_inf'):

    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(x_natural.device).detach()
    out_nat = model(x_natural)
    out_nat = out_nat[0] if isinstance(out_nat, tuple) else out_nat

    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                out_adv = model(x_adv)
                out_adv = out_adv[0] if isinstance(out_adv, tuple) else out_adv
                loss_kl = criterion_kl(F.log_softmax(out_adv, dim=1),
                                       F.softmax(out_nat, dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                out_adv = model(x_adv)
                out_adv = out_adv[0] if isinstance(out_adv, tuple) else out_adv

                loss = (-1) * criterion_kl(F.log_softmax(out_adv, dim=1),
                                           F.softmax(out_nat, dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    # calculate robust loss
    out_nat = model(x_natural)
    out_nat = out_nat[0] if isinstance(out_nat, tuple) else out_nat
    #EMA
    out_nat_ema = model_ema(x_natural)
    out_nat_ema = out_nat_ema[0] if isinstance(out_nat_ema, tuple) else out_nat_ema

    if "deit" in args.model_architecture:
        loss_ema_nat = dml_Loss(args, None, None, out_nat, out_nat_ema.detach())
    else:
        loss_ema_nat = dml_Loss(args, model.feat, [ft.detach() for ft in model_ema.feat], out_nat, out_nat_ema.detach())

    loss_natural = F.cross_entropy(out_nat, y)

    out_adv = model(x_adv)
    out_adv = out_adv[0] if isinstance(out_adv, tuple) else out_adv
    # EMA
    out_adv_ema = model_ema(x_adv)
    out_adv_ema = out_adv_ema[0] if isinstance(out_adv_ema, tuple) else out_adv_ema

    if "deit" in args.model_architecture:
        loss_ema_adv = dml_Loss(args, None, None, out_adv, out_adv_ema.detach())
    else:
        loss_ema_adv = dml_Loss(args, model.feat, [ft.detach() for ft in model_ema.feat], out_adv, out_adv_ema.detach())

    loss_ema_nat.loss_dml = {i : loss_ema_nat.loss_dml[i] + loss_ema_adv.loss_dml[i] for i in loss_ema_nat.loss_dml.keys()}

    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(out_adv, dim=1),
                                                    F.softmax(out_nat, dim=1))

    if args.reinit_mode == 'rgp_old':
        loss = (args.w_nat * loss_natural) + (args.w_rob * loss_robust)
    elif args.reinit_mode == 'rgp':
        loss_robust = F.cross_entropy(out_adv, y)
        loss = (args.w_nat * loss_natural) + (args.w_rob * loss_robust)
    elif args.reinit_mode == 'rgp_soft':
        loss_nat_ce = loss_natural
        loss_robust_ce = loss_robust
        loss = loss_natural + beta * loss_robust
        return loss_nat_ce, loss_robust_ce, loss, loss_ema_nat, out_adv
    else:
        loss = loss_natural + beta * loss_robust

    return loss, loss_ema_nat, out_adv, x_adv


def madry_loss(model1,
               x1,
               y,
               optimizer,
               step_size=0.007,
               epsilon=0.031,
               perturb_steps=10,
               reduce=True
               ):

    model1.eval()
    # generate adversarial example
    x_adv = x1.detach() + 0.001 * torch.randn(x1.shape).cuda().detach()

    for _ in range(perturb_steps):
        x_adv.requires_grad_()

        with torch.enable_grad():
            # loss_kl = F.cross_entropy(F.log_softmax(model(x_adv), dim=1), y)

            out_adv = model1(x_adv)
            out_adv = out_adv[0] if isinstance(out_adv, tuple) else out_adv
            loss = F.cross_entropy(out_adv, y)

        grad = torch.autograd.grad(loss, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x1 - epsilon), x1 + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model1.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    # calculate robust loss
    out_adv = model1(x_adv)
    out_adv = out_adv[0] if isinstance(out_adv, tuple) else out_adv

    if reduce:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss(reduction='none')

    loss_adv = criterion(out_adv, y)

    return loss_adv, out_adv, x_adv



