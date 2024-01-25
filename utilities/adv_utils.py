import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imsave
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn

from PIL import Image
import torchvision.transforms as transforms
# =============================================================================
# Robustness Evaluation
# =============================================================================
def cwloss(output, target,confidence=50, num_classes=10):
    # Compute the probability of the label class versus the maximum other
    # The same implementation as in repo CAT https://github.com/sunblaze-ucb/curriculum-adversarial-training-CAT
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)
    other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
    loss = -torch.clamp(real - other + confidence, min=0.)  # equiv to max(..., 0.)
    loss = torch.sum(loss)
    return loss

def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon,
                  num_steps,
                  step_size,
                  random,
                  device,
                  save_imgs = False,
                  ind = 0,
                  lip=False
                  ):

    out = model(X)
    out = out[0] if isinstance(out, tuple) else out
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)

    if random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            out = model(X_pgd)
            out = out[0] if isinstance(out, tuple) else out
            loss = nn.CrossEntropyLoss()(out, y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    if save_imgs:
        visualize(X, X_pgd, random_noise, epsilon, ind)

    out = model(X_pgd)
    out = out[0] if isinstance(out, tuple) else out
    err_pgd = (out.data.max(1)[1] != y.data).float().sum()
    # print('err pgd (white-box): ', err_pgd)

    if lip:
        out_pgd = model(X_pgd).data.max(1)[1].detach()
        out_clean = model(X).data.max(1)[1].detach()
        stable = (out_pgd.data == out_clean.data).float().sum()
        return err, err_pgd, X_pgd, stable.item()
    return err, err_pgd


def cnw(model,
        X,
        y,
        epsilon,
        num_steps,
        step_size,
        random,
        device,
        save_imgs = False,
        ind = 0,
        num_classes=10
        ):

    out = model(X)
    out = out[0] if isinstance(out, tuple) else out
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)

    if random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            out = model(X_pgd)
            out = out[0] if isinstance(out, tuple) else out
            loss = cwloss(out, y, num_classes=num_classes)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    if save_imgs:
        visualize(X, X_pgd, random_noise, epsilon, ind)

    out = model(X_pgd)
    out = out[0] if isinstance(out, tuple) else out
    err_pgd = (out.data.max(1)[1] != y.data).float().sum()
    # print('err pgd (white-box): ', err_pgd)
    return err, err_pgd


def visualize(X, X_adv, X_grad, epsilon, ind):

    X = X.to('cpu')
    X_adv = X_adv.to('cpu')
    X_grad = X_grad.to('cpu')
    dst = "/volumes2/feature_prior_project/art/feature_prior/vis/adv2/base"

    for i in range(X.shape[0]):
        x = X[i].squeeze().detach().cpu().numpy()  # remove batch dimension # B X C H X W ==> C X H X W
        # x = x.mul(torch.FloatTensor(std).view(3, 1, 1)).add(
        #     torch.FloatTensor(mean).view(3, 1, 1)).numpy()  # reverse of normalization op- "unnormalize"
        #x = upsamp(x)
        x = np.transpose(x, (1, 2, 0))  # C X H X W  ==>   H X W X C
        x = np.clip(x, 0, 1)

        x_adv = X_adv[i].squeeze().detach().cpu().numpy()
        # x_adv = x_adv.mul(torch.FloatTensor(std).view(3, 1, 1)).add(
        #     torch.FloatTensor(mean).view(3, 1, 1)).numpy()  # reverse of normalization op
        #x_adv = upsamp(x_adv)
        x_adv = np.transpose(x_adv, (1, 2, 0))  # C X H X W  ==>   H X W X C
        x_adv = np.clip(x_adv, 0, 1)

        x_grad = X_grad[i].squeeze().detach().cpu().numpy()
        x_grad = np.transpose(x_grad, (1, 2, 0))
        x_grad = np.clip(x_grad, 0, 1)
        #x_grad = upsamp(x_grad)
        diff = (x - x_adv) * 30
        diff = np.clip(diff, 0, 1)
        imsave(os.path.join(dst, 'noise_%i.png' % (i+ind)), diff, format="png")
        imsave(os.path.join(dst, 'noisegrad_%i.png' % (i+ind)), x_grad, format="png")

        figure, ax = plt.subplots(1, 3, figsize=(7, 3))
        ax[0].imshow(x)
        ax[0].set_title('Clean Example', fontsize=10)
        imsave(os.path.join(dst, 'orig_%i.png' % (i+ind)), x, format="png")

        ax[1].imshow(diff)
        ax[1].set_title('Perturbation', fontsize=10)
        ax[1].set_yticklabels([])
        ax[1].set_xticklabels([])
        ax[1].set_xticks([])
        ax[1].set_yticks([])

        ax[2].imshow(x_adv)
        ax[2].set_title('Adversarial Example', fontsize=10)
        imsave(os.path.join(dst, 'adv_%i.png' % (i+ind)), x_adv, format="png")

        ax[0].axis('off')
        ax[2].axis('off')

        ax[0].text(1.1, 0.5, "+{}*".format(round(epsilon, 3)), size=15, ha="center",
                   transform=ax[0].transAxes)

        #plt.show()
        figure.savefig(os.path.join(dst, '%i.png' % (i+ind)), dpi=300, bbox_inches='tight')

def upsamp(img):

    curr_size = img.shape[0]
    upsample_size = 224
    resize_up = transforms.Resize(max(curr_size, upsample_size), 3)
    img = transforms.ToPILImage()(img)
    upsamp_img = np.array(resize_up(Image.fromarray(img)))
    return upsamp_img

def local_lip(model, x, xp, top_norm=1, btm_norm=float('inf'), reduction='mean'):
    model.eval()
    down = torch.flatten(x - xp, start_dim=1)
    with torch.no_grad():
        if top_norm == "kl":
            criterion_kl = nn.KLDivLoss(reduction='none')
            top = criterion_kl(F.log_softmax(model(xp), dim=1),
                               F.softmax(model(x), dim=1))
            ret = torch.sum(top, dim=1) / torch.norm(down + 1e-6, dim=1, p=btm_norm)
        else:
            top = torch.flatten(model(x), start_dim=1) - torch.flatten(model(xp), start_dim=1)
            ret = torch.norm(top, dim=1, p=top_norm) / torch.norm(down + 1e-6, dim=1, p=btm_norm)

    if reduction == 'mean':
        return torch.mean(ret)
    elif reduction == 'sum':
        return torch.sum(ret)
    else:
        raise ValueError("Not supported reduction")


def eval_adv_robustness(
    model,
    data_loader,
    device='cuda',
    epsilon=0.031,
    num_steps=20,
    step_size=0.003,
    random=True,
    save_imgs = False,
    lip=False,
    attack = 'pgd',
    num_classes=10
):

    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    natural_err_total = 0
    lip_total = 0
    lip_count=0
    pert_stab = 0
    ind = 0

    for data, target in tqdm(data_loader, desc='robustness'):
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        if attack == 'pgd':
            if lip:
                err_natural, err_robust, X_pgd, stab = _pgd_whitebox(model, X, y, epsilon, num_steps, step_size, random, device, save_imgs, ind, True)
                lip = local_lip(model, X, X_pgd).item()
                lip_total += lip
                pert_stab += stab
            else:
                err_natural, err_robust = _pgd_whitebox(model, X, y, epsilon, num_steps, step_size, random, device, save_imgs, ind)
        elif attack == 'cnw':
            err_natural, err_robust = cnw(model, X, y, epsilon, num_steps, step_size, random, device, save_imgs, ind, num_classes)

        robust_err_total += err_robust
        natural_err_total += err_natural
        lip_count += 1
        ind += 65

    nat_err = natural_err_total.item()
    successful_attacks = robust_err_total.item()
    total_samples = len(data_loader.dataset)

    rob_acc = (total_samples - successful_attacks) / total_samples
    nat_acc = (total_samples - nat_err) / total_samples

    print('=' * 30)
    print(f"Adversarial Robustness = {rob_acc * 100} % ({total_samples - successful_attacks}/{total_samples})")
    print(f"Natural Accuracy = {nat_acc * 100} % ({total_samples - nat_err}/{total_samples})")

    if lip:
        lip_avg =lip_total/lip_count
        pert_stab_avg = pert_stab/total_samples
        return nat_acc, rob_acc, lip_avg, pert_stab_avg
    return nat_acc, rob_acc
