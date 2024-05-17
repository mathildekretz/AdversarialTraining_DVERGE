import torch
import torch.nn as nn
import torch.nn.functional as F
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def gradient_wrt_input(model, inputs, targets, criterion=nn.CrossEntropyLoss()):
    """Perform a classical gradient descent given a loss funtion, input and output"""
    inputs.requires_grad = True
    
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    model.zero_grad()
    loss.backward()

    data_grad = inputs.grad.data
    return data_grad.clone().detach()


def gradient_wrt_feature(model, source_data, target_data, layer, criterion=nn.MSELoss()):
    """Perform a GD within the layer ie GD with respect to features of the model"""
    source_data.requires_grad = True
    
    #respect to feature
    out = model.get_features(x=source_data, layer=layer)
    target = model.get_features(x=target_data, layer=layer).data.clone().detach()
    
    #gradient descent
    loss = criterion(out, target)
    model.zero_grad()
    loss.backward()

    data_grad = source_data.grad.data
    return data_grad.clone().detach()


def Linf_PGD(model, dat, lbl, eps, alpha, steps, is_targeted=False, rand_start=True, mu=1, criterion=nn.CrossEntropyLoss()):
    """Classical implementation of PGD on l infinity norm"""
    x_nat = dat.clone().detach()
    x_adv = None
    if rand_start:
        x_adv = dat.clone().detach() + torch.FloatTensor(dat.shape).uniform_(-eps, eps).to(device)
    else:
        x_adv = dat.clone().detach()
    x_adv = torch.clamp(x_adv, 0., 1.) # respect image bounds
    g = torch.zeros_like(x_adv)

    #iteration over PGD steps
    for _ in range(steps):
        # Calculate gradient w.r.t. data
        grad = gradient_wrt_input(model, x_adv, lbl, criterion)
        with torch.no_grad():
            new_grad = grad
            # Get the sign of the gradient
            sign_data_grad = new_grad.sign()
            if is_targeted:
                x_adv = x_adv - alpha * sign_data_grad # perturb the data to MINIMIZE loss on tgt class
            else:
                x_adv = x_adv + alpha * sign_data_grad # perturb the data to MAXIMIZE loss on gt class
            x_adv = torch.max(torch.min(x_adv, x_nat+eps), x_nat-eps)
            x_adv = torch.clamp(x_adv, 0., 1.)
    return x_adv.clone().detach()


def Linf_distillation(model, dat, target, eps, alpha, steps, layer, mu=1, momentum=True, rand_start=False):
    """Implementation of distillation in a PGD way"""
    x_nat = dat.clone().detach()
    x_adv = None
    if rand_start:
        x_adv = dat.clone().detach() + torch.FloatTensor(dat.shape).uniform_(-eps, eps).to(device)
    else:
        x_adv = dat.clone().detach()
    x_adv = torch.clamp(x_adv, 0., 1.) # respect image bounds
    g = torch.zeros_like(x_adv)

    # Iteratively Perturb data
    for _ in range(steps):
        # Calculate gradient w.r.t. data
        grad = gradient_wrt_feature(model, x_adv, target, layer)
        with torch.no_grad():
            new_grad = grad
            x_adv = x_adv - alpha * new_grad.sign() # perturb the data to MINIMIZE loss on tgt class
            x_adv = torch.max(torch.min(x_adv, x_nat+eps), x_nat-eps)
            x_adv = torch.clamp(x_adv, 0., 1.)
    return x_adv.clone().detach()