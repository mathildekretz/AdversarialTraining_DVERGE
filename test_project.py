#!/usr/bin/env python3

import os, os.path, sys
import argparse
import importlib 
import importlib.abc
import torch, torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
from utils import get_validation_loader


torch.seed()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def load_project(project_dir):
    module_filename = os.path.join(project_dir, 'model.py')
    if os.path.exists(project_dir) and os.path.isdir(project_dir) and os.path.isfile(module_filename):
        print("Found valid project in '{}'.".format(project_dir))
    else:
        print("Fatal: '{}' is not a valid project directory.".format(project_dir))
        raise FileNotFoundError 

    sys.path = [project_dir] + sys.path
    spec = importlib.util.spec_from_file_location("model", module_filename)
    project_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(project_module)

    return project_module


def test_natural(net, test_loader, num_samples):
    correct = 0
    total = 0
    with torch.no_grad():
        for i,data in enumerate(test_loader, 0):
            images, labels = data[0].to(device), data[1].to(device)
            total = 0
            correct = 0
            for _ in range(num_samples):
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

    return 100 * correct / total


def fgsm_attack(image, epsilon, data_grad):
    """performs FGSM attacks"""
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    return perturbed_image


def test_fgsm(net, test_loader, epsilon, save_dir=None):
    """test results on fgsm attacks"""
    correct = 0
    total = 0

    for i, data in enumerate(test_loader, 0):
        images, labels = data[0].to(device), data[1].to(device)
        images.requires_grad = True
        outputs = net(images)
        loss = F.cross_entropy(outputs, labels)
        net.zero_grad()
        loss.backward()
        data_grad = images.grad.data

        perturbed_images = fgsm_attack(images, epsilon, data_grad)
        outputs_perturbed = net(perturbed_images)
        outputs_original = net(images)
        _, predicted_original = torch.max(outputs_original.data, 1)
        _, predicted_perturbed = torch.max(outputs_perturbed.data, 1)
        total += labels.size(0)
        correct += (predicted_perturbed == labels).sum().item()

        if save_dir is not None:
            save_adversarial_images(images, labels, predicted_original, predicted_perturbed, save_dir)

    return 100 * correct / total


def pgd_attack(net, images, labels, epsilon, alpha, iters):
    """performs PGD linf attacks"""
    images = images.clone().detach().requires_grad_(True)

    for _ in range(iters):
        outputs = net(images)
        loss = F.cross_entropy(outputs, labels)
        net.zero_grad()

        # Retain the gradient for non-leaf tensors
        images.retain_grad()

        loss.backward(retain_graph=True)

        # Ensure the backward pass is executed before accessing the gradient
        if images.grad is not None:
            data_grad = images.grad.detach().sign()
        else:
            data_grad = torch.zeros_like(images)

        # PGD step
        perturbed_images = images + alpha * data_grad
        perturbed_images = torch.clamp(perturbed_images, 0, 1)

        # Project perturbed image back to epsilon-ball around the original image
        perturbation = perturbed_images - images
        perturbation = torch.clamp(perturbation, -epsilon, epsilon)
        images = images + perturbation
        images = torch.clamp(images, 0, 1)

    return images.detach()


def test_pgd(net, test_loader, epsilon, alpha, iters, save_dir=None):
    """test results on PGD linf attacks"""
    correct = 0
    total = 0

    for i, data in enumerate(test_loader, 0):
        images, labels = data[0].to(device), data[1].to(device)

        perturbed_images = pgd_attack(net, images, labels, epsilon, alpha, iters)

        outputs_original = net(images)
        outputs_perturbed = net(perturbed_images)

        _, predicted_original = torch.max(outputs_original.data, 1)
        _, predicted_perturbed = torch.max(outputs_perturbed.data, 1)

        total += labels.size(0)
        correct += (predicted_perturbed == labels).sum().item()

        if save_dir is not None:
            save_adversarial_images(images, labels, predicted_original, predicted_perturbed, save_dir)

    return 100 * correct / total


def pgd_attack_l2(net, images, labels, epsilon, alpha, iters):
    """performs PGD l2 attacks"""
    images = images.clone().detach().requires_grad_(True)

    for _ in range(iters):
        outputs = net(images)
        loss = F.cross_entropy(outputs, labels)
        net.zero_grad()

        # Retain the gradient for non-leaf tensors
        images.retain_grad()

        loss.backward(retain_graph=True)

        # Ensure the backward pass is executed before accessing the gradient
        if images.grad is not None:
            data_grad = images.grad.detach()
        else:
            data_grad = torch.zeros_like(images)

        # PGD step in L2 norm
        data_grad_norm = torch.norm(data_grad.view(data_grad.size(0), -1), p=2, dim=1, keepdim=True)
        scaled_data_grad = data_grad / (data_grad_norm.view(-1, 1, 1, 1) + 1e-10) # Add a small constant to avoid division by zero
        perturbed_images = images + alpha * scaled_data_grad
        perturbation = perturbed_images - images

        # Project perturbed image back to epsilon-ball around the original image in L2 norm
        perturbation_norm = torch.norm(perturbation.view(perturbation.size(0), -1), p=2, dim=1, keepdim=True)
        perturbation = epsilon * perturbation / (perturbation_norm.view(-1, 1, 1, 1) + 1e-10)
        images = images + perturbation

        images = torch.clamp(images, 0, 1)

    return images.detach()

def test_pgd_l2(net, test_loader, epsilon, alpha, iters, save_dir=None):
    """test results on PGD l2 attacks"""
    correct = 0
    total = 0

    for i, data in enumerate(test_loader, 0):
        images, labels = data[0].to(device), data[1].to(device)

        perturbed_images = pgd_attack_l2(net, images, labels, epsilon, alpha, iters)

        outputs_original = net(images)
        outputs_perturbed = net(perturbed_images)

        _, predicted_original = torch.max(outputs_original.data, 1)
        _, predicted_perturbed = torch.max(outputs_perturbed.data, 1)

        total += labels.size(0)
        correct += (predicted_perturbed == labels).sum().item()

        if save_dir is not None:
            save_adversarial_images(images, labels, predicted_original, predicted_perturbed, save_dir)

    return 100 * correct / total


def save_adversarial_images(images, labels, predicted_original, predicted_perturbed, save_dir):
    """Save perturbed data for vizualization"""
    for i in range(images.size(0)):
        if predicted_original[i] == labels[i] and predicted_perturbed[i] != predicted_original[i]:
            # Save misclassified adversarial examples
            image = images[i].detach()
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, f"adversarial_{i}_true_{labels[i]}_pred_original_{predicted_original[i]}_pred_perturbed_{predicted_perturbed[i]}.png")
            save_image(image, save_path)


def main():
    ##arguments for testing 
    parser = argparse.ArgumentParser()
    parser.add_argument("project_dir", metavar="project-dir", nargs="?", default=os.getcwd(),
                        help="Path to the project directory to test.")
    parser.add_argument("-b", "--batch-size", type=int, default=256,
                        help="Set batch size.")
    parser.add_argument("-s", "--num-samples", type=int, default=1,
                        help="Num samples for testing (required to test randomized networks).")
    parser.add_argument("--epsilon", type=float, default=0.03,
                        help="Epsilon value for FGSM and PGD attack.")

    args = parser.parse_args()
    project_module = load_project(args.project_dir)

    ##model loading
    net = project_module.Net()
    net.to(device)
    net.load_for_testing(project_dir=args.project_dir)

    transform = transforms.Compose([transforms.ToTensor()])
    cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=transform)
    valid_loader = get_validation_loader(cifar, batch_size=args.batch_size)

    # Natural accuracy
    net.eval()
    acc_nat_average = 0
    for i in range(10):
        valid_loader_acc_nat = get_validation_loader(cifar, batch_size=args.batch_size)
        acc_nat = test_natural(net, valid_loader_acc_nat, num_samples=args.num_samples)
        acc_nat_average += acc_nat
    acc_nat_average = acc_nat_average/10
    print("Model nat accuracy (averaged on 10 tests): {}".format(acc_nat_average))

    # FGSM attack accuracy
    net.eval()
    acc_fgsm = test_fgsm(net, valid_loader, epsilon=args.epsilon, save_dir=None)
    print("Model FGSM accuracy (test): {}".format(acc_fgsm))

    # PGD Linf attack accuracy
    net.eval()
    acc_pgd_linf = test_pgd(net, valid_loader, epsilon=args.epsilon, alpha=0.007,
                            iters=5, save_dir=None)
    print("Model PGD Linf accuracy (test): {}".format(acc_pgd_linf))

    # PGD L2 attack accuracy
    net.eval()
    acc_pgd_l2 = test_pgd_l2(net, valid_loader, epsilon=args.epsilon, alpha=0.007,
                             iters=5, save_dir=None)
    print("Model PGD L2 accuracy (test): {}".format(acc_pgd_l2))


if __name__ == "__main__":
    main()
