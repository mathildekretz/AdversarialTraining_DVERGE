import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from test_project import test_natural, test_fgsm, test_pgd, test_pgd_l2, get_validation_loader
from model import Conv
from utils import Ensemble


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


# Instantiating the submodels
submodel_paths = ["models/DVERGE_submodel_1.pth", "models/DVERGE_submodel_2.pth", "models/DVERGE_submodel_3.pth"]
submodels = []
for path in submodel_paths:
    submodel = Conv(path).to(device)
    submodel.load()
    submodels.append(submodel)

# Instantiating the Ensemble model
ensemble = Ensemble(submodels)
ensemble.to(device)

transform = transforms.Compose([transforms.ToTensor()])
cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=transform)
valid_loader = get_validation_loader(cifar, batch_size=256)

# Natural accuracy
ensemble.eval()
acc_nat = test_natural(ensemble, valid_loader, num_samples=1)
print("Model nat accuracy (test): {}".format(acc_nat))

# FGSM attack accuracy
ensemble.eval()
acc_fgsm = test_fgsm(ensemble, valid_loader, epsilon=0.03, save_dir=None)
print("Model FGSM accuracy (test): {}".format(acc_fgsm))

# PGD Linf attack accuracy
ensemble.eval()
acc_pgd_linf = test_pgd(ensemble, valid_loader, epsilon=0.03, alpha=0.01,
                        iters=5, save_dir=None)
print("Model PGD Linf accuracy (test): {}".format(acc_pgd_linf))

# PGD L2 attack accuracy
ensemble.eval()
acc_pgd_l2 = test_pgd_l2(ensemble, valid_loader, epsilon=0.03, alpha=0.01,
                         iters=5, save_dir=None)
print("Model PGD L2 accuracy (test): {}".format(acc_pgd_l2))
