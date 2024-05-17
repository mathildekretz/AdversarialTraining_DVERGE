import os
import argparse, random
from tqdm import tqdm
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms

import utils, distillation, model

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def train_dverge_model(models, train_distill_loader, optimizers, criterion, epoch, adv=False):
    """perform the training of the submodels according to the DVERGE method"""
    config = utils.load_json()

    for m in models:
        m.to(device)
        m.train()
    losses = [0 for _ in range(len(models))]
    tot = 0

    l = random.randint(1, config["num_layers"])     #randomly uniformly chosen layer

    ##training
    batch_it = tqdm(train_distill_loader, desc='Batch', leave=False, position=2)
    for batch_idx, (si,sl,ti,tl) in enumerate(batch_it):    #iteration on distill loader (ie two pairs of data points)
        tot+=1
        si, sl = si.to(device), sl.to(device)
        ti, tl = ti.to(device), tl.to(device)

        distilled_data_list = []
        if adv :
            adv_data_list = []

        #iteration to get the features distillation objectives (the x' = weak features for each models)
        for model in models:
            tmp = distillation.Linf_distillation(model,si,ti,config["distill_epsilon"], config["distill_alpha"], config["distill_steps"], layer=l)
            distilled_data_list.append(tmp)     #the distilled feature for each model

            if adv :
                tmp = distillation.Linf_PGD(model, si, sl, eps=config["adv_epsilon"], alpha=config["adv_alpha"], steps=config["adv_steps"])
                adv_data_list.append(tmp)
        
        #iteration to perform SGD update with respect to features for the weak features not to be shared between the models
        for m,model in enumerate(models):
            loss = 0

            for j, distilled_data in enumerate(distilled_data_list):
                if m==j:
                    continue                    #diversify is wanted accross the models

                outputs=model(distilled_data)   #evaluate how weak are weak features of other submodels
                loss += criterion(outputs, sl)

            if adv :
                outputs=model(adv_data_list[m])
                loss += criterion(outputs,sl)

            losses[m] += loss.item()
            optimizers[m].zero_grad()
            loss.backward()
            optimizers[m].step()                #update to avoid shared weak features
    
    printed_message = 'Epoch [%3d] | ' % epoch 
    for m in range(len(models)):
        printed_message += 'Model{m:d}: {loss:.4f} '.format(m=m+1, loss=losses[m]/(tot))
        losses[m] = losses[m]/(tot)
    tqdm.write(printed_message)

    return losses


def valid_dverge_model(models, test_loader, criterion):
    """perform the validation of the submodels trained together following DVERGE"""
    for m in models:
        m.to(device)
        m.eval()
    loss = 0
    correct = 0
    total = 0

    ensemble = utils.Ensemble(models)       #mixture of the submodels

    ##validation
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = ensemble(inputs)
            loss += criterion(outputs, targets).item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

            total += inputs.size(0)

    print_message = 'Evaluation  | Ensemble Loss {loss:.4f} Acc {acc:.2%}'.format(
        loss=loss/len(test_loader), acc=correct/total)
    tqdm.write(print_message)

    return loss/len(test_loader)

def run(models, train_loader, test_loader, saving_path, adv=False):
    """perform the training and the validation of the DVERGE method"""
    config = utils.load_json()
    criterion = nn.CrossEntropyLoss()
    optimizers = []
    train_losses, valid_losses = [], []
    for model in models:
        optimizer = optim.SGD(model.parameters(), lr=config["dverge_lr"], momentum=0.9)
        optimizers.append(optimizer)
        train_losses.append([])

    #iteration over the training epochs
    epoch_it = tqdm(list(range(1,config["dverge_epochs"]+1)), total=config["dverge_epochs"], desc='Epoch', leave=True, position=1)
    for epoch in epoch_it :
        train_distill_loader = utils.DistillationLoader(train_loader,train_loader)
        
        ##training
        training = train_dverge_model(models=models, train_distill_loader=train_distill_loader, optimizers=optimizers, criterion=criterion, epoch=epoch, adv=adv)
        for i in range(len(models)):
            train_losses[i].append(training[i])
        
        ##validation
        valid_losses.append(valid_dverge_model(models=models, test_loader=test_loader, criterion=criterion))

    ##saving
    for i,model in enumerate(models) :
        model.save(saving_path[i])
        print(f"Model {i+1} save to '{saving_path[i]}'.")


    #plot of the losses
    epochs=[e+1 for e in range(config["dverge_epochs"])]
    print(len(epochs), len(train_losses[0]))
    for i,train_loss in enumerate(train_losses):
        plt.plot(epochs, train_loss, label=f'training loss of submodel {i+1}')
    plt.plot(epochs, valid_losses, label='validation loss')
    plt.legend(loc='best')
    plt.title('Plot of losses during training')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig('plotdvergeadvloss.png')

def main():
    ###change the parameters in config.json
    config=utils.load_json()
    valid_size, batch_size = config["valid_size"], config["batch_size"]

    ####Loading the pretrained models 
    models_paths=["models/submodel_1.pth", "models/submodel_2.pth", "models/submodel_3.pth"]
    models=[]
    for path in models_paths:
        modeli = model.Conv(path)
        modeli.load(path)
        models.append(modeli)

    ###New models
    DVERGE_models_paths=["models/DVERGEadv_submodel_1_25epochs.pth", "models/DVERGEadv_submodel_2_25epochs.pth", "models/DVERGEadv_submodel_3_25epochs.pth"]

    #### Model DVERGE training 
    print("Training DVERGE model")
    adv = True      #to change if the DVERGE training is wanted to be adversarial
    if adv : 
        print("The traing of DVERGE is adversarial")

    train_transform = transforms.Compose([transforms.ToTensor()]) 
    cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=train_transform)
    train_loader = utils.get_train_loader(cifar, valid_size, batch_size)
    test_loader = utils.get_validation_loader(cifar, valid_size, batch_size)
    run(models=models, train_loader=train_loader, test_loader=test_loader, saving_path=DVERGE_models_paths, adv=adv)

    #### Individual Model testing (Ensemble model testing is performed in test_project.py)
    for i,m in enumerate(models):
        DVERGE_model_path = DVERGE_models_paths[i]
        print("Testing with model from '{}'. ".format(DVERGE_model_path))

        cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=transforms.ToTensor()) 
        valid_loader = utils.get_validation_loader(cifar, valid_size, batch_size)

        m.load(DVERGE_model_path)

        acc = model.test_natural(m, valid_loader)
        print("Model natural accuracy (valid): {}".format(acc))


if __name__ == "__main__":
    main()

