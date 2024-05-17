import os
import argparse, random
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from tqdm import tqdm

import utils, distillation, model
from model import Conv

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

config = utils.load_json()
batch_size = config["batch_size"]
valid_size = config["valid_size"]

def train_adv_model(models, train_loader, test_loader, pth_filename=None):
    '''Adversarial training function'''
    print("Starting training")
    config = utils.load_json()
    criterion = nn.CrossEntropyLoss()
    optimizers=[]
    training_losses,validation_losses=[],[]
    for model in models :
        optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)
        optimizers.append(optimizer)
        training_losses.append([])
        validation_losses.append([])
    
    #iteration over the epochs
    for epoch in tqdm(range(config["epochs"])):  
        for m, model in enumerate(models) :
            running_loss, training_loss, valid_loss = 0.0, 0, 0
            model.train()
            optimizer = optimizers[m]

            ##training on clean data 
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)

                # forward + backward + optimize
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                training_loss += loss.item()
                
            ##training on attacks
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                
                optimizer.zero_grad()
                adv_inputs = distillation.Linf_PGD(model, inputs, labels, eps=config["adv_epsilon"], alpha=config["adv_alpha"], steps=config["adv_steps"])
                outputs = model(adv_inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizers[m].step()            

                training_loss += loss.item()

            training_losses[m].append(training_loss/(len(train_loader)*2)) #loss on real data and attacked data

            ##validation
            model.eval()
            for i, data in enumerate(test_loader):
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                valid_loss += loss.item()
            validation_losses[m].append(valid_loss/len(test_loader))
    
    ##saving
    for model in models :
        model.save()
        print('Model saved in {}'.format(model.path))

    return training_losses, validation_losses


def main():
    ###Change the parameters in the config.json

    #### Create models and move it to whatever device is available (gpu/cpu)
    models_paths=["models/adv_submodel_1.pth", "models/adv_submodel_2.pth", "models/adv_submodel_3.pth"]
    models=[]
    for path in models_paths:
        modeli = Conv(path)
        modeli.to(device)
        models.append(modeli)

    #### Model training and validation
    cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=transforms.ToTensor())
    train_loader = utils.get_train_loader(cifar, valid_size, batch_size)
    valid_loader = utils.get_validation_loader(cifar, valid_size, batch_size)
    train_loss, valid_loss = train_adv_model(models, train_loader, valid_loader)

    ##Plot of losses 
    epochs = [i+1 for i in range(config["epochs"])]
    for m, model in enumerate(models):
        plt.plot(epochs, train_loss[m], label=f'training loss of submodel {m}')
        plt.plot(epochs, valid_loss[m], label=f'validation loss of submodel {m}')
    plt.title('Losses of baseline training')
    plt.legend(loc='best')
    plt.xlabel('epochs')
    plt.ylabel('losses')
    plt.savefig('plotadvloss.png')


if __name__ == "__main__":
    main()
