#!/usr/bin/env python3 
import os
import argparse, random, json
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from tqdm import tqdm
import utils

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

def load_json(file_path="./config.json"):
    """load the configuration file with """
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config 

config = load_json()
batch_size = config["batch_size"]
valid_size = config["valid_size"]

class Conv(nn.Module):
    '''Basic convolutional neural network architecture (from pytorch doc).'''

    def __init__(self, model_path):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.path = model_path
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def save(self, model_file=None):
        '''Helper function, use it to save the model weights after training.'''
        if model_file == None :
            torch.save(self.state_dict(), self.path)
        else : torch.save(self.state_dict(), model_file)

    def load(self, model_file=None):
        if model_file==None :
            self.load_state_dict(torch.load(self.path, map_location=torch.device(device)))
        else : self.load_state_dict(torch.load(model_file, map_location=torch.device(device)))

        
    def load_for_testing(self, project_dir='./'):
        '''This function will be called automatically before testing your
           project, and will load the model weights from the file
           specify in Net.model_file.
           
           You must not change the prototype of this function. You may
           add extra code in its body if you feel it is necessary, but
           beware that paths of files used in this function should be
           refered relative to the root of your project directory.
        '''        
        self.load(os.path.join(project_dir, self.path))
    
    def get_features(self, x, layer):
        """return the feature of layer 'layer' of the model"""
        #before_ReLu = True 
        x = self.conv1(x)
        if layer == 1 : 
            return x
        x = self.pool(F.relu(x))
        x = self.conv2(x)
        if layer == 2 : 
            return x
        x = self.pool(F.relu(x))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        if layer == 3 :
            return x 
        x = F.relu(x)
        x = self.fc2(x)
        if layer == 4 :
            return x
        x = F.relu(x)
        x = self.fc3(x)
        if layer == 5 :
            return x
        else : 
            raise ValueError('layer {:d} is out of index!'.format(layer))

class Net(nn.Module):
    """Ensemble class that output the mixture of models"""
    
    #default values to test on the plateform
    paths, models = ['models/DVERGE_submodel_1.pth', 'models/DVERGE_submodel_2.pth', 'models/DVERGE_submodel_3.pth'], []
    for path in paths:
        model = Conv(path)
        model.load()
        models.append(model)

    def __init__(self, models=models):
        super(Net, self).__init__()
        self.models = models
        assert len(self.models) > 0

    def forward(self, x):
        if len(self.models) > 1:
            outputs = 0
            for model in self.models:
                outputs += F.softmax(model(x), dim=-1)
            output = outputs / len(self.models)
            output = torch.clamp(output, min=1e-40)
            return torch.log(output)
        else:
            return self.models[0](x)
    
    def load_for_testing(self, project_dir='./'):
        for model in self.models:
            model.load()
            model.to(device)

def train_model(models, train_loader, test_loader):
    """train the vanilla submodels"""
    print("Starting training")
    config = utils.load_json()
    criterion = nn.CrossEntropyLoss()
    optimizers=[]           #one optimizer per submodels
    training_losses,validation_losses=[],[]
    for model in models : 
        optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)
        optimizers.append(optimizer)            
        training_losses.append([])
        validation_losses.append([])
    
    for epoch in tqdm(range(config["epochs"])):  # loop over the dataset multiple times
        for m, model in enumerate(models) :         # train all the submodels 
            running_loss, training_loss, valid_loss = 0.0, 0, 0
            model.train()
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)

                # forward + backward + optimize
                optimizer = optimizers[m]
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                training_loss += loss.item()
                if i % 500 == 499:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
            training_losses[m].append(training_loss/len(train_loader))

            model.eval()

            #validation 
            for i, data in enumerate(test_loader):
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                valid_loss += loss.item()
            validation_losses[m].append(valid_loss/len(test_loader))
    
    #saving of the trained submodels
    for model in models :
        model.save()
        print('Model saved in {}'.format(model.path))

    return training_losses, validation_losses

def test_natural(net, test_loader):
    '''Basic testing function.'''

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i,data in enumerate(test_loader, 0):
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

def main():
    ###Change the parameters in the config.json

    #### Create models and move it to whatever device is available (gpu/cpu)
    models_paths=["models/submodel_1.pth", "models/submodel_2.pth", "models/submodel_3.pth"]
    models=[]
    for path in models_paths:
        modeli = Conv(path)
        modeli.to(device)
        models.append(modeli)

    #### Model training and validation
    cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=transforms.ToTensor())
    train_loader = utils.get_train_loader(cifar, valid_size, batch_size)
    valid_loader = utils.get_validation_loader(cifar, valid_size, batch_size)
    train_loss, valid_loss = train_model(models, train_loader, valid_loader)

    ##Plot of losses 
    epochs = [i+1 for i in range(config["epochs"])]
    for m, model in enumerate(models):
        plt.plot(epochs, train_loss[m], label=f'training loss of submodel {m}')
        plt.plot(epochs, valid_loss[m], label=f'validation loss of submodel {m}')
    plt.title('Losses of baseline training')
    plt.legend(loc='best')
    plt.xlabel('epochs')
    plt.ylabel('losses')
    plt.savefig('plotbaselineloss.png')

    # Note: You should not change the transform applied to the
    # validation dataset since, it will be the only transform used
    # during final testing.
    


if __name__ == "__main__":
    main()

