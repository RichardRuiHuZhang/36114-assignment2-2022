# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 12:55:27 2022

@author: RRZ
"""
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

number_workers = 0
batch_size = 20
validation_size = 0.2

Converting_tensor_transform = transforms.ToTensor()

training_data = datasets.MNIST(root = 'data', train = True, download = True, transform = Converting_tensor_transform)

testing_data = datasets.MNIST(root = 'data', train = False, download = True, transform = Converting_tensor_transform)

number_of_train = len(training_data)

train_indices = list(range(number_of_train))

np.random.shuffle(train_indices)

splitting = int(np.floor(validation_size * number_of_train))

training_index, validation_index = train_indices[splitting:], train_indices[:splitting]

training_sampler = SubsetRandomSampler(training_index)

validation_sampler = SubsetRandomSampler(validation_index)

training_loader_batch = torch.utils.data.DataLoader(training_data, batch_size = batch_size, sampler = training_sampler, num_workers = number_workers)

validation_loader_batch = torch.utils.data.DataLoader(training_data, batch_size = batch_size,sampler = validation_sampler, num_workers = number_workers)

testing_loader_batch = torch.utils.data.DataLoader(testing_data, batch_size = batch_size, num_workers = number_workers)


class neuNet(nn.Module):
    def __init__(self):
      super(neuNet,self).__init__()
      ## This is the number of hidden nodes in each layer (512)
      hidden_1 = 512
      hidden_2 = 512

      ## This is the linear layer (784 -> hidden_1)
      self.fc1 = nn.Linear(28*28, 512)

      ## This is also linear layer but (n_hidden -> hidden_2)
      self.fc2 = nn.Linear(512,512)

      ##This is the linear layer with (n_hidden -> 10)
      self.fc3 = nn.Linear(512,10)

      #The dropout layer (p=0.2)
      #Also the dropout prevents overfitting of data
      self.droput = nn.Dropout(0.2)

        
    def forward(self,x):
      ## flattening the image input
          x = x.view(-1,28*28)

      ## adding the hidden layer, for activation we are using relu activation
          x = F.relu(self.fc1(x))

      ## adding the dropout layer
          x = self.droput(x)

      ## adding the hidden layer, for activation we are using relu activation
          x = F.relu(self.fc2(x))

      ## adding the dropout layer
          x = self.droput(x)

      ## adding the output layer
          x = self.fc3(x)
          return x
      
MLP_model = neuNet()

learning_rate = 0.01

MLP_criterion = nn.CrossEntropyLoss()

Model_optimizer = torch.optim.SGD(MLP_model.parameters(),lr =learning_rate)

num_epochs = 1

validation_loss_min = np.Inf

for epoch in range(num_epochs): # monitoring the losses
        training_loss = 0
        validation_loss = 0
        
        MLP_model.train()
        for data,label in training_loader_batch:
            Model_optimizer.zero_grad() 
            output = MLP_model(data)
            loss = MLP_criterion(output,label)
            loss.backward()
            Model_optimizer.step()
            training_loss += loss.item()
            
        MLP_model.eval()
        for data,label in validation_loader_batch:
            output = MLP_model(data)
            loss = MLP_criterion(output,label)
            validation_loss = loss.item() * data.size(0)
            training_loss = training_loss / len(training_loader_batch.sampler)
        
        print('Epoch: %5s, Test Loss: %6d\n' %(str(epoch), training_loss))
            

testing_loss = 0.0
class_corrected = list(0. for i in range(10))
class_total = list(0. for i in range(10))

MLP_model.eval() ## here we are preparing the model for evaluation

for data, target in testing_loader_batch:
    output = MLP_model(data) 
    loss = MLP_criterion(output,target) ## Calculating the loss
    testing_loss += loss.item()*data.size(0) ## updating the running validation loss
    _, predictions = torch.max(output, 1) ##converting the output probabilities to predicted class
    corrected = np.squeeze(predictions.eq(target.data.view_as(predictions))) ## comparing the predictions to true label

    for vals in range(len(target)): ## calculating the test accuracy for each object class
        label = target.data[vals]
        class_corrected[label] += corrected[vals].item()
        class_total[label] += 1

testing_loss = testing_loss/len(testing_loader_batch.sampler) 
print('The Test Loss: {:.6f}\n'.format(testing_loss))

for ele in range(10):
     if class_total[ele] > 0:
         print('The Testing Accuracy of %5s: %2d%% (%2d/%2d)' % (str(ele), 100 * class_corrected[ele] / class_total[ele], np.sum(class_corrected[ele]), np.sum(class_total[ele])))
     else:
         print('Test Accuracy of %5s: N/A (no training examples)' % (class_total[ele]))
        
print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (100. * np.sum(class_corrected) / np.sum(class_total), np.sum(class_corrected), np.sum(class_total)))
        
