import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepNeuralNet1(nn.Module):
    def __init__(self, input_dim):
      super(DeepNeuralNet1,self).__init__()
      hidden_1 = 512
      hidden_2 = 512
      self.fc1 = nn.Linear(input_dim, 512)
      self.fc2 = nn.Linear(512,512)
      self.fc3 = nn.Linear(512,103)
      self.droput = nn.Dropout(0.2)
        
    def forward(self,x):
          x = F.relu(self.fc1(x))
          x = self.droput(x)
          x = F.relu(self.fc2(x))
          x = self.droput(x)
          x = self.fc3(x)
          return x