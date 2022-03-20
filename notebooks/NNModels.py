from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import dump
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score #roc_auc_score
from category_encoders.ordinal import OrdinalEncoder
from sklearn.preprocessing import label_binarize

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


#df_data_raw = pd.read_csv('G:/Data Science/UTS Courses/36114 Advanced Data Science for Innovation/Assignment2/36114-assignment2-2022/data/raw/beer_reviews.csv')
df_data_raw = pd.read_csv('G:/Data Science/UTS Courses/36114 Advanced Data Science for Innovation/Assignment2/36114-assignment2-2022/data/raw/beer_reviews_reduced.csv')
#df_data_raw = pd.read_csv('../data/raw/beer_reviews.csv')

df_data_clean1 = df_data_raw.copy()

col_usable = ['brewery_name','review_aroma','review_appearance','review_palate','review_taste','beer_abv','beer_style']
independent_cols = ['brewery_name','review_aroma','review_appearance','review_palate','review_taste','beer_abv']
numerical_cols = ['review_aroma','review_appearance','review_palate','review_taste','beer_abv']
factor_cols = ['brewery_name']
target_col = ['beer_style']
#target = df_data_clean1.pop('beer_style')

df_data_reduced1 = df_data_clean1.loc[:,col_usable]

df_data_reduced2 = df_data_reduced1.copy()
df_data_reduced2 = df_data_reduced2.dropna()
target = df_data_reduced2.pop('beer_style')

factor_encoder = OrdinalEncoder()
df_data_reduced2[factor_cols] = factor_encoder.fit_transform(df_data_reduced2[factor_cols])

target_encoder = LabelEncoder()
target_out = target_encoder.fit_transform(target)



X_train, X_test, y_train, y_test = train_test_split(df_data_reduced2, target_out, train_size=0.7, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.7, random_state=42)

def proba_to_class(probs):
    return np.argmax(probs, axis=1)

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu') # don't have GPU 
    return device

class PytorchDataset(Dataset):
    """
    Pytorch dataset
    ...

    Attributes
    ----------
    X_tensor : Pytorch tensor
        Features tensor
    y_tensor : Pytorch tensor
        Target tensor

    Methods
    -------
    __getitem__(index)
        Return features and target for a given index
    __len__
        Return the number of observations
    to_tensor(data)
        Convert Pandas Series to Pytorch tensor
    """
        
    def __init__(self, X, y):
        self.X_tensor = self.to_tensor(X)
        self.y_tensor = self.to_tensor(y)
    
    def __getitem__(self, index):
        return self.X_tensor[index], self.y_tensor[index]
        
    def __len__ (self):
        return len(self.X_tensor)
    
    def to_tensor(self, data):
        if type(data) == pd.core.frame.DataFrame:
            data_out = data.values
        if type(data) == np.ndarray:
            data_out = data
        return torch.Tensor(data_out)
 
class LogisticRegressionModel(nn.Module):
     def __init__(self, input_dim):
        super(LogisticRegressionModel,self).__init__()
        
        self.layer1 = nn.Linear(input_dim, 128)
        self.layerout = nn.Linear(128, 103)
        
     def forward(self, x):
         x = F.relu(self.layer1(x))
         x = F.sigmoid(self.layerout(x))
         return x

class LogisticRegressionModel2(nn.Module):
     def __init__(self, input_dim):
        super(LogisticRegressionModel2,self).__init__()
        
        self.layer1 = nn.Linear(input_dim, 256)
        self.layerout = nn.Linear(256, 103)
        
     def forward(self, x):
         x = F.relu(self.layer1(x))
         x = F.sigmoid(self.layerout(x))
         return x

class DeepNeuralNet1(nn.Module):
    def __init__(self, input_dim):
      super(DeepNeuralNet1,self).__init__()
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

model1 = LogisticRegressionModel(X_train.shape[1])   
model2 = LogisticRegressionModel2(X_train.shape[1]) 
model3 = DeepNeuralNet1(X_train.shape[1]) 
 
device = get_device()

train_dataset = PytorchDataset(X=X_train, y=y_train)
val_dataset = PytorchDataset(X=X_val, y=y_val)
test_dataset = PytorchDataset(X=X_test, y=y_test)


batch_size_test = y_test.shape[0]

num_epochs = 50
batch_size = 100
batch_size_test = y_test.shape[0]
batch_size_train = y_train.shape[0]
batch_size_val = y_val.shape[0]
learning_rate = 0.01


criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model1.parameters(),lr =learning_rate)

def training_loop(model,num_epochs,batch_size):
    for epoch in range(num_epochs): # monitoring the losses
        train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_data = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        training_loss = 0
        validation_loss = 0
        size = 0
        accuracy = 0
        
        model.train()
        for batch_idx, (data,label) in enumerate(train_data):
            optimizer.zero_grad() 
            output = model(data)
            label = label.to(torch.long)
            loss = criterion(output,label)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            size += label.shape[0]
            values, indices = output.max(1)
            accuracy += (indices == label).sum()
        
        model.eval()
        for batch_idx, (data,label) in enumerate(val_data):
            output = model(data)
            label = label.to(torch.long)
            loss = criterion(output,label)
            validation_loss += loss.item()
    
        training_loss /= size
        validation_loss /= size
        accuracy = accuracy.float()/size*100
        print('Epoch: %5s, Train Loss: %6f, Validation Loss: %6f, Accuracy: %6f\n' %(str(epoch), training_loss, validation_loss, accuracy))
    return model        

train_data = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
val_data = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True)
test_data = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True)

def prediction_generation(model,test_data,batch_size):
    model.eval()
    testing_loss = 0.0
    
    for batch_idx, (data,label) in enumerate(test_data):
        output = model(data) 
        label = label.to(torch.long)
        loss = criterion(output,label)
        testing_loss += loss.item()
        _, predictions = torch.max(output, 1)
        testing_loss /= len(test_dataset)
    return predictions

model1 = training_loop(model1,num_epochs=num_epochs,batch_size=batch_size)
y_pred_nnlogreg1_train = prediction_generation(model1,train_data,batch_size=batch_size_train)
y_pred_nnlogreg1_val = prediction_generation(model1,val_data,batch_size=batch_size_val)
y_pred_nnlogreg1_test = prediction_generation(model1,test_data,batch_size=batch_size_test)

acc_nnlogreg1_train = accuracy_score(y_train,y_pred_nnlogreg1_train.numpy())
acc_nnlogreg1_val = accuracy_score(y_val,y_pred_nnlogreg1_val.numpy())
acc_nnlogreg1_test = accuracy_score(y_test,y_pred_nnlogreg1_test.numpy())

model2 = training_loop(model2,num_epochs=num_epochs,batch_size=batch_size)
y_pred_nnlogreg2_train = prediction_generation(model2,train_data,batch_size=batch_size_train)
y_pred_nnlogreg2_val = prediction_generation(model2,val_data,batch_size=batch_size_val)
y_pred_nnlogreg2_test = prediction_generation(model2,test_data,batch_size=batch_size_test)

acc_nnlogreg2_train = accuracy_score(y_train,y_pred_nnlogreg2_train.numpy())
acc_nnlogreg2_val = accuracy_score(y_val,y_pred_nnlogreg2_val.numpy())
acc_nnlogreg2_test = accuracy_score(y_test,y_pred_nnlogreg2_test.numpy())

model3 = training_loop(model3,num_epochs=num_epochs,batch_size=batch_size)
y_pred_nn3_train = prediction_generation(model3,train_data,batch_size=batch_size_train)
y_pred_nn3_val = prediction_generation(model3,val_data,batch_size=batch_size_val)
y_pred_nn3_test = prediction_generation(model3,test_data,batch_size=batch_size_test)

acc_nn3_train = accuracy_score(y_train,y_pred_nn3_train.numpy())
acc_nn3_val = accuracy_score(y_val,y_pred_nn3_val.numpy())
acc_nn3_test = accuracy_score(y_test,y_pred_nn3_test.numpy())

g = y_pred_nn3.numpy()

#torch.save(model3, '../models/NN01.pt')
torch.save(model3, 'G:/Data Science/UTS Courses/36114 Advanced Data Science for Innovation/Assignment2/36114-assignment2-2022/models/NN01.pt')