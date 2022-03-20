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

b = df_data_reduced2.describe()
c = target.unique()
d = df_data_reduced1.brewery_name.unique()

def proba_to_class(probs):
    return np.argmax(probs, axis=1)

abcde= proba_to_class(logreg.predict_proba(X_train))


ad = logreg.predict_proba(X_train)
ab = np.amax(logreg.predict_proba(X_train), axis=1)
ae = (ad == ab)
#df_data_reduced1['brewery_name'] = df_data_reduced1['brewery_name'].fillna('n/a')
#df_data_reduced1['beer_abv'] = df_data_reduced1['beer_abv'].fillna(0.0)

df_data_reduced1[df_data_reduced1.isnull().any(axis=1)]
df_data_reduced2[df_data_reduced1.isnull().any(axis=1)]


# df_data_reduced3 = df_data_reduced1.copy()
# # factor_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
# # df_data_reduced3[factor_cols] = factor_encoder.fit_transform(df_data_reduced3[factor_cols])
# factor_encoder = LabelEncoder()
# df_data_reduced3['brewery_name_code'] = factor_encoder.fit_transform(df_data_reduced3[factor_cols])

factor_encoder = OrdinalEncoder()
df_data_reduced2[factor_cols] = factor_encoder.fit_transform(df_data_reduced2[factor_cols])

# target = df_data_reduced3.pop('beer_style')
target_encoder = LabelEncoder()
target_out = target_encoder.fit_transform(target)
#brewery_name_list = df_data_reduced3.pop('brewery_name')
logreg = LogisticRegression(max_iter=10000)
model = logreg.fit(X_train,y_train)
y_pred_train = proba_to_class(model.predict_proba(X_train))
y_pred_val = proba_to_class(model.predict_proba(X_val))
y_pred_test = proba_to_class(model.predict_proba(X_test))

# labels_target = range(103)
# y_train_array = label_binarize(y_train,classes=labels_target)
# y_val_array = label_binarize(y_val,classes=labels_target)
# y_test_array = label_binarize(y_test,classes=labels_target)

# y_pred_train_array = label_binarize(logreg.predict(X_train),classes=labels_target)
# y_pred_val_array = label_binarize(logreg.predict(X_train),classes=labels_target)
# y_pred_test_array = label_binarize(logreg.predict(X_train),classes=labels_target)

# acddd = logreg.predict(X_train)

# AUC_logreg_train = roc_auc_score(pd.DataFrame(y_train_array),pd.DataFrame(y_pred_train_array),multi_class='ovr')
# AUC_logreg_val = roc_auc_score(y_val,proba_to_class(logreg.predict_proba(X_val)),multi_class='ovr')
# AUC_logreg_test = roc_auc_score(y_test,proba_to_class(logreg.predict_proba(X_test)),multi_class='ovr')

acc_logreg_train = accuracy_score(y_train,y_pred_train)
acc_logreg_val = accuracy_score(y_val,y_pred_val)
acc_logreg_test = accuracy_score(y_test,y_pred_test)

from sklearn.ensemble import RandomForestClassifier
randfor = RandomForestClassifier(random_state=42,max_depth=4)
model2 = randfor.fit(X_train,y_train)

y_pred_train_rf = proba_to_class(model2.predict_proba(X_train))
y_pred_val_rf = proba_to_class(model2.predict_proba(X_val))
y_pred_test_rf = proba_to_class(model2.predict_proba(X_test))

acc_rf_train = accuracy_score(y_train,y_pred_train_rf)
acc_rf_val = accuracy_score(y_val,y_pred_val_rf)
acc_rf_test = accuracy_score(y_test,y_pred_test_rf)


numerical_encoder = MinMaxScaler()
df_data_reduced2[numerical_cols] = numerical_encoder.fit_transform(df_data_reduced2[numerical_cols])

dump(target_encoder,'G:/Data Science/UTS Courses/36114 Advanced Data Science for Innovation/Assignment2/36114-assignment2-2022/models/target_decoder.joblib')

# model.coef_

# dump(model,'../models/test01.joblib')

# df_data_reduced4 = df_data_reduced3.iloc[:2000,:]

# out = model.predict(df_data_reduced4)
# out_text = target_encoder.inverse_transform(out)

# Model pipeline setup
cat_var_transformer = Pipeline(
    steps=[
        ('brewery_name_encoder', MinMaxScaler())
    ]
)

num_var_transformer = Pipeline(
    steps=[
        ('beer_measures_encoder', OrdinalEncoder())
    ]
)

# target_var_transformer = Pipeline(
#     steps=[
#         ('beer_style_encoder', LabelEncoder())
#     ]
# )

preprocessor = ColumnTransformer(
    transformers=[
        ('fac_cols', cat_var_transformer, factor_cols),
        ('num_cols', num_var_transformer, numerical_cols)
        # ('fac_cols', cat_var_transformer, factor_cols),
        # ('target_col', target_var_transformer, target_col)
    ]
)

model_pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        #('log_regression', LogisticRegression(max_iter=10000))
    ]
)

model_pipeline.fit(df_data_reduced2,target_out)

#dump(model_pipeline,'G:/Data Science/UTS Courses/36114 Advanced Data Science for Innovation/Assignment2/36114-assignment2-2022/models/test02.joblib')
dump(model_pipeline,'G:/Data Science/UTS Courses/36114 Advanced Data Science for Innovation/Assignment2/36114-assignment2-2022/models/pipeline.joblib')


model0 = model_pipeline.fit(X_train,y_train)
aaa = model0.predict(X_test)
aaab = sum(aaa==y_test)/y_test.shape[0]

def format_features(brewery: str, aroma: float, appearance: float, palate: float, taste: float, alcohol: float):
    return {
        'brewery_name': [brewery],
        'review_aroma': [aroma],
        'review_appearance': [appearance],
        'review_palate': [palate],
        'review_taste': [taste],
        'beer_abv':[alcohol]
    }


g1 = pd.DataFrame(format_features('Crow Peak Brewing',1.5,2.5,4.0,5.5,6.0))

y_pred = model_pipeline.predict(g1)

y_pred_name = target_encoder.inverse_transform(y_pred)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_data_reduced2, target_out, train_size=0.7, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.7, random_state=42)

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu') # don't have GPU 
    return device

device = get_device()

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
       super(LogisticRegressionModel,self).__init__()
       
       self.layer1 = nn.Linear(input_dim, 128)
       self.layerout = nn.Linear(128, 103)
       
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.sigmoid(self.layerout(x))
        return x

model = LogisticRegressionModel(X_train.shape[1])

from torch.utils.data import Dataset, DataLoader

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
    
train_dataset = PytorchDataset(X=X_train, y=y_train)
val_dataset = PytorchDataset(X=X_val, y=y_val)
test_dataset = PytorchDataset(X=X_test, y=y_test)

criterion = nn.CrossEntropyLoss()
#criterion = nn.MSELoss()

from torch import optim

optimiser = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

def train_regression(train_data, model, criterion, optimizer, batch_size, device, scheduler=None, collate_fn=None):
    """Train a Pytorch regresssion model

    Parameters
    ----------
    train_data : torch.utils.data.Dataset
        Pytorch dataset
    model: torch.nn.Module
        Pytorch Model
    criterion: function
        Loss function
    optimizer: torch.optim
        Optimizer
    bacth_size : int
        Number of observations per batch
    device : str
        Name of the device used for the model
    scheduler : torch.optim.lr_scheduler
        Pytorch Scheduler used for updating learning rate
    collate_fn : function
        Function defining required pre-processing steps

    Returns
    -------
    Float
        Loss score
    Float:
        RMSE Score
    """
    
    # Set model to training mode
    model.train()
    train_loss = 0

    # Create data loader
    data = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # Iterate through data by batch of observations
    for feature, target_class in data:
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Load data to specified device
        feature, target_class = feature.to(device), target_class.to(device)
        
        # Make predictions
        output = model(feature)
        if type(criterion)==torch.nn.modules.loss.CrossEntropyLoss:
            output = output
        target_class = target_class.to(torch.long)
        
        # Calculate loss for given batch
        loss = criterion(output, target_class)
        
        # Calculate global loss
        train_loss += loss.item()
        
        # Calculate gradients
        loss.backward()
        
        # Update Weights
        optimiser.step()
        
    # Adjust the learning rate
    if scheduler:
        scheduler.step()

    return model, train_loss / len(train_data), torch.sum(output == train_data.y_tensor) / len(train_data)
    #return train_loss / len(train_data), torch.exp(train_loss) / torch.exp(train_loss).sum(), torch.sum(output == train_data.y_tensor) / len(train_data)

def test_regression(test_data, model, criterion, batch_size, device, collate_fn=None):
    """Calculate performance of a Pytorch regresssion model

    Parameters
    ----------
    test_data : torch.utils.data.Dataset
        Pytorch dataset
    model: torch.nn.Module
        Pytorch Model
    criterion: function
        Loss function
    bacth_size : int
        Number of observations per batch
    device : str
        Name of the device used for the model
    collate_fn : function
        Function defining required pre-processing steps

    Returns
    -------
    Float
        Loss score
    Float:
        RMSE Score
    """    
    
    # Set model to evaluation mode
    model.eval()
    test_loss = 0

    # Create data loader
    data = DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn)
    
    # Iterate through data by batch of observations
    for feature, target_class in data:
        
        # Load data to specified device
        feature, target_class = feature.to(device), target_class.to(device)
        
        # Set no update to gradients
        with torch.no_grad():
            
            # Make predictions
            output = model(feature)
            target_class = target_class.to(torch.long)
            # Calculate loss for given batch
            loss = criterion(output, target_class)
            
            # Calculate global loss
            test_loss += loss.item()
    
    return test_loss / len(test_data), torch.sum(output == test_data.y_tensor) / len(test_data)
    #return test_loss / len(test_data), torch.exp(test_loss) / torch.exp(test_loss).sum(), torch.sum(output == test_data.y_tensor) / len(test_data)

def MakePrediction(test_data, model, criterion, batch_size, device):
    model.eval()
    # Create data loader
    data = DataLoader(test_data, batch_size=batch_size)
    for feature, target in data:
         # Load data to specified device
        feature, target = feature.to(device), target.to(device)
        with torch.no_grad():       
    #         # Generate prediction
    #         prediction = model(feature)
    #         # Predicted class value using argmax
    #         predicted_class = np.argmax(prediction)
            predicted_class = model(feature)
            loss = criterion(predicted_class, target)
            _, predictions = torch.max(predicted_class, 1)
            #corrected = np.squeeze(predictions.eq(target.data.view_as(predictions)))
            # for vals in range(len(target)): ## calculating the test accuracy for each object class
            #     label = target.data[vals]
            #     class_corrected[label] += corrected[vals].item()
            #     class_total[label] += 1
    return predictions.cpu().detach().numpy()
            
    # for data, target in testing_loader_batch:
    #     output = MLP_model(data) 
    #     loss = MLP_criterion(output,target) ## Calculating the loss
    #     testing_loss += loss.item()*data.size(0) ## updating the running validation loss
    #     _, predictions = torch.max(output, 1) ##converting the output probabilities to predicted class
    #     corrected = np.squeeze(predictions.eq(target.data.view_as(predictions))) ## comparing the predictions to true label

    #     for vals in range(len(target)): ## calculating the test accuracy for each object class
    #         label = target.data[vals]
    #         class_corrected[label] += corrected[vals].item()
    #         class_total[label] += 1
    # return predicted_class.cpu().detach().numpy()

N_EPOCHS = 20
BATCH_SIZE = 100
batch_size_test = y_test.shape[0]

for epoch in range(N_EPOCHS):
    model, train_loss, train_acc = train_regression(train_dataset, model=model, criterion=criterion, optimizer=optimiser, batch_size=BATCH_SIZE, device=device)
    valid_loss, valid_acc = test_regression(val_dataset, model=model, criterion=criterion, batch_size=BATCH_SIZE, device=device)

    print(f'Epoch: {epoch}')
    print(f'\t(train)\tLoss: {train_loss:.4f}\t|\(train)\tAcc: {train_acc:.4f}')
    print(f'\t(valid)\tLoss: {valid_loss:.4f}\t|\(valid)\tAcc: {valid_acc:.4f}')
    #print(f'\t(train)\tLoss: {train_loss:.4f}\t|\tRMSE: {train_rmse:.1f}\t|\(train)\tAcc: {train_acc:.4f}')
    #print(f'\t(valid)\tLoss: {valid_loss:.4f}\t|\tRMSE: {valid_rmse:.1f}\t|\(valid)\tAcc: {valid_acc:.4f}')
    
test_loss, test_acc = test_regression(test_dataset, model=model, criterion=criterion, batch_size=BATCH_SIZE, device=device)  
abcc = MakePrediction(test_dataset, model=model, criterion=criterion, batch_size=batch_size_test, device=device)

dataabcde = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
x1 = dataabcde.dataset.X_tensor
y1 = dataabcde.dataset.y_tensor

outputmodel = model
outputmodel.eval()
dataabcde = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True)

for feature, target in dataabcde:
         # Load data to specified device
    feature, target = feature.to(device), target.to(device)
    with torch.no_grad():
        predicted_class = outputmodel(feature)
        #loss = criterion(feature, target)
        _, predictions = torch.max(predicted_class, 1)
        #corrected = np.squeeze(predictions.eq(target.data.view_as(predictions)))   
defghj = predictions.cpu().detach().numpy()
    
acc_nn01_train = accuracy_score(y_train,predicted_class.numpy().astype(int))

# More models
class MultiLayerPreceptronModel(nn.Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MultiLayerPreceptronModel, self).__init__()
        # input to first hidden layer
        self.hidden1 = nn.Linear(n_inputs, 10)
        nn.init.kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = nn.ReLU()
        # second hidden layer
        self.hidden2 = nn.Linear(10, 8)
        nn.init.kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = nn.ReLU()
        # third hidden layer and output
        self.hidden3 = nn.Linear(8, 1)
        nn.init.xavier_uniform_(self.hidden3.weight)
        self.act3 = nn.Sigmoid()
 
    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
         # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer and output
        X = self.hidden3(X)
        X = self.act3(X)
        return X
 
model2 = MultiLayerPreceptronModel(X_train.shape[1])
 
for epoch in range(N_EPOCHS):
    train_loss, train_rmse, train_acc = train_regression(train_dataset, model=model2, criterion=criterion, optimizer=optimiser, batch_size=BATCH_SIZE, device=device)
    valid_loss, valid_rmse, valid_acc = test_regression(val_dataset, model=model2, criterion=criterion, batch_size=BATCH_SIZE, device=device)

    print(f'Epoch: {epoch}')
    print(f'\t(train)\tLoss: {train_loss:.4f}\t|\tRMSE: {train_rmse:.1f}\t|\(train)\tAcc: {train_acc:.4f}')
    print(f'\t(valid)\tLoss: {valid_loss:.4f}\t|\tRMSE: {valid_rmse:.1f}\t|\(valid)\tAcc: {valid_acc:.4f}')
    
test_loss, tes_rmse, test_acc = test_regression(test_dataset, model=model2, criterion=criterion, batch_size=BATCH_SIZE, device=device)  
   
class MultiLayerNNModel(nn.Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MultiLayerNNModel, self).__init__()
        # input to first hidden layer
        self.hidden1 = nn.Linear(n_inputs, 10)
        self.act1 = nn.ReLU()
        # second hidden layer
        self.hidden2 = nn.Linear(10, 8)
        self.act2 = nn.ReLU()
        # third hidden layer and output
        self.hidden3 = nn.Linear(8, 1)
        self.act3 = nn.Sigmoid()
 
    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
         # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer and output
        X = self.hidden3(X)
        X = self.act3(X)
        return X
 
model3 = MultiLayerNNModel(X_train.shape[1])
 
for epoch in range(N_EPOCHS):
    train_loss, train_rmse, train_acc = train_regression(train_dataset, model=model3, criterion=criterion, optimizer=optimiser, batch_size=BATCH_SIZE, device=device)
    valid_loss, valid_rmse, valid_acc = test_regression(val_dataset, model=model3, criterion=criterion, batch_size=BATCH_SIZE, device=device)

    print(f'Epoch: {epoch}')
    print(f'\t(train)\tLoss: {train_loss:.4f}\t|\tRMSE: {train_rmse:.1f}\t|\(train)\tAcc: {train_acc:.4f}')
    print(f'\t(valid)\tLoss: {valid_loss:.4f}\t|\tRMSE: {valid_rmse:.1f}\t|\(valid)\tAcc: {valid_acc:.4f}')
    
test_loss, test_rmse, test_acc = test_regression(test_dataset, model=model3, criterion=criterion, batch_size=BATCH_SIZE, device=device)  
   
class MultiLayer2Model(nn.Module):
    def __init__(self, input_dim):
       super(MultiLayer2Model,self).__init__()
       
       self.layer1 = nn.Linear(input_dim, 128)
       self.layer2 = nn.Linear(128, 10)
       self.layer3 = nn.Linear(10, 8)
       self.layer4 = nn.Linear(8, 128)
       self.layerout = nn.Linear(128, 1)
       
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.sigmoid(self.layerout(x))
        return x

model4 = MultiLayer2Model(X_train.shape[1])
 
for epoch in range(N_EPOCHS):
    train_loss, train_rmse, train_acc = train_regression(train_dataset, model=model4, criterion=criterion, optimizer=optimiser, batch_size=BATCH_SIZE, device=device)
    valid_loss, valid_rmse, valid_acc = test_regression(val_dataset, model=model4, criterion=criterion, batch_size=BATCH_SIZE, device=device)

    print(f'Epoch: {epoch}')
    print(f'\t(train)\tLoss: {train_loss:.4f}\t|\tRMSE: {train_rmse:.1f}\t|\(train)\tAcc: {train_acc:.4f}')
    print(f'\t(valid)\tLoss: {valid_loss:.4f}\t|\tRMSE: {valid_rmse:.1f}\t|\(valid)\tAcc: {valid_acc:.4f}')
    
test_loss, test_rmse, test_acc = test_regression(test_dataset, model=model4, criterion=criterion, batch_size=BATCH_SIZE, device=device)  
