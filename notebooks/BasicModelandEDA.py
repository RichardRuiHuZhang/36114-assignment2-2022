from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import dump
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from category_encoders.ordinal import OrdinalEncoder

#df_data_raw = pd.read_csv('G:/Data Science/UTS Courses/36114 Advanced Data Science for Innovation/Assignment2/36114-assignment2-2022/data/raw/beer_reviews.csv')
df_data_raw = pd.read_csv('G:/Data Science/UTS Courses/36114 Advanced Data Science for Innovation/Assignment2/36114-assignment2-2022/data/raw/beer_reviews_reduced.csv')
#df_data_raw = pd.read_csv('../data/raw/beer_reviews.csv')

df_data_clean1 = df_data_raw.copy()

col_usable = ['brewery_name','review_aroma','review_appearance','review_palate','review_taste','beer_abv','beer_style']
independent_cols = ['brewery_name','review_aroma','review_appearance','review_palate','review_taste','beer_abv']
factor_cols = ['brewery_name']
target_col = ['beer_style']
#target = df_data_clean1.pop('beer_style')

df_data_reduced1 = df_data_clean1.loc[:,col_usable]

df_data_reduced2 = df_data_reduced1.copy()
df_data_reduced2 = df_data_reduced2.dropna()
target = df_data_reduced2.pop('beer_style')

b = df_data_reduced2.describe()

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
# logreg = LogisticRegression()
# model = logreg.fit(df_data_reduced3,target_out)

dump(target_encoder,'G:/Data Science/UTS Courses/36114 Advanced Data Science for Innovation/Assignment2/36114-assignment2-2022/models/target_decoder.joblib')

# model.coef_

# dump(model,'../models/test01.joblib')

# df_data_reduced4 = df_data_reduced3.iloc[:2000,:]

# out = model.predict(df_data_reduced4)
# out_text = target_encoder.inverse_transform(out)

# Model pipeline setup
cat_var_transformer = Pipeline(
    steps=[
        ('brewery_name_encoder', OrdinalEncoder())
    ]
)

# target_var_transformer = Pipeline(
#     steps=[
#         ('beer_style_encoder', LabelEncoder())
#     ]
# )

preprocessor = ColumnTransformer(
    transformers=[
        ('fac_cols', cat_var_transformer, factor_cols)
        # ('fac_cols', cat_var_transformer, factor_cols),
        # ('target_col', target_var_transformer, target_col)
    ]
)

model_pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('log_regression', LogisticRegression())
    ]
)

model_pipeline.fit(df_data_reduced2,target_out)

dump(model_pipeline,'G:/Data Science/UTS Courses/36114 Advanced Data Science for Innovation/Assignment2/36114-assignment2-2022/models/test02.joblib')


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


from src.models.null import NullModel

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_data_reduced2, target_out, train_size=0.2, random_state=42)

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
model.to(device)

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
       super(LogisticRegressionModel,self).__init__()
       
       self.layer1 = nn.Linear(input_dim, 128)
       self.layerout = nn.Linear(128, 1)
       
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.sigmoid(self.layerout(x))
        return x

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
#val_dataset = PytorchDataset(X=X_val, y=y_val)
test_dataset = PytorchDataset(X=X_test, y=y_test)

#criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()

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

    return train_loss / len(train_data), np.sqrt(train_loss / len(train_data))

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
            
            # Calculate loss for given batch
            loss = criterion(output, target_class)
            
            # Calculate global loss
            test_loss += loss.item()
            
    return test_loss / len(test_data), np.sqrt(test_loss / len(test_data))

N_EPOCHS = 5
BATCH_SIZE = 100
model = LogisticRegressionModel(X_train.shape[1])

for epoch in range(N_EPOCHS):
    train_loss, train_rmse = train_regression(train_dataset, model=model, criterion=criterion, optimizer=optimiser, batch_size=BATCH_SIZE, device=device)
    valid_loss, valid_rmse = test_regression(test_dataset, model=model, criterion=criterion, batch_size=BATCH_SIZE, device=device)

    print(f'Epoch: {epoch}')
    print(f'\t(train)\tLoss: {train_loss:.4f}\t|\tRMSE: {train_rmse:.1f}')
    print(f'\t(valid)\tLoss: {valid_loss:.4f}\t|\tRMSE: {valid_rmse:.1f}')