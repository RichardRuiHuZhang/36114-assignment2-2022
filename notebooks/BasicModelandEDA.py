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
target = df_data_clean1.pop('beer_style')

df_data_reduced1 = df_data_clean1.loc[:,independent_cols]
df_data_reduced1['brewery_name'] = df_data_reduced1['brewery_name'].fillna('n/a')
df_data_reduced1['beer_abv'] = df_data_reduced1['beer_abv'].fillna(0.0)
df_data_reduced2 = df_data_reduced1.copy()

df_data_reduced1[df_data_reduced1.isnull().any(axis=1)]


# df_data_reduced3 = df_data_reduced1.copy()
# # factor_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
# # df_data_reduced3[factor_cols] = factor_encoder.fit_transform(df_data_reduced3[factor_cols])
# factor_encoder = LabelEncoder()
# df_data_reduced3['brewery_name_code'] = factor_encoder.fit_transform(df_data_reduced3[factor_cols])

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