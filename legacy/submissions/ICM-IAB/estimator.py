import os
import pandas as pd

from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import xgboost as xgb

merging_keys = ['DateOfDeparture', 'Departure', 'Arrival'] # keys to merge external_data.csv and the main data


ext_data_predictors = ["Frequency", "Delta", "Price"]
int_data_predictors = ["WeeksToDeparture"]

all_predictors = merging_keys + ext_data_predictors + int_data_predictors

def _merge_external_data(X):
    filepath = os.path.join(
        os.path.dirname(__file__), 'external_data.csv'
    )

    X = X.copy()
    X.loc[:, "DateOfDeparture"] = pd.to_datetime(X['DateOfDeparture'])
    
    # Parse date to also be of dtype datetime
    data_w = pd.read_csv(filepath)
    data_w.loc[:, "DateOfDeparture"] = pd.to_datetime(data_w['DateOfDeparture'])
    # data_w = data_w.drop(['internat_dep_month_vol', 'population_arrival', 'population_departure'], axis=1)
    #data_w = data_w.fillna(0)

    X_merged = X.merge(data_w, how='left', on=['DateOfDeparture', 'Departure', 'Arrival'])
    #X_merged = X_merged.drop(columns=["density_departure", "a_airlines", "WeeksToDeparture", "population_departure", "Holiday", "density_departure"])
    X_merged = X_merged.loc[:, ['DateOfDeparture', 'Departure', 'Arrival', 'WeeksToDeparture', 'delta', 'a_airlines', 'booking', 'Holiday']]
    return X_merged

def _encode_dates(X):
    # With pandas < 1.0, we wil get a SettingWithCopyWarning
    # In our case, we will avoid this warning by triggering a copy
    # More information can be found at:
    # https://github.com/scikit-learn/scikit-learn/issues/16191
    X_encoded = X.copy()

    # Make sure that DateOfDeparture is of datetime format
    X_encoded.loc[:, "DateOfDeparture"] = pd.to_datetime(X_encoded["DateOfDeparture"])
    # Encode the DateOfDeparture
    # X_encoded.loc[:, 'year'] = X_encoded["DateOfDeparture"].dt.year
    X_encoded.loc[:, 'month'] = X_encoded['DateOfDeparture'].dt.month
    # X_encoded.loc[:, 'day'] = X_encoded["DateOfDeparture"].dt.day
    X_encoded.loc[:, 'weekday'] = X_encoded["DateOfDeparture"].dt.weekday
    # X_encoded.loc[:, 'week'] = X_encoded['DateOfDeparture'].dt.week
    # X_encoded.loc[:, 'n_days'] = X_encoded['DateOfDeparture'].apply(
    #    lambda date: (date - pd.to_datetime("1970-01-01")).days)
    # Once we did the encoding, we will not need DateOfDeparture
    return X_encoded.drop(columns=["DateOfDeparture"])


def get_estimator():

    data_merger = FunctionTransformer(_merge_external_data)
    date_encoder = FunctionTransformer(_encode_dates)
    

    date_cols = ["DateOfDeparture"]

    # categorical_encoder = make_pipeline(
    #     SimpleImputer(strategy='constant', fill_value="missing"),
    #     OrdinalEncoder()
    # )

    categorical_encoder = make_pipeline(
        OneHotEncoder()
    )
    

    categorical_cols = ['Arrival', 'Departure']
    # categorical_cols=[]

    preprocessor = make_column_transformer(
        (date_encoder, date_cols),
        (categorical_encoder, categorical_cols),
        remainder='passthrough',  # passthrough numerical columns as they are
    )


    regressor = xgb.XGBRegressor(objective ='reg:squarederror', 
    # colsample_bylevel=0.3, colsample_bynode=0.5,
    max_depth = 8, alpha = 2, n_estimators = 2500, max_leaves=15, eval_metric='rmse')



    return make_pipeline(data_merger, preprocessor, regressor)