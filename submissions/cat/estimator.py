import os
import pandas as pd

from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from catboost import CatBoostRegressor, Pool

merging_keys = ['DateOfDeparture', 'Departure', 'Arrival'] # keys to merge external_data.csv and the main data



def _merge_external_data(X):
    filepath = os.path.join(
        os.path.dirname(__file__), 'external_data.csv'
    )

    X = X.copy()
    X.loc[:, "DateOfDeparture"] = pd.to_datetime(X['DateOfDeparture'])
    
    # Parse date to also be of dtype datetime
    data_w = pd.read_csv(filepath)
    data_w.loc[:, "DateOfDeparture"] = pd.to_datetime(data_w['DateOfDeparture'])

    X_merged = X.merge(data_w, how='left', on=['DateOfDeparture', 'Departure', 'Arrival'])
    #X_merged = X_merged.drop(columns=["population_departure", "Holiday"])
    return X_merged



def get_estimator():

    data_merger = FunctionTransformer(_merge_external_data)
    regressor = CatBoostRegressor(n_estimators=2500, colsample_bylevel=0.3, learning_rate=0.1, verbose=1000, reg_lambda=2,
    cat_features=["Departure", "Arrival"])


    return make_pipeline(data_merger, regressor)