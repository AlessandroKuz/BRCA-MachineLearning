from brca_preprocessing_pipeline import brca_preprocessing_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn import set_config
import pandas as pd
import numpy as np

set_config(transform_output="pandas")


def impute_dataset(df, imputer, impute_columns: list):
    for column, strategy in impute_columns:
        df[column] = imputer[column].transform(df[[column]])
    return df

def make_onehot(onehot_columns: list):
    onehot = make_column_transformer(
        (
            OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.int8),
            onehot_columns,
        ),
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    return onehot


def make_imputer(impute_columns: list):
    imputer = make_column_transformer(
        (
            KNNImputer(weights="distance", n_neighbors=5, add_indicator=True),
            impute_columns,
        ),
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    return imputer


def make_scaler(scaler_columns: list):
    scaler = make_column_transformer(
        (
            MinMaxScaler(),
            scaler_columns,
        ),
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    return scaler
    

def get_brca_pipeline(cols_to_encode, cols_to_scale, imputation=False, cols_to_impute=None):
    onehot = make_onehot(cols_to_encode)
    scale = make_scaler(cols_to_scale)
    if imputation:
        impute = make_imputer(cols_to_impute)
        preprocess_pipeline = Pipeline(
            [
            ("preprocessing", brca_preprocessing_pipeline),
            ("onehot", onehot),
            ("impute", impute),
            ("scale", scale)
            ]
        )
    else:
        preprocess_pipeline = Pipeline(
            [
            ("preprocessing", brca_preprocessing_pipeline),
            ("onehot", onehot),
            ("scale", scale)
            ]
        )

    return preprocess_pipeline
