from brca_preprocessing_pipeline import brca_preprocessing_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.compose import make_column_transformer
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


def make_numerical_imputer(impute_columns: list):
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

def make_categorical_imputer(impute_columns: list):
    imputer = make_column_transformer(
        (
            SimpleImputer(strategy="constant", fill_value="missing", add_indicator=True),
            impute_columns,
        ),
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    return imputer
    

def get_brca_pipeline(cols_to_encode:list, cols_to_scale:list, imputation:bool=False, cols_to_impute:(list,list)=None):
    """
    params:
        cols_to_encode: list of columns to be one-hot encoded
        cols_to_scale: list of columns to be scaled
        imputation: whether to impute or not
        cols_to_impute: tuple of lists, the first list contains columns to be imputed numerically, 
                        the second list contains columns to be imputed categorically
    
    outputs:
        preprocess_pipeline: sklearn pipeline object
    """
    onehot = make_onehot(cols_to_encode)
    scale = make_scaler(cols_to_scale)
    if imputation:
        cols_to_impute_num, cols_to_impute_cat = cols_to_impute
        preprocess_pipeline = Pipeline(
            [
            ("preprocessing", brca_preprocessing_pipeline()),
            ("categorical_impute", make_categorical_imputer(cols_to_impute_cat)),
            ("numerical_impute", make_numerical_imputer(cols_to_impute_num)),
            ("onehot", onehot),
            ("scale", scale)
            ]
        )
    else:
        preprocess_pipeline = Pipeline(
            [
            ("preprocessing", brca_preprocessing_pipeline()),
            ("onehot", onehot),
            ("scale", scale)
            ]
        )

    return preprocess_pipeline
