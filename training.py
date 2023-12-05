from data_preprocessing.brca_full_pipeline import get_brca_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, roc_auc_score, cohen_kappa_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn import set_config
import pandas as pd
import numpy as np

# Setting pandas to display all columns
set_config(transform_output="pandas")

# Disabling numpy scientific notation (to easily visualize label predictions)
np.set_printoptions(suppress=True)


def model_training(data_path, cols_to_encode, cols_to_scale,
                   imputation=False, cols_to_impute=None,
                   verbosity=3) -> None:
    """
    Main function to run the pipeline

    Parameters
    ----------
    data_path : str
        Path to the dataset
    cols_to_encode : list
        List of columns to perform one hot encoding
    cols_to_scale : list
        List of columns to scale (MinMaxScaler)
    imputation : bool, optional
        Whether to perform imputation or not, by default False
        when True, cols_to_impute must be specified;
        if True the model will be RandomForestClassifier, otherwise XGBClassifier
    cols_to_impute : tuple, optional
        Tuple of lists, the first one contains numerical columns to impute with KNNImputer,
        the second one with the "MISSING" string, by default None
    verbosity : int, optional
        Verbosity level of training, by default 3

    Returns
    -------
    None
    """
    df = pd.read_csv(data_path)
    if imputation:
        preprocessing_pipeline = get_brca_pipeline(cols_to_encode, cols_to_scale, imputation=True, cols_to_impute=cols_to_impute)
    else:
        preprocessing_pipeline = get_brca_pipeline(cols_to_encode, cols_to_scale)

    # encoding the labels
    df['LABEL'] = df['LABEL'].map({'NEG': 0, 'POS': 1, 'VUS': 2})
    # keeping df where LABEL != 2, save the rest into a to predict dataset
    df, predict_df = df[df['LABEL'] != 2].copy(), df[df['LABEL'] == 2].copy()

    # splitting the dataset into X and y
    X, y = df.drop(columns=['LABEL']).copy(), df['LABEL'].copy()

    # splitting the dataset into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)

    # preprocessing the train and validation sets
    X_train = preprocessing_pipeline.fit_transform(X_train)
    y_train = y_train[X_train.index]
    
    X_val = preprocessing_pipeline.transform(X_val)
    y_val = y_val[X_val.index]

    # creating the model
    if imputation:
        model = RandomForestClassifier(random_state=21)
        grid_params = {
            'model__n_estimators': [100, 200, 300, 400, 500],
            'model__max_depth': [None, 1, 2, 3, 4, 5],
            'model__min_samples_split': [2, 3, 4, 5],
            'model__min_samples_leaf': [1, 2, 3, 4, 5],
            'model__max_features': ['sqrt', 'log2'],
            'model__criterion': ['gini', 'entropy']
        }
    else:
        model = XGBClassifier(random_state=21)
        grid_params = {
            'model__n_estimators': [100, 200, 300, 400, 500],
            'model__max_depth': [None, 1, 2, 3, 4, 5],
            'model__learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
            'model__booster': ['gbtree', 'gblinear', 'dart'],
            'model__base_score': [0.5, 0.75, 1.0],
        }

    # creating the pipeline
    pipeline = Pipeline([
        ('model', model)
    ])

    # creating the grid search
    grid = GridSearchCV(pipeline, grid_params, cv=10, scoring='recall_macro', n_jobs=-1, verbose=verbosity)

    # fitting the model
    grid.fit(X_train, y_train)

    # printing the best params and score
    print('='*100)
    print('Grid search results:')
    print(grid.best_params_)
    print(grid.best_score_)
    print('='*100)

    # printing precision, recall, f1_score, cohen's kappa, roc auc and confusion matrix on the validation set
    y_pred = grid.predict(X_val)
    print(f"Recall: {recall_score(y_val, y_pred, average='macro')}")
    print(f"Precision: {precision_score(y_val, y_pred, average='macro')}")
    print(f"F1_score: {f1_score(y_val, y_pred, average='macro')}")
    print(f"Cohen's Kappa: {cohen_kappa_score(y_val, y_pred)}")
    print(f"ROC AUC: {roc_auc_score(y_val, y_pred, multi_class='ovr')}")
    print(confusion_matrix(y_val, y_pred))
    print('='*100)

    # preprocessing the to predict dataset
    X_predict = predict_df.drop(columns=['LABEL']).copy()  # Dropping the label column as it's only VUS
    X_predict = preprocessing_pipeline.transform(X_predict)

    # printing the predictions
    print('Predictions:')
    print(grid.predict_proba(X_predict)[:, 1].round(3))
    print('-'*100)
    print(grid.predict(X_predict))
    print('='*100)


if __name__ == "__main__":
    file_path = ''  # Insert the path to the dataset here

    cols_to_encode = ['REF', 'ALT', 'GENE_SYMBOL', 'TYPE', 'VARIANT_TYPE', 
                      'MOST_SEVERE_CONSEQUENCE', 'IMPACT', 'EXON_INTRON_TYPE', 'CLINPRED_PRED']
    cols_to_impute = (['CLINPRED_RANKSCORE', 'POLYPHEN2_HDIV_RANKSCORE',  'SIFT_CONVERTED_RANKSCORE', 
                      'SIFT4G_CONVERTED_RANKSCORE',  'MUTATIONASSESSOR_RANKSCORE', 'MUTATIONTASTER_CONVERTED_RANKSCORE'],
                      ['CLINPRED_PRED'])
    cols_to_scale = ['STRAND', 'VARIANT_OCCURRENCES', 'EXON_INTRON_N', 'DOMAINS_COUNT', 'PUBMED_COUNT']

    # Random Forest Classifier
    model_training(file_path, cols_to_encode, cols_to_scale, imputation=True, cols_to_impute=cols_to_impute, verbosity=3)

    # XGBoost Classifier
    model_training(file_path, cols_to_encode, cols_to_scale, verbosity=3)
