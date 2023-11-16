from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from collections import Counter
from sklearn import set_config
import pandas as pd
import numpy as np

set_config(transform_output='pandas')


def find_empty_columns(df):
    empty_columns = []
    for column in df.columns:
        if isinstance(df[column], str):
            massimo = df[column].max()
            if massimo == 0:
                empty_columns.append(column)
    return empty_columns


def drop_column(df, columns):
    df.drop(columns=columns, inplace=True)
    return df


def to_uppercase(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].apply(lambda x: x.upper() if isinstance(x, str) else x)
    return df


def clean_columns(df, column_to_clean: list):
    for column, replacements in column_to_clean:
        for old, new in replacements.items():
            df[column] = df[column].apply(lambda x: x.replace(old, new))
    return df


def rename_columns(df, columns_to_rename: list):
    for column, new_name in columns_to_rename:
        df.rename(columns={column: new_name}, inplace=True)
    return df


def convert_column_types(df, conversions):
    for column, new_type in conversions.items():
        df[column] = df[column].astype(new_type)
    return df


def drop_NaN_rows(df, column):
    df.dropna(subset=column, inplace=True)
    return df


def fill_NaN_rows(df, fill_values):
    for column, value in fill_values.items():
        df[column].fillna(value, inplace=True)
    return df


def Preprocess_Dataset(df):
    columns_to_drop = ['ANNO', 'MSP', 'DIAGNOSI', 'ALTRO ISOTIPO', 'N/D', 'Data Firma Referto Formato Corto', 'BUILT',
                       'GENE_ENST', 'SAMPLE', 'POS', 'CAMPIONE', 'PANNELLO', 'GT', 'ETA_DIAGNOSI', 'DATA_NASCITA', 'EXON/INTRON_POS', 'AOP',
                       'SEDE_TUMORE_FAMILIARITA', 'ANNO_NASCITA', 'ETA']

    df = rename_columns(df, [('STORIA FAMILIARE POS ', 'STORIA_FAMILIARE_POS'),
                             ('SEDE TUMORE FAMILIARITA\'', 'SEDE_TUMORE_FAMILIARITA'),
                             ('ETA ALLA DIAGNOSI', 'ETA_DIAGNOSI'),
                             ('Data di nascita formato breve', 'DATA_NASCITA'),
                             ('A.O.P.', 'AOP'),
                             ('HGVSG', 'VARTYPE')])

    fill_values = {'STORIA_FAMILIARE_POS': '0', 'SEDE_TUMORE_FAMILIARITA': 'ND', 'ETA_DIAGNOSI': 'ND', 'AOP': 'ND'}
    df = fill_NaN_rows(df, fill_values)

    df = to_uppercase(df)

    columns_to_clean = [('SEDE_TUMORE_FAMILIARITA', {' ': '',
                                                     'COLON-RETTO': 'COLON,RETTO',
                                                     'COLONRETTO': 'COLON,RETTO',
                                                     'COLONMAMMELLA': 'COLON,MAMMELLA',
                                                     'POMONE': 'POLMONI',
                                                     'POLMONE': 'POLMONI',
                                                     'RENALE': 'RENE',
                                                     'TRATTOGASTROINT': 'GASTROINTESTINALE',
                                                     'STOMACO': 'GASTROINTESTINALE',
                                                     'GASTRICO': 'GASTROINTESTINALE',
                                                     'INSTESTINO': 'GASTROINTESTINALE',
                                                     'INTESTINO': 'GASTROINTESTINALE',
                                                     'MAMMELL': 'MAMMELLA',
                                                     'SENO': 'MAMMELLA',
                                                     'OVIAO': 'OVAIO',
                                                     'ENDOMETRIO': 'OVAIO',
                                                     'OVARICA': 'OVAIO',
                                                     'POLMONARE': 'POLMONI',
                                                     'NEOPLASIAOVAIO': 'OVAIO',
                                                     'MELANOMAOVAIO': 'OVAIO',
                                                     'OVARICO': 'OVAIO',
                                                     'OVAIE': 'OVAIO',
                                                     'MAMMLLA': 'MAMMELLA',
                                                     'MAMMELLAA': 'MAMMELLA',
                                                     'MAMMARIA': 'MAMMELLA',
                                                     'OSSEA': 'OSSEO',
                                                     'OSSA': 'OSSEO',
                                                     'CAVOORALE': 'ORALE',
                                                     'BOCCA': 'ORALE',
                                                     'COLONSENO': 'COLON,MAMMELLA',
                                                     'NEOPLASIAMAMMARIA': 'MAMMELLA',
                                                     'NEOPLASIAMAMMELLA': 'MAMMELLA',
                                                     'TESTICOLO': 'TESTICOLI',
                                                     'NEOPLASIAOVARICA': 'OVAIO',
                                                     'ENCEFALICA': 'CERVELLO',
                                                     'NEOPLASIACEREBRALE': 'CERVELLO',
                                                     'CEREBRALE': 'CERVELLO',
                                                     'ENCEFALO': 'CERVELLO',
                                                     'EPATICA': 'FEGATO',
                                                     'EPATICO': 'FEGATO',
                                                     'EPATOCARCINOMA': 'FEGATO',
                                                     'PROSTAT': 'PROSTATA',
                                                     'PROSTATAA': 'PROSTATA',
                                                     'TIDOIDE': 'TIROIDE',
                                                     'VIEBILLIARI': 'VIEBILIARI',
                                                     'VIABILIARE': 'VIEBILIARI',
                                                     'BRCABILATERALE': 'BRCA',
                                                     'NO': 'ND',
                                                     '\n': ',',
                                                     'GH.': '',
                                                     '.': ','}),

                        ('STORIA_FAMILIARE_POS', {'SI': '1', 'SÌ': '1', 'S': '1', 'NO': '0', 'N': '0', 'ND': '0'}),

                        ('AOP', {' ': '',
                                 '?': 'ND',
                                 'N/D': 'ND',
                                 'PR': 'P',
                                 'PA': 'P',
                                 'RENE': 'A',
                                 'COLON': 'C',
                                 'PELVICA': 'A',
                                 'M,RENE': 'M/A',
                                 'M, C.G, TI': 'M/A/TI',
                                 '-': '/',
                                 '.': '/',
                                 ',': '/'}),

                        ('ETA_DIAGNOSI', {'-': '/', ',': '/', 'N.D': 'ND'}),

                        ]

    df = clean_columns(df, columns_to_clean)

    df['GT_1'] = df['GT'].str.split('/').str[0]
    df['GT_2'] = df['GT'].str.split('/').str[1]

    df['IS_EXON'] = df['EXON'].apply(lambda x: 0 if pd.isna(x) else 1)
    df['IS_INTRON'] = df['INTRON'].apply(lambda x: 0 if pd.isna(x) else 1)

    df['EXON/INTRON_POS'] = '0/0'
    df.loc[df['IS_EXON'] == 1, 'EXON/INTRON_POS'] = df['EXON']
    df.loc[df['IS_INTRON'] == 1, 'EXON/INTRON_POS'] = df['INTRON']
    df.drop(columns=['EXON', 'INTRON'], inplace=True)
    df['EXON/INTRON_POS'] = df['EXON/INTRON_POS'].astype('str')

    df['EXON/INTRON_CURR_POS'] = df['EXON/INTRON_POS'].apply(lambda x: x.split('/')[0])

    columns_to_convert = {'GT_1': int, 'GT_2': int, 'EXON/INTRON_CURR_POS': int, 'CLINPRED_PRED': 'category', 'STORIA_FAMILIARE_POS': bool}
    df = convert_column_types(df, columns_to_convert)

    df['ANNO_NASCITA'] = '19' + df['DATA_NASCITA'].str[-2:]
    df['ANNO_NASCITA'] = pd.to_numeric(df['ANNO_NASCITA'])
    df['ETA'] = df['ANNO'] - df['ANNO_NASCITA']

    df['ETA_DIAGNOSI_MIN'] = df['ETA_DIAGNOSI'].str.split('/').str[0]
    df['ETA_DIAGNOSI_MAX'] = df['ETA_DIAGNOSI'].str.split('/').str[-1]
    df['N_DIAGNOSI'] = df['ETA_DIAGNOSI'].str.count('/') + 1

    df['LABEL'] = df['LABEL'].map({'NEG': 0, 'POS': 1, 'VUS': 2})
    df['SESSO'] = df['SESSO'].map({'M': 1, 'F': 0})
    df['VARTYPE'] = df['VARTYPE'].apply(lambda x: "SOST" if ">" in x else "INS" if "_" in x else "DEL")

    family_counter = Counter()

    for index, row in df.iterrows():
        family_keys = row['SEDE_TUMORE_FAMILIARITA'].split(',')
        family_keys = set([f'FAMILIARITA_{k.strip()}' for k in family_keys if k])
        family_counter.update(family_keys)

    updated_counter = {k: ('FAMILIARITA_ALTRO' if v < 150 else k) for k, v in family_counter.items()}

    for key in updated_counter.keys():
        df[key] = 0
    df['FAMILIARITA_ALTRO'] = 0

    for index, row in df.iterrows():
        family_keys = row['SEDE_TUMORE_FAMILIARITA'].split(',')
        family_keys = set([f'FAMILIARITA_{k.strip()}' for k in family_keys if k])
        updated_keys = {updated_counter[k] for k in family_keys}
        df.loc[index, list(updated_keys)] = 1

    aop_counter = Counter()

    for index, row in df.iterrows():
        aop_keys = row['AOP'].split('/')
        aop_keys = set([f'AOP_{k.strip()}' for k in aop_keys if k])
        aop_counter.update(aop_keys)

    updated_counter = {k: ('AOP_A' if v < 150 else k) for k, v in aop_counter.items()}

    for key in updated_counter.keys():
        df[key] = 0

    for index, row in df.iterrows():
        aop_keys = row['AOP'].split('/')
        aop_keys = set([f'AOP_{k.strip()}' for k in aop_keys if k])
        updated_keys = {updated_counter[k] for k in aop_keys}
        df.loc[index, list(updated_keys)] = 1

    empty_columns = find_empty_columns(df)
    columns_to_drop.extend(empty_columns)
    df = drop_column(df, columns_to_drop)

    df_c = df.copy()
    df_c = df_c[df_c[['REF', 'ALT']].apply(tuple, 1).isin(df_c[['REF', 'ALT']].value_counts()[:12].index)]

    df['REF'] = 'X'
    df['ALT'] = 'Y'

    df.loc[df_c.index, 'REF'] = df_c['REF']
    df.loc[df_c.index, 'ALT'] = df_c['ALT']

    return df


onehot = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.int8),
     ['GENE_SYMBOL', 'REF', 'ALT', 'TYPE', 'MOST_SEVERE_CONSEQUENCE', 'IMPACT', 'VARTYPE']),
    remainder='passthrough',
    verbose_feature_names_out=False
)


def drop_columns(df):
    suffixes_to_drop = ['_nan', '_ND', '_?']
    for col in df.columns:
        if any([col.endswith(suffix) for suffix in suffixes_to_drop]):
            df.drop(columns=col, inplace=True)

    return df


def Create_Pipeline():
    preprocess = FunctionTransformer(Preprocess_Dataset)
    drop_cols = FunctionTransformer(drop_columns)
    preprocess_pipeline = Pipeline([
        ('preprocess', preprocess),
        ('onehot', onehot),
        ('drop_cols', drop_cols)
    ])
    return preprocess_pipeline


if __name__ == '__main__':
    # ADD HERE THE PATHS OF THE DATASET TO PREPROCESS AND THE OUTPUT PATH
    input_path = ''
    output_path = ''
    
    df = pd.read_csv(input_path)
    pipeline = Create_Pipeline()
    finished_df = pipeline.fit_transform(df)
    finished_df.to_csv(output_path)
