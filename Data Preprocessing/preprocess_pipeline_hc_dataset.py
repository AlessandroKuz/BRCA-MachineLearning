from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn import set_config
import pandas as pd
import numpy as np

set_config(transform_output="pandas")


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
        if df[column].dtype == "object":
            df[column] = df[column].apply(
                lambda x: x.upper().strip() if isinstance(x, str) else x
            )
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


def Process_Tresholds_Columns(df, column_name, sep, threshold):
    formatted_values = df[column_name].str.split(sep).explode()
    value_counts = formatted_values.value_counts(dropna=False)
    value_counts = value_counts[value_counts >= threshold]

    for index, row in df.iterrows():
        values_list = row[column_name].split(sep)
        for value in values_list:
            if value == "":
                continue
            if value in value_counts.index:
                value_col_name = f"{column_name}_{value}"
                df.loc[index, value_col_name] = 1
            else:
                other_col_name = (
                    f"{column_name}_ALTRO" if column_name != "AOP" else f"{column_name}_A"
                )
                df.loc[index, other_col_name] = 1

    return df


def fill_rankscore_columns(df, fill_value):
    for column in df.columns:
        if column.endswith("RANKSCORE"):
            df[column] = df[column].apply(lambda x: fill_value if pd.isna(x) else x)

    return df


def Preprocess_Dataset(df):
    columns_to_drop = [
        "ANNO",
        "MSP",
        "DIAGNOSI",
        "ALTRO ISOTIPO",
        "N/D",
        "Data Firma Referto Formato Corto",
        "BUILT",
        "GENE_ENST",
        "SAMPLE",
        "POS",
        "CAMPIONE",
        "PANNELLO",
        "GT",
        "ETA_DIAGNOSI",
        "DATA_NASCITA",
        "EXON/INTRON_POS",
        "AOP",
        "SEDE_TUMORE_FAMILIARITA",
        "ANNO_NASCITA",
        "STORIA_FAMILIARE_POS",
        "PUBMED",
        "1000GP3_EUR_AF",
        "CLINPRED_PRED",
        "DOMAINS",
        "GNOMAD_EXOMES_NON_CANCER_NFE_AF",


        "CLINVAR_ID"
    ]

    df = rename_columns(
        df,
        [
            ("STORIA FAMILIARE POS ", "STORIA_FAMILIARE_POS"),
            ("SEDE TUMORE FAMILIARITA'", "SEDE_TUMORE_FAMILIARITA"),
            ("ETA ALLA DIAGNOSI", "ETA_DIAGNOSI"),
            ("Data di nascita formato breve", "DATA_NASCITA"),
            ("A.O.P.", "AOP"),
            ("HGVSG", "VARTYPE"),
            ("TYPE", "IS_SOMATIC"),
            ("SESSO", "IS_MALE"),
        ],
    )

    fill_values = {
        "STORIA_FAMILIARE_POS": "0",
        "SEDE_TUMORE_FAMILIARITA": "ND",
        "MOST_SEVERE_CONSEQUENCE": "ND",
        "ETA_DIAGNOSI": "ND",
        "AOP": "ND",
        "STRAND": 0
    }
    df = fill_NaN_rows(df, fill_values)

    df = to_uppercase(df)

    columns_to_clean = [
        (
            "SEDE_TUMORE_FAMILIARITA",
            {
                " ": "",
                "\n": ",",
                "GH.": "",
                ".": ",",
                "COLONRETTO": "COLON,RETTO",
                "COLON-RETTO": "COLON,RETTO",
                "COLONSENO": "COLON,MAMMELLA",
                "COLONMAMMELLA": "COLON,MAMMELLA",
                "TRATTOGASTROINT": "GASTROINTESTINALE",
                "INSTESTINO": "GASTROINTESTINALE",
                "INTESTINO": "GASTROINTESTINALE",
                "GASTRICO": "GASTROINTESTINALE",
                "STOMACO": "GASTROINTESTINALE",
                "NEOPLASIAMAMMARIA": "MAMMELLA",
                "NEOPLASIAMAMMELLA": "MAMMELLA",
                "MAMMARIA": "MAMMELLA",
                "MAMMLLA": "MAMMELLA",
                "MAMMELL": "MAMMELLA",
                "MAMMELLAA": "MAMMELLA",
                "SENO": "MAMMELLA",
                "NEOPLASIAOVARICA": "OVAIO",
                "NEOPLASIAOVAIO": "OVAIO",
                "MELANOMAOVAIO": "OVAIO",
                "ENDOMETRIO": "OVAIO",
                "OVARICO": "OVAIO",
                "OVARICA": "OVAIO",
                "OVIAO": "OVAIO",
                "OVAIE": "OVAIO",
                "POLMONARE": "POLMONI",
                "POLMONE": "POLMONI",
                "POMONE": "POLMONI",
                "RENALE": "RENE",
                "OSSEA": "OSSEO",
                "OSSA": "OSSEO",
                "BOCCA": "ORALE",
                "CAVOORALE": "ORALE",
                "TESTICOLO": "TESTICOLI",
                "NEOPLASIACEREBRALE": "CERVELLO",
                "ENCEFALICA": "CERVELLO",
                "CEREBRALE": "CERVELLO",
                "ENCEFALO": "CERVELLO",
                "PROSTAT": "PROSTATA",
                "PROSTATAA": "PROSTATA",
                "VIEBILLIARI": "VIEBILIARI",
                "VIABILIARE": "VIEBILIARI",
                "BRCABILATERALE": "BRCA",
                "TIDOIDE": "TIROIDE",
                "EPATICA": "FEGATO",
                "EPATICO": "FEGATO",
                "EPATOCARCINOMA": "FEGATO",
            },
        ),
        (
            "STORIA_FAMILIARE_POS",
            {"SI": "1", "SÌ": "1", "S": "1", "NO": "0", "ND": "0", "N": "0"},
        ),
        (
            "AOP",
            {
                " ": "",
                "?": "ND",
                "N/D": "ND",
                "PR": "P",
                "PA": "P",
                "RENE": "A",
                "COLON": "C",
                "PELVICA": "A",
                "M,RENE": "M/A",
                "M, C.G, TI": "M/A/TI",
                "-": "/",
                ".": "/",
                ",": "/",
            },
        ),
        ("ETA_DIAGNOSI", {"-": "/", ",": "/", "N.D": "ND"}),
    ]

    df = clean_columns(df, columns_to_clean)

    df["GT_1"] = df["GT"].str.split("/").str[0]
    df["GT_2"] = df["GT"].str.split("/").str[1]

    df["IS_EXON"] = df["EXON"].apply(lambda x: 0 if pd.isna(x) else 1)
    df["IS_INTRON"] = df["INTRON"].apply(lambda x: 0 if pd.isna(x) else 1)

    df["EXON/INTRON_POS"] = "0/0"
    df.loc[df["IS_EXON"] == 1, "EXON/INTRON_POS"] = df["EXON"]
    df.loc[df["IS_INTRON"] == 1, "EXON/INTRON_POS"] = df["INTRON"]
    df.drop(columns=["EXON", "INTRON"], inplace=True)
    df["EXON/INTRON_POS"] = df["EXON/INTRON_POS"].astype("str")

    df["EXON/INTRON_CURR_POS"] = df["EXON/INTRON_POS"].apply(lambda x: x.split("/")[0])

    df["ANNO_NASCITA"] = "19" + df["DATA_NASCITA"].str[-2:]
    df["ANNO_NASCITA"] = pd.to_numeric(df["ANNO_NASCITA"])
    df["ETA"] = df["ANNO"] - df["ANNO_NASCITA"]

    df["ETA_DIAGNOSI_MIN"] = df["ETA_DIAGNOSI"].str.split("/").str[0]
    df["ETA_DIAGNOSI_MAX"] = df["ETA_DIAGNOSI"].str.split("/").str[-1]
    df["N_DIAGNOSI"] = df["ETA_DIAGNOSI"].str.count("/") + 1

    df.loc[
        df["ETA_DIAGNOSI_MIN"] == "ND", ["ETA_DIAGNOSI_MIN", "ETA_DIAGNOSI_MAX"]
    ] = df.loc[df["ETA_DIAGNOSI_MIN"] == "ND", "ETA"]

    df["IS_MALE"] = df["IS_MALE"].map({"M": 1, "F": 0})
    df["IS_SOMATIC"] = df["IS_SOMATIC"].map({"SOMATIC": 1, "GERMLINE": 0})
    df["VARTYPE"] = df["VARTYPE"].apply(
        lambda x: "SOST" if ">" in x else "INS" if "_" in x else "DEL"
    )

    value_counts = df["MOST_SEVERE_CONSEQUENCE"].value_counts(dropna=False)
    value_counts = value_counts[value_counts >= 120]
    df["MOST_SEVERE_CONSEQUENCE"] = df["MOST_SEVERE_CONSEQUENCE"].where(
        df["MOST_SEVERE_CONSEQUENCE"].isin(value_counts.index), "OTHER"
    )

    Process_Tresholds_Columns(df, "AOP", "/", 200)
    Process_Tresholds_Columns(df, "SEDE_TUMORE_FAMILIARITA", ",", 200)

    fill_rankscore_columns(df, 0.5)

    columns_to_convert = {
        "GT_1": int,
        "GT_2": int,
        "EXON/INTRON_CURR_POS": int,
        "ETA_DIAGNOSI_MIN": int,
        "ETA_DIAGNOSI_MAX": int,
    }

    df = convert_column_types(df, columns_to_convert)
    empty_columns = find_empty_columns(df)
    columns_to_drop.extend(empty_columns)
    df = drop_column(df, columns_to_drop)

    df_c = df.copy()
    df_c = df_c[
        df_c[["REF", "ALT"]]
        .apply(tuple, 1)
        .isin(df_c[["REF", "ALT"]].value_counts()[:12].index)
    ]

    df["REF"] = "X"
    df["ALT"] = "Y"

    df.loc[df_c.index, "REF"] = df_c["REF"]
    df.loc[df_c.index, "ALT"] = df_c["ALT"]

    df = drop_useless_columns(df)

    return df


onehot = make_column_transformer(
    (
        OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.int8),
        [
            "GENE_SYMBOL",
            "REF",
            "ALT",
            "IMPACT",
            "VARTYPE",
            "MOST_SEVERE_CONSEQUENCE",
        ],
    ),
    remainder="passthrough",
    verbose_feature_names_out=False,
)


def drop_useless_columns(df):
    suffixes_to_drop = ["_nan", "_ND", "_?", "_NO"]
    for col in df.columns:
        if any([col.endswith(suffix) for suffix in suffixes_to_drop]):
            df.drop(columns=col, inplace=True)

    return df


def Create_Pipeline():
    preprocess = FunctionTransformer(Preprocess_Dataset)
    preprocess_pipeline = Pipeline(
        [("preprocess", preprocess), ("onehot", onehot)]
    )
    return preprocess_pipeline



if __name__ == "__main__":
    df = pd.read_csv(r"C:\Users\AlessandroKuz\Desktop\ITS\PW01 - Project Work\ULTIMATE WORK\Clean data and scripts\DATASET_COMPLETO_HC.csv")
    pipeline = Create_Pipeline()
    finished_df = pipeline.fit_transform(df)

    finished_df.to_csv(r"C:\Users\AlessandroKuz\Desktop\ITS\PW01 - Project Work\ULTIMATE WORK\Clean data and scripts\HC_TRAIN_DF.csv", index=False)
