# Project: Unification and Classification of GMO HC and BRCA Files for Diagnosis

## Project Scope

The project aims to merge GMO(Report table), HC(Excel HC panel), VCF and VEP(Variant Effect Predictor) files and GMO BRCA(Excel BRCA) VCF and VEP files to classify Variants of Unknown Significance (VUS) using Machine Learning models (Random Forest and Support Vector Machine). The ultimate goal is to classify VUS as POS (Positive) or NEG (Negative) based on the available data.

## Link Slides on Google Drive
They are working in progress, they will be available soon completed
(https://docs.google.com/presentation/d/16vIr4YETAOycf6kZIynbmBwbPGMWkvEMFpeLpVITlrU/edit?usp=sharing)

## Process Documentation

### Processing VCF Files

#### Step 1. Data Exploration

Initially, an exploratory analysis was conducted on VCF and Excel files. Key observations included:

- Identification of duplicate files and non-standard naming conventions.
- Detection of files lacking content.
- Identification of a partially loaded file.

#### Step 2. Duplicate Removal

A Python application was developed to identify and remove duplicate files based on their content.  

#### Step 3. Conversion of VCF Files

VCF files were converted into more manageable data structures, such as pandas.Dataframe, to facilitate linking with information present in Excel. Key information in the files included CHROM, POS, REF, ALT_1, VERSIONE, and GT.

#### Step 4. Use of VEP for Additional Information

Utilizing VEP's REST APIs, a JSON file containing detailed variant information was obtained. This information was filtered and added to the Pandas Dataframe containing the VCF extractions.

#### Step 5. Merging VCF Files with Excel

To merge the two data sources, the SAMPLE parameter was used to link VCF files to data in Excel. Transformations were made to standardize the SAMPLE field.

### Processing Excel Files

#### Step 1 - Data Exploration
With a lot of data to process, it was important to understand the data and its structure. The data was explored to identify the following:

- Columns with missing data and bad data
- Columns with duplicate data
- Data types of columns and their suitability for machine learning
- Columns with data that could be used for feature engineering

#### Step 2 - Data Normalization e Data cleaning 
We have normalized the data by removing the columns that were not useful for the analysis and by removing the rows with missing data. We have also cleaned the data by removing the duplicate rows and by removing the rows with bad data.

We standardized the data by converting the data types of the columns to the appropriate data types and by converting the data to uppercase.

Using regex, we cleaned the data in the columns by removing the special characters and by replacing the values with the appropriate values.

#### Step 3 - Merge with VCF Files

With standardized data, the 3 files(For both hc and brca) were merged based on patient code, creating two single semifinal files.

#### Step 4 - Merge with VEP Files

The semifinal files were merged with the VEP files, creating two final files, doing and outer join on the patient code.

#### Step 5 - Dataset Preparation for Learning Model (Pipeline)

This repository contains code for a data preprocessing pipeline and feature engineering for a genetic dataset. The code utilizes Python libraries like pandas, numpy, and scikit-learn.

##### Libraries Used
- pandas
- numpy
- scikit-learn (`sklearn`)
  - `OneHotEncoder`
  - `FunctionTransformer`
  - `make_column_transformer`
  - `Pipeline`

##### Steps Implemented

1. **Data Preprocessing Functions:**
    - `find_empty_columns`: Identifies empty columns in the dataset.
    - `drop_columns`: Drops specified columns from the DataFrame.
    - `to_uppercase`: Converts string values in a DataFrame to uppercase.
    - `clean_columns`: Cleans specific columns based on predefined replacements.
    - `rename_columns`: Renames columns as per the provided mapping.
    - `convert_column_types`: Converts specified columns to defined data types.
    - `drop_NaN_rows`: Drops rows with NaN values in specified columns.
    - `fill_NaN_rows`: Fills NaN values in specified columns with provided values.

2. **Data Preprocessing Function - `Preprocess_Dataset`:**
    - Implements a series of data cleaning steps, including dropping unnecessary columns, renaming columns, filling NaN values, cleaning text in columns, converting data types, handling missing values, splitting and manipulating columns, and engineering new features.

3. **One-Hot Encoding:**
    - Utilizes `make_column_transformer` and `OneHotEncoder` to perform one-hot encoding on categorical columns.

4. **Pipeline Creation - `Create_Pipeline`:**
    - Constructs a pipeline using `FunctionTransformer` and `make_column_transformer`.
    - The pipeline encapsulates the `Preprocess_Dataset` function and one-hot encoding.

##### Instructions

1. **Setup:**
    - Ensure the necessary libraries (`pandas`, `numpy`, `scikit-learn`) are installed.
    - The code is written in Python.

2. **Execution:**
    - Load your dataset using `pd.read_excel` or any suitable method and pass it to the `Create_Pipeline` function.
    - Execute the pipeline using `fit_transform` method on the loaded dataset.
    - The resulting processed DataFrame can be accessed and saved using appropriate file I/O functions (`to_csv`, etc.).

3. **Example:**
    - Below is a sample script to execute the pipeline:
    ```python
    import pandas as pd
    from pipeline_script import Create_Pipeline

    if __name__ == '__main__':
        df = pd.read_excel('path_to_your_dataset.xlsx')  
        pipeline = Create_Pipeline()
        finished_df = pipeline.fit_transform(df)

        print(finished_df)  
        finished_df.to_csv('path_to_save_processed_data.csv')  
    ```

Replace `'path_to_your_dataset.xlsx'` with the path to your dataset, and `'path_to_save_processed_data.csv'` with the desired path to save the processed data. 

## Project Participants

- Leonardo Rocca
- Gianluca Meneghetti
- Alessandro Kuz
- Stefano Bonfanti
- Gabriele Laguna

## Contacts

For further information or queries, please contact project members:

- Leonardo Rocca (Email: l.rocca@itsrizzoli.it)
- Gianluca Meneghetti (Email: g.meneghetti@itsrizzoli.it)
- Alessandro Kuz (Email: a.kuz@itsrizzoli.it)
- Stefano Bonfanti (Email: s.bonfanti@itsrizzoli.it)
- Gabriele Laguna (Email: g.laguna@itsrizzoli.it)
