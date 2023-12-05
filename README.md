# Project: Unification and Classification of GMO HC and BRCA Files for Diagnosis

## Project Scope

The project aims to merge GMO(Report table), HC(Excel HC panel), VCF and VEP(Variant Effect Predictor) files and GMO BRCA(Excel BRCA) VCF and VEP files to classify Variants of Unknown Significance (VUS) using Machine Learning models (Random Forest and Support Vector Machine). The ultimate goal is to classify VUS as POS (Positive) or NEG (Negative) based on the available data.

## Link Slides on Google Drive
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

With standardized data, the 3 files(For both HC and BRCA) were merged based on patient code, creating two single semifinal files.

#### Step 4 - Merge with VEP Files

The semifinal files were merged with the VEP files, creating two final files, doing and outer join on the patient code.

#### Step 5 - Dataset Preparation for Learning Model (Pipeline)

This repository contains code for a data preprocessing pipeline and feature engineering for a genetic dataset. The code utilizes Python libraries like pandas, numpy, and scikit-learn.

##### Libraries Used
- pandas
- numpy
- scikit-learn
- xgboost

##### Instructions

1. **Setup:**
    - Ensure the necessary libraries (requirements.txt) are installed, if not you can install them by using `pip install -r requirements.txt`
    - The code is written in Python.

2. **Execution:**
    - Load your dataset using and pass it to the `file_path` variable.
    - Execute the training on the loaded dataset.
    - See the results and predictions on your training data.

3. **Example:**
    - Below is a sample script to execute the pipeline:
    ```python
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
    ```

Replace `'file_path'` with the path to your dataset and see the results for yourself! 

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
