from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class MapColumn(BaseEstimator, TransformerMixin):
    def __init__(self, column, mapping):
        self.column = column
        self.mapping = mapping
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X[self.column] = X[self.column].map(self.mapping)
        return X
    
class DropColumns(BaseEstimator, TransformerMixin): 
    def __init__(self, columns):
        self.columns = columns
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.drop(columns=self.columns)
        return X

class FillNA(BaseEstimator, TransformerMixin):
    def __init__(self, columns, value):
        self.columns = columns
        self.value = value
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X[self.columns] = X[self.columns].fillna(self.value)
        return X

class KeepRowsWhereColumnIsIn(BaseEstimator, TransformerMixin):
    def __init__(self, column, values):
        self.column = column
        self.values = values
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X[X[self.column].isin(self.values)]
        return X
    
class VariantOccurrences(BaseEstimator, TransformerMixin): 
    def __init__(self, column):
        self.column = column
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X['VARIANT_OCCURRENCES'] = 0
        hgvsg_counts = X[self.column].value_counts(dropna=False)
        for variante in hgvsg_counts.index:
            X.loc[X[self.column] == variante, 'VARIANT_OCCURRENCES'] = hgvsg_counts[variante]
        X = X.drop_duplicates(subset=[self.column], keep='first')
        return X

class ExonIntronType(BaseEstimator, TransformerMixin):
    def __init__(self, exon, intron):
        self.exon = exon
        self.intron = intron
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X.loc[X[self.exon].notna(), 'EXON_INTRON_TYPE'] = 'EXON'
        X.loc[X[self.intron].notna(), 'EXON_INTRON_TYPE'] = 'INTRON'
        X.loc[X[self.exon].isna() & X[self.intron].isna(), 'EXON_INTRON_TYPE'] = 'NA'
        X.loc[X[self.exon].notna(), 'EXON_INTRON_N'] = X[self.exon]
        X.loc[X[self.intron].notna(), 'EXON_INTRON_N'] = X[self.intron]
        X['EXON_INTRON_N'].fillna(0, inplace=True)
        X["EXON_INTRON_N"] = X["EXON_INTRON_N"].apply(lambda x: str(x).split("/")[0])
        return X

class AlleleFreq(BaseEstimator, TransformerMixin): 
    def __init__(self, af_1000, af_gnomad, variant_occurrences):
        self.af_1000 = af_1000
        self.af_gnomad = af_gnomad
        self.variant_occurrences = variant_occurrences
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X.loc[X[self.af_1000].notna(), 'ALLELE_FREQ'] = X[self.af_1000]
        X.loc[X[self.af_gnomad].notna(), 'ALLELE_FREQ'] = X[self.af_gnomad]
        X.loc[X[self.af_gnomad].isna() & X[self.af_1000].isna(), 'ALLELE_FREQ'] = X[self.variant_occurrences] / len(X)
        return X

class SplitGT(BaseEstimator, TransformerMixin):
    def __init__(self, gt):
        self.gt = gt
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X[['GT1', 'GT2']] = X[self.gt].str.split('/', expand=True)
        X[['GT1', 'GT2']] = X[['GT1', 'GT2']].astype(int)
        return X
    
class MostSevereConsequence(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X.loc[X[self.column].str.contains('splice'), self.column] = 'splice'
        conseq_counts = X[self.column].value_counts(dropna=False)
        X[self.column] = X[self.column].str.upper()
        X.loc[X[self.column].isin(conseq_counts[conseq_counts < 50].index), self.column] = 'other'
        return X
    
class VariantType(BaseEstimator, TransformerMixin):  
    def __init__(self, column):
        self.column = column
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X.loc[X[self.column].str.contains('>'), self.column] = 'SOST'
        X.loc[X[self.column].str.contains('delins'), self.column] = 'DELINS'
        return X
    
class RefAlt(BaseEstimator, TransformerMixin):  
    def __init__(self, ref, alt):
        self.ref = ref
        self.alt = alt
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X.loc[~X[self.ref].isin(['A', 'C', 'G', 'T']), self.ref] = 'X'
        X.loc[~X[self.alt].isin(['A', 'C', 'G', 'T']), self.alt] = 'Y'
        return X
    
class RenameColumns(BaseEstimator, TransformerMixin):  
    def __init__(self, columns):
        self.columns = columns
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.rename(columns=dict(self.columns))
        return X

def brca_preprocessing_pipeline():
    brca_pipeline = Pipeline([
        ('map_column', MapColumn('LABEL', {'NEG': 0, 'POS': 1, 'VUS': 2})),
        ('fill_na', FillNA('FAMILIARI', 0)),
        ('keep_rows_where_gene_symbol_brca', KeepRowsWhereColumnIsIn('GENE_SYMBOL', ['BRCA1', 'BRCA2'])),
        ('keep_rows_where_familiari', KeepRowsWhereColumnIsIn('FAMILIARI', ['NO'])),
        ('variant_occurrences', VariantOccurrences('HGVSG')),
        ('rename_column', RenameColumns([('CLINVAR_ID', 'CLINVAR_ID_COUNT'), ('DOMAINS', 'DOMAINS_COUNT'), ('HGVSG', 'VARIANT_TYPE')])),
        ('exon_intron_type', ExonIntronType('EXON', 'INTRON')),
        ('allele_freq', AlleleFreq('1000GP3_EUR_AF', 'GNOMAD_EXOMES_NON_CANCER_NFE_AF', 'VARIANT_OCCURRENCES')),
        ('split_gt', SplitGT('GT')),
        ('most_severe_consequence', MostSevereConsequence('MOST_SEVERE_CONSEQUENCE')),
        ('ref_alt', RefAlt('REF', 'ALT')),
        ('variant_type', VariantType('VARIANT_TYPE')),
        ('drop_columns', DropColumns(['ANNO', 'MSP', 'A.O.P.', 'ETA ALLA DIAGNOSI', 'DUTTALE', 'LOBULARE', 'BILATERALE', 
                                    'TRIPLO NEG', 'INFILTRANTE', 'DUTTALE IN SITU', 'MICROPAPILLARE', 'MUCINOSO', 'IN SITU', 
                                    'SIEROSO', 'ENDOMETRIOIDE', 'ALTRO', 'N/D', 'ALTRO ISTOTIPO', 'ALTRO TUMORE', 'POS', 
                                    'STORIA FAMILIARE POS \r\n\r\nPER K MAMMARIO/ OVARICO', 'CAMPIONE', 'BUILT','INTRON',
                                    'STORIA FAMILIARE POS PER PATOLOGIA \r\n\r\nONCOLOGICA DIVERSA', 'FAMILIARI', 'CHROM', 
                                    'Data Firma Referto Formato Corto', 'Data di nascita formato breve', 'PANNELLO', 'EXON',
                                    'GENE_ENST', 'SAMPLE', 'PUBMED', '1000GP3_EUR_AF', 'GNOMAD_EXOMES_NON_CANCER_NFE_AF',
                                    'SESSO', 'GT'])),
        ('fillna_with_0', FillNA(['CLINVAR_ID_COUNT', 'DOMAINS_COUNT'], 0)),
    ])

    return brca_pipeline
