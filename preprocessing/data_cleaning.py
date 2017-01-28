import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer

def clean_data( df, imp=Imputer(missing_values='NaN', strategy='mean', axis=0) ):
    """
    NaN cleaner.
    For all columns, replace NaN with mean values.

    Parameters:
    df  : pandas dataframe
    imp : Imputer class

    Returns:
    df  : pandas dataframe
    """
    df = df.fillna(df.mean())
    #imp = imp.fit( df.values )
    #imp_data = imp.transform( df.values )
    #df=df.update(imp_data)
    return df
