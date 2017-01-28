import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer

def clean_data( data, imp=Imputer(missing_values='NaN', strategy='mean', axis=0) ):
    """
    NaN cleaner.
    For all columns, replace NaN with mean values.

    Parameters:
    data : numpy array (n_samples,n_features)
    imp  : Imputer class

    Returns:
    clean_data  : numpy array (n_samples,n_features)
    """
    imp = imp.fit( data )
    clean_data = imp.transform( data )
    return clean_data
