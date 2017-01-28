import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def scale_features( df, sc=StandardScaler() ):
    """
    Algorithms behave better if features are roughly
    of the same scale. This uses a standardisation
    procedure by default:

    x_i^std = ( x_i - mean_x ) / ( stddev_x)
    where x is a given feature column

    This centers feature columns over zero.
    Standardisation is less sensitive to outliers than
    normalisation, and usually more practical.

    Parameters:
    >   df : pandas dataframe
        Unscaled data
    >   sc : sklearn preprocessing scaler
        The scaling algorithm to use.
        StandardScaler by default

    Returns:
    >   df_scaled : pandas dataframe
        Standardised data
    """
    scaled = np.zeros([len(df.index),len(df.columns)])
    for col_num in range(len(df.columns)) :
        if df[df.columns[col_num]].dtype == np.float :
            X = df[df.columns[col_num]].values
            sc = sc.fit(X.reshape(-1,1))
            X_scaled = sc.transform(X.reshape(-1,1))
            scaled[:,col_num] = X_scaled[:].flatten()
    df_scaled = pd.DataFrame( scaled, df.index, df.columns)
    return df_scaled
