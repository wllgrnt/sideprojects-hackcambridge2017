import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def scale_features( data, sc=StandardScaler() ):
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
    >   data : numpy array (n_samples,n_features)
        Unscaled data
    >   sc : sklearn preprocessing scaler
        The scaling algorithm to use.
        StandardScaler by default

    Returns:
    >   data : numpy array (n_samples,n_features)
        Standardised data
    """
    for col_num in range(data.shape[1]) :
        X = data[:,col_num]
        sc = sc.fit(X.reshape(-1,1))
        X_scaled = sc.transform(X.reshape(-1,1))
        data[:,col_num] = X_scaled[:].flatten()
    return data
