#!/usr/bin/env python
""" Stuff in this file:

    * some PCA
"""

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class QuantumBlackPCA:
    """ Do PCA on the data passed into it.

    Input: data <pd.DataFrame>,
           n_components <int> (DEFAULT: num_columns)

    """
    def __init__(self, data, columns=None, n_components=None):
        """ Construct the PCA and fit.
        """
        self.data = data
        # construct PCA model
        self.pca = dict()
        self.n_components = list(n_components)
        # run PCA on data at different desired components
        for n_comp in self.n_components:
            self.pca[n_comp] = PCA(n_components=n_comp)
            self.pca[n_comp].fit(self.data)
        if columns is None:
            self.columns = self.data.columns
        self.max_key = max(self.n_components)
        # explained variance ratios
        self.cum_explained_variance_ratio = np.cumsum(self.pca[self.max_key].explained_variance_ratio_)

        # self.total_weights = dict()
        # self.columns = dict()
        # for key in self.pca:
            # self.total_weights[key] = np.zeros((len(self.data.columns)))
            # self.columns[key] = np.asarray(self.data.columns)
            # for component in self.pca[key].components_:
                # self.total_weights[key] += component
            # self.columns[key] = self.columns[key][np.argsort(np.abs(self.total_weights[key]))]
            # self.total_weights[key] = np.sort(np.abs(self.total_weights[key]))

    def get_derived_dataframe(self, num_components):
        """ Return a pd.DataFrame with the derived components,
        with column headers as simple numbers.

        Also returns a dict of what column header went into the
        component, and their weights.
        """

        if num_components in self.n_components:
            key = num_components
        else:
            key = self.max_key

        derived_df_dict = []

        for ind, component in enumerate(self.pca[key].components_):
            for idx, row in enumerate(self.data.iterrows()):
                print(row)
                # derived_df_dict[idx]['Feature' + str(ind)] = component * row
        self.derived_df = pd.DataFrame.from_dict(derived_df_dict)
        return self.derived_df

    def plot_pca_explained_variance(self):
        """ Plot explained variance against the number
        of components used in the PCA.
        """
        plt.plot(self.cum_explained_variance_ratio)
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        return

    def plot_column_weights(self):
        """ Sum over weights per column and plot. """
        cmap = plt.cm.get_cmap('Dark2', len(self.data.columns))(np.linspace(0, 0.8, len(self.data.columns)))
        for key in self.pca:
            plt.plot(range(0, len(self.columns[key])), self.total_weights[key], label=key, c=cmap[key-1])
        plt.xlabel('column weight rank')
        plt.ylabel('total weight over feature space')
        # plt.xticks(range(0, len(columns)), np.argsort(total_weights))
        return
