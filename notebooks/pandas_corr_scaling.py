# coding: utf-8

# In[5]:

# import sklearn
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('../preprocessing')
from data_cleaning import *
from feature_scaling import *

data = pd.DataFrame.from_csv('../DataFiles/train.csv')
data = clean_data(data)
data = scale_features(data)

corrmat = data.corr()

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(20, 15))

# Draw the heatmap using seaborn
sns.heatmap(corrmat, square=True)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(5)
# Use matplotlib directly to emphasize known networks
# networks = corrmat.columns.get_level_values("network")
# for i, network in enumerate(networks):
#     if i and network != networks[i - 1]:
#         ax.axhline(len(networks) - i, c="w")
#         ax.axvline(i, c="w")
f.tight_layout()
plt.savefig("corr.png", dpi=600)
plt.show()
