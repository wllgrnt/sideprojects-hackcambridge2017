import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

sys.path.append('../preprocessing')
from data_cleaning import *
from feature_scaling import *


data = pd.DataFrame.from_csv('../DataFiles/train.csv')

cols = [col for col in data.columns if col != "class"]
features = data[cols]
bankrupt = data['class']

x_train, x_test, y_train, y_test = train_test_split(
    features, bankrupt, test_size=0.2, random_state=1)

x_train = clean_data(x_train)
x_test  = clean_data(x_test)
#x_train = scale_features(x_train)
#x_test  = scale_features(x_test)

# Create the random forest object which will include all the parameters
# for the fit
forest = RandomForestClassifier(n_estimators=100,    # Number of trees in ensemble
                                criterion='entropy', # Decision trees maximise information gain
                                n_jobs=8,            # Go faster number
                                random_state=1)      # Set RNGesus
# smaller version
glade = RandomForestClassifier(n_estimators=10,     # Number of trees in ensemble
                               criterion='entropy', # Decision trees maximise information gain
                               n_jobs=8,            # Go faster number
                               random_state=1)      # Set RNGesus

# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(x_train, y_train)
glade = glade.fit(x_train,y_train)
# Take the same decision trees and run it on the test data
output = forest.predict(x_test)
print(forest.score(x_test, y_test))
output = glade.predict(x_test)
print(glade.score(x_test, y_test))
