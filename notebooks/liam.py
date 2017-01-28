import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

sys.path.append('../preprocessing')
from data_cleaning import *
from feature_scaling import *


data = pd.DataFrame.from_csv('../DataFiles/train_withNaNinfo.csv')

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
forest = RandomForestClassifier(n_estimators=1000,     # Number of trees in ensemble
                                criterion='entropy', # Decision trees maximise information gain
                                n_jobs=8,            # Go faster number
                                random_state=1)      # Set RNGesus
# smaller version
#glade = RandomForestClassifier(n_estimators=10,     # Number of trees in ensemble
#                               criterion='entropy', # Decision trees maximise information gain
#                               n_jobs=8,            # Go faster number
#                               random_state=1)      # Set RNGesus

# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(x_train, y_train)
#glade = glade.fit(x_train,y_train)
# Take the same decision trees and run it on the test data
print(forest.score(x_test, y_test))
#output = glade.predict(x_test)
#print(glade.score(x_test, y_test))

# Probabilities
output = forest.predict(x_test)
proba = forest.predict_proba(x_test)

errors = np.where(y_test!=output,y_test,output)
true_positives  = np.where(np.logical_and(y_test,output))
true_negatives  = np.where(np.logical_not(np.logical_or(y_test,output)))
false_positives = np.where(y_test<output)
false_negatives = np.where(y_test>output)

n_true_positives = len(true_positives[0])
n_true_negatives = len(true_negatives[0])
n_false_negatives = len(false_negatives[0])
n_false_positives = len(false_positives[0])

print('sum: ', n_true_positives+n_false_positives+n_false_negatives+n_true_negatives)
print('true positives: ' , n_true_positives)
print('true negatives: ' , n_true_negatives)
print('false positives: ' , n_false_positives)
print('false negatives: ' , n_false_negatives)
print('ratio pos/neg: ', n_false_positives/n_false_negatives)
print('test ratio pos/neg: ', np.sum(y_test)/(len(y_test)-np.sum(y_test)))
