import numpy as np


def bayes_output(output, score, class_test):
    """ Return num false pos and neg. """
    true_positives = np.where(np.logical_and(class_test, output))
    true_negatives = np.where(np.logical_not(np.logical_or(class_test, output)))
    false_positives = np.where(class_test < output)
    false_negatives = np.where(class_test > output)

    n_true_positives = len(true_positives[0])
    n_true_negatives = len(true_negatives[0])
    n_false_negatives = len(false_negatives[0])
    n_false_positives = len(false_positives[0])

    print('sum: ', n_true_positives+n_false_positives+n_false_negatives+n_true_negatives)
    print('true positives: ', n_true_positives)
    print('true negatives: ', n_true_negatives)
    print('false positives: ', n_false_positives)
    print('false negatives: ', n_false_negatives)
    print('ratio pos/neg: ', n_false_positives/n_false_negatives)
    print('test ratio pos/neg: ', np.sum(class_test)/(len(class_test)-np.sum(class_test)))
    print('bs score', score)
