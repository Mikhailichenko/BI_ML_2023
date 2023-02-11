import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    tp = 0
    fp = 0 
    fn = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            if y_pred[i] == 1:
                tp += 1
        elif y_pred[i] == 1:
            fp += 1
        else:
            fn += 1
    #The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.
    if (tp + fp) == 0:
        precision_score = 0
    else:
        precision_score = tp / (tp + fp)
    print(f'precision_score {precision_score}')
    #The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.
    if (tp + fn) == 0:
        recall_score = 0
    else:
        recall_score = tp / (tp + fn)
    print(f'recall_score {recall_score}')
    
    #F1 = 2 * (precision * recall) / (precision + recall)
    if (precision_score + recall_score) == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision_score * recall_score) / (precision_score + recall_score)
    print(f'f1_score {f1_score}')
    
    accuracy_score = (fp + fn)/len(y_pred)
    print(f'accuracy_score {accuracy_score}')

    
def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    pos = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            pos +=1
    return pos/len(y_pred)


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """
    
    
    r2 = 1 - np.sum((y_true - y_pred)**2)/np.sum((y_true - np.mean(y_true))**2)
    return r2


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    """
    YOUR CODE IS HERE
    """
    mse = sum(np.square(y_true - y_pred))/len(y_pred)
    return mse


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    """
    YOUR CODE IS HERE
    """
    mae = sum(abs(y_true - y_pred))/len(y_pred)
    return mae