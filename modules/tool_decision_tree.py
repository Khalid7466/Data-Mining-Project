# tool_decision_tree.py
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

def importdata(path):
    """
    Import dataset from a CSV file and display basic information.
    
    Parameters:
        path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: The loaded dataset.
    """
    balance_data = pd.read_csv(path)
    
    print("Dataset Length: ", len(balance_data))
    print("Dataset Shape: ", balance_data.shape)
    print("Dataset: ", balance_data.head())
    
    return balance_data

def prepare_dataset(df, target_column="stroke", test_size=0.2, random_state=100):
    """
    Split dataset into features and target, then into training and test sets.
    
    Parameters:
        df (pd.DataFrame): The dataset.
        target_column (str): Name of the target column. Default is "stroke".
        test_size (float): Proportion of test set. Default is 0.2.
        random_state (int): Random seed for reproducibility. Default is 100.
    
    Returns:
        tuple: (X, Y, X_train, X_test, y_train, y_test)
    """
    Y = df[target_column].values
    X = df.drop(columns=[target_column]).values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )
    
    return X, Y, X_train, X_test, y_train, y_test

def train_using_entropy_smote(X_train, y_train, max_depth=None, min_samples_leaf=1, random_state=100):
    """
    Train a Decision Tree classifier using Entropy criterion with SMOTE to balance classes.
    
    Parameters:
        X_train (np.array): Training features.
        y_train (np.array): Training labels.
        max_depth (int): Maximum depth of the tree. Default is None.
        min_samples_leaf (int): Minimum samples per leaf. Default is 1.
        random_state (int): Random seed. Default is 100.
    
    Returns:
        DecisionTreeClassifier: Trained Decision Tree classifier.
    """
    # Apply SMOTE to balance classes
    sm = SMOTE(random_state=random_state)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    
    # Train Decision Tree
    clf = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        class_weight='balanced'

    )
    clf.fit(X_train_res, y_train_res)
    
    return clf

def prediction(X_test, clf_object):
    """
    Predict labels for test data using trained classifier.
    
    Parameters:
        X_test (np.array): Test features.
        clf_object (DecisionTreeClassifier): Trained classifier.
    
    Returns:
        np.array: Predicted labels.
    """
    y_pred = clf_object.predict(X_test)
    print("Predicted values:")
    print(y_pred)
    return y_pred

def cal_accuracy(y_test, y_pred):
    """
    Calculate and print confusion matrix, accuracy, and classification report.
    
    Parameters:
        y_test (np.array): True labels.
        y_pred (np.array): Predicted labels.
    """
    print("Confusion Matrix: ")
    print(confusion_matrix(y_test, y_pred))
    print("Accuracy : ", accuracy_score(y_test, y_pred)*100)
    print("Report : ")
    print(classification_report(y_test, y_pred))

def plot_decision_tree(clf_object, feature_names, class_names):
    """
    Plot a trained Decision Tree.
    
    Parameters:
        clf_object (DecisionTreeClassifier): Trained Decision Tree.
        feature_names (list): List of feature names.
        class_names (list): List of class names as strings.
    """
    plt.figure(figsize=(15, 10))
    plot_tree(clf_object, filled=True, feature_names=feature_names, class_names=class_names, rounded=True)
    plt.title("Decision Tree")
    plt.show()
