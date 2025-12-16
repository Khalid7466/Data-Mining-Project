import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Setup path to import from 'modules' directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from modules import tool_decision_tree

# Import dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "encoded_data.csv")
data = tool_decision_tree.importdata(data_path)

# Split dataset
X, Y, X_train, X_test, y_train, y_test = tool_decision_tree.prepare_dataset(data)

# Train model
clf = tool_decision_tree.train_using_entropy(X_train, y_train)

# Make predictions
y_pred = tool_decision_tree.prediction(X_test, clf)

# Evaluate accuracy
tool_decision_tree.cal_accuracy(y_test, y_pred)

# Plot the decision tree
feature_names = list(data.columns[:-1])  
class_names = [str(cls) for cls in sorted(data.iloc[:, -1].unique())]  
tool_decision_tree.plot_decision_tree(clf, feature_names, class_names)