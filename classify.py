# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 22:41:28 2022

@author: teresa
"""

import pandas as pd
import argparse
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing as preproc
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC  
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score, f1_score
import numpy

parser = argparse.ArgumentParser(description='Comparative study of logistic regression and svm')
parser.add_argument('-d',metavar='dataset',required=True,dest='dataset', action='store',help='path and name of dataset')
args = parser.parse_args()

dataset = pd.read_csv(args.dataset)
#dataset = pd.read_csv("diabetes.csv")
#profile = dataset.profile_report(title='Diabetes Profiling Report')
#profile
print(" = 1 Summary of dataset = ")
# shape
print(" == 1.1. number of species, attributes == ")
print(dataset.shape)
# head
print(" == 1.2 first 10 items == ")
print(dataset.head(10))

# data types
print(" == 1.3 data types for each attributes == ")
print(dataset.dtypes)

# descriptions
print(" == 1.4 Statistical Summary == ")
print(dataset.describe())

# class distribution
print(" == 1.5 class distribution  ==")
print(dataset.groupby('Outcome').size())


print(" = 2. Data Visualization = ")
# box and whisker plots
print(" == 2.1 box and whisker plots == ")
dataset.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
fig = plt.show()
print(fig)

print(" == 2.2 histograms == ")
dataset.hist()
fig=plt.show()
print(fig)

# =============================================================================
# 
# =============================================================================

print(" == 3 Generation of dataset == ")
# Extract Features
X = dataset.iloc[:, :8]

# Extract Class Labels
y = dataset["Outcome"]


print(" == 3 Split of dataset, training size=0.8 == ")
# Split Dataset
x_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

print(" == training set ==")
print(x_train.shape)
print(y_train.shape)
print(" == testing set == ")
print(X_test.shape)
print(y_test.shape)
print(" ==  first 5 training items == ")
print(x_train.head())

# =============================================================================
print("== 4 Replacement of missing values ==")
print("=== Assuming, zero indicates missing values === ")
print("missing values by count")
print((X == 0).sum())
# make a copy of original data set
x_train_cp = x_train.copy(deep=True)

#replace 0 with NaN
x_train_cp[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = x_train_cp[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, numpy.NaN)

print("=== imputing by replacing missing values with mean column values ===")

x_train_cp = x_train_cp.fillna(x_train_cp.mean())
# count the number of NaN values in each column
print(x_train_cp.isnull().sum())
print((x_train_cp == 0).sum())


# =============================================================================
print(" == 5 Normalization of dataset == ")


print("normalized_attr: range of 0 to 1")
scaler_n = preproc.MinMaxScaler().fit(x_train_cp)
normalized_attr = scaler_n.transform(x_train_cp)
normalized_df = pd.DataFrame(normalized_attr)
print(normalized_df.describe())

# =============================================================================
# print("standardized_attr: mean of 0 and stdev of 1")
# scaler = preproc.StandardScaler().fit(x_train_cp)
# standardized_attr = scaler.transform(x_train_cp)
# standardized_df = pd.DataFrame(standardized_attr)
# print(standardized_df.describe())
# =============================================================================

# =============================================================================

#SVM
# =============================================================================
print(" == 6 SVM model == ")
x_train=normalized_attr
print("=== Selection of SVM ===")
#select attribute
for k in ('linear', 'poly', 'rbf', 'sigmoid'):
    model = svm.SVC(kernel=k)
    model.fit(x_train, y_train)
    scores = cross_val_score(model, x_train, y_train, cv=5)
    print(k,"%0.3f accuracy with a standard deviation of %0.3f" % (scores.mean(), scores.std()))


# Using the best model
print("=== kernel=linear is selected ===")

for k in (0.01,0.1,1, 10, 100):
        model = svm.SVC(kernel='linear',C=k)
        model.fit(x_train, y_train)
        scores = cross_val_score(model, x_train, y_train, cv=5)
        print('C=',k,"%0.3f accuracy with a standard deviation of %0.3f" % (scores.mean(), scores.std()))


print("=== C=10, is selected ===")
model = svm.SVC(kernel='linear',C=10)
model.fit(x_train, y_train)

print("=== Evaluation of SVM ===")
# Accuracy on Testing Set
X_test_svm = scaler_n.transform(X_test)
y_pred = model.predict(X_test_svm)
print("Accuracy Score:", accuracy_score(y_test, y_pred))



    
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
    
print("Precision is", precision)
print("Recall is", recall)
print("F1 score is", f1)

# Generate classification report
print(classification_report(y_test, y_pred))
print("=== Confustion matrix ===")
cf_matrix = confusion_matrix(y_test, y_pred)
print(cf_matrix)


# 
# =============================================================================
#Logistic Regression
print(" == 7 Logistic Regression model == ")
print("=== Selection of logsitic regression ===")
print("=== Solver=liblinear is selected ===")

for k in (0.1, 1, 10, 100):
    lrmodel=LogisticRegression(solver='liblinear',penalty='l2',C=k)
    lrmodel.fit(x_train, y_train)
    scores = cross_val_score(lrmodel, x_train, y_train, cv=5)
    print("l2",'c=',k)
    print("%0.3f accuracy with a standard deviation of %0.3f" % (scores.mean(), scores.std()))
    
    lrmodel=LogisticRegression(solver='liblinear',penalty='l1',C=k)
    lrmodel.fit(x_train, y_train)
    scores = cross_val_score(lrmodel, x_train, y_train, cv=5)
    print('l1','c=',k)
    print("%0.3f accuracy with a standard deviation of %0.3f" % (scores.mean(), scores.std()))

print("=== C=10, penalty=l1 is selected ===")
lrmodel = LogisticRegression(C=10, penalty='l1',solver='liblinear')
lrmodel.fit(x_train, y_train)

print("=== Evaluation of SVM ===")
X_test_lr = scaler_n.transform(X_test)
y_pred = lrmodel.predict(X_test_lr)
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Compute precision, recall and f1 score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
    
print("Precision is", precision)
print("Recall is", recall)
print("F1 score is", f1)
print(classification_report(y_test, y_pred))

print("=== Confustion matrix ===")
cf_matrix = confusion_matrix(y_test, y_pred)
print(cf_matrix)

print("=== Importance of each feature ===")
#get importance
importance = lrmodel.coef_[0]
#summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))









