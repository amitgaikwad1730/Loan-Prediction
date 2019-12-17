# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 22:43:49 2019

@author: Amit
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

trainingDataset = pd.read_csv("C:/Users/Dell/Project/Loan Prediction Application/train_LoanPrediction.csv")
testingDataset = pd.read_csv("C:/Users/Dell/Project/Loan Prediction Application/test_LoanPrediction.csv")
print (trainingDataset.head())
combined = pd.concat([trainingDataset, testingDataset],ignore_index=True, sort=False)
print (combined.shape)
print (combined.columns)
y = trainingDataset['Loan_Status']

print ("TRAINING DATA DETAILS")
print ("Total  number of records present in the dataset -", trainingDataset.shape[0])
print ("Total  number of columns present in the dataset -", trainingDataset.shape[1])

print ("\n TESTING DATA DETAILS")
print ("Total  number of records present in the dataset -", testingDataset.shape[0])
print ("Total  number of columns present in the dataset -", testingDataset.shape[1])

print ("Following are the columns present in the dataset - ", trainingDataset.columns)
print (trainingDataset.dtypes)

print ("TOTAL NUMBER OF RECORDS IN THE COMBINED DATASET  - ", combined.shape[0])
print ("\n")
categoricalColNames = combined.iloc[:,1:].select_dtypes(include=['object'])
requiredCategoricalVariables = list(categoricalColNames.columns.values)
for x in requiredCategoricalVariables:
    print ("Number of value counts for -", x)
    print (combined[x].value_counts())
    print ('Number of Missing values: %d'% sum(combined[x].isnull()))
    print ("\n")
    
numericalColNames = combined.iloc[:,1:].select_dtypes(include=['int64','float64'])
requiredCategoricalVariables = list(numericalColNames.columns.values)
for x in requiredCategoricalVariables:
    print ('Number of missing values in ', x ,': %d'% sum(combined[x].isnull()))
    
combined.info()

### FILLING THE MISSING VALUES IN THE REQUIRED COLUMNS
combined['LoanAmount'].fillna(combined['LoanAmount'].mean(), inplace=True)
combined['Loan_Amount_Term'].fillna(combined['Loan_Amount_Term'].mean(), inplace=True)
combined['Self_Employed'].fillna('No',inplace=True)
combined['Married'].fillna('NA',inplace=True)
combined['Gender'].fillna('NA',inplace=True)
combined['Dependents'].fillna('0',inplace=True)
combined['Credit_History'].fillna(0,inplace=True)
    

requiredColumns = list(combined.columns.values)
print ("Checking if there are any missing values in the dataset - ")
for col in requiredColumns:
    print ("column name  -", col)
    print ('Final #missing: %d'% sum(combined[col].isnull()))
    print ("\n")
    
import seaborn as sns
plt.figure(figsize=(12,12))
sns.heatmap(combined.iloc[:, 2:].corr(), annot=True, square=True, cmap='BuPu')
plt.show()


plt.figure(figsize=(20,20))
temp = trainingDataset.iloc[:,2:].select_dtypes(include=['int64','float64'])
requiredColumns = list(temp.columns.values)
counter = 1
for col in requiredColumns:
    plt.subplot(3, 3, counter)
    trainingDataset[col].hist(color = 'green')
    plt.title(col)
    counter = counter + 1
    
    
plt.figure(figsize=(10,10))
temp3 = pd.crosstab(trainingDataset['Credit_History'], trainingDataset['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['green','red'], grid=False)

nrow_train = trainingDataset.shape[0]
X_train = combined[:nrow_train]
X_test = combined[nrow_train:]
from sklearn.preprocessing import LabelEncoder
var_mod = ['Gender','Married','Education','Self_Employed','Property_Area','Loan_Status','Dependents']
le = LabelEncoder()
for i in var_mod:
    X_train[i] = le.fit_transform(X_train[i])
    X_test[i] = le.fit_transform(X_test[i])
print ("CONVERTED THE CATEGORICAL VARIABLES INTO NUMERICALS")


sns.pairplot(trainingDataset[trainingDataset.columns.values], hue='Loan_Status', diag_kind='kde', height=2);

### THE BOX PLOT SHOW THE OUTLIERS IN YOUR DATA. 
### AS YOU CAN SEE COLUMNS NAMED "APPLICANT INCOME" AND "CO APPLICANT INCOME" HAVE OUTLIERS
temp = trainingDataset.iloc[:,2:].select_dtypes(include=['int64','float64'])
requiredColumns = list(temp.columns.values)
plt.figure(figsize=(10,10))
#trainingDataset[trainingDataset.columns.values].plot.box();
sns.boxplot(data=X_train[requiredColumns], palette="Set2")

#The above diagram tells us that there are outliers in columns such as "Applicant Income" and "Co Applicant Income"

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split #For K-fold cross validation

from sklearn import metrics
X = X_train.iloc[:, 2:11].values
y = X_train.iloc[:, 12].values
#X = X.reshape(X.shape[0],1)
X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.20, random_state=0)

LR_model = LogisticRegression(solver='sag')
LR_model.fit(X_tr,y_tr)
#Make predictions on training set:
predictions = LR_model.predict(X_te)

#Print accuracy
accuracy = metrics.accuracy_score(predictions,y_te)
print ("Accuracy : %s" % "{0:.3%}".format(accuracy))
print("\n Classification report for classifier %s:\n%s\n"
      % (LR_model, metrics.classification_report(y_te, predictions)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_te, predictions))

count = 0
print ("TOTAL NUMBER OF TESTING RECORD - ",X_te.shape[0])
for x in range(len(X_te)):
    if(y_te[x] == predictions[x]):
        count = count + 1
print ("NUMBER OF CORRECTLY PREDICTED OUTPUTS - ",count)
print("\n")
for x in range(0,5):
    if(y_te[x] == predictions[x]):
        print ("TESTING RECORD - ",X_te[x])
        print ("ACTUAL OUTPUT - ", y_te[x])
        print ("PREDICTED OUTPUT - ",predictions[x])
        print ("_________________________________")


temp = trainingDataset.iloc[:,2:11]
requiredColumns = list(temp.columns.values)
print (requiredColumns)
FinalData = list(zip(X_te, y_te, predictions))
print (FinalData[0])
my_submission = pd.DataFrame( X_te, columns=[requiredColumns])
my_submission['Actual_Loan_Status'] = y_te
my_submission['Predicted_Loan_Status'] = predictions
my_submission.head()
my_submission.to_csv('LoanPredictionSubmissions.csv', index=False)


#Import models from scikit learn module:
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split #For K-fold cross validation

from sklearn import metrics

model = LogisticRegression(solver='sag')
X = X_train.iloc[:, 10:11].values
y = X_train.iloc[:, 12].values
#X = X.reshape(X.shape[0],1)
X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.25,random_state=0)

model.fit(X_tr,y_tr)
#Make predictions on training set:
predictions = model.predict(X_te)

#Print accuracy
accuracy = metrics.accuracy_score(predictions,y_te)
print ("Accuracy : %s" % "{0:.3%}".format(accuracy))
print("\n Classification report for classifier %s:\n%s\n"
      % (model, metrics.classification_report(y_te, predictions)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_te, predictions))

count = 0
print ("TOTAL NUMBER OF TESTING RECORD - ",X_te.shape[0])
for x in range(len(X_te)):
    if(y_te[x] == predictions[x]):
        count = count + 1
print ("NUMBER OF CORRECTLY PREDICTED OUTPUTS - ",count)
print("\n")
for x in range(0,5):
    if(y_te[x] == predictions[x]):
        print ("TESTING RECORD - ",X_te[x])
        print ("ACTUAL OUTPUT - ", y_te[x])
        print ("PREDICTED OUTPUT - ",predictions[x])
        print ("_________________________________")
        
 import seaborn as sns
sns.set(style="darkgrid")
plt.figure(figsize=(12,6))
ax = sns.countplot(x=predictions)
ax.set_xticklabels(["Loan Not Approved","Loan Approved"])
print ("FOLLOWING IS THE COUNTPLOT DISPLAYING THE PREDICTIONS - ")      



from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier()
rf_model.fit(X_tr,y_tr)
predictions = rf_model.predict(X_te)

#Print accuracy
accuracy = metrics.accuracy_score(predictions,y_te)
print ("Accuracy : %s" % "{0:.3%}".format(accuracy))
print("\n Classification report for classifier %s:\n%s\n"
      % (model, metrics.classification_report(y_te, predictions)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_te, predictions))

count = 0
print ("TOTAL NUMBER OF TESTING RECORD - ",X_te.shape[0])
for x in range(len(X_te)):
    if(y_te[x] == predictions[x]):
        count = count + 1
print ("NUMBER OF CORRECTLY PREDICTED OUTPUTS - ",count)
print("\n")
for x in range(0,5):
    if(y_te[x] == predictions[x]):
        print ("TESTING RECORD - ",X_te[x])
        print ("ACTUAL OUTPUT - ", y_te[x])
        print ("PREDICTED OUTPUT - ",predictions[x])
        print ("_________________________________")
 
    
    import seaborn as sns
sns.set(style="darkgrid")
plt.figure(figsize=(12,6))
ax = sns.countplot(x=predictions)
ax.set_xticklabels(["Loan Not Approved","Loan Approved"])
print ("FOLLOWING IS THE COUNTPLOT DISPLAYING THE PREDICTIONS - ")



from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(X_tr,y_tr)
predictions = clf.predict(X_te)

#Print accuracy
accuracy = metrics.accuracy_score(predictions,y_te)
print ("Accuracy : %s" % "{0:.3%}".format(accuracy))
print("\n Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(y_te, predictions)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_te, predictions))

count = 0
print ("TOTAL NUMBER OF TESTING RECORD - ",X_te.shape[0])
for x in range(len(X_te)):
    if(y_te[x] == predictions[x]):
        count = count + 1
print ("NUMBER OF CORRECTLY PREDICTED OUTPUTS - ",count)
print("\n")
for x in range(0,5):
    if(y_te[x] == predictions[x]):
        print ("TESTING RECORD - ",X_te[x])
        print ("ACTUAL OUTPUT - ", y_te[x])
        print ("PREDICTED OUTPUT - ",predictions[x])
        print ("_________________________________")




from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_tr,y_tr)
predictions = knn.predict(X_te)

#Print accuracy
accuracy = metrics.accuracy_score(predictions,y_te)
print ("Accuracy : %s" % "{0:.3%}".format(accuracy))
print("\n Classification report for classifier %s:\n%s\n"
      % (knn, metrics.classification_report(y_te, predictions)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_te, predictions))

count = 0
print ("TOTAL NUMBER OF TESTING RECORD - ",X_te.shape[0])
for x in range(len(X_te)):
    if(y_te[x] == predictions[x]):
        count = count + 1
print ("NUMBER OF CORRECTLY PREDICTED OUTPUTS - ",count)
print("\n")
for x in range(0,5):
    if(y_te[x] == predictions[x]):
        print ("TESTING RECORD - ",X_te[x])
        print ("ACTUAL OUTPUT - ", y_te[x])
        print ("PREDICTED OUTPUT - ",predictions[x])
        print ("_________________________________")
