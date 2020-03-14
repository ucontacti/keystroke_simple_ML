# In[1]: Header
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split #for split the data
from sklearn.metrics import accuracy_score  #for accuracy_score
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.metrics import make_scorer, roc_curve #for eer
from scipy.optimize import brentq #for eer
from scipy.interpolate import interp1d #for eer


# In[2]: Calculate eer rate
def calculate_eer(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer


# In[3]: Read data and split train and test data
in_df=pd.read_csv("final_dataset.csv")
in_df.loc[in_df['Label'] > 0, 'Label'] = 1
in_df.drop("Unnamed: 0", axis=1, inplace = True)
X = in_df.drop("Label",axis=1)
y = in_df["Label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[3]: Feature Selection
#apply SelectKBest class to extract top 10 best features
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

bestfeatures = SelectKBest(score_func=f_classif, k=10)
fit = bestfeatures.fit(X_train,y_train)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print('-------------- Univariate Selection Feature Selection----------------------------')
print(featureScores.nlargest(10,'Score'))  #print 10 best features



# In[4]: SVM classifier
from sklearn import svm
accuracy_list = []
err_list = []
for i in range(len(X_train.columns)):
    X_train_feature_less = X_train.drop([X_train.columns[i]], axis='columns')    
    X_test_feature_less = X_test.drop([X_test.columns[i]], axis='columns')    
    clf = svm.SVC(gamma='scale').fit(X_train_feature_less, y_train)
    prediction_rm=clf.predict(X_test_feature_less)
    accuracy_list.append(round(accuracy_score(prediction_rm,y_test)*100,2))
    err_list.append(round(calculate_eer(prediction_rm, y_test)*100,2))
dfscores = pd.DataFrame(accuracy_list)
dfcolumns = pd.DataFrame(X_train.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print('--------------Feature Selection by accuracy----------------------------')
print(featureScores.nsmallest(10,'Score'))  #print 10 best features

dfscores = pd.DataFrame(err_list)
dfcolumns = pd.DataFrame(X_train.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print('--------------Feature Selection by eer----------------------------')
print(featureScores.nlargest(10,'Score'))  #print 10 best features

# print(featureScores.nlargest(10,'Score'))  #print 10 best features
# print(accuracy_list)
# print(np.argsort(accuracy_list))
# print(err_list)
# print(np.argsort(err_list))
# print(accuracy_list.index(max(accuracy_list)))
# print(err_list.index(min(err_list)))
# print('--------------The Accuracy of the model----------------------------')
# print('The accuracy of the SVM Classifier is', round(accuracy_score(prediction_rm,y_test)*100,2))
# kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts
# result_rm=cross_val_score(clf, X, y, cv=10,scoring='accuracy')
# print('The cross validated score for SVM Classifier is:',round(result_rm.mean()*100,2))

import joblib
joblib.dump(clf, 'svm_model.pkl', compress=9)

# In[5]: KNN classifier
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(metric='manhattan').fit(X_train, y_train)
prediction_rm=clf.predict(X_test)
print('--------------The Accuracy of the model----------------------------')
print('The accuracy of the KNN Classifier is', round(accuracy_score(prediction_rm, y_test)*100,2))
print('--------------The Accuracy of the model----------------------------')
print('The EER value of the KNN Classifier is', round(calculate_eer(prediction_rm, y_test)*100,2))


# In[6]: One-Class SVM
from sklearn.svm import OneClassSVM
clf = OneClassSVM(kernel='rbf',gamma=26).fit(X_train, y_train)
prediction_rm=clf.predict(X_test)
print('--------------The Accuracy of the model----------------------------')
print('The accuracy of the One-Class SVM Classifier is', round(accuracy_score(prediction_rm, y_test)*100,2))
print('--------------The Accuracy of the model----------------------------')
print('The EER value of the One-Class SVM Classifier is', round(calculate_eer(prediction_rm, y_test)*100,2))
