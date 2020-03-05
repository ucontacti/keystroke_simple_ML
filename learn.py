import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split #for split the data
from sklearn.metrics import accuracy_score  #for accuracy_score
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation


in_df=pd.read_csv("final_dataset.csv")
in_df.loc[in_df['Label'] > 0, 'Label'] = 1
in_df.drop("Unnamed: 0", axis=1, inplace = True)
# in_df.to_csv("final_dataset_1.csv")
X = in_df.drop("Label",axis=1)
y = in_df["Label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print(X_train.shape,X_test.shape,y_train.shape,y_test.shape,in_df.shape)

from sklearn import svm
clf = svm.SVC(gamma='scale').fit(X_train, y_train)
prediction_rm=clf.predict(X_test)
print('--------------The Accuracy of the model----------------------------')
print('The accuracy of the SVM Classifier is', round(accuracy_score(prediction_rm,y_test)*100,2))
kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts
result_rm=cross_val_score(clf, X, y, cv=10,scoring='accuracy')
print('The cross validated score for SVM Classifier is:',round(result_rm.mean()*100,2))
