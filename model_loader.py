import joblib
model_clone = joblib.load('svm_model.pkl')

import pandas as pd
from sklearn.model_selection import train_test_split #for split the data
from sklearn.metrics import accuracy_score  #for accuracy_score

# In[3]: Read data and split train and test data
in_df=pd.read_csv("final_dataset.csv")
in_df.loc[in_df['Label'] > 0, 'Label'] = 1
in_df.drop("Unnamed: 0", axis=1, inplace = True)
X = in_df.drop("Label",axis=1)
y = in_df["Label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

prediction_rm=model_clone.predict(X_test)
print('--------------The Accuracy of the model----------------------------')
print('The accuracy of the SVM Classifier is', round(accuracy_score(prediction_rm,y_test)*100,2))
