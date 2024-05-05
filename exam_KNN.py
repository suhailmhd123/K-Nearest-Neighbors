# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 11:18:01 2024

@author: H P
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

NB = pd.read_csv(r"C:\SUHAIL\python data\NB_Car_Ad.csv")
NB.head()
NB.columns
# ILOC
NB_1=NB.iloc[:,1:4]
# outlier
numerical_columns = NB_1.select_dtypes(include=[np.number]).columns.tolist()
for column in numerical_columns:
    plt.boxplot(NB_1[column])
    plt.title("Boxplot of "+column)
    plt.show()

# DUMMY
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
NB_1["Gender"]=labelencoder.fit_transform(NB_1["Gender"])

# MISSING VALUES
NB_1.isna().sum()

# NORMALIZATION
def norm_fuct(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)
NB_1=norm_fuct(NB_1)

NB_1.describe()

## KNN
x = np.array(NB_1)
y = np.array(NB["Purchased"])

from sklearn.model_selection import  train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2,random_state=42)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=17)
knn.fit(x_train,y_train)
pred = knn.predict(x_test)
pred
#accuracy_score
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,pred))
pd.crosstab(y_test, pred, rownames=["Actual"],colnames=["Predictions"])

## NAIVE BIASE
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train,y_train)
#prediction
predict = gnb.predict(x_test)
predict

#accuracy_score
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predict))
pd.crosstab(y_test, predict,rownames=["Actual"],colnames=["Preditions"])