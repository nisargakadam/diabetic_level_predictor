#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 21:21:26 2023

@author: nisar
"""


import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge


tr = pd.read_csv("/Users/nisar/Desktop/psu/SPRING/DS 310 /diabetic-level-prediction (1)/train.csv")
x_test = pd.read_csv("/Users/nisar/Desktop/psu/SPRING/DS 310 /diabetic-level-prediction (1)/x_test.csv")

## create X or y

X = tr.drop(['y'],axis=1)
y = tr['y']




# split data into training test
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# create ridge model with alpha value .5 
ri = Ridge(alpha=.102)

# fit it to the training data
ri.fit(X, y)

# predict y values for test set using model
y_hat_ri = ri.predict(x_test)
print(y_hat_ri)


# create df and export as csv

#m = pd.DataFrame({'id': x_test['id'], 'y': y_hat_ri})
#print(m)
#m.to_csv("yri1.csv", index=False)
