# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 21:33:35 2022

@author: bagar
"""

import pandas as pd

#read csv file
dataset = pd.read_csv("02Students.csv")
df = dataset.copy()

#split the data vertically into the dependent and independent variables
x_dt = df.iloc[:, :-1]
y_dt = df.iloc[:, -1]

#split the data horizontally into train and test datasets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = \
    train_test_split(x_dt, y_dt, test_size=0.3, random_state=1234)
    
#create and train the simple/multiple linear regression
from sklearn.linear_model import LinearRegression
mlReg = LinearRegression()

#train the model
mlReg.fit(x_train, y_train)

#predict from the model
y_predict = mlReg.predict(x_test)

#calculate the r-squared and equation of the line
mlr_score = mlReg.score(x_test, y_test)

#coefficient of the line
mlr_coeff = mlReg.coef_
mlr_intercept = mlReg.intercept_

#root mean squared error
from sklearn.metrics import mean_squared_error
import math
mlr_rmse = math.sqrt(mean_squared_error(y_test, y_predict))




