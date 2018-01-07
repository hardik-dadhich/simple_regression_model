# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 23:45:40 2018

@author: DeLL
"""

#Simple linear regressison
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

#split the dataset into trainning and test set!
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

#feature scalling alredy done!

#fitting the simple regression to trauing set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#prediction of test set results
y_pred = regressor.predict(X_test)

#visulaize the trainning set data
plt.scatter(X_train, Y_train, color = 'Red')
plt.plot(X_train, regressor.predict(X_train), color = 'Blue')
plt.title('Salary vs Experience(Traning set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

