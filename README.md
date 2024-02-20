# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Hema Dharshini N
RegisterNumber: 212223220034 
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
![Screenshot 2024-02-20 060703](https://github.com/hema-dharshini5/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147117728/c14095f6-e95e-437e-9f97-ba450ffecb01)
![Screenshot 2024-02-20 060721](https://github.com/hema-dharshini5/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147117728/e2772efa-8be1-4ce3-b2ea-91cc24bd4ec7)
![Screenshot 2024-02-20 060752](https://github.com/hema-dharshini5/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147117728/e4443048-b0a2-4a39-900e-171e02e8d8f5)
![Screenshot 2024-02-20 060759](https://github.com/hema-dharshini5/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147117728/9a5bbf17-ff68-4bea-9731-6bb7320f5a2e)





## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
