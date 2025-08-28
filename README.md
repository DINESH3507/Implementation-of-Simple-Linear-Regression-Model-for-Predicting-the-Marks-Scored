# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

Name:Dinesh.V

Reg no:212224040076

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()



df.tail()


x=df.iloc[:,:-1].values

y=df.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

y_pred


y_test





plt.scatter(x_train,y_train,color="black")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()





plt.scatter(x_test,y_test,color="blue")
plt.plot(x_test,regressor.predict(x_test),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()




mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)

```

## Output:
HEAD

<img width="237" height="133" alt="Screenshot from 2025-08-28 14-54-55" src="https://github.com/user-attachments/assets/de4cd4ed-43be-462a-8114-e2b6a71e8a5c" />

TAIL

<img width="237" height="133" alt="Screenshot from 2025-08-28 14-55-05" src="https://github.com/user-attachments/assets/cd5881f1-0287-4bb3-a40a-5f95bc92399f" />

X VALUE

<img width="256" height="461" alt="Screenshot from 2025-08-28 14-55-34" src="https://github.com/user-attachments/assets/409fc0ad-9f06-4153-89be-fe7b45cbb9a4" />

Y VALUE

<img width="661" height="49" alt="Screenshot from 2025-08-28 20-52-06" src="https://github.com/user-attachments/assets/1fcb4809-b057-43a4-b8ac-10144e8e9c17" />

PRED

<img width="661" height="49" alt="Screenshot from 2025-08-28 20-52-18" src="https://github.com/user-attachments/assets/c4e97425-64dd-49c7-a13d-b2b76264dc8f" />

TEST

<img width="661" height="49" alt="Screenshot from 2025-08-28 20-52-38" src="https://github.com/user-attachments/assets/0cebe958-c4a3-4107-87a1-7507d4b79867" />

TRAINING

<img width="594" height="469" alt="Screenshot from 2025-08-28 20-53-44" src="https://github.com/user-attachments/assets/9f6c8aa8-d235-4556-a41f-a01a8adab6f8" />

TESTING

<img width="594" height="469" alt="Screenshot from 2025-08-28 20-54-01" src="https://github.com/user-attachments/assets/a7f1dd9d-aad9-4784-8782-98c63000be7f" />

MSE MAE RMSE

<img width="286" height="91" alt="Screenshot from 2025-08-28 20-54-19" src="https://github.com/user-attachments/assets/82113b92-3884-44fb-a04f-78d6e3e989e0" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
