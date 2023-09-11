# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
 
1. Import the required libraries and read the dataframe.  
3. Assign hours to X and scores to Y. 
4.Implement training set and test set of the dataframe
5.Plot the required graph both for test data and training data.
6.Find the values of MSE , MAE and RMSE.


## Program:

```
Program to implement the simple linear regression model for predicting the marks scored.

Developed by: T K Sharangini
RegisterNumber:  212222230143
```
# implement a simple regression model for predicting the marks scored by the students

import pandas as pd
import numpy as np
dataset=pd.read_csv('/content/student_scores.csv')
print(dataset)

# assigning hours to X & Scores to Y
X=dataset.iloc[:,:1].values
Y=dataset.iloc[:,1].values
print(X)
print(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)

Y_pred=reg.predict(X_test)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error

plt.scatter(X_train,Y_train,color='green')
plt.plot(X_train,reg.predict(X_train),color='red')
plt.title('Training set (H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show

plt.scatter(X_test,Y_test,color='purple')
plt.plot(X_test,reg.predict(X_test),color='blue')
plt.title('Test set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)



## Output:
![simple linear regression model for predicting the marks scored](sam.png)

![230738524-8e33f10f-4a92-4ced-94b0-c9c4118e61ae](https://github.com/hariprasath5106/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/111515488/d2ed3e48-aecf-49de-937b-3a96eacae192)

![230738547-ad67ab98-e30c-469c-973f-cca617f6c4a3](https://github.com/hariprasath5106/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/111515488/9b0c70ab-8d11-4c68-941b-a5aab83bc637)

![230738567-0469492e-6ceb-4c10-8304-faea06237701](https://github.com/hariprasath5106/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/111515488/7650dd3e-5342-4cf5-a05c-f0216abb8742)

![230738581-db53bfa1-1ac3-4f8c-aff0-4ee0a4668cb7](https://github.com/hariprasath5106/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/111515488/f57b2cb4-d86f-43b2-89cf-a5aeea487171)

![230738623-47ff43ca-da1f-4588-8e70-0e2f49396f91](https://github.com/hariprasath5106/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/111515488/0e888d23-0e6b-4c21-a287-518eec2744d9)

![230738637-83fd8a0b-34bc-4ddf-bf91-d76b5f248a91](https://github.com/hariprasath5106/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/111515488/9d6e26bb-3849-4e71-ba9e-8521b8de3903)

![230738645-55522693-34bc-4a9e-b6d5-c8a0b78259bd](https://github.com/hariprasath5106/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/111515488/dee5c457-34d7-40f1-afb5-cd5b1684303f)

![230738645-55522693-34bc-4a9e-b6d5-c8a0b78259bd](https://github.com/hariprasath5106/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/111515488/67d85371-ae1a-4306-b433-82b8e314b72f)

![230738707-f763a26a-b18e-445d-98c8-027a37819a21](https://github.com/hariprasath5106/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/111515488/c847054f-9169-4ed2-bc0d-5970c04f9b21)

![230738713-14729867-3192-48df-a0e3-38a4afe0fffb](https://github.com/hariprasath5106/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/111515488/fc17f469-e9d3-4bb1-b728-2bdaa923b656)
### result :

Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
