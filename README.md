# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
 1.Import the required packages.
 
2.Import the dataset to operate on.

3.Split the dataset.

4.Predict the required output.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: swathi.s
RegisterNumber:  212223040219
*/

```
import chardet
file='spam.csv'
with open (file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:

![image](https://github.com/user-attachments/assets/8d78863c-4066-47c9-a11e-58a579ea224e)

![image](https://github.com/user-attachments/assets/1eec7a56-c35e-4796-a0a0-d315654cef53)

![image](https://github.com/user-attachments/assets/42d9f734-c2ee-4558-a4fd-27830a338a24)

![image](https://github.com/user-attachments/assets/d4b9c406-4d4a-4230-85c9-c665df94a50d)

![image](https://github.com/user-attachments/assets/33e3d17d-d9cd-4524-815b-c94f46c4149b)

![image](https://github.com/user-attachments/assets/90857a17-1e4e-40a5-9476-52cb09b9f3eb)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
