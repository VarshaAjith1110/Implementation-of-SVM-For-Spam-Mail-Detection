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
5.End the program. 


## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Varsha Ajith
RegisterNumber:  212221230118
*/
```
import pandas as pd
data=pd.read_csv("spam.csv",encoding='Windows-1252')

import chardet
file='spam.csv'
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))
result

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

## Output:
![v1](https://user-images.githubusercontent.com/94222288/204584685-b35c749c-3039-458b-af85-eb846870fb08.png)
![v2](https://user-images.githubusercontent.com/94222288/204584825-d9b1a69b-d5a2-42de-a686-05f0f0141180.png)
![v3](https://user-images.githubusercontent.com/94222288/204584862-93968ed3-3ce0-4a8c-80fe-2a9fc573ad5b.png)
![v4](https://user-images.githubusercontent.com/94222288/204584920-b0ceee3c-50f7-4279-b8d1-558d2efdf8e3.png)
![v5](https://user-images.githubusercontent.com/94222288/204584969-26247c53-2a8e-4462-bf09-f330425255d0.png)
![v6](https://user-images.githubusercontent.com/94222288/204585087-9e524a16-2e2f-412a-92da-cad15306773d.png)
![v7](https://user-images.githubusercontent.com/94222288/204585137-cee630d8-fa2b-4afa-8c6b-34e4ec0194af.png)





## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
