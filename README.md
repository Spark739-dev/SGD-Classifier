# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import all the necessary python libraries to perform the given SGDClassifier program.
2. Use the Iris datasets from sklearn.datasets for this program.
3. Take x and y input values from the iris dataset.
4. Use y_pred to store predicted values.
5. Calculate the accuracy score for y_test and y_pred
6. Create the heatmap with attributes for confusion matrix with matplotlib.pyplot attributes.
7. Show the heapmap for confusion matrix.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: VESHWANTH.
RegisterNumber: 212224230300
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
iris=load_iris()
df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
df['target']=iris.target
print(df.head())
x=df.drop('target',axis=1)
y=df['target']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
sgf_clf=SGDClassifier(max_iter=1000,tol=1e-3)
sgf_clf.fit(x_train,y_train)
y_pred=sgf_clf.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
print(f"accuracy score is :{accuracy:.3f}")
cm=confusion_matrix(y_test,y_pred)
print("confusion matrix")
print(cm)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt='d', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
*/
```

## Output:
![1](https://github.com/user-attachments/assets/e873e793-dd12-4944-9200-e4eb35ed5c4b)
![2](https://github.com/user-attachments/assets/0fc52a2a-8eed-4ff7-81f4-7732545a563f)
![3](https://github.com/user-attachments/assets/e2491cde-172e-4044-852d-9bbcd79d4e23)
![4](https://github.com/user-attachments/assets/f6be55df-5695-4d68-b118-3d1b29212288)



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
