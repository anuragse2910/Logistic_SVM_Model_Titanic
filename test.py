import pickle
import pandas as pd
import numpy as np


test_data = pd.read_csv('D:/Logistic_SVM_Model_Titanic/titanic_test.csv')
test_data

X = test_data.drop(columns=['PassengerId','Cabin','Name','Ticket'],axis=1)

X.isnull().sum()

# filling missing values with mean value
X['Age'].fillna(X['Age'].mean(), inplace= True)
# finding the most appered values in embarked column and fill with same for that we using mode
X['Embarked'].fillna('S', inplace=True)

X['Fare'].fillna(X['Fare'].mean(), inplace= True)


X.isnull().sum()

# changeing the data type of few columns
X['Pclass'] = X['Pclass'].astype('category')
X['Sex'] = X['Sex'].astype('category')
X['Age'] = X['Age'].astype('int')
X['Embarked'] = X['Embarked'].astype('category')

# One Hot Encoding
X = pd.get_dummies(data= X, columns=['Pclass','Sex','Embarked'],drop_first=True)
X.shape

clf = pickle.load(open('logistic_model.pkl','rb'))
clf.predict(X)

svm = pickle.load(open('svm_model.pkl','rb'))
svm.predict(X)
