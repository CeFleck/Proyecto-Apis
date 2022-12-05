import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import svm 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
from flask_restful import Resource


data=pd.read_csv("train.csv")
data=data.drop(columns=['Cabin'],axis=1)
data['Initial']=0
for i in data:
    data['Initial']=data.Name.str.extract('([A-Za-z]+)\.') 
data['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)
data.loc[(data.Age.isnull())&(data.Initial=='Mr'),'Age']=33
data.loc[(data.Age.isnull())&(data.Initial=='Mrs'),'Age']=36
data.loc[(data.Age.isnull())&(data.Initial=='Master'),'Age']=5
data.loc[(data.Age.isnull())&(data.Initial=='Miss'),'Age']=22
data.loc[(data.Age.isnull())&(data.Initial=='Other'),'Age']=46
data['Embarked'].fillna('S',inplace=True)
data['Sex'].replace(['male','female'],[0,1],inplace=True)
data['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
data['Initial'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)
data.drop(['Name','Age','Ticket','Fare','PassengerId'],axis=1,inplace=True)
train,test=train_test_split(data,test_size=0.3,random_state=0,stratify=data['Survived'])
train_X=train[train.columns[1:]]
train_Y=train[train.columns[:1]]
test_X=test[test.columns[1:]]
test_Y=test[test.columns[:1]]
X=data[data.columns[1:]]
Y=data['Survived']

class graficas(Resource):
    def get(self):
        return "El mejor seleccionado es RBF con una prediccion de 0.8208955223880597"+data.to_json()
class lin(Resource):
    def get(self):
        model=svm.SVC(kernel='linear',C=0.1,gamma=0.1)
        model.fit(train_X,train_Y)
        prediction=model.predict(test_X)
        respu=metrics.accuracy_score(prediction,test_Y)
        return'Segun el entrenamiento con regresion lineal tenemos un score de '+ str(respu)

class ran(Resource):
    def get(self):
        model=RandomForestClassifier(n_estimators=100)
        model.fit(train_X,train_Y)
        prediction=model.predict(test_X)
        respu=metrics.accuracy_score(prediction,test_Y)
        return 'Segun el entrenamiento con Random Forest tenemos un score de '+ str(respu)



class rbf(Resource):
    def get(self):
        model=svm.SVC(kernel='rbf',C=1,gamma=0.1)
        model.fit(train_X,train_Y)
        prediction=model.predict(test_X)
        metrics.accuracy_score(prediction,test_Y)
        return 'Segun el entrenamiento con Radial Basis Function tenemos un scor de '+ str(respu)

