#1. kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

#2.veri ön işleme

#2.1 veri yükleme
data = pd.read_csv('../data/data.csv')

#bağımlı ve bağımsız değişkenleri ayırma
x = data.iloc[:,1:4].values
y=data.iloc[:,-1:].values

#verilerin test ve train olarak bölünmesi
from sklearn.cross_validation import train_test_split
x_train ,x_test , y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=0)

#verilerin ölçeklendirilmesi
from sklearn.preprocessing import StandardScaler
st = StandardScaler()
X_train=st.fit_transform(x_train)
X_test=st.transform(x_test)

#Logistic regression
from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression(random_state=0)
log_reg.fit(X_train,y_train)

y_pred=log_reg.predict(X_test)

#The confusion matrix logistic regression
from sklearn.metrics import confusion_matrix
print('logistic regression - confusion matrix')
con_met=confusion_matrix(y_test,y_pred)
print(con_met)

#KNN algoritması 1 komşuya bakarak
from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier(n_neighbors=1,metric='minkowski')
knn.fit(X_train,y_train)

y_pred=knn.predict(X_test)

#The confusion matrix KNN
print('KNN - confusion matrix')
con_met=confusion_matrix(y_test,y_pred)
print(con_met)

#svm (svc)
from sklearn.svm import SVC
svc=SVC(kernel='rbf')#kernel ayrım çizgisinin nasıl olacağını gösteriyor.
svc.fit(X_train,y_train)

y_pred= svc.predict(X_test)

#The confusion matrix - svc
print('svc - confusion matrix')
con_met=confusion_matrix(y_test,y_pred)
print(con_met)

#Navie Bayes
from sklearn.naive_bayes import GaussianNB
gnb =  GaussianNB()
gnb.fit(X_train,y_train)

y_pred=gnb.predict(X_test)

#The confusion matrix - navie bayes
print('svc - confusion matrix')
con_met=confusion_matrix(y_test,y_pred)
print(con_met)

#decision tree
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(criterion='entropy')
dtc.fit(X_train,y_train)

y_pred=dtc.predict(X_test)

#The confusion matrix - desicion tree
print('decision tree - confusion matrix')
con_met=confusion_matrix(y_test,y_pred)
print(con_met)

#random forest
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=10,criterion='entropy')
rfc.fit(X_train,y_train)

y_pred=rfc.predict(X_test)

#The confusion matrix - random forest
print('random forest - confusion matrix')
con_met=confusion_matrix(y_test,y_pred)
print(con_met)

#roc tpr ve fpr değerleri
y_proba=rfc.predict_proba(X_test) #X_test için olasılık hesaplıyor.
print(y_proba[:,0])
from sklearn import metrics
fpr, tpr, thold = metrics.roc_curve(y_test,y_proba[:,0],pos_label='e')#pos label hangi değeri pozitif olarak seçtiğimiz
print(fpr)
print(tpr)
