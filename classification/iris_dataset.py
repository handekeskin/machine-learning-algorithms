#1. kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

#http://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
#iris kümesi görselleştirme kaynağı burdan görselleştirmesi görülebilir.
#http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
#bunuda çalışmak için algoritmayı çalıştırıp bakabiliriz.

#2.veri ön işleme

#2.1 veri yükleme
data = pd.read_excel('../data/iris.xls')

#bağımlı ve bağımsız değişkenleri ayırma
x = data.iloc[:,0:4].values
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

y_pred1=log_reg.predict(X_test)

#The confusion matrix logistic regression
from sklearn.metrics import confusion_matrix
print('logistic regression - confusion matrix')
con_met1=confusion_matrix(y_test,y_pred1)
print(con_met1)

#roc tpr ve fpr değerleri
y_proba1=log_reg.predict_proba(X_test) #X_test için olasılık hesaplıyor.
from sklearn import metrics
fpr1, tpr1, thold1 = metrics.roc_curve(y_test,y_proba1[:,0],pos_label='Iris-setosa')#pos label hangi değeri pozitif olarak seçtiğimiz
print('logistic regression - ftr and tpr')
print(fpr1)
print(tpr1)

#KNN algoritması 1 komşuya bakarak
from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier(n_neighbors=5,metric='minkowski')
knn.fit(X_train,y_train)

y_pred2=knn.predict(X_test)

#The confusion matrix KNN
print('KNN - confusion matrix')
con_met2=confusion_matrix(y_test,y_pred2)
print(con_met2)

#roc tpr ve fpr değerleri
y_proba2=log_reg.predict_proba(X_test) #X_test için olasılık hesaplıyor.
from sklearn import metrics
fpr2, tpr2, thold2 = metrics.roc_curve(y_test,y_proba2[:,0],pos_label='Iris-setosa')#pos label hangi değeri pozitif olarak seçtiğimiz
print('KNN - ftr and tpr')
print(fpr2)
print(tpr2)

#svm (svc)
from sklearn.svm import SVC
svc=SVC(kernel='linear')#kernel ayrım çizgisinin nasıl olacağını gösteriyor.
svc.fit(X_train,y_train)

y_pred3= svc.predict(X_test)

#The confusion matrix - svc
print('svc - confusion matrix')
con_met3=confusion_matrix(y_test,y_pred3)
print(con_met3)

#roc tpr ve fpr değerleri
y_proba3=log_reg.predict_proba(X_test) #X_test için olasılık hesaplıyor.
from sklearn import metrics
fpr3, tpr3, thold3 = metrics.roc_curve(y_test,y_proba3[:,0],pos_label='Iris-setosa')#pos label hangi değeri pozitif olarak seçtiğimiz
print('svm - ftr and tpr')
print(fpr3)
print(tpr3)

#Navie Bayes
from sklearn.naive_bayes import GaussianNB
gnb =  GaussianNB()
gnb.fit(X_train,y_train)

y_pred4=gnb.predict(X_test)

#The confusion matrix - navie bayes
print('navie bayes - confusion matrix')
con_met4=confusion_matrix(y_test,y_pred4)
print(con_met4)

#roc tpr ve fpr değerleri
y_proba4=log_reg.predict_proba(X_test) #X_test için olasılık hesaplıyor.
from sklearn import metrics
fpr4, tpr4, thold4 = metrics.roc_curve(y_test,y_proba4[:,0],pos_label='Iris-setosa')#pos label hangi değeri pozitif olarak seçtiğimiz
print('Navie Bayes - ftr and tpr')
print(fpr4)
print(tpr4)

#decision tree
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(criterion='entropy')
dtc.fit(X_train,y_train)

y_pred5=dtc.predict(X_test)

#The confusion matrix - desicion tree
print('decision tree - confusion matrix')
con_met5=confusion_matrix(y_test,y_pred5)
print(con_met5)

#roc tpr ve fpr değerleri
y_proba5=log_reg.predict_proba(X_test) #X_test için olasılık hesaplıyor.
from sklearn import metrics
fpr5, tpr5, thold5 = metrics.roc_curve(y_test,y_proba5[:,0],pos_label='Iris-setosa')#pos label hangi değeri pozitif olarak seçtiğimiz
print('decision-tree - ftr and tpr')
print(fpr5)
print(tpr5)

#random forest
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=10,criterion='entropy')
rfc.fit(X_train,y_train)

y_pred6=rfc.predict(X_test)

#The confusion matrix - random forest
print('random forest - confusion matrix')
con_met6=confusion_matrix(y_test,y_pred6)
print(con_met6)

#roc tpr ve fpr değerleri
y_proba6=rfc.predict_proba(X_test) #X_test için olasılık hesaplıyor.
from sklearn import metrics
fpr6, tpr6, thold6 = metrics.roc_curve(y_test,y_proba6[:,0],pos_label='Iris-setosa')#pos label hangi değeri pozitif olarak seçtiğimiz
print('random-forest - ftr and tpr')
print(fpr6)
print(tpr6)