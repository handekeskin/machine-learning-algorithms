#birden fazla karar ağacı algoritmasının aynı veri seti için kullanılması ve
#problemin çözümünde bu algoritmaların hepsinin beraber kullanılarak hesaplama yapılmasıdır.
#buna çoğunluğun oyuda(majority vote) deniyor.
#normal decision tree de tüm data setini verip algoritmayı eğitiyorsun. Bu yüzden aynı veri setini bir kaç kez decision tree algoritmasınada versek aynı sonucu üretiyor.
#fakat random forest da data setinden rastgele datalar seçilerek küçük veriseti kümeleri ayrılıyor ve her ayrılan parça için decision tree algoritması üretiliyor.
#böylece elimizde birden fazla decision tree algoritması oluyor ve yeni datayı tahmin ederken tüm decision treelerden gelen sonuçlar alınıyor
#bir değerin sonucuna bakılırken tüm decision treelerden gelen değerlerden en fazla değere sahip sınıf o değerin sınıflamasında kullanılır.

#1. kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

#random forest
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=10,criterion='entropy')
rfc.fit(X_train,y_train)

y_pred=rfc.predict(X_test)
print(y_pred)

#The confusion matrix
from sklearn.metrics import confusion_matrix
con_met=confusion_matrix(y_test,y_pred)
print(con_met)