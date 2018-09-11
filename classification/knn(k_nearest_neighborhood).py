#sınıflandrılmış veri kümesi elimize verildiğinde yeni bir değerin hangi kategoriye girdiğini bulmak için
#verilen değerin en yakın 3 komşusunu buluyoruz. Bu komşulardan hangi kategorideki komşu daha fazla ise yeni gelen değeri o değerdir diye sınıflandırıyoruz.
#burada kullanılan mesafe minkovski algoritmasıdır. yani x-y karesinin karekökü bildiğimiz mesefe kavramı.
#örneğin boyda 20 cm değişim ile kiloda 20 kilo değişim aynı değildir böyle durmlar için farklı mesafe özlçüm tipleri kullanılmakatdır.
#lazy learning yeni gelen datayı en yakın komşularını bulup sınıfdırma yapması
#eager learning elimizdeki datadan sınıflandırma bölgeleri çıkarıp yeni gelen datayı o bölgelerden hangisine denk gelirse o sınıfa yazıyor
#eager da önce öğrenip sonra sınıflıyor. lazyde ise yeni veri gelince elindeki örneklere bakıp hesaplama yapıyor.
#lazy learningde tüm veri kümesine bakmak gerekiyor her seferinde
#http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
#aşağıdaki örnekte görülebileceği gibi paremetreleri ayarlamak önemlidir. 5 komşuya bakarken daha az doğruluk payımız varken 1 komşuya baktığımızda doğruluğumuz arttı
#her örnekte böyle olacak değil ama paremetreleri değiştirerek daha başarılı sonuçlar elde edilebileceğinin örneğidir.

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

#KNN algoritması 5komşuya bakarak
from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier(n_neighbors=5,metric='minkowski')
knn.fit(X_train,y_train)

y_pred=knn.predict(X_test)
print(y_pred)

#The confusion matrix
from sklearn.metrics import confusion_matrix
con_met=confusion_matrix(y_test,y_pred)
print(con_met)

#KNN algoritması 1 komşuya bakarak
from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier(n_neighbors=1,metric='minkowski')
knn.fit(X_train,y_train)

y_pred=knn.predict(X_test)
print(y_pred)

#The confusion matrix
from sklearn.metrics import confusion_matrix
con_met=confusion_matrix(y_test,y_pred)
print(con_met)
