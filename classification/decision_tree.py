#uzayı bölgelere ayıyoruz ve ayrımlardan bir karar ağacı oluşuruz.
#burada datayı karar ağacı kullanarak belli bölgelere ayırıyoruz. Bu ayırma işlemi entropy'ye göre yapılıyor.
#tüm bölgede aynı sınıftan değerler varsa o sonuca ilgili sınıf yazılır fakat bazen bölgeyi böldüğümüzde
#başka sınıflardan da değerler gelebilir birinci yöntemde
#hangi sınıftan örnek daha yoğunlukta varsa karar ağacının sonucuna o sınıfı yazarız. yani çoğunluğun dediği olur
#ikinci yöntemde ise aynı sınfıtan adamlar kalana kadar uzay bölünmeye devam eder.
#iki yönteminde kendine göre faydasız olduğu yerler vardır.
#birinci yöntemde detayları kaçırma olasılığımız yüksektir.
#çok bölersek overfiting yani ezber riski var. her bir bölgede tek bir örnek kalana kadar bölüm yapılabilir buda ezberlemeye gidiyor.
#bu bölme işlemi entropy'e göre karar veriliyor. Entropy formülü ve anlatımı ekran görüntüleride mevcut
#Class P : bilgisayar alanlar
#Class N : bilgisayar almayanlar
#info(D)= I(bilgisayar alan sayısı, bilgisayar almayan sayısı)= - (bilgisayar alma olasılığı)* (log 2 tabanında)(bilgisayar alma olasılığı)  - (bilgisayar almama olasılığı)* (log 2 tabanında)(bilgisayar almama olasılığı)
#sonrasında köke koyulacak olan değerin kümeyi ne kadar iy dağıttığına bakmak istiyoruz yani bize sağlayacağı enformasyon değeri ne olur sonusuna yanıt bulmak istiyoruz.
#info(köke koyulacak kolon(bu örnekte age)) (D)= kökte yer alan kolonun değerlerin birinin olasılığı * I(olma olasılığı, olmama olasılığı) +kökte yer alan kolonun başka bir değerin olasılığı * I(olma olasılığı, olmama olasılığı) +...
#bu örnekte infoage(D) = 30 yaşından küçük olma olasılığı * I(30 yaşından küçük olanların bilgisayar alma sayısı,30 yaşından küçük olanların bilgisayar almama sayısı) + 30 -40 yaş arası olma olasılığı * I(30 -40 yaş arası olanların bilgisayar alma sayısı,30 -40 yaş arası  olanların bilgisayar almama sayısı)+...
#bir satır yukarıda yer alan  I(x,y) değeride ilk başka yazdığımız info(D) formülü gibi hesaplacak.
#gain(age)=info(D)-infoage(D)
#ageden elde ettiğimiz kazancı ilk elde ettiğimiz değerden ageden elde ettiğimiz değeri çıkararak elde ederiz.
#tüm değişkenlerin kazancı hesaplandıktan sonra bölgelerin nerelerden bölüneceğine karar veriyoruz.
#her ayrım için tekrar aynı yöntemle alt kırılımların bölünmesi için hesaplama yapılır
#en sonunda tüm küme sınıflara ayrılana yada artık soracak soru kalmayana kadar bu işlem devam eder.
#datalar pure kalmadığı ve sorulacak soruda olmadığı zaman çoğunluğun dediği karar sınıfın sonucu olur.
#criterion gini ve entropy gelebilir. ginide info(D) hesaplamında entropy log 2 tabanında hesaplasıyla alırken ginide olasılığı log 2 tabına almadan hesaplama yapar. yani olasılığı karesidir.

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

#decision tree
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(criterion='entropy')
dtc.fit(X_train,y_train)

y_pred=dtc.predict(X_test)
print(y_pred)

#The confusion matrix
from sklearn.metrics import confusion_matrix
con_met=confusion_matrix(y_test,y_pred)
print(con_met)