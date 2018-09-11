#iki sınıfı birbirinden ayıracak çizgiyi bulmaya yarar.
#iki sınıfır birbirinden ayırabilecek sonsuz tane doğru bulunabilir. Bu doğrulardan hangisi en iyi sınıf ayrımını oluşturacaktır bunu bulmaya yarar.
#marjini arasındaki uzaklığı maksimize eden doğru olarak hesaplama yapıyor.
# 1 1 1 | | | 2 2  2 ortadaki ayrım olsun iki kümeyi arırabilecek doğrulardan biri olsun ve ya 1 1 1|  |  |2 2  2 burada bir ayrım vardır.
#2. ayrımda ortaki ayrım çizgisinin marjin aralığı daha geniştir. yani birini tercih edecek olursa 2. tercih ede algoritma
#ayıran doğru hem kümleri en hatasız ayıracak hemde marjin aralığı en geniş olan olacak
#sınıflandırmada marjin aralığı belirlerken ayrılacak olan iki sınıfdan herhangi bir eleman bu marjin aralığında olmaması gerekiyor.
#biz açtığımız gerçek vektorümüze destek olan support vectorlerden biri vaya ikisi sınıflandırma algoritmasının öğrenmesinde kullanılan verilerden biri olacaktır.
#bu destek vektörlerin dayandığı noktalarada destek noktası deniyor.
#tahminde bu marjin aralığına en fazla nokta dğşsün istiyorduk sınıflandırmada ise bu marjin aralığına hiç bir nokta düşmesin istiyoruz.
#bu ayrım yapılırken lineer - polymonial - rbf - exponential(üstsel) fonksiyonlar kullanılabilir.
#bu svm modelini kullanırken dataları scale etmemiz gerekiyor yani ölçeklendirmemiz gerekiyor.
#daha önceki örnekleri unutarak bir marjin doğrusu çıkıyor ve bu marjin doğrusuna göre yeni gelen örneği sınıflandırıyor.
#knn algoritmasında her noktanın çevresinde bir sınır öğrenerek çok detaylı bir algoritma öğrenebilir.
#svm de ise sınır ayrımının bir fonksiyon ile ifade edilebileceği kavramı üstünden ilerler. KNN kadar komplex bir algoritma çıkarmaz.
#çok kopmlex ayrım yapılması gerektiğinde bazı sorunları olabilir.
#bu durumun bir avantajıda fonksiyonu hesaplayıp kümleri ayırdığı için yeni değeri sınıflandırmak istidiğimizde daha hızlı işlem yapıyor ve verileri daha iyi saklıyor.
#yani işlem gücü yüksek.
#parelelleştirme algoritmalarında kullanılamıyor.(paralelleştirme elimizde mesela 1 milyon data var 100binlik parçalara bölüp 10 makinede çalıştırma.)
#http://scikit-learn.org/stable/modules/svm.html
#3 sınıfı ayrıştırırken her sınıfı kendi arasında değerlndirip ayırır sonrasında bunları birleştirir.yani 3ün 2li kombimasyonundan 3 farklı sınıflama yapıp birleştirirdi.

#kernel trick
#marjin aralığında öğrenme kümesindeki hiçbir noktayı kabul etmeyen svm algoritmalarına hard svm algoritmaları denir.
#softlarda ise minumum sayıda örneği marjin aralığına alabiliyor.
#doğrusal olarak ayrulamayan kümeler için svm kullanıcağımızda kernek trickleri kullanmamız gerekiyor.
#boyut artırma kullanılabilir. Yani iki boyutlu bir veriyi 3. boyuta yükseltebiliriz.
#doğrusal ayrılamayan örneklerde knn faydalı bir ayrıştırma yöntemidir.
#3. boyuta çıkarken bir çekirdek nokta belirlesek çekirdeğe uzak olan değerler aşağıda çekirdeğe yakın olan değerleri yukarı taşırsak bu 2 kümeyi birbinden ayırabiliriz..
#aldığım ekran görüntülerinde bununla ilgili örnek görülebilir.
#3. boyutta lineer yada bir farklı doğru ile ayrım yapılabilir.
#artık svm kullanılabilir.
#yani bir alt sınıfta doğru ile ayrılamayan bir kümeyi doğru ile ayırmayı ve tekrar alt boyuta indiğimizde bu doğru alt uzayda bir eğriye denk gelir.
#rbf için sigma değeri ne kadar küçükse grafik o kadar dar dik yukarı çıkar. mü(ortalam değer) değeride merkez noktayı farklı noktalara kaydırır.
#çoklu kernel da kullanılabilir.
#yani eğer bir kernel noktası belirleyerek verimizi ayrıştıramıyorsak daha fazla çekirdek kullanıp ayrıştırma yapabiliriz.



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

#svm
from sklearn.svm import SVC
svc=SVC(kernel='rbf')#kernel ayrım çizgisinin nasıl olacağını gösteriyor.
svc.fit(X_train,y_train)

y_pred= svc.predict(X_test)
print(y_pred)

#The confusion matrix
from sklearn.metrics import confusion_matrix
con_met=confusion_matrix(y_test,y_pred)
print(con_met)


