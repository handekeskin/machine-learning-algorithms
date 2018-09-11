#koşullu olasılık
#P(A|B) burada B nin olması durumunda A nın olması olasığı
#P(A|B)=P(A kesişim B) / P(B)  a ve b nin kesişim olasılığının b nin olma olasılığına bölümü yani koşullu olasılık formülü
#bayes teoremi
#P(Y|X)=( P(X|Y) * P(Y))/ P(X)
#bu formul Xin gerçekleştiğinde Y nin olma olasılığı ile Ynin gerçekleştiğinde Xin gerçekleşme olasılığını veriyor.
#dengesiz veri kümelerinde çalışmayı sağlayabiliyor. Yani mesela bir problemde bilgisayar alan ve almayan kişiler var elimizde ve
#bilgisayar alanlar almayanlardan çok daha fazla yani alanlar ile ilgili daha fazla veri bilgimiz var.
#bu tip sonuclarda bir sonuç daha fazla ise navie bayes bu durumu kendi içinde düzenleyip sonuç üretebiliyor.
#ekran görüntülerini aldığım örneği biraz anlatmak gerekirse
#öncelikle bilgisayar alanların ve almayanların olasılığı hesaplanır.
#sonrasında bize 30 yaş altında | gelir düzeyi orta seviyede | öğrenci | kredi skoru ortalama olan birinin bilgisayar alma olasığını bulmak istiyorsak
#her bir koşul için bu koşullarda bilgisayar alan ve almayanların olasılıpı bulmamız gerekir.
#mesela P(30 yaş altında | bilgisayar alanlar), P(30 yaş altında | bilgisayar almayanlar),P(gelir düzeyi ortala | bilgisayar alanlar) ...
# P(30 yaş altında | bilgisayar alanlar) açıklaması bilgisayar alanların kaçı 30 un altında olasılığı
# tüm değerler için ayrı olasılık hesaplaması yapılmalıdır.
#sonrasında her biri için hesaplama yaptıktan sonra bilgisayar alma koşuluna bağlı ihtimalleri birbiri ile çarpılır.
#aynı şekilde bilgisayar almayan koşuluna bağlı ihtimallerde birbiri ile çarpılır.
#X = (30 yaş altında , gelir düzeyi orta seviyede , öğrenci , kredi skoru ortalama )
#C1 = bilgisayar alan
#C2 = bilgisayar almayan
#P(X|C1) = P(30 yaş altında | bilgisayar alanlar)* P(gelir düzeyi orta seviyede | bilgisayar alanlar)*...
#P(X|C2) = P(30 yaş altında | bilgisayar almayanlar)* P(gelir düzeyi orta seviyede | bilgisayar almayanlar)*...
#sonra çıkan sonuçlar (P(X|C1),P(X|C2)) normalize edilir. Bunun nedeni bilgisayar alan kişi sayısı ile bilgisayar almayan kişi sayısı birbiyle eşit değildi.
#normalizasyon işlemi aşağıdaki gibi yapılıyor.
#P(X|C1)*P(C1) (yani bilgisayar alan ve bizim koşulumuzu sağlama olasılığı ile bilgisayarın alınma olasılığını çarpıyoruz.)
#P(X|C2)*P(C2) (bilgisayar alanlar ile aynı mantık bilgisayar almayanlara yapıyoruz.)
#daha sonra normalize edilen değerler üstünden elde ettiğimiz sonuçları karşılaştırarak bilgisayar alır ve ya almaz diye yorum yapabiliyoruz.
#sınıflandırma yaparken olasılık kullanıyor.
#knn uzaklık hesabı yaparak ayrım yapıyor ama navie bayes tüm kümenin olasılıklarını hesaplayarak ilerliyor.
#burada yapılan lazy learning yani sistem sonradan öğreniyor. Biz dışardan veri girişi yapana kadar bir öğrenme olmuyor.
#yada eager olarak hesaplama dahil olabilecek olan tüm kombinasyon değerlerinin olasılığı bir yerde hesaplayabilir sonrasında ise veri kümesini unutarak yeni gelen değeri hespalayabilir
#çok büyük ve çok karmaşık kümlerde eager navie bayes yöntemi maliyetli oluyor.

#http://scikit-learn.org/stable/modules/naive_bayes.html
#eğer tahmin edeceğimiz veri continious(sürekli) bir değerse Gaussian Navie bayes kullanılır.
#eğer tahmin verimiz nominal yani sürekli olmayan bir değişkense multinominalNB kullanılır.
#şayet değişkenimiz 0-1 gibi nominal değer alan ama sadece 2 değer alan bir değişkense BernoulliNB kullanılır
#gaussianNB değerleri tüm değerler için kullanılabildiği için multinominal ve bernoulli nb yöntemlerini aslında kapsar.


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

#Navie Bayes
from sklearn.naive_bayes import GaussianNB
gnb =  GaussianNB()
gnb.fit(X_train,y_train)

y_pred=gnb.predict(X_test)
print(y_pred)

#The confusion matrix
from sklearn.metrics import confusion_matrix
con_met=confusion_matrix(y_test,y_pred)
print(con_met)


