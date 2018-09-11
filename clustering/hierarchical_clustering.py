#hierarchical clustering
#agglomerative:bottom-up approximation
#her veri bir küme olarak başlar
#en yakın iki komşuyu alıp ikili küme oluşturulur. Bu birleştirme veri kümemizde n eleman n-1 ile n/2 arasında bir sayıya azalıyor. Yani sadece yakın noktalar varsa birleştirme yapılır.
#en yakın iki kümeyi alıp yeni bölüt oluşturulur
#bir önceki adım,tek küme kalana kadar devam eder.
#divisive:top-down approximation
#agglomerative farklı olarak burada önce herkes tüm uzay bir küme oluyor . Sonra işlemler yapılıp her veri bir cluster olana kadar kümeleme devam ediyor
#burada en büyük sorunlardan biri aradaki mesefe nasıl olçülmesi gerektiğine karar vermek lazım.
#hem iki nokta arasındaki mesafe nasıl ölçülecek hemde clusterlar arası mesefe nasıl ölçülecek.
#clusterlarda birden fazla eleman olduğundan 2 cluster arası mesefa ölçme sorunu çıkabiliyor.
#mesefa ölçümü?
#1 metrik problemi:uzaklık hangi yöntemle olçüleceği
#öklit mesafesi:öklüd bir çözüm ama başka uzaklık ölçüm yöntemleride mevcut
#2 referanslar:cluster arası mesafeye bakılma çözümler
#en yakın noktalar: iki clusterin birbirine en yakın noktaları arasındaki mesafeye bakma
#en uzak noktalar: iki clusterin birbirine en uzak noktaları arasındaki mesafeye bakma
#ortalama:bütün elemanların ortalaması alınıp bu ortalama noktalar arası mesafe bakma
#merkezler arası mesafe:clusterin merkezi hesaplanır ve merkezler arası mesafeye bakılır
#k sayısına göre hangi aşamada duracağımızı belirleyebiliriz. İlgili k sayısında kümeleme yapıldıysa sorunsuz hesaplama yapar

#dendogram kavramı
#dendogram grafiği ekran görüntüsünü ekledim.
#dendogramda y ekseninde mesafeler x ekseninde ise noktalar yer alır. x ekseninde ise ilk başta en yakın olan 2 nokta ver alır.
#sonrasında uzaklık mesafelerine göre noktalar yer alır ve bu noktalar arasındaki mesefeler yer alır.
#bu ölçümü bilgisayar yaparken bir mesefe matrisi çıkarıyor. Her noktanın diğer nokta ile arasındaki uzaklığı ölçüyor.
#bu msefeleri hesapladıktan sonra en yakınları birbiriyle ilişkilendirilir ve dendogram yapısı oluşur. Bir birleştirme işlmei yapıldığında mesafee matrisi yeni birleşen kümelre göre tekrar düzenlenir.
#clusterlar arsındaki mesafe ölçme yöntemi kümelemeleri değiştirir. Nasıl değiştirdiğine dair göresel örneği ekran görüntülerinde görebilirsin.
#yukarda referanslarda clusterlar arasında verilen mesafe ölçümlere ek olarak ward's method da veriler bilir.
#Wards mesefesi=wcss1+wcss2+....
#wards mesafesi her clusterin wcss değeri hesaplanarak bulunur. wcss (clusterın her bir noktasının clusterın merkezine uzaklığının kareleri toplamı)
#yani cluster1'in wcss değerini hesapla cluster2'in wcss değerini hesapla sonra bu iki küme birleştiğinde wcss değeri hesapla bu 3 değeri topla oluşan değer 2 küme arasındaki uzaklık hesafesini verir.

#burada da küme/bölüt sayısı kullanıcıdan alınır.
#dengodramda yerleştirdiğimiz mesafelerde kullanıcı kaç cluster isterse o kadar bölme yapılabilir
#en optimum k değeri dendogramdaki hiyeraşi yapısında 2 clusteri birleştirdiğinde en uzun mefasedeki aradan kesmektir.
#veri kümesi büyüdükçe hiyerarşik bölütleme k-means den daha iyi çalışmaz.
#büyük veri kümesi için uygun dağildir.

#kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#2.1 veri yükleme
data = pd.read_csv('../data/customer.csv')
X=data.iloc[:,3:].values

#hierarchical clustering
from sklearn.cluster import AgglomerativeClustering
agc=AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage='ward')
y_prediction=agc.fit_predict(X)
print(y_prediction)
#n_clusters kaç küme olacak
#affinity mesafe ne ölçüsü ile alınacak
#linkage clusterlar arası mesafe nasıl ölçülecek
#ward kullanacaksak sadece euclidean ölçü birimi kullanılmak zorunda
#fit inşa ediyor fit_predict hem inşa et hemde tahmin et

#yapılan kümeleme işleminin grafiğini çizmek
plt.scatter(X[y_prediction==0,0],X[y_prediction==0,1],s=100,c='red')
plt.scatter(X[y_prediction==1,0],X[y_prediction==1,1],s=100,c='blue')
plt.scatter(X[y_prediction==2,0],X[y_prediction==2,1],s=100,c='green')
plt.scatter(X[y_prediction==3,0],X[y_prediction==3,1],s=100,c='yellow')
plt.title('Agglomerative Clustering')
plt.show()

#SciPy kütüphanesinde farklı görselleştirme yöntemleri bulunuyor. Burada da sadece göreselleştirme değil farklı yöntemler mevcut
#dendogram
import scipy.cluster.hierarchy as sch
dendogram=sch.dendrogram(sch.linkage(X,method='ward'))
plt.show()
#grafikten görülebileceği gibi 2 ve 4 kümeleme için mantıklı değerler