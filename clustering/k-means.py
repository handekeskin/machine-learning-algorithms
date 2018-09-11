#nasıl çalışır
#kaç segment olacağı kullanıcıdan paremetse alınarak belirlenir.(bu bir dezevantaj) (kaç segment olacağına sen karar vermelisin)
#rastgale k merkez noktası seçilir
#her veri kümesien yakın merkez noktaya göre ilgili kümeye atanır
#her küme için yeni merkez noktaları seçilerek merkez kaydırılır

#diyelim mi 2 segment olacak. Biz 2 tane noktayı rast gelen uzayda yerleştirdik.
#bu iki nokta arsında bir doğru çizelim. Daha bu doğruya dik başka bir doğru çizdiğimizde bu noktanın 2 tarafında kalan noktalar segmente ayrılmış olur.
#dha sonra merkez noktası kaydırılır.
#yine aynı işlemle kümeler ayrılır.
#data merkezi satabil olana kadar kaydırma işlemi yapılıyor.
#bu noktalar stabil hala gelice küme sınırları belirlenmiş oluyor

#k means başlangıç tuzağı
#k-means algoritmasının bazı dezavantajları var
#merkez rastgele seçildiğinde aslında daha iyi bir kümeleme yapabilecekken daha kötü bir ayrım yapabilir.
#rastgele merkez seçince k means kümelemeleri hatalı yapma şansı büyük
#bunun çzöümlerinden biri her noktanın teker teker merkeez noktası seçilmesi olabilir. ama k^n tane işlem yapamak gerekir.
#bunun kolay yolu için k means ++ algoritması hbulunmuş.
#k means ++ da rastgele seçilen noktaların en yakındaki her nokta ile arasındaki mesafeye bakılır.bununa D(x) dersek
#bir sonraki merkez D(X)^2 olasılığına bakarak yer değişitirir
#bu k means için iyileştirme yapmak için kullnılabilir.

#k means algoritmeasında küme sayısına karar verme
#within cluster sum of squares(wcss) yöntemi ile en iyi k değerini bulmak için kullanılıyor.
#her k değeri için tek tek bu değer hesaplanıp karşılatırılıyor.
#hesaplama yöntemi ilgili clusterrin merkezinin o clusterda yer alan noktalar uzaklığının kareleri alınır ve her cluster için
#hesaplama yapıldıktan sonra bu her cluster için için hespalanan değer toplanır bu toplma değere wcss değeri denir.
#bu değerleri alıp grafik çzidiğimizde her cluster artığında wcss değeri düşer ama biz
#ciddi bir kırılım noktası bulduğumuzda bu değeri k değeri olarak mantıklı olacaktır.
#eğimin ciddi olarak değiştiği nokta bizim için önemlidir.bununla ilgili bir ekran görüntüsü ekledim. Burada 2 veya 3 dirsek noktaları. 2 ve ya 3 ü tercih etmek mantıklı
#zaten kaç gruba ayırabileceğimizi biliyorsak bu değeri hesaplamaya gerek olmaz

#http://scikit-learn.org/stable/modules/clustering.html--genel clustering algoritmalarındaki farklılıkları gösteriyor.
#yukardaki dokümandan hangi algoritmayı kullanacağımıza bakılabilir.

#kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#2.1 veri yükleme
data = pd.read_csv('../data/customer.csv')
X=data.iloc[:,3:].values

#k-means
#n cluster kaç parçaya bölmek gerek
#init merkez seçme yöntemi
from sklearn.cluster import KMeans
km=KMeans(n_clusters=4,init='k-means++')
km.fit(X)
y_prediction=km.predict(X)

print(km.cluster_centers_)#cluster central değerleri nerede
print(km.predict(X))

#yapılan kümeleme işleminin grafiğini çizmek
plt.scatter(X[y_prediction==0,0],X[y_prediction==0,1],s=100,c='red')
plt.scatter(X[y_prediction==1,0],X[y_prediction==1,1],s=100,c='blue')
plt.scatter(X[y_prediction==2,0],X[y_prediction==2,1],s=100,c='green')
plt.scatter(X[y_prediction==3,0],X[y_prediction==3,1],s=100,c='yellow')
plt.title('KMeans')
plt.show()

#k means de en iyi değer deneyerek bulunur

sonuclar=[]

#wcss
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++',random_state=123)#random_state önemli her seferinde cluster merkezi ayrı bir noktadna başlamasın diye random state değerine herhangi bir değer atamak önemli. Buradaki 123 de öylesine bir rakam
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)#inertia_ değeri wcss değerlerine denk geliyor.

print(sonuclar)
#wcss değerlerini grafik üstünde görelim
plt.plot(range(1,11),sonuclar)
plt.show()#k değeri 2-3 veya 4 seçilebilir grafiktende görüldüğü gibi

