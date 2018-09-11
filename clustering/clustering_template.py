#kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#2.1 veri yükleme
data = pd.read_csv('../data/customer.csv')
X=data.iloc[:,3:].values

#k-means
from sklearn.cluster import KMeans
km=KMeans(n_clusters=4,init='k-means++')
km.fit(X)
y_prediction=km.predict(X)

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
    kmeans = KMeans(n_clusters=i, init='k-means++',random_state=123)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)
print(sonuclar)
#wcss değerlerini grafik üstünde görelim
plt.plot(range(1,11),sonuclar)
plt.show()

#hierarchical clustering
from sklearn.cluster import AgglomerativeClustering
agc=AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage='ward')
y_prediction=agc.fit_predict(X)
print(y_prediction)

#yapılan kümeleme işleminin grafiğini çizmek
plt.scatter(X[y_prediction==0,0],X[y_prediction==0,1],s=100,c='red')
plt.scatter(X[y_prediction==1,0],X[y_prediction==1,1],s=100,c='blue')
plt.scatter(X[y_prediction==2,0],X[y_prediction==2,1],s=100,c='green')
plt.scatter(X[y_prediction==3,0],X[y_prediction==3,1],s=100,c='yellow')
plt.title('Agglomerative Clustering')
plt.show()

#dendogram
import scipy.cluster.hierarchy as sch
dendogram=sch.dendrogram(sch.linkage(X,method='ward'))
plt.show()


