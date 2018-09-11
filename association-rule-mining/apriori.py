#1. kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#2.1 veri yükleme
data=pd.read_csv('../data/basket.csv',header=None)

#csv dosyasındaki her satırı bir listeye çevirdik.
t=[]
for i in range(0,7501):
    t.append([str(data.values[i,j]) for j in range (0,20)])

#bir dosya içinde bulunan apyori kütüphanesini import ettik
#ilişkili olan ürünleri bulmak istiyoruz.
from apriori_library import apriori
rule = apriori(t,min_support=0.01,min_confidence=0.2,min_lift=3,min_lenght=2)
print(list(rule))

#eclat algoritması küçük verilerde işe yarar fakat büyük verilerde sorun yaşar. Apriori büyük yerilerde daha iyi

