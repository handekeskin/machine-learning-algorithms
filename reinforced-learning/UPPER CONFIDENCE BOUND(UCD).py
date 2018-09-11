#1. kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as math

#örnek olarak yüklediğimiz datalarda satırlar müşterileri sütunlarda 10 farklı reklamı gösteriyor.
#1 olan değerlerde müşteri reklama tıklamış demek oluyor.
#her satırdan sonra bir sonraki satır gerçekleşiyor yani zaman sırası var.
#reklamların gösterilme sıklığı aynı değil.

#verileri okuma
data=pd.read_csv('../data/Ads_CTR_Optimisation.csv')

'''
#öncelikle random seçim yaparak reklam seçtiğimizde sonuçları inceleyeceğiz.
#ramdom kütüphanesi
import random

N=10000
d=10
toplam=0
secilenler=[]
for n in range(0,N):
    ad = random.randrange(d) #10 a kadar herhangi rastgele bir sayı üret
    secilenler.append(ad)
    odul = data.values[n,ad] #datadaki n.satırdaki değer 1 ise ödülde 1 olur
    toplam = toplam + odul

print(toplam) #rastgele seçin yaptığımızda kaç defa bildik.

plt.hist(secilenler)
plt.show()
'''

import random

N=10000 #10000 seçim var
d=10 #toplamda 10 ilan var
#Ri(n)
oduller=[0]*d #her bir elemanı o olan 10luk bir dizi #ilk başta bütün ilanların ödülü 0
#Ni(n)
tiklamalar=[0]*d # o ana kadarki tıklamalar
toplam =0  #toplam ödül
secilenler=[]
odul=0
for n in range(0,N):
    ad=0#seçilen ilan
    max_ucb=0
    for i in range (0,d): #bu for dögüsü en yüksek ucb değerini hesaplamaya yarıyor.
        if (tiklamalar[i]>0):
            ortalama = oduller[i] /tiklamalar[i]
            delta = math.sqrt(3/2 * math.log(n)/tiklamalar[i])
            ucb=ortalama+delta
        else:
            ucb=N*10
        if max_ucb < ucb: #maxdan daha büyük ucb çıktı max'ı yeni gelen değerler değiştir
            max_ucb = ucb
            ad=i
    secilenler.append(ad)
    tiklamalar[ad]=tiklamalar[ad]+1
    odul = data.values[n,ad] #datadaki n.satırdaki değer 1 ise ödülde 1 olur
    oduller[ad] = oduller[ad] + odul
    toplam = toplam + odul#toplam ödülü bulma
print('toplam ödül:')
print(toplam)

plt.hist(secilenler)
plt.show()