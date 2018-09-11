#takviyeli/pekiştirmeli öğrenme(reinforced learning)
#burada makine (veya robot) için bir hedefe blirleniyor ve bu robot yazılımını kulanarak ve olayı deneyimleyerek
#dışardan öğnereke kendini gelitiriyor. Her yanlış hareketini cezalandırarak kendini geliştiriyor.
#A/B testing mesela reklam gösteriyorsun günde 1000 kişiye göstericeksin 500'üne bir rekalam 500'üne diğer rekalmı gösterip
#hangisinin daha çok tıkladığına göre hangi reklamında daha iyi olduğunu bilgisayar gözlemleye bilir.
#tek kollo canavar(kumar makinesi) için hangi dağılımla para kazandığı gözlemlenerek hangi makinede kumar oynamamın mantıklı olacağı
#makine kandi kendine öğrenebilir.

#UPPER CONFIDENCE BOUND(UCD)(UST GUVEN SINIRI)
#algoritmanın arkasındaki mantık her olayın arkasında bir dağılım olduğunu kabul etmek.
#kullanıcı her seferinde bir eylem yapar
#bu eylem karşılığında bir skor dönüyor (web sitesi tıklanması 1 tıklanmaması 0)
#amacımız tıklanmayı maximuma çıkarmak
#buarada sitenin minumumda ve maximumda tıklanma ihtimalleri var aralık confidence bound oluyor
#mesela kazanabileceğimiz en düşük para ile en yüksek para
#her eylem sonucunda bizim confidence bound sınırımız biraz değişir.
#her oynamada bu aralık küçüldükçe bizim bilmediğimiz ama keşfetmez istediğimiz istatistiksel olarak dağılımı bulmuş oluruz.
#tabi buarada çok fazla oynamak mümkün olmadığı için belli bir oynamadan sonra elimizdeki en yüksek sınıra sabip olan aralıkta yer alan
#değeri almak gerektiğini söylüyor.
#elimizde birden fazla alternatif olduğunda bu alternatiflerden en yüksek sınıra sahip olan alternatifin seçilmesi.

#random selection :  herhangi zeki bir seçim yapmadan rast gele yapılan seçim
#UCB düzgün dağılım göstermeyen verileri işlemede random seletiondan daha iyi çalşıyor. (burada düzegün dağılımdan kastım örneğin 1. reklam 200 kere 2. reklam 10 defa gösterilmiş olsun müşterilere)
#randomda herhangi bir zeka belirtisi göstermeden sistem rastgele seçimler yapıyor ve bu seçimlerin sonucunda kazanıyoruz ve ya kaybediyoruz.

#UCB 'de
#1.adımda her turda (tur sayısı n) ,her reklam alternetifi (i için) aşağıdaki sayılar tutulur
#Ni(n) :o reklamın o ana kadarki tıklanma sayısı
#Ri(n): o ana kadarki i reklamlardan gelen ödül
#2.adımda yukardaki bu değerlerden aşağıdaki değerler hesaplanır
#O ana kadarki her reklamın ortalama ödülü Ri(n)/Ni(n)
#Güven aralığı içinde aşağı yukarı oynama olasılığı di(n)=karekö içinde(3/2* log(n)/Ni(n) )
#3.adımda en yüksek UCB olanı alırız.