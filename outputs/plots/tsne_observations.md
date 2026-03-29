# Faz 6 - t-SNE Gözlemleri

Bu dosyada, farklı embedding modelleri kullanılarak üretilen t-SNE görselleştirmelerinden elde edilen ilk gözlemler yer almaktadır.

## Gözlemler

- **`bge-m3`**: Soru (mavi) ve cevap (turuncu) kümeleri genel olarak birbirinden ayrışmış durumda. İki ana küme belirgin, ancak aralarında bir miktar geçişkenlik ve iç içe geçmiş noktalar mevcut. Bu durum, modelin soru ve cevap anlamsal uzaylarını büyük ölçüde ayırt edebildiğini gösteriyor.

- **`gte-multilingual-base`**: `bge-m3` modeline benzer şekilde soru ve cevap kümeleri arasında bir ayrım gözlemleniyor, ancak kümeler arasındaki sınır daha az belirgin ve daha fazla örtüşme var. Kümeler daha dağınık bir yapıya sahip.

- **`kalm-embedding-multilingual-mini`**: Bu modelde soru ve cevap noktaları büyük ölçüde iç içe geçmiş durumda. Belirgin bir kümelenme veya ayrım yapmak oldukça zor. Bu, modelin soru ve cevapların anlamsal temsillerini ayırt etmede diğer modellere göre daha zayıf kaldığını düşündürüyor.

- **`multilingual-e5-large-instruct`**: En net ayrışma bu modelde gözlemleniyor. Soru ve cevap kümeleri neredeyse hiç örtüşme olmadan, yoğun ve belirgin iki ayrı küme olarak konumlanmış. `query:` ve `passage:` gibi özel direktiflerin kullanılması, bu anlamsal ayrımı keskinleştirmiş görünüyor.

- **`multilingual-e5-large`**: `instruct` versiyonu kadar olmasa da, bu modelde de soru ve cevap kümeleri arasında belirgin bir ayrım mevcut. Kümeler, `gte-multilingual-base` modeline göre daha net bir şekilde ayrışmış.

- **`turkish-e5-large`**: Türkçe'ye özel bu model, `multilingual-e5-large-instruct` modelinden sonra en iyi ayrışmayı gösteren modellerden biri. Soru ve cevap kümeleri oldukça net bir şekilde ayrılmış. Bu, modelin Türkçe soru-cevap çiftlerinin anlamsal farklılıklarını başarılı bir şekilde yakaladığını gösteriyor.

