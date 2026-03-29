# Faz 7 - Analiz ve Yorum

Bu bölümde, `patient_doctor_qa_tr_1500` veri kümesi üzerinde yapılan retrieval değerlendirmesi sonuçları ve t-SNE görselleştirmeleri temel alınarak model performansları analiz edilmiştir.

## 1. En İyi ve En Zayıf Model Davranışları

Retrieval metrikleri (`retrieval_summary.csv`) ve t-SNE grafikleri incelendiğinde, modellerin performansları şu şekilde özetlenebilir:

**En İyi Modeller:**

- **`ytu-ce-cosmos/turkish-e5-large`**: Hem Soru-Cevap (Q->A) hem de Cevap-Soru (A->Q) retrieval görevlerinde en yüksek Top-1 ve Top-5 başarı oranlarına sahip modeldir. t-SNE grafiğinde de soru ve cevap kümelerini oldukça net bir şekilde ayırması, bu modelin Türkçe anlamsal temsilleri ne kadar başarılı yakaladığını göstermektedir.
- **`intfloat/multilingual-e5-large-instruct`**: `turkish-e5-large` modelini çok yakından takip eden bu model, özellikle `query:` ve `passage:` direktifleri sayesinde anlamsal ayrımı çok keskin bir şekilde yapabilmektedir. t-SNE görselleştirmesindeki neredeyse mükemmel kümelenme, bu başarının bir kanıtıdır.

**En Zayıf Model:**

- **`KARDES/kalm-embedding-multilingual-mini`**: Bu model, hem metriklerde hem de görselleştirmede en düşük performansı sergilemiştir. Retrieval başarı oranları diğer modellere göre oldukça düşüktür. t-SNE grafiğinde soru ve cevap noktalarının neredeyse tamamen iç içe geçmiş olması, modelin bu iki farklı anlamsal uzayı ayırt etmekte zorlandığını açıkça göstermektedir. Modelin "mini" olması, yani daha az parametreye sahip olması, bu performans düşüklüğünün ana nedenlerinden biri olabilir.

## 2. Top-1 ve Top-5 Farklarını Açıklayan Örnekler

Metrikler, Top-1 ve Top-5 başarıları arasında önemli farklar olduğunu göstermektedir. Bu durum, doğru cevabın genellikle ilk 5 aday arasında yer almasına rağmen, her zaman en benzer aday olarak ilk sırada bulunamadığını gösterir.

- **`ytu-ce-cosmos/turkish-e5-large`** (En İyi Model):
  - **Q->A Başarısı**: Top-1: %51.3, Top-5: %80.8
  - **Yorum**: Bu modelde, sorguların yaklaşık %29.5'i için doğru cevap ilk sırada olmasa da ilk 5'te yer almıştır. Bu, modelin anlamsal olarak çok yakın veya benzer konuları içeren birden fazla cevabı adeta bir "kısa liste" olarak döndürme eğiliminde olduğunu gösterir. Örneğin, "baş ağrısı için ne yapmalıyım?" gibi bir soruya, doğrudan cevabın yanı sıra "migren belirtileri" veya "gerilim tipi baş ağrısı tedavisi" gibi çok yakın anlamdaki diğer cevapları da ilk 5'e dahil etmiş olabilir.

- **`KARDES/kalm-embedding-multilingual-mini`** (En Zayıf Model):
  - **Q->A Başarısı**: Top-1: %26.7, Top-5: %49.8
  - **Yorum**: Burada Top-1 ve Top-5 arasındaki fark %23.1'dir. Ancak daha önemlisi, sorguların yarısından fazlasında (%50.2) doğru cevabın ilk 5 aday arasında bile yer almamasıdır. Bu, modelin anlamsal olarak alakasız cevapları bile yüksek benzerlik skoruyla döndürdüğünü gösterir. t-SNE grafiğindeki soru ve cevap kümelerinin tamamen iç içe geçmiş olması bu durumu doğrulamaktadır. Model, anlamsal ayrım yapamadığı için, kelime örtüşmesi olan ancak anlamsal olarak tamamen farklı bir cevabı en iyi aday olarak getirebilir.

## 3. Hatalı Retrieval Örnekleri Üzerinden Hata Analizi

`kalm-embedding-multilingual-mini` modelinin düşük performansı, birkaç temel nedene bağlanabilir:

- **Anlamsal Belirsizlik**: Model, soru ve cevapların anlamsal nüanslarını yakalayamamaktadır. Örneğin, "enfeksiyon" kelimesi geçen bir soruya, alakasız bir "enfeksiyon" türünden bahseden bir cevabı, sırf anahtar kelime eşleşmesi nedeniyle getirebilir.
- **Parametre Yetersizliği**: "Mini" bir model olması, karmaşık anlamsal ilişkileri öğrenebilmesi için yeterli kapasiteye sahip olmadığı anlamına gelir. Bu nedenle, daha genel ve yüzeysel özelliklere odaklanır.
- **Çok Dilli Olmanın Etkisi**: Model çok dilli olsa da, Türkçe'nin spesifik morfolojik ve anlamsal yapısına yeterince adapte olamamış olabilir. Bu durum, özellikle ekler ve deyimsel ifadeler içeren sorgularda hatalı eşleşmelere yol açar.

Özetle, `kalm-embedding-multilingual-mini` modelindeki hatalar, modelin anlamsal derinlikten yoksun olmasından ve daha çok yüzey seviyesindeki kelime eşleşmelerine dayanmasından kaynaklanmaktadır.

## 4. Genel Değerlendirme

Tek bir veri kümesi üzerinde çalışılmış olsa da, sonuçlar model seçiminin ne kadar kritik olduğunu göstermektedir. Özellikle Türkçe diline özel olarak eğitilmiş veya `instruct` gibi özel yönlendirmelerle kullanılan büyük modellerin, genel amaçlı ve daha küçük çok dilli modellere kıyasla belirgin bir şekilde daha iyi performans gösterdiği açıktır. `kalm-embedding-multilingual-mini` gibi küçük modeller, hızlı ve az kaynak gerektiren senaryolar için uygun olabilirken, yüksek doğruluk gerektiren anlamsal arama görevlerinde yetersiz kalmaktadır.
