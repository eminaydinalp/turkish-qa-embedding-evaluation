# Faz 3 - Model Seçimi (Nihai Liste)

Bu dosya, ödev için kullanılacak 6 embedding modelini ve kısa seçim gerekçelerini içerir.

## Seçilen Modeller

| # | Model | Kısa gerekçe |
|---|---|---|
| 1 | intfloat/multilingual-e5-large-instruct | E5 ailesinin instruction-tuned sürümü; çok dilli retrieval için güçlü bir referans. |
| 2 | BAAI/bge-m3 | Çok dilli ve güçlü genel amaçlı embedding modeli; retrieval benchmarklarında yaygın. |
| 3 | intfloat/multilingual-e5-large | E5 ailesinin temel çok dilli sürümü; instruct varyantıyla karşılaştırma sağlar. |
| 4 | HIT-TMG/KaLM-embedding-multilingual-mini-v1 | Daha hafif bir çok dilli model; kalite-hız dengesi karşılaştırması için faydalı. |
| 5 | Alibaba-NLP/gte-multilingual-base | GTE ailesinin çok dilli taban modeli; farklı model ailesi kıyası sağlar. |
| 6 | ytu-ce-cosmos/turkish-e5-large | Ödevde zorunlu model; Türkçe odaklı E5 varyantı. |

## Ödev Kısıtı Uyum Notu
- İlk 5 model çok dilli model grubu olarak kullanılacaktır.
- 6. model olarak ytu-ce-cosmos/turkish-e5-large eklenmiştir.
- MTEB top-100 ve <1B parametre koşulu, ders kapsamında raporda model kartı referanslarıyla ayrıca belirtilecektir.

## E5 Task Text Stratejisi

E5 tabanlı modellerde query/passage önekleri kullanılacaktır:

- intfloat/multilingual-e5-large-instruct
  - query: Instruct + Query formatı
  - passage: passage: <cevap>
- intfloat/multilingual-e5-large
  - query: query: <soru>
  - passage: passage: <cevap>
- ytu-ce-cosmos/turkish-e5-large
  - query: query: <soru>
  - passage: passage: <cevap>

E5 dışındaki modellerde varsayılan olarak ham metin kullanılacaktır (önek yok).