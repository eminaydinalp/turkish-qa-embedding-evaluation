# Hesaplamalı Anlambilim Ödev 1 - İlerleme Takibi

Bu dosyayı birlikte takip edeceğiz. Tamamlanan maddeleri `[x]` yaparak ilerleyelim.

## Faz 1 - Planlama ve Kurulum
- [x] Proje klasör yapısını oluştur
- [x] Gerekli Python ortamını ve paketleri kur (`transformers`, `sentence-transformers`, `datasets`, `scikit-learn`, `matplotlib`, `seaborn`, `pandas`, `numpy`, `torch`)
- [x] Deney ayarlarını tek bir config dosyasında tanımla (seed, batch size, cihaz, vb.)

## Faz 2 - Veri Kümelerini Bulma ve Hazırlama
- [x] En az 1000 QA içeren 1. Türkçe veri kümesini seç
- [x] En az 1000 QA içeren 2. Türkçe veri kümesini seç
- [x] Lisans ve kullanım uygunluğunu not et
- [x] Her veri kümesini ortak formata dönüştür (`id`, `question`, `answer`)
- [x] Temizlik yap (boş/kısa kayıtlar, tekrarlar, bozuk metin)
- [x] Veri kümesi istatistiklerini çıkar (örnek sayısı, ortalama uzunluk, vb.)

## Faz 3 - Model Seçimi
- [x] MTEB leaderboard'dan ilk 100 içinde, multilingual ve 1B'den küçük 5 modeli belirle
- [x] `ytu-ce-cosmos/turkish-e5-large` modelini ekle
- [x] Seçilen 6 modelin adlarını ve kısa gerekçelerini tabloya yaz
- [x] E5 için task text stratejisini netleştir (`query:` / `passage:`)

## Faz 4 - Embedding Üretimi
- [x] Her veri kümesi için soruların embedding'lerini üret
- [x] Her veri kümesi için cevapların embedding'lerini üret
- [x] Her model için embedding'leri diske kaydet (tekrar kullanım için)

## Faz 5 - Benzerlik ve Retrieval Değerlendirmesi
- [x] Açı tabanlı benzerlik hesaplamasını uygula (cosine -> arccos)
- [x] Soru -> Cevap için top-5 adayları bul
- [x] Top1 ve Top5 başarılarını hesapla (Soru -> Cevap)
- [x] Cevap -> Soru için top-5 adayları bul
- [x] Top1 ve Top5 başarılarını hesapla (Cevap -> Soru)
- [x] Sonuçları model bazında karşılaştırma tablosuna dök

## Faz 6 - Görselleştirme (t-SNE)
- [x] Her model için soru+cevap embedding'lerine t-SNE uygula
- [x] Soru/cevap türüne göre renklendirilmiş 2B grafik üret
- [x] 6 model için toplam 6 grafik dosyasını kaydet
- [x] Grafiklerden kısa gözlemleri not et

## Faz 7 - Analiz ve Yorum
- [ ] En iyi ve en zayıf model davranışlarını yorumla
- [ ] Top1/Top5 farklarını açıklayan örnekler seç
- [ ] Hatalı retrieval örnekleri üzerinden hata analizi yap
- [ ] Veri kümesi farklarının etkisini tartış

## Faz 8 - Sunum Hazırlığı
- [ ] Sunum iskeletini hazırla (problem, veri, yöntem, sonuç, yorum)
- [ ] Sonuç tablolarını ve grafikleri slaytlara yerleştir
- [ ] Her slayt için kısa konuşma notu ekle
- [ ] Son kontrol ve prova yap

## Faz 9 - Teslim
- [ ] Kodları düzenle ve çalışır halde paketle
- [ ] Sunum dosyasını son haline getir
- [ ] Teslim öncesi checklist tamamla
- [ ] online.yildiz.edu.tr üzerinden zamanında yükle (31 Mart 2026, 09:30)
