# Hesaplamalı Anlambilim Ödev 1 - İlerleme Takibi

Bu dosyayı birlikte takip edeceğiz. Tamamlanan maddeleri `[x]` yaparak ilerleyelim.

## Faz 1 - Planlama ve Kurulum
- [x] Proje klasör yapısını oluştur
- [x] Gerekli Python ortamını ve paketleri kur (`transformers`, `sentence-transformers`, `datasets`, `scikit-learn`, `matplotlib`, `seaborn`, `pandas`, `numpy`, `torch`)
- [x] Deney ayarlarını tek bir config dosyasında tanımla (seed, batch size, cihaz, vb.)

## Faz 2 - Veri Kümelerini Bulma ve Hazırlama
- [ ] En az 1000 QA içeren 1. Türkçe veri kümesini seç
- [ ] En az 1000 QA içeren 2. Türkçe veri kümesini seç
- [ ] Lisans ve kullanım uygunluğunu not et
- [ ] Her veri kümesini ortak formata dönüştür (`id`, `question`, `answer`)
- [ ] Temizlik yap (boş/kısa kayıtlar, tekrarlar, bozuk metin)
- [ ] Veri kümesi istatistiklerini çıkar (örnek sayısı, ortalama uzunluk, vb.)

## Faz 3 - Model Seçimi
- [ ] MTEB leaderboard'dan ilk 100 içinde, multilingual ve 1B'den küçük 5 modeli belirle
- [ ] `ytu-ce-cosmos/turkish-e5-large` modelini ekle
- [ ] Seçilen 6 modelin adlarını ve kısa gerekçelerini tabloya yaz
- [ ] E5 için task text stratejisini netleştir (`query:` / `passage:`)

## Faz 4 - Embedding Üretimi
- [ ] Her veri kümesi için soruların embedding'lerini üret
- [ ] Her veri kümesi için cevapların embedding'lerini üret
- [ ] Her model için embedding'leri diske kaydet (tekrar kullanım için)

## Faz 5 - Benzerlik ve Retrieval Değerlendirmesi
- [ ] Açı tabanlı benzerlik hesaplamasını uygula (cosine -> arccos)
- [ ] Soru -> Cevap için top-5 adayları bul
- [ ] Top1 ve Top5 başarılarını hesapla (Soru -> Cevap)
- [ ] Cevap -> Soru için top-5 adayları bul
- [ ] Top1 ve Top5 başarılarını hesapla (Cevap -> Soru)
- [ ] Sonuçları model bazında karşılaştırma tablosuna dök

## Faz 6 - Görselleştirme (t-SNE)
- [ ] Her model için soru+cevap embedding'lerine t-SNE uygula
- [ ] Soru/cevap türüne göre renklendirilmiş 2B grafik üret
- [ ] 6 model için toplam 6 grafik dosyasını kaydet
- [ ] Grafiklerden kısa gözlemleri not et

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
