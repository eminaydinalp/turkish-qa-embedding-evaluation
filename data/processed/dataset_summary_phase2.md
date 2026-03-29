# Faz 2 Veri Kümeleri - Seçim ve Hazırlama Özeti

Bu dosya, ödev kapsamındaki iki Türkçe soru-cevap veri kümesinin kaynak ve ön işleme özetini içerir.

## 1) patient_doctor_qa_tr
- HF repo: `kayrab/patient-doctor-qa-tr-5695`
- Kaynak split boyutları: train=22776, test=2532, toplam=25308
- Karttaki lisans: MIT
- Ön işleme sonrası örnek sayısı: 11736
- Kullanım amacı: Sağlık alanı QA retrieval değerlendirmesi

## 2) tquad
- HF repo: `dilanbakr/tquad`
- Kaynak split boyutları: train=8308, validation=2676, toplam=10984
- Karttaki lisans: belirtilmemiş (HF metadata içinde `license` alanı yok)
- Ön işleme sonrası örnek sayısı: 8184
- Kullanım amacı: Genel alan QA retrieval değerlendirmesi

## Uygulanan ortak ön işleme
- Ortak şema: `id`, `question`, `answer`
- Metin temizliği: satır sonu/boşluk normalizasyonu, boş metinlerin temizlenmesi
- Kısa kayıt filtreleme: question < 10 karakter, answer < 3 karakter olanlar çıkarıldı
- Tekilleştirme: (`question`, `answer`) bazında duplicate kayıtlar silindi

## Üretilen dosyalar
- Ham standart veri: `data/raw/*_raw.jsonl`
- İşlenmiş veri: `data/processed/*_qa.parquet` ve `data/processed/*_qa.jsonl`
- İstatistik dosyaları: `data/processed/*_stats.json`