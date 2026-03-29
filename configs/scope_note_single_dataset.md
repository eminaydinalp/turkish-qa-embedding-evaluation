# Proje Kapsam Notu (Tek Dataset Çalışma Modu)

Bu repository içinde deneyler yalnızca `patient_doctor_qa_tr` veri kümesi ile yürütülmektedir.

Gerekçe:
- Grup çalışmasında dataset iş paylaşımı yapıldı.
- Bu repo, patient-doctor veri kümesi tarafını üretmek ve raporlamak için kullanılacak.

Teknik uygulama:
- `configs/experiment.yaml` içinde `datasets` alanında `patient_doctor_qa_tr` için `enabled: true`, `tquad` için `enabled: false` ayarlandı.
- Faz 4 embedding scripti (`src/generate_embeddings_phase4.py`) varsayılan olarak yalnızca `enabled: true` datasetleri işler.

Not:
- Final teslimde ders gereksinimleri raporlanırken grup düzeyindeki toplam kapsam ayrıca belirtilmelidir.