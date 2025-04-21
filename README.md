# Yazılım Geliştirici Aday Seçimi (SVM)

Bu proje, bir teknoloji firmasında işe alım sürecinde adayların `tecrübe yılı` ve `teknik puan` bilgilerine göre işe alınıp alınmayacağını tahmin etmek için geliştirilmiştir.

##  Kullanılan Teknolojiler
- Python (numpy, pandas, matplotlib, sklearn)
- Support Vector Machine (SVM)
- FastAPI (REST API)
- GridSearchCV (Parametre ayarı)

##  Dosyalar
- `svm_odev.py` – model eğitimi ve değerlendirmesi
- `svm_api.py` – tahmin servisi (FastAPI)
- `svm_model.pkl` ve `scaler.pkl` – eğitilmiş model ve ölçekleyici

##  API Kullanımı
1. `uvicorn svm_api:app --reload` komutuyla başlatılır.
2. `http://127.0.0.1:8000/docs` üzerinden Swagger arayüzüyle test edilebilir.
3. POST isteğiyle şu formatta veri gönderilir:

```json
{
  "tecrube": 2.5,
  "puan": 78
}
