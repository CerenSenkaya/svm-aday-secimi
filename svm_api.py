from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Model ve scaler dosyalarını yükle
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

# Uygulamayı başlat
app = FastAPI()

# Giriş verisi için model tanımı
class Aday(BaseModel):
    tecrube: float
    puan: float

# Ana tahmin endpoint'i
@app.post("/tahmin")
def tahmin_yap(aday: Aday):
    girdi = np.array([[aday.tecrube, aday.puan]])
    girdi_scaled = scaler.transform(girdi)
    sonuc = model.predict(girdi_scaled)[0]
    return {"tahmin": "İşe alindi " if tahmin == 0 else "İşe alinmadi "}
