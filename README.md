# LSTM ile Oda Doluluk Oranı Tahmini

Bu proje, sensör verilerinden (Temperature, Humidity, Light, CO2, HumidityRatio) yararlanarak LSTM modeli ile odanın dolu/boş durumunu (Occupancy: 0/1) tahmin eder ve seçilen zaman aralıkları için doluluk oranını (%) hesaplar.

## Klasör Yapısı
- `data/`: Veri dosyaları (datatraining.txt, datatest.txt, datatest2.txt)
- `model.py`: LSTM modeli
- `train.py`: Eğitim + değerlendirme + model kaydetme
- `serve.py`: Gradio arayüzü ile doluluk oranı hesaplama
- `requirements.txt`: Bağımlılıklar

## Kurulum
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
