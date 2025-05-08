import joblib
import numpy as np
from scipy.io import wavfile
import python_speech_features as psf

# Modeli yükle
model = joblib.load("ses_tanima_modeli.pkl")
print(type(model))
print(model.classes_)

# MFCC çıkaran fonksiyon
def extract_mfcc(filename, sr=16000):
    rate, signal = wavfile.read(filename)
    mfcc = psf.mfcc(signal, rate)
    mfcc_mean = np.mean(mfcc, axis=0)
    return mfcc_mean

# Ses dosyasından MFCC özelliklerini çıkar
fake_mfcc = extract_mfcc("recorded_7.wav")
print("MFCC shape:", fake_mfcc.shape)

# Modeli kullanarak tahmin yap
prediction = model.predict([fake_mfcc])

# Sonucu yazdır
if prediction[0] == 0:
    print("Şuan Muhammed konuşuyor!")
elif prediction[0] == 1:
    print("Bu bir kedi sesi!")
elif prediction[0] == 2:    
    print("Bu bir köpek sesi!")
else:                       
    print("Bilinmeyen bir sınıf!")
