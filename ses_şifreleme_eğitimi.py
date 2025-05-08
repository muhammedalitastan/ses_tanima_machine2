import librosa
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Ses dosyasından MFCC özniteliklerini çıkaran fonksiyon
def extract_features(file_path):
    # Ses dosyasını yükle
    y, sr = librosa.load(file_path, sr=None)

    # MFCC özelliklerini çıkar
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # MFCC'yi 1D vektöre dönüştür (ortalama)
    mfcc_mean = np.mean(mfcc, axis=1)

    return mfcc_mean

# Veri setini ve etiketleri hazırlama
def prepare_data(data_dir):
    features = []
    labels = []

    for label in ['Muhammed', 'cat', 'dog']:  # Etiketler (Muhammed, Kedi ve Köpek)
        label_dir = os.path.join(data_dir, label)

        for file_name in os.listdir(label_dir):
            if file_name.endswith('.wav'):
                file_path = os.path.join(label_dir, file_name)
                feature = extract_features(file_path)
                features.append(feature)
                labels.append(label)

    return np.array(features), np.array(labels)

# Veri kümesini yükleyin ve özellikleri çıkarın
data_dir = 'Muhammed' # Ses dosyalarınızın bulunduğu ana klasör
X, y = prepare_data(data_dir)

# Etiketleri sayısal hale getirin (Muhammed=0, cat=1, dog=2)
# y = np.where(y == 'Muhammed', 0, y)
# y = np.where(y == 'cat', 1, y)
# y = np.where(y == 'dog', 2, y)

y = np.array([0 if label == 'Muhammed' else 1 if label == 'cat' else 2 for label in y])

# Veriyi eğitim ve test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM modelini eğitme
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Test verisi ile doğruluğu değerlendirme
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Test Doğruluğu: {accuracy * 100:.2f}%")

# Modeli kaydetme
import joblib
joblib.dump(model, 'ses_tanima_modeli.pkl')
