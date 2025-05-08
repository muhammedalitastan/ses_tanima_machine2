import pyaudio
import wave
import time

def record_audio(duration=10, total_duration=1800, rate=16000, output_dir="recordings"):
    p = pyaudio.PyAudio()
    
    # Toplam kaydın süresi kadar döngü oluştur
    num_files = total_duration // duration  # 30 dakikalık bir süreyi 10 saniyelik parçalara bölelim

    for i in range(num_files):
        # Ses kaydını başlat
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=rate,
                        input=True,
                        frames_per_buffer=1024)
        
        print(f"{i+1}. 10 saniyelik ses kaydınız başlatılıyor...")

        frames = []
        for _ in range(0, int(rate / 1024 * duration)):
            data = stream.read(1024)
            frames.append(data)

        print(f"{i+1}. 10 saniyelik ses kaydınız tamamlandı.")
        
        # Kayıt tamamlandı, dosyaya kaydet
        output_filename = f"{output_dir}/recorded_{i+1}.wav"
        with wave.open(output_filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(rate)
            wf.writeframes(b''.join(frames))

        print(f"{output_filename} dosyasına kaydedildi.")
        
        # Kısa bir süre bekle (kaydı dosyaya yazarken bir aralık koymak amacıyla)
        time.sleep(1)

        # Kayıt akışını kapat
        stream.stop_stream()
        stream.close()

    p.terminate()
    print("Tüm kayıtlar tamamlandı.")

# 30 dakikalık ses kaydet, her biri 10 saniye sürecek
record_audio(duration=10, total_duration=3600)
