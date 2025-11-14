import librosa
import numpy as np

# En esta sección se trabaja la normalización y recorte de silencios.

def normalize_audio(x):
    # la idea es normalizar la señal dentro del rango [-1,1]
    if np.max(np.abs(x))>0:
        return (x / np.max(np.abs(x)))
    return x

def remove_silence(x, threshold=0.02):
    # se recortan los silencios de inicio y fin usando energía por ventana
    energy = librosa.feature.rms(y=x)[0]
    frames = np.where(energy > threshold)[0]

    if len(frames) == 0:
        return x
    
    start = max(librosa.frames_to_samples(frames[0]),0)
    end = min(librosa.frames_to_samples(frames[-1] +1), len(x))

    return x[start:end]

def preprocess_wav(filepath, sr=16000):
    # se carga un archivo wav, lo normaliza y recorta los silencios
    x, _ = librosa.load(filepath,sr=sr)
    x = normalize_audio(x)
    x = remove_silence(x)

    return x