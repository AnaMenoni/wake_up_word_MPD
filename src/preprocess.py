import librosa
import numpy as np

def normalize_audio(x):
    """
    Normaliza la señal en el rango [-1, 1].
    """
    m = np.max(np.abs(x))
    if m > 0:
        return x / m
    return x

def remove_silence(x, top_db=25):
    """
    Recorta silencios al inicio y fin usando energía logarítmica.
    top_db controla qué tan agresivo es el recorte.
    """
    y, _ = librosa.effects.trim(x, top_db=top_db)
    return y

def preprocess_wav(filepath, sr=16000):
    """
    Carga un WAV, lo normaliza y recorta silencios.
    NO ajusta la duración.
    """
    # Cargar audio
    x, _ = librosa.load(filepath, sr=sr)

    # Normalizar amplitud
    x = normalize_audio(x)

    # Recortar silencio
    x = remove_silence(x)

    return x
