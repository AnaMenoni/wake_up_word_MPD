"""
Realiza la detección de la palabra de activación en un archivo de audio.
El procedimiento carga un modelo previamente entrenado, aplica el
preprocesamiento definido sobre la señal (normalización y recorte de
silencios) y extrae un mel-espectrograma como representación
tiempo frecuencia. Esta matriz se aplana y ajusta a una dimensión fija
para poder ser procesada por el clasificador. Finalmente, el modelo emite
una predicción binaria indicando si la wake-word está presente o no.
"""

import librosa
import joblib
from preprocess import preprocess_wav
from features import compute_mel_spectrogram
import numpy as np

def detect_wakeword(filepath, model_path="modelo.pkl"):
    """
    Detecta si un audio contiene la wake-word.
    """
    model = joblib.load(model_path)

    x = preprocess_wav(filepath)
    mel = compute_mel_spectrogram(x)

    mel_flat = mel.flatten()[:4096]
    mel_flat = np.pad(mel_flat, (0, max(0, 4096 - len(mel_flat))), 'constant')

    pred = model.predict([mel_flat])[0]

    if pred == 1:
        print("Wake-word detectada.")
    else:
        print("No detectada.")

