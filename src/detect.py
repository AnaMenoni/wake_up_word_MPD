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
