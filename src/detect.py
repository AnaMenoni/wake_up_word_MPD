"""
Realiza la detección de la wake-word utilizando el mismo pipeline
que durante el entrenamiento: preprocesamiento → mel → features estadísticas
(mean + std) → escalado → predicción con RandomForest.
"""

import joblib
import numpy as np
from src.preprocess import preprocess_wav
from src.features import compute_mel_spectrogram


def detect_wakeword(filepath, model_path="modelo.pkl", scaler_path="scaler.pkl"):
    """
    Detecta si un audio contiene la wake-word.
    """

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    x = preprocess_wav(filepath)

    mel = compute_mel_spectrogram(x)

    mel_mean = mel.mean(axis=1)   # promedio por cada banda Mel
    mel_std  = mel.std(axis=1)    # desviación estándar por banda Mel

    features = np.concatenate([mel_mean, mel_std])

    features_scaled = scaler.transform([features])

    pred = model.predict(features_scaled)[0]

    # Salida
    if pred == 1:
        print("Wake-word detectada.")
    else:
        print("No detectada.")
