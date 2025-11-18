"""
Realiza la detección de la wake-word utilizando el mismo pipeline
que durante el entrenamiento: preprocesamiento → mel → features estadísticas
(mean + std) → escalado → predicción con RandomForest.
"""

import joblib
import numpy as np
from src.preprocess import preprocess_wav
from src.features import compute_mel_spectrogram
import matplotlib.pyplot as plt
import librosa.display


def detect_wakeword(filepath, model_path="modelo.pkl", scaler_path="scaler.pkl"):
    """
    Detecta si un audio contiene la wake-word.
    """

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    x = preprocess_wav(filepath)

    mel = compute_mel_spectrogram(x)

    graficar_senal(x)
    graficar_stft(x)
    graficar_mel(mel)
    graficar_features(mel)

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

def graficar_senal(x, sr=16000):
    plt.figure(figsize=(12,3))
    plt.title("Señal temporal preprocesada")
    plt.plot(x)
    plt.xlabel("Muestras")
    plt.ylabel("Amplitud")
    plt.tight_layout()
    plt.show()


def graficar_stft(x, sr=16000):
    X = librosa.stft(x)
    X_db = librosa.amplitude_to_db(abs(X))

    plt.figure(figsize=(12,4))
    librosa.display.specshow(X_db, sr=sr, x_axis='time', y_axis='hz', cmap='magma')
    plt.colorbar(format="%+2.f dB")
    plt.title("Espectrograma STFT")
    plt.tight_layout()
    plt.show()


def graficar_mel(mel, sr=16000):
    mel_db = librosa.power_to_db(mel, ref=np.max)

    plt.figure(figsize=(12,4))
    librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel', cmap='magma')
    plt.colorbar(format="%+2.f dB")
    plt.title("Mel-Espectrograma")
    plt.tight_layout()
    plt.show()

def graficar_features(mel):
    mel_mean = mel.mean(axis=1)
    mel_std  = mel.std(axis=1)

    plt.figure(figsize=(12,4))
    plt.plot(mel_mean, label="mean")
    plt.plot(mel_std, label="std")
    plt.title("Características Mel (mean / std)")
    plt.xlabel("Banda Mel")
    plt.ylabel("Valor")
    plt.legend()
    plt.tight_layout()
    plt.show()


