import os
import librosa
import numpy as np
import pandas as pd
from src.preprocess import preprocess_wav
from src.features import compute_mel_spectrogram

def load_dataset(wav_folder, etiquetas_csv):
    """
    Carga todos los audios WAV y genera X (features) e y (etiquetas).
    """
    df = pd.read_csv(etiquetas_csv)
    X = []
    y = []

    for _, row in df.iterrows():
        filepath = os.path.join(wav_folder, row['archivo'])
        label = row['clase']

        x = preprocess_wav(filepath)
        mel = compute_mel_spectrogram(x)

        # Aplanar el Mel-espectrograma
        mel_flat = mel.flatten()[:4096]  # limitar tamaño
        mel_flat = np.pad(mel_flat, (0, max(0, 4096 - len(mel_flat))), 'constant')

        X.append(mel_flat)
        y.append(label)

    return np.array(X), np.array(y)
