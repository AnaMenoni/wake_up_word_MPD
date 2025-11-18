import os
import librosa
import numpy as np
import pandas as pd
from src.preprocess import preprocess_wav
from src.features import compute_mel_spectrogram

def load_dataset(wav_folder, etiquetas_csv, sr=16000):
    """
    Carga audios WAV, los preprocesa y extrae features robustas.
    Devuelve X (características) e y (etiquetas).
    """

    df = pd.read_csv(etiquetas_csv)
    X = []
    y = []

    for _, row in df.iterrows():
        filename = row["archivo"]
        label = row["clase"]

        filepath = os.path.join(wav_folder, filename)

        x = preprocess_wav(filepath, sr=sr)

        # Mel - espectrograma
        mel = compute_mel_spectrogram(x, sr=sr)   # mel.shape = (n_mels, n_frames)

        # 
        mel_mean = mel.mean(axis=1)   # vector de tamaño n_mels
        mel_std  = mel.std(axis=1)    # vector de tamaño n_mels

        # vector final = concatenación (tamaño = 2*n_mels)
        features = np.concatenate([mel_mean, mel_std])

        X.append(features)
        y.append(label)

    return np.array(X), np.array(y)
