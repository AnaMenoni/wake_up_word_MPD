import librosa
import numpy as np

def compute_stft(x, sr=16000, n_fft=512, hop_length=160, win_length=400):
    """
    Calcula el espectrograma STFT (magnitud).
    """
    S = librosa.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    S_mag = np.abs(S)
    return S_mag


def compute_mel_spectrogram(x, sr=16000, n_mels=64, n_fft=512, hop_length=160):
    """
    Calcula Mel-espectrograma.
    """
    mel = librosa.feature.melspectrogram(
        y=x,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db
