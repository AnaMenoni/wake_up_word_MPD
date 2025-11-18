import os
from pydub import AudioSegment

def convert_to_wav(input_path, output_path, target_sr=16000):
    """
    Convierte un archivo cualquiera (.m4a/.mp3) a WAV mono 16 kHz.

    Parámetros:
        input_path (str): Ruta del archivo de entrada.
        output_path (str): Ruta donde se guarda el WAV convertido.
        target_sr (int): Frecuencia de muestreo objetivo. Por defecto 16000 Hz.
    """
    # Cargar el audio con pydub (detecta el formato automáticamente)
    print(f'El formato del archivo a convertir es {input_path.format()}')
    audio = AudioSegment.from_file(input_path)

    # Convertir a mono
    audio = audio.set_channels(1)

    # Ajustar la frecuencia de muestreo
    audio = audio.set_frame_rate(target_sr)

    # Exportar como WAV
    audio.export(output_path, format="wav")

def batch_convert_raw_to_wav(raw_folder, wav_folder, target_sr=16000):
    """
    Convierte TODOS los archivos de data/raw/ a WAV 16 kHz mono,
    usando convert_to_wav().
    """

    os.makedirs(wav_folder, exist_ok=True)

    for filename in os.listdir(raw_folder):
        print("entró al for")

        if not filename.lower().endswith((".wav", ".mp3", ".m4a", ".ogg", ".flac", ".opus")):
            continue  # ignora archivos no-audio

        input_path = os.path.join(raw_folder, filename)
        name_no_ext = os.path.splitext(filename)[0]
        output_path = os.path.join(wav_folder, name_no_ext + ".wav")

        convert_to_wav(input_path, output_path, target_sr)

    print("Conversión en lote finalizada.")
    