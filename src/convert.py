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
    audio = AudioSegment.from_file(input_path)

    # Convertir a mono
    audio = audio.set_channels(1)

    # Ajustar la frecuencia de muestreo
    audio = audio.set_frame_rate(target_sr)

    # Exportar como WAV
    audio.export(output_path, format="wav")

