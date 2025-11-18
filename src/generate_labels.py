import os
import pandas as pd

wav_folder = "../data/wav"
positivo_folder = "../data/positivo"
negativo_folder = "../data/negativo"

rows = []

positivos_ori = {os.path.splitext(f)[0] for f in os.listdir(positivo_folder)}
negativos_ori = {os.path.splitext(f)[0] for f in os.listdir(negativo_folder)}

for filename in os.listdir(wav_folder):
    if filename.endswith(".wav"):
        base = os.path.splitext(filename)[0]

        if base in positivos_ori:
            rows.append([filename, 1])
        elif base in negativos_ori:
            rows.append([filename, 0])
        else:
            print("Advertencia: no se encontró etiqueta:", filename)

df = pd.DataFrame(rows, columns=["archivo", "clase"])
df.to_csv("../data/etiquetas.csv", index=False)

print(df)