import os
import pandas as pd

positivo_folder = "../data/positivo"
negativo_folder = "../data/negativo"
raw_folder = "../data/raw"

# Crear raw si no existe
os.makedirs(raw_folder, exist_ok=True)

rows = []

# procesar positivos
for f in os.listdir(positivo_folder):
    if f.lower().endswith((".wav", ".mp3", ".m4a", ".opus", ".ogg", ".flac")):
        rows.append([f, 1])

# procesar negativos
for f in os.listdir(negativo_folder):
    if f.lower().endswith((".wav", ".mp3", ".m4a", ".opus", ".ogg", ".flac")):
        rows.append([f, 0])

df = pd.DataFrame(rows, columns=["archivo", "clase"])
df.to_csv("../data/etiquetas.csv", index=False)

print("etiquetas.csv generado correctamente:")
print(df)


import shutil

raw_folder = "../data/raw"
os.makedirs(raw_folder, exist_ok=True)

# copiar positivos
for f in os.listdir(positivo_folder):
    src = os.path.join(positivo_folder, f)
    dst = os.path.join(raw_folder, f)
    shutil.copy(src, dst)

# copiar negativos
for f in os.listdir(negativo_folder):
    src = os.path.join(negativo_folder, f)
    dst = os.path.join(raw_folder, f)
    shutil.copy(src, dst)

print("Todos los archivos copiados a data/raw.")
