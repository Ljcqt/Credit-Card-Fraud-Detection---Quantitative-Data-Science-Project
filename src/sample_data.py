import pandas as pd
import numpy as np

#Chemin des datasets d'entrée et de sortie
file_path = "/home/luka/Bureau/cours/4A/projet_fraude/data/raw/data/credit_card_transactions-ibm_v2.csv" 
output_path = "/home/luka/Bureau/cours/4A/projet_fraude/data/processed/sampled_transactions.csv"

#Lecture des users
print("Lecture de la colonne User du csv")
df_users = pd.read_csv(file_path, usecols=["User"])

#Séléction des utilisateurs uniques
print("Identification des utilisateurs uniques")
unique_users = df_users["User"].unique()
print(f"Nombre total d'utilisateurs : {len(unique_users)}")

#Séléction de l'échantillon (~20%)
selected_users = set(np.random.choice(unique_users, size = int(0.2*len(unique_users)), replace = False))

#Sauvegarde des informations des utilisateurs séléctionnés 
print("Lecture par chunk")
chunk_size = 10000
first_chunk = True

for chunk in pd.read_csv(file_path, chunksize=chunk_size):
    filtered_chunk = chunk[chunk["User"].isin(selected_users)]
    if first_chunk:
        filtered_chunk.to_csv(output_path, index=False, mode='w', header=True)
        first_chunk = False
    else:
        filtered_chunk.to_csv(output_path, index=False, mode='a', header=False)

print("Échantillonnage fait")

