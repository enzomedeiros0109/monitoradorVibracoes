import scipy.io
import numpy as np
import pandas as pd
import os
from scipy import signal

taxa_original = 12000
taxa_alvo = 1000

lista_dfs = []

print("Iniciando extração e processamento dos dados do CWRU...")

for root, dirs, files in os.walk("dados_vibracao_defeitos/CWRU", topdown=False):
    for file_name in files:
        if not file_name.endswith('.mat'):
            continue
            
        path = os.path.join(root, file_name)
        mat = scipy.io.loadmat(path)

        # Busca dinâmica pela chave de tempo do Drive End (DE)
        key_name = next((key for key in mat.keys() if 'DE_time' in key), None)

        if key_name is None:
            print(f"Aviso: Chave 'DE_time' não encontrada em {file_name}. Pulando arquivo!")
            continue

        DE_data = mat.get(key_name).flatten()

        # Downsampling de 12kHz para 1kHz
        sinal_1kHz = signal.resample_poly(DE_data, up=taxa_alvo, down=taxa_original)

        # Cria a matriz de rótulos baseada no nome do arquivo
        fault = np.full((len(sinal_1kHz), 1), file_name[:-4])

        # Adiciona os dados à lista
        df_temp = pd.DataFrame({'DE_data': np.ravel(sinal_1kHz), 'fault': np.ravel(fault)})
        lista_dfs.append(df_temp)

df = pd.concat(lista_dfs, ignore_index=True)

os.makedirs('sample_data', exist_ok=True)

df.to_csv('sample_data/all_faults.csv', index=False)
print("Dataset CWRU processado para 1kHz e salvo com sucesso em all_faults.csv!")