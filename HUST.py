import scipy.io
import numpy as np
import pandas as pd
import os
from scipy import signal

taxa_alvo = 1000
taxa_original = 51200

lista_dfs = []

for root, dirs, files in os.walk("dados_vibracao_defeitos/HUST", topdown=False):
    for file_name in files:
        if not file_name.endswith('.mat'):
            continue
            
        path = os.path.join(root, file_name)
        print(f"\nProcessando: {path}")

        try:
            mat = scipy.io.loadmat(path)
        except Exception as e:
            print(f"Erro ao ler {file_name}: {e}")
            continue

        if 'data' not in mat:
            print(f"Aviso: Chave 'data' não encontrada em {file_name}. Pulando arquivo!")
            continue

        # Extrai sinal de vibração
        vibration_data = mat.get('data').flatten()

        # Calcula a duração e a quantidade de amostras para o resample
        duracao_sinal = len(vibration_data) / taxa_original
        num_amostras_alvo = int(duracao_sinal * taxa_alvo)
        
        # Faz a reamostragem (resample) para 1000 Hz
        # Filtro anti-aliasing para evitar artefatos
        sinal_1kHz = signal.resample_poly(vibration_data, taxa_alvo, taxa_original)

        print(f"Shape do sinal original ({taxa_original}Hz): {vibration_data.shape}")
        print(f"Shape do novo sinal ({taxa_alvo}Hz): {sinal_1kHz.shape}")

        # Cria a label/falha baseada no nome do arquivo (ex: B500)
        fault_name = file_name[:-4]
        fault_array = np.full((len(sinal_1kHz), 1), fault_name)

        # Cria um DataFrame temporário e adiciona na lista
        df_temp = pd.DataFrame({'DE_data': np.ravel(sinal_1kHz), 'fault': np.ravel(fault_array)})
        lista_dfs.append(df_temp)
    
if lista_dfs:
    # Concatena todos os arquivos processados do HUST
    df_hust = pd.concat(lista_dfs, axis=0, ignore_index=True)
    
    os.makedirs('sample_data', exist_ok=True)

    # Anexa na mesma planilha global 'all_faults.csv'
    df_hust.to_csv('sample_data/all_faults.csv', mode='a', header=False, index=False)
    
    print("\nDataset HUST processado e adicionado ao 'all_faults.csv' com sucesso!")
    print(f"Classes adicionadas: {df_hust['fault'].unique()}")
else:
    print("\nNenhum dado do HUST foi processado.")