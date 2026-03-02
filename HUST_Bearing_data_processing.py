import pandas as pd
import numpy as np
import os
from scipy import signal
import scipy.io

# Configurações do HUST
HUST_FOLDER = "dados_vibracao_defeitos\HUST" 
TARGET_SAMPLE_RATE = 1000        

# Taxa de amostragem oficial do dataset HUST
ORIGINAL_SAMPLE_RATE = 51200 

FILE_EXTENSION = ".mat"
CSV_DESTINO = 'sample_data/all_faults.csv'

def processar_hust():
    print(f"[{'HUST Extractor'}] A iniciar leitura na pasta: {HUST_FOLDER}...")
    
    if os.path.exists(CSV_DESTINO):
        base_df = pd.read_csv(CSV_DESTINO)
        print(f"[{'HUST Extractor'}] Encontrado CSV base com {len(base_df)} linhas.")
    else:
        base_df = pd.DataFrame(columns=['DE_data', 'fault'])
        print(f"[{'HUST Extractor'}] CSV base não encontrado. Um novo será criado.")

    new_data_list = []

    for root, dirs, files in os.walk(HUST_FOLDER):
        for file_name in files:
            if file_name.endswith(FILE_EXTENSION):
                path = os.path.join(root, file_name)
                fault_name = file_name.replace(FILE_EXTENSION, '')
                print(f"A extrair falha: {fault_name}...", end=" ")

                try:
                    mat = scipy.io.loadmat(path)
                    
                    # Pega apenas a vibração
                    raw_signal = mat['data'].flatten()
                    
                    # Faz o Resample usando a taxa manual
                    signal_duration = len(raw_signal) / ORIGINAL_SAMPLE_RATE
                    num_target_samples = int(signal_duration * TARGET_SAMPLE_RATE)
                    signal_1kHz = signal.resample(raw_signal, num_target_samples)

                    # Prepara a coluna
                    fault_col = np.full((len(signal_1kHz), 1), fault_name)

                    # Adiciona à lista
                    processed_df = pd.DataFrame({'DE_data': np.ravel(signal_1kHz), 'fault': np.ravel(fault_col)})
                    new_data_list.append(processed_df)
                    
                    print(f"OK! (Resample: {ORIGINAL_SAMPLE_RATE}Hz -> {TARGET_SAMPLE_RATE}Hz)")

                except KeyError as e:
                    print(f"ERRO: Chave {e} não encontrada no ficheiro {file_name}.")
                except Exception as e:
                    print(f"ERRO inesperado ao ler {file_name}: {e}")

    if new_data_list:
        print("\nA unir os novos dados ao dataset principal...")
        new_df = pd.concat(new_data_list, axis=0)
        final_df = pd.concat([base_df, new_df], axis=0)
        
        final_df.to_csv(CSV_DESTINO, index=False)
        
        print("\n Dataset HUST processado e adicionado com sucesso!")
        print(f"Novas falhas injetadas: {new_df['fault'].unique()}")
    else:
        print("\n⚠️ Nenhum ficheiro processado. Verifique a pasta.")

if __name__ == '__main__':
    processar_hust()