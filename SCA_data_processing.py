import pandas as pd
import os
from scipy.io import loadmat
from scipy import signal
import numpy as np
import math

csv_path = "sample_data/all_faults.csv"
sample_target = 1000
new_data_list = []

if os.path.exists(csv_path):
    base_df = pd.read_csv(csv_path)
    print(f"Dataset base encontrado com {len(base_df)} amostras.")
else:
    base_df = pd.DataFrame()

def processar_dados():
    for root, dirs, files in os.walk("dados_vibracao_defeitos/SCA", topdown=False):
        for file_name in files:
            if file_name.endswith(".mat"):
                path = os.path.join(root, file_name)
                print(f"Extraindo de {path}: {file_name}")

                try:
                    mat = loadmat(path)
                    chaves_dados = ["DS", "FS"]

                    for chave in chaves_dados:
                        if chave not in mat:
                            continue

                        estrutura = mat[chave][0][0]
                        sinais_vibracao = estrutura['rawData']
                        taxas_amostragem = estrutura['samplingRate'][0]
                        labels = estrutura['label'][0]

                        dias_validos = 0

                        for i in range(len(labels)):
                            label = int(labels[i])
                            if label == -1:
                                continue

                            fs = float(taxas_amostragem[i])
                            if fs <= 0:
                                continue

                            # Garante que o sinal é 1D
                            sinal_do_dia = sinais_vibracao[i].flatten()

                            # --- CÁLCULO DINÂMICO PARA RESAMPLE_POLY ---
                            fs_int = int(fs)
                            divisor_comum = math.gcd(sample_target, fs_int)
                            up_val = sample_target // divisor_comum
                            down_val = fs_int // divisor_comum

                            sinal_1kHz = signal.resample_poly(sinal_do_dia, up=up_val, down=down_val)
                            # -------------------------------------------

                            if label == 0:
                                fault = "SCA_Normal"
                            elif label == 1:
                                fault = "SCA_AnelInterno" 
                            elif label == 2:
                                fault = "SCA_Bola"
                            elif label == 3:
                                fault = "SCA_AnelExterno"
                            else:
                                fault = "SCA_Desconhecido"

                            fault_col = np.full((len(sinal_1kHz), 1), fault)

                            processed_df = pd.DataFrame({'DE_data': np.ravel(sinal_1kHz), 'fault': np.ravel(fault_col)})
                            new_data_list.append(processed_df)
                                    
                            dias_validos += 1

                        if dias_validos > 0:
                            print(f"  -> Sensor: {chave} | Medições processadas: {dias_validos} | {fs_int}Hz -> 1000Hz")
                            
                except Exception as e:
                    print(f"Erro ocorreu ao tentar ler o arquivo {file_name}: {e}")

    if new_data_list:
        print("\nAdicionando à all_faults.csv")
        # ignore_index garante a integridade das linhas
        new_df = pd.concat(new_data_list, ignore_index=True)
        final_df = pd.concat([base_df, new_df], ignore_index=True)
        
        final_df.to_csv(csv_path, index=False)
        
        print("\nDataset SCA processado e adicionado com sucesso!")
        print(f"Novas falhas injetadas: {new_df['fault'].unique()}")
    else:
        print("\nNenhum arquivo processado.")

if __name__ == '__main__':
    processar_dados()