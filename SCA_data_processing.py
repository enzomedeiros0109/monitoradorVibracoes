import pandas as pd
import os
from scipy.io import loadmat
from scipy import signal
import numpy as np

csv_path = "sample_data/all_faults.csv"
sample_target = 1000
base_df = pd.read_csv(csv_path)
new_data_list = []

def processar_dados():
    for root, dirs, files in os.walk("dados_vibracao_defeitos/SCA", topdown=False):
        for file_name in files:
            if file_name.endswith(".mat"):
                path = os.path.join(root, file_name)
                print(f"Extraindo de {path}: {file_name}")

                try:

                    mat = loadmat(path)
                    #   DS (Lado do Motor) e FS (Lado Livre)
                    chaves_dados = ["DS", "FS"]

                    for chave in chaves_dados:
                            if chave not in mat:
                                continue

                    #mat["DS"] ou mat["FS"]
                    estrutura = mat[chave][0][0]
                    nomes_gavetas = estrutura.dtype.names
                    
                    # Pega o índice em que a 'gaveta' equivale ao nome dentro do parenteses

                    sinais_vibracao = estrutura['rawData']
                    taxas_amostragem = estrutura['samplingRate'][0]
                    labels = estrutura['label'][0]

                    dias_validos = 0

                    for i in range(len(labels)):

                        label = int(labels[i])
                        if label == -1:
                            continue

                        fs = float(taxas_amostragem[i])
                        if fs == 0:
                            continue

                        sinal_do_dia = sinais_vibracao[i]

                        duracao_sinal = len(sinal_do_dia) / fs
                        num_amostras_alvo = int(duracao_sinal*sample_target)
                        sinal_1kHz = signal.resample(sinal_do_dia, num_amostras_alvo)

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

                        #print(f"Shape do sinal original {fs}: {fs.shape}")
                        #print(f"Shape do novo sinal (1kHz): {sinal_1kHz.shape}")

                        fault_col = np.full((len(sinal_1kHz), 1), fault)

                        processed_df = pd.DataFrame({'DE_data': np.ravel(sinal_1kHz), 'fault': np.ravel(fault_col)})
                        new_data_list.append(processed_df)
                                
                        dias_validos += 1

                        print(f"Sensor: {chave} | Medições processados: {dias_validos} | {fs} -> 1000")
                        # print(f"FS: {fs}\n\n")
                        # print(f"DS: {ds}")
                    
                except Exception as e:
                    print(f"Um erro ocorreu a tentar ler o arquivo {file_name} em {path}")

    if new_data_list:
        print("\nAdicionando à all_faults.csv")
        new_df = pd.concat(new_data_list, axis=0)
        final_df = pd.concat([base_df, new_df], axis=0)
        
        final_df.to_csv(csv_path, index=False)
        
        print("\nDataset processado e adicionado com sucesso!")
        print(f"Novas falhas injetadas: {new_df['fault'].unique()}")
    else:
        print("\nNenhum arquivo processado.")

if __name__ == '__main__':
    processar_dados()