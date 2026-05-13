import scipy.io
import numpy as np
import pandas as pd
import os
from scipy import signal
from math import gcd

taxa_alvo = 1000

mapa_sca = {
    -1: 'SCA_Desligada',
    0: 'SCA_Normal',
    1: 'SCA_AnelInterno',
    2: 'SCA_Bola',
    3: 'SCA_AnelExterno'
}

lista_dfs = []

for root, dirs, files in os.walk("dados_vibracao_defeitos/SCA", topdown=False):
    for file_name in files:
        if not file_name.endswith('.mat'):
            continue
            
        path = os.path.join(root, file_name)
        print(f"\nProcessando: {path}")

        try:
            mat = scipy.io.loadmat(path)
        except Exception as e:
            print(f"Erro ao ler arquivo {file_name}: {e}")
            continue

        for sensor in ['DS', 'FS', 'Upper', 'Lower']:
            if sensor not in mat:
                continue
                
            try:
                dados_sensor = mat[sensor][0, 0]
                matriz_vibracao = dados_sensor[3]
                array_taxa_orig = dados_sensor[4][0]
                array_labels = dados_sensor[7][0]
            except Exception as e:
                print(f"Aviso: Estrutura inesperada na variável {sensor}. Pulando.")
                continue
            
            sinais_resampled = []
            labels_expandidos = []

            for i in range(matriz_vibracao.shape[0]):
                try:
                    sinal = matriz_vibracao[i]
                    if sinal.dtype == object:
                        sinal = np.hstack(sinal)
                    
                    # Força a ser array unidimensional e numérico
                    sinal = np.ravel(sinal).astype(float)
                    
                    # Força a extração de um único número mesmo se vier como array
                    taxa_orig = int(np.ravel(array_taxa_orig[i])[0])
                    label_num = int(np.ravel(array_labels[i])[0])
                    
                except Exception as e:
                    print(f"  -> Aviso: Linha {i} ignorada (dados corrompidos ou formato irregular)")
                    continue

                label_str = mapa_sca.get(label_num, f'SCA_Desconhecido_{label_num}')
                
                # Reamostragem
                duracao = len(sinal) / taxa_orig
                num_amostras = int(duracao * taxa_alvo)
                # upsample
                if taxa_orig != taxa_alvo:
                    g = gcd(taxa_alvo, taxa_orig)
                    sinal_1kHz = signal.resample_poly(sinal, taxa_alvo // g, taxa_orig // g)

                else:
                    # Se a frequência já for a alvo, pula o processamento pesado
                    sinal_1kHz = sinal
                
                sinais_resampled.append(sinal_1kHz)
                labels_expandidos.append(np.full(len(sinal_1kHz), label_str))

            if sinais_resampled:
                dados_finais = np.concatenate(sinais_resampled)
                labels_finais = np.concatenate(labels_expandidos)
                
                df_temp = pd.DataFrame({'DE_data': dados_finais, 'fault': labels_finais})
                lista_dfs.append(df_temp)
                
                print(f"- Sensor {sensor}: {len(sinais_resampled)} medições reamostradas com sucesso.")

if lista_dfs:
    df_final = pd.concat(lista_dfs, axis=0, ignore_index=True)
    os.makedirs('sample_data', exist_ok=True)
    
    df_final.to_csv('sample_data/all_faults.csv', mode='a', header=False, index=False)
    
    print(f"\nSucesso! Dataset SCA adicionado.")
    print(f"Rótulos inseridos na planilha: {df_final['fault'].unique()}")
else:
    print("\nNenhum dado válido encontrado para processar.")