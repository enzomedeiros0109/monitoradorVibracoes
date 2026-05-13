import pandas as pd
import numpy as np
from scipy import stats

WINDOW_SIZE = 2048
STEP_SIZE = 1024
TAXA_ALVO = 1000 # Frequência MPU6050

df_raw = pd.read_csv('sample_data/all_faults.csv')
labels = df_raw['fault'].unique()

all_rows = []

def calcular_features(janela):
    janela_ac = janela - np.mean(janela)
    rms = np.sqrt(np.mean(janela_ac**2))
    
    if rms == 0: return None
    
    std = np.std(janela_ac)
    skew = float(stats.skew(janela_ac))
    kurtosis = float(stats.kurtosis(janela_ac))
    pico_pico = np.max(janela_ac) - np.min(janela_ac)
    fator_crista = np.max(np.abs(janela_ac)) / rms
    
    # Domínio da Frequência
    fft_vals = np.abs(np.fft.rfft(janela_ac))
    freqs = np.fft.rfftfreq(len(janela_ac), d=1.0/TAXA_ALVO)
    
    energia_0_100 = np.sum(fft_vals[(freqs >= 0) & (freqs < 100)]**2)
    energia_100_300 = np.sum(fft_vals[(freqs >= 100) & (freqs < 300)]**2)
    energia_300_500 = np.sum(fft_vals[(freqs >= 300) & (freqs <= 500)]**2)
    
    return [rms, std, skew, kurtosis, pico_pico, fator_crista, energia_0_100, energia_100_300, energia_300_500]

print("Iniciando extração de features...")

for label in labels:
    # Pega todo o sinal daquela falha
    signal_segment = df_raw[df_raw['fault'] == label]['DE_data'].values

    # Aplica o janelamento em todo o sinal contínuo
    for i in range(0, len(signal_segment) - WINDOW_SIZE, STEP_SIZE):
        window = signal_segment[i:i + WINDOW_SIZE]

        if len(window) < WINDOW_SIZE or np.std(window) == 0:
            continue

        features = calcular_features(window)
        
        if features:
            row = features + [label]
            all_rows.append(row)

print(f"Extração concluída. Total de {len(all_rows)} janelas processadas.")

if not all_rows:
    print("Nenhum dado foi processado. Encerrando")
else:
    # Nomes das colunas de features robustas
    feature_columns = [
        'RMS', 'Std', 'Skew', 'Kurtosis', 'Pico_Pico', 'FatorCrista', 
        'Energia_0_100', 'Energia_100_300', 'Energia_300_500', 'fault'
    ]
    
    df_features = pd.DataFrame(all_rows, columns=feature_columns)
    df_features.to_csv('sample_data/dataset_de_features.csv', index=False)

    print("Novo dataset 'dataset_de_features.csv' salvo com sucesso!")
    print(df_features.head())