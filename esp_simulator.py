import os
import glob
import serial
import time
import struct
import numpy as np
import scipy.io
from scipy import signal
from math import gcd

COM_PORT = 'COM9' 
BAUD_RATE = 921600
HEADER = b'\xAA\xBB'
TAXA_ALVO = 1000

mapa_sca = {0: 'SCA_Normal', 1: 'SCA_AnelInterno', 2: 'SCA_Bola', 3: 'SCA_AnelExterno'}

def carregar_arquivos_reais():
    print("Carregando e reamostrando arquivos reais (.mat) para 1kHz. Aguarde...")
    sinais = []
    
    # 1. Carregar do CWRU
    arquivos_cwru = glob.glob("dados_vibracao_defeitos/CWRU/*.mat")[:5]
    for f in arquivos_cwru:
        try:
            mat = scipy.io.loadmat(f)
            key = next((k for k in mat.keys() if 'DE_time' in k), None)
            if key:
                sinal = mat[key].flatten()
                g = gcd(TAXA_ALVO, 12000)
                sinais.append((f"[CWRU] Arquivo: {os.path.basename(f)}", signal.resample_poly(sinal, TAXA_ALVO // g, 12000 // g)))
        except: pass

    # 2. Carregar do HUST
    arquivos_hust = glob.glob("dados_vibracao_defeitos/HUST/*.mat")[:5]
    for f in arquivos_hust:
        try:
            mat = scipy.io.loadmat(f)
            if 'data' in mat:
                sinal = mat['data'].flatten()
                g = gcd(TAXA_ALVO, 51200)
                sinais.append((f"[HUST] Arquivo: {os.path.basename(f)}", signal.resample_poly(sinal, TAXA_ALVO // g, 51200 // g)))
        except: pass

    # 3. Carregar do SCA (Buscando diversidade: Falhas e Normais)
    pastas_sca = [p.path for p in os.scandir("dados_vibracao_defeitos/SCA") if p.is_dir()]
    labels_encontrados = set() # Vamos tentar achar um de cada tipo!
    
    for pasta in pastas_sca:
        for tipo_arq in ["train.mat", "test.mat"]:
            f = os.path.join(pasta, tipo_arq)
            if os.path.exists(f):
                try:
                    mat = scipy.io.loadmat(f)
                    for sensor in ['DS', 'FS', 'Upper', 'Lower']:
                        if sensor in mat:
                            dados = mat[sensor][0, 0]
                            vibs, taxas, labels = dados[3], dados[4][0], dados[7][0]
                            
                            for i in range(vibs.shape[0]):
                                lbl = int(np.ravel(labels[i])[0])
                                # Só adiciona se for um defeito novo que ainda não pegamos, ou se tiver poucos
                                if lbl != -1 and lbl not in labels_encontrados:
                                    sinal = np.ravel(vibs[i]).astype(float)
                                    tx = int(np.ravel(taxas[i])[0])
                                    g = gcd(TAXA_ALVO, tx)
                                    sinal_1k = signal.resample_poly(sinal, TAXA_ALVO // g, tx // g)
                                    nome_falha = mapa_sca.get(lbl, f"SCA_Erro_{lbl}")
                                    sinais.append((f"[SCA] {tipo_arq} - Pasta {os.path.basename(pasta)} | {nome_falha}", sinal_1k))
                                    labels_encontrados.add(lbl)
                                    break 
                except: pass
        if len(labels_encontrados) >= 4: # Já achou os 4 tipos de label do SCA
            break

    print(f"Pronto! {len(sinais)} arquivos reais (com defeitos variados) carregados.\n")
    return sinais

def iniciar_simulador():
    sinais_reais = carregar_arquivos_reais()
    if not sinais_reais:
        print("Erro: Nenhum dado carregado. Verifique os caminhos das pastas.")
        return

    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE)
        print(f"Simulador aguardando conexão do app na porta {COM_PORT}...")
    except Exception as e:
        print(f"Erro ao abrir porta {COM_PORT}: {e}")
        return

    estado = 0 # 0 = Esperando, 1 = Transmitindo
    idx_sinal = 0

    while True:
        # Handshake
        if ser.in_waiting > 0:
            cmd = ser.read(1)
            if cmd == b'S':
                estado = 1
                print("\n*** CONEXÃO ESTABELECIDA. Iniciando injeção de dados... ***")
            elif cmd == b'C':
                estado = 0
                print("\nApp desconectou. Aguardando novo ciclo...")

        if estado == 0:
            ser.write(b'PRONTO\n')
            time.sleep(0.5)
            continue

        # == ESTADO DE TRANSMISSÃO ==
        nome_arquivo, array_sinal = sinais_reais[idx_sinal]
        tamanho_sinal = len(array_sinal)
        pos = 0
        t_inicio = time.time()
        
        print(f"\n>> Transmitindo agora: {nome_arquivo} <<")
        
        # Fica injetando o mesmo arquivo repetidamente por 15 segundos
        while time.time() - t_inicio < 15.0:
            if ser.in_waiting > 0:
                cmd = ser.read(1)
                if cmd == b'C': 
                    estado = 0
                    break
            
            y_val = array_sinal[pos]
            ser.write(HEADER + struct.pack('<3f', 0.0, float(y_val), 0.0))
            
            pos += 1
            if pos >= tamanho_sinal: 
                pos = 0 # Dá um loop no array se ele acabar antes dos 15s
                
            time.sleep(0.0008) # ~1000 Hz

        # Avança para o próximo arquivo do Dataset
        idx_sinal = (idx_sinal + 1) % len(sinais_reais)

if __name__ == '__main__':
    iniciar_simulador()