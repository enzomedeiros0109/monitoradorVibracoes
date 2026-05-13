import serial
import time
import struct
import numpy as np
import math
import serial, time, struct, numpy as np, math

COM_PORT = 'COM9' 
BAUD_RATE = 921600
SAMPLE_RATE = 1000
HEADER = b'\xAA\xBB'

def iniciar_simulador():
    print(f"Iniciando Simulador ESP32 na porta {COM_PORT}...")
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE)
    except Exception as e:
        print(f"Erro ao abrir porta {COM_PORT}. Você criou as portas virtuais? Erro: {e}")
        return

    while True:
        try:
            # 1. Envia sinal de que está vivo
            print("Enviando 'PRONTO' e aguardando app.py conectar...")
            ser.write(b'PRONTO\n')
            
            # 2. Aguarda o comando 'S' (Start) do app.py
            if ser.in_waiting > 0:
                comando = ser.read(1)
                if comando == b'S':
                    print("Comando 'S' recebido. Iniciando transmissão a 1000Hz...")
                    transmitir_dados(ser)
            time.sleep(1)
            
        except serial.SerialException:
            print("Conexão perdida. Reiniciando...")
            time.sleep(2)
        except KeyboardInterrupt:
            print("Simulador encerrado pelo usuário.")
            ser.close()
            break

def transmitir_dados():
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE)
        print(f"Simulador rodando em {COM_PORT}...")
    except: return print("Erro: Porta ocupada ou inexistente.")

    t = 0.0
    estado = 0 
    ultima_troca = time.time()

    while True:
        if ser.in_waiting > 0:
            cmd = ser.read(1)
            if cmd == b'S': estado = 1
            if cmd == b'C': break
        else:
            ser.write(b'PRONTO\n')
            time.sleep(0.5)
            if ser.in_waiting == 0: continue

        agora = time.time()
        if agora - ultima_troca > 10:
            estado = 1 if estado == 2 else 2 # Alterna entre Normal e Falha
            ultima_troca = agora
            print(f"Mudando para: {'FALHA' if estado == 2 else 'NORMAL'}")

        # Simulação mais agressiva para a IA detectar
        noise = np.random.normal(0, 0.05)
        if estado == 1: # Normal
            y = 0.2 * math.sin(2 * math.pi * 60 * t) + noise
        else: # Falha (Picos de impacto que geram Kurtosis > 10)
            impacto = 5.0 if (t % 0.1) < 0.005 else 0.0
            y = 0.2 * math.sin(2 * math.pi * 60 * t) + noise + impacto

        ser.write(HEADER + struct.pack('<3f', 0.0, float(y), 0.0))
        t += 1.0/SAMPLE_RATE
        time.sleep(0.0009) # Estabiliza 1kHz

if __name__ == '__main__': transmitir_dados()