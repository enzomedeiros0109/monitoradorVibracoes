import sys
import serial
import time
import struct
import pandas as pd
import numpy as np
from PyQt6 import QtWidgets
import app

class VirtualESP32:
    def __init__(self, port, baudrate, timeout=1):
        print(f"\n[ESP32 VIRTUAL] \nConectado na porta virtual {port} a {baudrate} bps")
        self.is_open = True
        self.running = False
        self.sent_ready = False
        self.last_fault = None

        print("Carregando all_faults.csv")        
        try:
            df = pd.read_csv('sample_data/all_faults.csv')
            df_turbo = df.groupby('fault').head(4096)
            self.dados_y = df_turbo['DE_data'].values
            self.labels = df_turbo['fault'].values

            print(f"{len(self.dados_y)} dados prontos.")
        except Exception as e:
            print(f"Erro ao ler CSV: {e}.")
        
        self.index = 0

    def readline(self):
        if not self.sent_ready:
            time.sleep(2)
            self.sent_ready = True
            return b'PRONTO\n'
        return b''
    
    def write(self, command):
        if command == b'S':
            print("Comando 'S' Start recebido. Iniciando envio.")
            
            self.running = True
        elif command == b'C':
            print("Commando 'C' Close recebido. Encerrando envio.")
            self.running = False

    def flush(self):
        pass

    def read(self, size=12):
        if not self.running:
            time.sleep(0.1)
            return b''
        
        if self.index >= len(self.dados_y):
            self.index = 0

        y_val = float(self.dados_y[self.index])
        current_fault = self.labels[self.index]

        if current_fault != self.last_fault:
            print(f"Inserindo nova falha mecânica: {current_fault}")
            self.last_fault = current_fault
            

        self.index += 1

        x_val = float(np.random.normal(0, 0.01))
        z_val = float(np.random.normal(0, 0.01))

        return struct.pack('<3f', x_val, y_val, z_val)
        
    def close(self):
        self.is_open = False
        print("Conexão encerrada")

serial.Serial = VirtualESP32

if __name__ == '__main__':
    print("Iniciando ambiente simulado")

    app_qt = QtWidgets.QApplication(sys.argv)
    main_window = app.VibrationAnalyzer()
    main_window.show()

    sys.exit(app_qt.exec())