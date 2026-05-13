import sys
import serial
import time
import struct
import numpy as np
import pyqtgraph as pg
import joblib
from scipy import stats
from datetime import datetime
from PyQt6 import QtWidgets, QtCore

COM_PORT = 'COM8' # PORTA ESP32 (Par do simulador)
BAUD_RATE = 921600
BUFFER_SIZE = 2048
SAMPLE_RATE = 1000

try:
    scaler = joblib.load('./sample_data/scaler.pkl')
    modelo_rf = joblib.load('./sample_data/modelo_rf.pkl')
    print("Sucesso: Modelos de IA carregados no app.py!")
except Exception as e:
    print(f"AVISO: Arquivos .pkl não encontrados. Treine o modelo primeiro. Erro: {e}")
    scaler, modelo_rf = None, None

def processamento(data_buffer):
    N = len(data_buffer)
    if N == 0 or np.all(data_buffer == 0):
        return np.zeros(N//2)
    
    data_buffer_ac = data_buffer - np.mean(data_buffer)
    window = np.hanning(N)
    windowed_data = data_buffer_ac * window
    fft_result_complex = np.fft.rfft(windowed_data)
    fft_magnitudes_raw = np.abs(fft_result_complex)
    scaling_factor = N / 4.0
    return fft_magnitudes_raw / scaling_factor if scaling_factor > 0 else fft_magnitudes_raw

def decodificar_label(erro):
    """Traduz o label interno da IA para um texto único e limpo"""
    erro = str(erro)
    if erro.startswith('SCA_'):
        mapa = {'SCA_Desligada': 'Máquina desligada', 'SCA_Normal': 'Vibração normal', 
                'SCA_AnelInterno': 'Falha: Anel Interno', 'SCA_Bola': 'Falha: Esfera',
                'SCA_AnelExterno': 'Falha: Anel Externo'}
        return f"[SCA] {mapa.get(erro, erro)}"
    elif erro.startswith('CWRU_'):
        mapa = {'CWRU_Normal': 'Vibração Normal', 'CWRU_IR': 'Falha: Pista Interna',
                'CWRU_B': 'Falha: Esfera', 'CWRU_OR': 'Falha: Pista Externa'}
        return f"[CWRU] {mapa.get(erro, erro)}"
    elif erro.startswith('HUST_'):
        return f"[HUST] Falha: {erro.replace('HUST_', '')}"
    return erro

class SerialWorker(QtCore.QObject):
    dataReady = QtCore.pyqtSignal(np.ndarray, np.ndarray, np.ndarray)
    predictionReady = QtCore.pyqtSignal(str, str)
    errorOccurred = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._is_running = True
        self.data_buffer_x = np.zeros(BUFFER_SIZE, dtype=np.float64)
        self.data_buffer_y = np.zeros(BUFFER_SIZE, dtype=np.float64)
        self.data_buffer_z = np.zeros(BUFFER_SIZE, dtype=np.float64)
        self.current_index = 0
        self.ser = None

    def stop(self):
        self._is_running = False

    def run(self):
        try:
            print("Worker: Abrindo porta serial...")
            self.ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
            time.sleep(2)

            print("Worker: Esperando o ESP 32...")
            while self._is_running:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if line == 'PRONTO':
                    print("Worker: ESP32 pronto!")
                    self.ser.write(b'S')
                    break
                elif not self._is_running:
                    if self.ser and self.ser.is_open:
                        self.ser.close()
                    return
                
            self.ser.flush()
            print("Worker: Lendo dados...")

            while self._is_running:
                if self.ser.in_waiting >= 14: 
                    byte1 = self.ser.read(1)
                    if byte1 == b'\xAA':
                        byte2 = self.ser.read(1)
                        if byte2 == b'\xBB':
                            data = self.ser.read(12)
                            
                            if len(data) == 12:
                                try:
                                    x, y, z = struct.unpack('<3f', data)
                                    
                                    if not np.isfinite(y): y = 0.0
                                    
                                    if self.current_index < BUFFER_SIZE:
                                        self.data_buffer_x[self.current_index] = x
                                        self.data_buffer_y[self.current_index] = y
                                        self.data_buffer_z[self.current_index] = z
                                        self.current_index += 1

                                    if self.current_index >= BUFFER_SIZE:
                                        self.current_index = 0
                                        
                                        # 1. Gráficos
                                        fx = processamento(self.data_buffer_x)[1:]
                                        fy = processamento(self.data_buffer_y)[1:]
                                        fz = processamento(self.data_buffer_z)[1:]
                                        self.dataReady.emit(fx, fy, fz)

                                        # 2. Inteligência Artificial
                                        buffer_ac = self.data_buffer_y - np.mean(self.data_buffer_y)
                                        rms = np.sqrt(np.mean(buffer_ac**2))
                                        hora = datetime.now().strftime("%H:%M:%S")

                                        if rms < 0.005:
                                            self.predictionReady.emit("Máquina parada", hora)
                                        elif modelo_rf is not None and scaler is not None:
                                            std = np.std(buffer_ac)
                                            skew = float(stats.skew(buffer_ac, bias=False))
                                            kurtosis = float(stats.kurtosis(buffer_ac, bias=False))
                                            pico_pico = np.max(buffer_ac) - np.min(buffer_ac)
                                            fator_crista = np.max(np.abs(buffer_ac)) / rms
                                            
                                            fft_vals = np.abs(np.fft.rfft(buffer_ac))
                                            freqs = np.fft.rfftfreq(BUFFER_SIZE, d=1.0/SAMPLE_RATE)
                                            
                                            energia_0_100 = np.sum(fft_vals[(freqs >= 0) & (freqs < 100)]**2)
                                            energia_100_300 = np.sum(fft_vals[(freqs >= 100) & (freqs < 300)]**2)
                                            energia_300_500 = np.sum(fft_vals[(freqs >= 300) & (freqs <= 500)]**2)
                                            
                                            features = np.array([[rms, std, skew, kurtosis, pico_pico, fator_crista, energia_0_100, energia_100_300, energia_300_500]])
                                            
                                            features_scaled = scaler.transform(features)
                                            pred_label = modelo_rf.predict(features_scaled)[0]
                                            confianca = np.max(modelo_rf.predict_proba(features_scaled)) * 100
                                            
                                            texto_traduzido = decodificar_label(pred_label)
                                            texto_final = f"{texto_traduzido} | Confiança: {confianca:.1f}%"

                                            self.predictionReady.emit(texto_final, hora)
                                except struct.error:
                                    pass

            print("Worker: Loop encerrado. Enviando comando CLOSE...")
            if self.ser and self.ser.is_open:
                self.ser.write(b'C')
                self.ser.flush()

        except serial.SerialException as e:
            self.errorOccurred.emit(f"Erro na porta serial: {e}")
        finally:
            if self.ser and self.ser.is_open:
                self.ser.close()
            print("Worker: Thread encerrada.")

class VibrationAnalyzer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Monitorador de Vibração (IA Integrada)")
        self.setGeometry(100, 100, 1000, 800)

        pg.setConfigOption('background', '#111')
        pg.setConfigOption('foreground', 'w')
        pg.setConfigOption('antialias', True)

        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QtWidgets.QVBoxLayout(self.central_widget)

        self.start_button = QtWidgets.QPushButton("Iniciar monitoramento")
        font = self.start_button.font()
        font.setPointSize(16)
        self.start_button.setFont(font)
        self.start_button.clicked.connect(self.start_monitoring)
        self.main_layout.addWidget(self.start_button)

        self.graph_container = QtWidgets.QWidget()
        self.graph_layout = QtWidgets.QVBoxLayout(self.graph_container)
        self.main_layout.addWidget(self.graph_container)

        self.result_label = QtWidgets.QLabel("Aguardando dados...")
        font = self.result_label.font()
        font.setPointSize(16)
        font.setBold(True)
        self.result_label.setFont(font)
        self.result_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("color: #AAAAAA; padding: 20px;")
        self.main_layout.addWidget(self.result_label)

        self.freq_axis = np.fft.rfftfreq(BUFFER_SIZE, d=1.0 / SAMPLE_RATE)[1:]
        self.thread = None
        self.worker = None

    def start_monitoring(self):
        self.start_button.setEnabled(False)
        self.start_button.setText("Monitorando...")

        self.plot_y, self.line_y = self.create_plot("Eixo Y", color='#2ECC40')
        self.graph_layout.addWidget(self.plot_y)
        
        self.thread = QtCore.QThread()
        self.worker = SerialWorker()
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.dataReady.connect(self.update_graphs)
        self.worker.errorOccurred.connect(self.show_error)
        self.worker.predictionReady.connect(self.update_prediction_label)

        self.thread.start()

    @QtCore.pyqtSlot(np.ndarray, np.ndarray, np.ndarray)
    def update_graphs(self, fft_mags_x, fft_mags_y, fft_mags_z):
        if len(self.freq_axis) == len(fft_mags_y):
            self.line_y.setData(self.freq_axis, fft_mags_y)

    @QtCore.pyqtSlot(str, str)
    def update_prediction_label(self, texto, hora):
        if "parada" in texto.lower() or "normal" in texto.lower():
            cor = "#2ECC40" # Verde
        else:
            cor = "#FF4136" # Vermelho
            
        self.result_label.setStyleSheet(f"color: {cor}; padding: 20px;")
        self.result_label.setText(f"[{hora}] {texto}")

    def create_plot(self, title, color):
        plot_widget = pg.PlotWidget()
        plot_item = plot_widget.getPlotItem()
        plot_item.setTitle(f"Espectro de Frequência - {title}", size="16pt")
        plot_item.setLabel('left', 'Amplitude (g)')
        plot_item.setLabel('bottom', 'Frequência (Hz)')
        plot_item.showGrid(x=True, y=True, alpha=0.3)
        plot_item.setYRange(0, 0.5)
        
        line = plot_widget.plot([], [], pen=pg.mkPen(color=color, width=2))
        return plot_widget, line
    
    def show_error(self, error_message):
        print(f"Erro: {error_message}")
        self.start_button.setText("Falha na Conexão")
        msgBox = QtWidgets.QMessageBox()
        msgBox.setIcon(QtWidgets.QMessageBox.Icon.Critical)
        msgBox.setText(error_message)
        msgBox.exec()

    def closeEvent(self, event):
        print("Fechando aplicação...")
        if self.worker:
            self.worker.stop()
        if self.thread:
            self.thread.quit()
            self.thread.wait()
        event.accept()