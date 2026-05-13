import scipy.io as sio
import numpy as np

files = [
    r'dados_vibracao_defeitos\SCA\7\train.mat', 
    #r'dados_vibracao_defeitos\HUST\B500.mat', 
    #r'dados_vibracao_defeitos\CWRU\B007_0.mat'
]

for f in files:
    try:
        # squeeze_me=True remove dimensões unitárias redundantes do MATLAB
        mat_data = sio.loadmat(f, squeeze_me=True)
        print(f"--- Análise do arquivo {f} ---")
        for key, value in mat_data.items():
            if not key.startswith('__'): # Ignora variáveis de sistema do MATLAB
                shape = getattr(value, 'shape', 'Escalar')
                if type(value) == np.ndarray:
                    shape = value.shape
                elif isinstance(value, (int, float, str)):
                    shape = 'Escalar'
                    
                print(f"Variável: {key} | Grandeza/Shape: {shape}")
                
                try:
                    if isinstance(value, np.ndarray):
                        if value.dtype.names is not None:
                            print(f"  Array Estruturado com campos: {value.dtype.names}")
                            for name in value.dtype.names:
                                # Mostra a primeira amostra do campo para evitar poluição
                                print(f"    Campo: {name} | Amostra: {value[name][()]}")
                        else:
                            print(f"  5 Amostras: {value.flatten()[:5]}")
                    elif isinstance(value, (int, float, str)):
                        print(f"  Amostra (escalar): {value}")
                except Exception as e:
                    print(f"  Erro ao extrair amostras: {e}")
        print("\n")
    except Exception as e:
        print(f"Erro ao carregar {f}: {e}")