import scipy.io
import numpy as np

def listar_variaveis_mat(caminho_arquivo):
    try:
        # Carrega o arquivo .mat
        mat_data = scipy.io.loadmat(caminho_arquivo)
        
        # Filtra as chaves para remover os metadados do arquivo
        variaveis = [chave for chave in mat_data.keys() if not chave.startswith('__')]
        
        print(f"Variáveis e primeiros valores no arquivo '{caminho_arquivo}':\n")
        for var in variaveis:
            print(f"--- {var} ---")
            dados = mat_data[var]
            
            try:
                # Transforma em 1D e pega até os 15 primeiros elementos
                valores = np.ravel(dados)[:15]
                print(valores)
            except Exception as e:
                print(f"Não foi possível ler os valores: {e}")
            print("") # Linha em branco para separar
            
    except FileNotFoundError:
        print("Erro: O arquivo especificado não foi encontrado.")
    except Exception as e:
        print(f"Ocorreu um erro ao tentar ler o arquivo: {e}")

# Diretório ajustado com barras normais (funciona em Windows, Mac e Linux)
listar_variaveis_mat('dados_vibracao_defeitos\HUST\B500.mat')