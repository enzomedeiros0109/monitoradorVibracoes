import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from collections import Counter
import os

dataset_path = './sample_data/dataset_de_features.csv'

if os.path.exists(dataset_path):
    print("Iniciando o treinamento do modelo de Machine Learning do zero.")
    
    # exist_ok=True previne erros caso as pastas já existam
    os.makedirs("dados_vibracao_defeitos", exist_ok=True)
    os.makedirs("ML", exist_ok=True)
    os.makedirs("sample_data", exist_ok=True)
    
    df_sample = pd.read_csv(dataset_path)

    def agrupar_rotulos(rotulo):
        rotulo = str(rotulo)
        if rotulo.startswith('SCA_'):
            return rotulo
        elif '_' in rotulo:
            partes = rotulo.split('_')
            return partes[0] + '_'
        else:
            letras = ''.join([char for char in rotulo if char.isalpha()])
            return letras
        
    df_sample['fault'] = df_sample['fault'].apply(agrupar_rotulos)

    # Filtra as classes que têm pelo menos 2 amostras para não quebrar o stratify
    contagem_classes = df_sample['fault'].value_counts()
    classes_validas = contagem_classes[contagem_classes >= 2].index
    df_sample = df_sample[df_sample['fault'].isin(classes_validas)]

    X = df_sample.drop('fault', axis=1).values
    y = df_sample['fault']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.25, 
        random_state=42, 
        stratify=y
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Primeiro treinamento
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_

    # Seleciona as 200 features mais importantes
    idx = np.argsort(importances)[-200:]

    X_train_sel = X_train[:, idx]
    X_test_sel = X_test[:, idx]

    # Segundo treinamento com as 200 features
    rf2 = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )

    rf2.fit(X_train_sel, y_train)
    y_pred = rf2.predict(X_test_sel)
    accuracy = accuracy_score(y_test, y_pred) * 100
    print(f"Acurácia do modelo: {accuracy:.4f}%")

else:
    print(f"Erro: O arquivo '{dataset_path}' não foi encontrado.")
    scaler, rf2, idx = None, None, None

def erroLabel(erro):
    erro = str(erro)
    erroProv, diametroFalha, cargaMotor = '', 'N/A', 'N/A'
    
    try:
        # DATASET SCA
        if erro.startswith('SCA_'):
            if erro == 'SCA_Desligada': erroProv = 'Máquina desligada / Sem dados'
            elif erro == 'SCA_Normal': erroProv = 'Vibração normal'
            elif erro == 'SCA_AnelInterno': erroProv = 'Anel Interno'
            elif erro == 'SCA_Bola': erroProv = 'Bola'
            elif erro == 'SCA_AnelExterno': erroProv = 'Anel Externo'
            else: erroProv = 'Desconhecido'
            
            return f"[SCA] ERRO: {erroProv} | DIAMETRO: N/A | CARGA: N/A"

        # DATASET CWRU
        elif '_' in erro:
            indexUnderscore = erro.index('_')
            
            # Identificação do Defeito e Posição (CWRU)
            if 'normal' in erro.lower(): erroProv = 'Vibração normal'
            elif 'IR' in erro: erroProv = 'Pista interna'
            elif 'B' in erro: erroProv = 'Bola'
            elif 'OR' in erro:
                if '@6' in erro: erroProv = 'Pista externa (Centrada @6:00)'
                elif '@3' in erro: erroProv = 'Pista externa (Ortogonal @3:00)'
                elif '@12' in erro: erroProv = 'Pista externa (Oposta @12:00)'
                else: erroProv = 'Pista externa'

            # Identificação do Diâmetro da Falha (CWRU)
            if '007' in erro: diametroFalha = '0.007 polegadas'
            elif '014' in erro: diametroFalha = '0.014 polegadas'
            elif '021' in erro: diametroFalha = '0.021 polegadas'
            elif '028' in erro: diametroFalha = '0.028 polegadas'

            # Identificação da Carga do Motor (CWRU)
            carga = erro[indexUnderscore+1:] if len(erro) > indexUnderscore+1 else "N/A"
            
            return f"[CWRU] ERRO: {erroProv} | DIAMETRO: {diametroFalha} | CARGA: {carga} HP"

        # DATASET HUST
        else:
            if erro.startswith('IB'): erroProv = 'Rachadura Interna e Bola'
            elif erro.startswith('IO'): erroProv = 'Rachadura Interna e Externa'
            elif erro.startswith('OB'): erroProv = 'Rachadura Externa e Bola'
            elif erro.startswith('N'): erroProv = 'Vibração normal'
            elif erro.startswith('I'): erroProv = 'Rachadura Interna'
            elif erro.startswith('O'): erroProv = 'Rachadura Externa'
            elif erro.startswith('B'): erroProv = 'Rachadura Bola'
            else: erroProv = f'Desconhecido ({erro})'
            
            return f"[HUST] ERRO: {erroProv} | DIAMETRO: N/A | CARGA: N/A"

    except Exception as e:
        print(f"Erro ao converter label '{erro}': {e}")
        return f"ERRO PROVÁVEL: {erro} | DIAMETRO: N/A | CARGA: N/A"

def previsao(dados):
    if rf2 is None or scaler is None or idx is None:
        return "Erro: O modelo não foi treinado. Verifique o dataset."
        
    X_data = scaler.transform(dados)
    X_data_sel = X_data[:, idx]
    y_pred_data = rf2.predict(X_data_sel)
    contagem = Counter(y_pred_data)
    probs = rf2.predict_proba(X_data_sel)
    confidence = np.max(probs) * 100
    resultado = f"{erroLabel(contagem.most_common(1)[0][0])} ; confiança = {confidence:.2f}%"
    return resultado

# TESTE PRÁTICO #

def identifica_base(rotulo):
        rotulo = str(rotulo)
        if rotulo.startswith('SCA_'):
            return 'SCA'
        elif '_' in rotulo:
            return 'CWRU'
        else:
            return 'HUST'

def realizar_teste_pratico():
    print("\n" + "="*50)
    print(" INICIANDO TESTE COM AMOSTRAS (SCA, CWRU, HUST) ")
    print("="*50)
    
    # Carrega o dataset completo
    df = pd.read_csv('./sample_data/dataset_de_features.csv')
    
    # Separa os dados de cada máquina com base nas regras dos rótulos
    df['Origem'] = df['fault'].apply(identifica_base)
    
    # Agora a separação fica perfeita e blindada contra erros de tipagem
    df_sca = df[df['Origem'] == 'SCA']
    df_cwru = df[df['Origem'] == 'CWRU']
    df_hust = df[df['Origem'] == 'HUST']
    
    # Sorteia aleatoriamente 10 amostras de cada base (ou o máximo disponível)
    amostras_sca = df_sca.sample(n=min(10, len(df_sca))) if not df_sca.empty else df_sca
    amostras_cwru = df_cwru.sample(n=min(10, len(df_cwru))) if not df_cwru.empty else df_cwru
    amostras_hust = df_hust.sample(n=min(10, len(df_hust))) if not df_hust.empty else df_hust
    
    # Junta as 30 amostras sorteadas
    todas_amostras = pd.concat([amostras_sca, amostras_cwru, amostras_hust])
    
    # Faz o teste uma por uma
    for index, row in todas_amostras.iterrows():
        rotulo_real = row['fault']
        
        # Pega as features da linha e transforma num array 2D para a IA ler
        features = row.drop(['fault', 'Origem']).values.reshape(1, -1)
        
        # Faz a predição!
        resultado_ia = previsao(features)
        
        print(f"Original : {rotulo_real}")
        print(f"IA previu: {resultado_ia}")
        print("-" * 50)

if __name__ == "__main__":
    realizar_teste_pratico()