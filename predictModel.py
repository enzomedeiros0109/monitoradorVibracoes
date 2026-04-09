import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from collections import Counter
import joblib
import os

# Diretório da "memória" da máquina
MODEL_PATH = './ML/modelo_rf.joblib'
SCALER_PATH = './ML/scaler.joblib'
IDX_PATH = './ML/features_idx.joblib'

dataset_path = './sample_data/dataset_de_features.csv'

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(IDX_PATH):
    print(f"Carregando arquivo de {MODEL_PATH}")
    rf2 = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    idx = joblib.load(IDX_PATH)

elif os.path.exists(dataset_path):
    print("Arquivo de ML não encontrado. Começando treinamento do zero.")
    os.makedirs("dados_vibracao_defeitos", exist_ok=True)
    os.makedirs("ML", exist_ok=True)
    os.makedirs("sample_data", exist_ok=True)
    
    df_sample = pd.read_csv(dataset_path)

    X = df_sample.drop('fault', axis=1).values
    y = df_sample['fault']

    contagem_classes = df_sample['fault'].value_counts()
    
    # Define o número mínimo de amostras por falha
    MIN_AMOSTRAS = 2
    
    # Filtra apenas as classes que têm dados suficientes
    classes_validas = contagem_classes[contagem_classes >= MIN_AMOSTRAS].index
    
    # Cria um novo DataFrame apenas com essas classes
    df_filtrado = df_sample[df_sample['fault'].isin(classes_validas)]
    
    # Imprime no console quais classes foram removidas
    classes_removidas = set(df_sample['fault']) - set(df_filtrado['fault'])
    if classes_removidas:
        print(f"Aviso: As seguintes classes foram removidas por terem menos de {MIN_AMOSTRAS} amostras: {classes_removidas}")

    X = df_filtrado.drop('fault', axis=1).values
    y = df_filtrado['fault']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.25, 
        random_state=42, 
        stratify=y
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 1º Modelo: Seleção das características mais importantes (Feature Selection)
    rf = RandomForestClassifier(n_estimators=500, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_

    # Seleciona as 200 frequências mais relevantes
    idx = np.argsort(importances)[-200:]

    X_train_sel = X_train[:, idx]
    X_test_sel = X_test[:, idx]

    # 2º Modelo: Treinamento final focado apenas nos dados relevantes
    rf2 = RandomForestClassifier(n_estimators=500, max_depth=15, random_state=42, n_jobs=-1)
    rf2.fit(X_train_sel, y_train)
    
    y_pred = rf2.predict(X_test_sel)
    accuracy = accuracy_score(y_test, y_pred) * 100
    print(f"Acurácia do modelo: {accuracy:.4f}%")

    # Salva o aprendizado da máquina
    joblib.dump(rf2, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(idx, IDX_PATH)
else:
    scaler, rf2, idx = None, None, None

def erroLabel(erro):
    # (Sua função erroLabel original permanece inalterada - lógica de string está ótima!)
    erro = str(erro)
    erroProv, diametroFalha, cargaMotor = '', 'N/A', 'N/A'
    
    try:
        if erro.startswith('SCA_'):
            if erro == "SCA_Normal": erroProv = 'Vibração normal'
            elif erro == "SCA_AnelInterno": erroProv = 'Anel Interno'
            elif erro == "SCA_Bola": erroProv = 'Bola'
            elif erro == "SCA_AnelExterno": erroProv = 'Anel Externo'
            else: erroProv = 'Desconhecido'

        elif '_' in erro: # CWRU dataset prefix
            indexUnderscore = erro.index('_')
            if 'normal' in erro: erroProv = 'Vibração normal'
            elif 'IR' in erro: erroProv = 'Pista interna'
            elif 'OR' in erro:
                if '@' in erro:
                    indexArrob = erro.index('@')
                    positionRelative = erro[indexArrob+1:indexUnderscore]
                    if positionRelative == '6': erroProv = 'Pista externa, Centrada'
                    elif positionRelative == '3': erroProv = 'Pista externa, Ortogonal'
                    elif positionRelative == '12': erroProv = 'Pista externa, Oposta'
                else:
                    erroProv = 'Pista externa'
            elif 'B' in erro: erroProv = 'Bola'

            if '007' in erro: diametroFalha = '0.007 polegadas'
            elif '014' in erro: diametroFalha = '0.014 polegadas'
            elif '021' in erro: diametroFalha = '0.021 polegadas'
            elif '028' in erro: diametroFalha = '0.028 polegadas'

            cargaMotor = erro[indexUnderscore:]
            carga = cargaMotor[1] if len(cargaMotor) > 1 else "N/A"
            return f"ERRO PROVÁVEL: {erroProv} | DIAMETRO DA FALHA: {diametroFalha} | CARGA DO MOTOR: {carga}"

        else: # HUST dataset prefix
            if 'IB' in erro: erroProv = 'Rachadura interna e Bola'
            elif 'IO' in erro: erroProv = 'Rachadura interna e externa'
            elif 'OB' in erro: erroProv = 'Rachadura externa e Bola'
            elif 'N' in erro: erroProv = 'Vibração normal'
            elif 'I' in erro: erroProv = 'Rachadura interna'
            elif 'O' in erro: erroProv = 'Rachadura externa'
            elif 'B' in erro: erroProv = 'Rachadura Bola'
        
        return f"ERRO PROVÁVEL: {erroProv} | DIAMETRO DA FALHA: {diametroFalha} | CARGA DO MOTOR: {cargaMotor}"
    except Exception as e:
        return f"ERRO PROVÁVEL: {erro} | DIAMETRO DA FALHA: N/A | CARGA DO MOTOR: N/A"

def previsao(dados):
    # Transforma e seleciona as features
    X_data = scaler.transform(dados)
    X_data_sel = X_data[:, idx]
    
    # Faz as 15 previsões
    y_pred_data = rf2.predict(X_data_sel)
    contagem = Counter(y_pred_data)
    
    # Define a classe vencedora (votação majoritária)
    classe_vencedora = contagem.most_common(1)[0][0]
    
    # Extrai as matrizes de probabilidade
    probs = rf2.predict_proba(X_data_sel)
    
    # Descobre qual é a coluna (índice) correspondente à classe vencedora no modelo
    indice_classe = np.where(rf2.classes_ == classe_vencedora)[0][0]
    
    # Calcula a confiança MÉDIA apenas dos votos da classe que ganhou
    confidence = np.mean(probs[:, indice_classe]) * 100
    
    resultado = f"{erroLabel(classe_vencedora)} ; confiança = {confidence:.2f}%"
    return resultado