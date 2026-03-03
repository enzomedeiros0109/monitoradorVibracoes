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

#Diretório da "memória" da máquina
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
    print("Arquivo de ML não encontrado. Começando ML do zero.")
    os.mkdir("ML")
    df_sample = pd.read_csv('./sample_data/dataset_de_features.csv')

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

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_

    idx = np.argsort(importances)[-200:]

    X_train_sel = X_train[:, idx]
    X_test_sel = X_test[:, idx]

    rf2 = RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )

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
    erro = str(erro)

    erroProv, diametroFalha, cargaMotor = '', 'N/A', 'N/A'
    
    try:
    
        # SCA Dataset prefix
        if erro.startswith('SCA_'):
            if erro == "SCA_Normal": erroProv = 'Vibração normal'
            elif erro == "SCA_AnelInterno": erroProv = 'AnelInterno'
            elif erro == "SCA_Bola": erroProv = 'Bola'
            elif erro == "SCA_AnelExterno": erroProv = 'PAnelExterno'
            else: erroProv = 'Desconhecido'

        #CWRU dataset prefix
        if '_' in erro:

            indexUnderscore = erro.index('_')
            if 'normal' in erro: erroProv = 'Vibração normal'
            elif 'IR' in erro: erroProv = 'Pista interna'
            elif 'OR' in erro:
                if '@' in erro:
                    indexArrob = erro.index('@')
                    # Correção extra: +1 para ignorar o caractere '@'
                    positionRelative = erro[indexArrob+1:indexUnderscore]
                    if positionRelative == '6': erroProv = 'Pista externa, Centrada'
                    elif positionRelative == '3': erroProv = 'Pista externa, Ortogonal'
                    elif positionRelative == '12': erroProv = 'Pista externa, Oposta'
                else:
                    erroProv = 'Pista externa'
            elif 'B' in erro: erroProv = 'Bola'

            # CWRU Dataset Ball Diameter prefix

            if '007' in erro: diametroFalha = '0.007 polegadas'
            elif '014' in erro: diametroFalha = '0.014 polegadas'
            elif '021' in erro: diametroFalha = '0.021 polegadas'
            elif '028' in erro: diametroFalha = '0.028 polegadas'

            cargaMotor = erro[indexUnderscore:]
            carga = cargaMotor[1] if len(cargaMotor) > 1 else "N/A"
            return f"ERRO PROVÁVEL: {erroProv} | DIAMETRO DA FALHA: {diametroFalha} | CARGA DO MOTOR: {carga}"

        # Apenas o dataset CWRU e SCA possui '_' no label
        if '_' not in erro:
            #HUST dataset prefix
            if 'IB' in erro: erroProv = 'Rachadura interna e Bola'
            elif 'IO' in erro: erroProv = 'Rachadura interna e externa'
            elif 'OB' in erro: erroProv = 'Rachadura externa e Bola'
            elif 'N' in erro: erroProv = 'Vibração normal'
            elif 'I' in erro: erroProv = 'Rachadura interna'
            elif 'O' in erro: erroProv = 'Rachadura externa'
            elif 'B' in erro: erroProv = 'Rachadura Bola'
        
        
        return f"ERRO PROVÁVEL: {erroProv} | DIAMETRO DA FALHA: N/A | CARGA DO MOTOR: N/A"
    except Exception as e:
        print(f"Erro Exception Linha 100: {e}")
        return f"ERRO PROVÁVEL: {erro} | DIAMETRO DA FALHA: N/A | CARGA DO MOTOR: N/A"

def previsao(dados):
    X_data = scaler.transform(dados)
    X_data_sel = X_data[:, idx]
    y_pred_data = rf2.predict(X_data_sel)
    contagem = Counter(y_pred_data)
    probs = rf2.predict_proba(X_data_sel)
    confidence = np.max(probs) * 100
    resultado = f"{erroLabel(contagem.most_common(1)[0][0])} ; confiança = {confidence:.2f}%"
    return resultado