import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from collections import Counter
import joblib
import os

dataset_path = './sample_data/dataset_de_features.csv'

if os.path.exists(dataset_path):
    print("Iniciando o treinamento do modelo de Machine Learning...\n")
    
    os.makedirs("sample_data", exist_ok=True)
    
    df_sample = pd.read_csv(dataset_path)

    def agrupar_rotulos(rotulo):
        rotulo = str(rotulo)
        if rotulo.startswith('SCA_'):
            return rotulo
        elif '_' in rotulo: # CWRU
            if 'normal' in rotulo.lower(): return 'CWRU_Normal'
            elif 'IR' in rotulo: return 'CWRU_IR'
            elif 'OR' in rotulo: return 'CWRU_OR'
            elif 'B' in rotulo: return 'CWRU_B'
            return 'CWRU_Outro'
        else: # HUST
            letras = ''.join([char for char in rotulo if char.isalpha()])
            return 'HUST_' + letras
        
    df_sample['fault'] = df_sample['fault'].apply(agrupar_rotulos)

    contagem_classes = df_sample['fault'].value_counts()
    classes_validas  = contagem_classes[contagem_classes >= 2].index
    df_sample = df_sample[df_sample['fault'].isin(classes_validas)]

    # Separa os dados (X) e os rótulos (y)
    X = df_sample.drop('fault', axis=1).values
    y = df_sample['fault'].values

    # 70% Treino, 30% Teste
    # stratify=y garante que todas as falhas tenham a mesma proporção no treino e no teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Treinamento único
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train_scaled, y_train)
    
    y_pred = rf_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred) * 100
    print(f"Acurácia real do modelo: {accuracy:.2f}%\n")

    # Salva o modelo
    joblib.dump(scaler, './sample_data/scaler.pkl')
    joblib.dump(rf_model, './sample_data/modelo_rf.pkl')
    print("Modelo e Normalizador salvos em '.pkl' com sucesso!\n")

else:
    print(f"Erro: O arquivo '{dataset_path}' não foi encontrado.")
    scaler, rf_model = None, None

def erroLabel(erro):
    erro = str(erro)
    erroProv = 'Desconhecido'
    
    if erro.startswith('SCA_'):
        if erro == 'SCA_Desligada': erroProv = 'Máquina desligada / Sem dados'
        elif erro == 'SCA_Normal': erroProv = 'Vibração normal'
        elif erro == 'SCA_AnelInterno': erroProv = 'Anel Interno'
        elif erro == 'SCA_Bola': erroProv = 'Bola'
        elif erro == 'SCA_AnelExterno': erroProv = 'Anel Externo'
        return f"[SCA] ERRO: {erroProv}"

    elif erro.startswith('CWRU_'):
        if 'Normal' in erro: erroProv = 'Vibração normal'
        elif 'IR' in erro: erroProv = 'Pista interna'
        elif 'B' in erro: erroProv = 'Bola'
        elif 'OR' in erro: erroProv = 'Pista externa'
        return f"[CWRU] ERRO: {erroProv} | TIPO AGRUPADO"

    elif erro.startswith('HUST_'):
        if 'IB' in erro: erroProv = 'Rachadura Interna e Bola'
        elif 'IO' in erro: erroProv = 'Rachadura Interna e Externa'
        elif 'OB' in erro: erroProv = 'Rachadura Externa e Bola'
        elif 'N' in erro: erroProv = 'Vibração normal'
        elif 'I' in erro: erroProv = 'Rachadura Interna'
        elif 'O' in erro: erroProv = 'Rachadura Externa'
        elif 'B' in erro: erroProv = 'Rachadura Bola'
        return f"[HUST] ERRO: {erroProv}"

    return f"ERRO PROVÁVEL: {erro}"

def previsao(dados):
    if rf_model is None or scaler is None:
        return "Erro: O modelo não foi treinado. Verifique o dataset."

    # Transformação direta (sem o 'idx' antigo)
    X_data = scaler.transform(dados)
    y_pred_data = rf_model.predict(X_data)

    contagem = Counter(y_pred_data)
    classe_vencedora = contagem.most_common(1)[0][0]
    classe_idx = rf_model.classes_.tolist().index(classe_vencedora)
    probs = rf_model.predict_proba(X_data)
    confidence = np.mean(probs[:, classe_idx]) * 100

    resultado = f"{erroLabel(classe_vencedora)} ; confiança = {confidence:.2f}%"
    return resultado

# TESTE PRÁTICO #
def identifica_base(rotulo):
    rotulo = str(rotulo)
    if rotulo.startswith('SCA_'): return 'SCA'
    elif '_' in rotulo: return 'CWRU'
    else: return 'HUST'

def realizar_teste_pratico():
    print("="*50)
    print(" INICIANDO TESTE COM AMOSTRAS (SCA, CWRU, HUST) ")
    print("="*50)
    
    df = pd.read_csv('./sample_data/dataset_de_features.csv')
    df['Origem'] = df['fault'].apply(identifica_base)
    
    df_sca = df[df['Origem'] == 'SCA']
    df_cwru = df[df['Origem'] == 'CWRU']
    df_hust = df[df['Origem'] == 'HUST']
    
    amostras_sca = df_sca.sample(n=min(10, len(df_sca))) if not df_sca.empty else df_sca
    amostras_cwru = df_cwru.sample(n=min(10, len(df_cwru))) if not df_cwru.empty else df_cwru
    amostras_hust = df_hust.sample(n=min(10, len(df_hust))) if not df_hust.empty else df_hust
    
    todas_amostras = pd.concat([amostras_sca, amostras_cwru, amostras_hust])
    
    for index, row in todas_amostras.iterrows():
        rotulo_real = row['fault']
        
        # Pega apenas as features, ignorando a coluna de falha e a de origem
        features = row.drop(['fault', 'Origem']).values.reshape(1, -1)
        
        resultado_ia = previsao(features)
        
        print(f"Original : {rotulo_real}")
        print(f"IA previu: {resultado_ia}")
        print("-" * 50)

if __name__ == "__main__":
    realizar_teste_pratico()