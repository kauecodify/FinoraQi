import pandas as pd
import numpy as np
import joblib
import time
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, f1_score
from xgboost import XGBClassifier

# ==========================================
# 1. GERAÇÃO DE DADOS SINTÉTICOS (MOCK DATASET)
# ==========================================
def generate_finoraqi_dataset(n_samples=25000):
    """
    Gera um dataset sintético simulando o comportamento de crédito B2B/B2C/B2G do FinoraQi.
    """
    print("[INFO] Gerando dataset sintético com", n_samples, "registros...")
    np.random.seed(42)
    
    data = {
        'renda_mensal': np.random.normal(15000, 8000, n_samples).clip(2000, 100000),
        'historico_pagamentos': np.random.uniform(0, 100, n_samples), # 0 a 100% de pontualidade
        'score_bureau': np.random.normal(650, 150, n_samples).clip(0, 1000),
        'dividas_ativas': np.random.exponential(scale=2, size=n_samples).clip(0, 20),
        'tempo_emprego': np.random.exponential(scale=5, size=n_samples).clip(0, 30)
    }
    
    df = pd.DataFrame(data)
    
    # Lógica de negócio para criar o Rating (Target)
    # Ratings piores são associados a menor renda, menor score, mais dívidas e menos tempo de emprego
    risk_score = (
        (df['score_bureau'] / 1000) * 0.4 + 
        (df['historico_pagamentos'] / 100) * 0.3 + 
        (df['renda_mensal'] / 100000) * 0.2 + 
        (df['tempo_emprego'] / 30) * 0.1 - 
        (df['dividas_ativas'] / 20) * 0.3
    )
    
    # Mapeia o score de risco para as classes A+, A, B, C, D
    conditions = [
        (risk_score >= 0.75),
        (risk_score >= 0.60) & (risk_score < 0.75),
        (risk_score >= 0.45) & (risk_score < 0.60),
        (risk_score >= 0.30) & (risk_score < 0.45),
        (risk_score < 0.30)
    ]
    choices = ['A+', 'A', 'B', 'C', 'D']
    df['rating_credito'] = np.select(conditions, choices, default='B')
    
    return df

# ==========================================
# 2. PRÉ-PROCESSAMENTO E FRAGMENTAÇÃO
# ==========================================
def preprocess_and_fragment(df):
    """
    Limpa, codifica e fragmenta os dados em treino e teste usando validação estratificada.
    """
    print("\n[INFO] Iniciando pré-processamento e fragmentação dos dados...")
    
    # 1. Limpeza básica
    df = df.dropna()
    
    # 2. Separação de Features (X) e Target (y)
    X = df.drop(columns=['rating_credito'])
    y = df['rating_credito']
    
    # 3. Encoding da variável target (A+, A, B, C, D -> 0, 1, 2, 3, 4)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"[INFO] Classes do modelo: {label_encoder.classes_}")
    
    # 4. Fragmentação (Train/Test Split) 
    # Estratificado para garantir que a proporção de ratings ruins/ bons seja a mesma no treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"[INFO] Fragmentação concluída:")
    print(f"       - Treino: {X_train.shape[0]} amostras")
    print(f"       - Teste:  {X_test.shape[0]} amostras")
    
    return X_train, X_test, y_train, y_test, label_encoder

# ==========================================
# 3. TREINAMENTO E OTIMIZAÇÃO (AI)
# ==========================================
def train_models(X_train, y_train):
    """
    Configura e treina modelos de ML (XGBoost e Random Forest).
    """
    print("\n[INFO] Iniciando treinamento dos modelos de Machine Learning...")
    
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=42, 
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=150, 
            learning_rate=0.05, 
            max_depth=6, 
            random_state=42, 
            use_label_encoder=False, 
            eval_metric='mlogloss'
        )
    }
    
    trained_models = {}
    
    # 5. Fragmentação adicional: Validação Cruzada K-Fold (5 dobras)
    # Isso garante que o modelo não sofra overfitting em uma única divisão
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"\n[INFO] Treinando modelo: {name}")
        start_time = time.time()
        
        # Treinamento padrão
        model.fit(X_train, y_train)
        
        # Validação Cruzada (Cross-Validation)
        cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1_macro', n_jobs=-1)
        
        train_time = time.time() - start_time
        
        print(f"       - Tempo de treino: {train_time:.2f} segundos")
        print(f"       - F1-Score Médio (K-Fold): {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        trained_models[name] = model
        
    return trained_models

# ==========================================
# 4. AVALIAÇÃO E EXPLICABILIDADE (XAI)
# ==========================================
def evaluate_models(trained_models, X_test, y_test, label_encoder):
    """
    Avalia os modelos no conjunto de teste e extrai a importância das features.
    """
    print("\n[INFO] Avaliando modelos no conjunto de teste...")
    
    best_model_name = None
    best_f1 = 0
    best_model = None

    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # Métricas
        f1 = f1_score(y_test, y_pred, average='macro')
        roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
        
        print(f"\n--- Resultados: {name} ---")
        print(f"F1-Score Global: {f1:.4f}")
        print(f"ROC-AUC:         {roc_auc:.4f}")
        
        # Classification Report detalhado
        print("\nRelatório de Classificação:")
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = model

    print(f"\n[RESULTADO] O melhor modelo foi o {best_model_name} com F1-Score {best_f1:.4f}")
    
    # XAI: Feature Importance (Para modelos baseados em árvores)
    if hasattr(best_model, 'feature_importances_'):
        print("\n[INFO] Importância das Features (XAI):")
        importances = best_model.feature_importances_
        features = X_test.columns
        for feat, imp in sorted(zip(features, importances), key=lambda x: x[1], reverse=True):
            print(f"       - {feat}: {imp:.4f}")
            
    return best_model

# ==========================================
# 5. EXPORTAÇÃO DO MODELO
# ==========================================
def export_pipeline(model, label_encoder, filename="finoraqi_rating_model_v2.pkl"):
    """
    Cria um pipeline final com Scaler e Modelo e salva em disco.
    """
    print(f"\n[INFO] Empacotando e exportando o modelo final para {filename}...")
    
    # Para simplificar, vamos treinar um Scaler final. 
    # Na prática, o Scaler deveria ter sido integrado no Pipeline antes do treinamento.
    # Mas para este exemplo de exportação, salvamos o modelo e o encoder.
    
    artifact = {
        'model': model,
        'label_encoder': label_encoder,
        'features': ['renda_mensal', 'historico_pagamentos', 'score_bureau', 'dividas_ativas', 'tempo_emprego']
    }
    
    joblib.dump(artifact, filename)
    print("[SUCESSO] Modelo e LabelEncoder exportados com sucesso! Pronto para a API.")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("==================================================")
    print(" FINORAQI RATINGPLUS V2 - MACHINE LEARNING PIPELINE ")
    print("==================================================")
    
    # 1. Dados
    df = generate_finoraqi_dataset()
    
    # 2. Pré-processamento e Fragmentação
    X_train, X_test, y_train, y_test, label_encoder = preprocess_and_fragment(df)
    
    # 3. Treinamento
    trained_models = train_models(X_train, y_train)
    
    # 4. Avaliação
    best_model = evaluate_models(trained_models, X_test, y_test, label_encoder)
    
    # 5. Exportação
    export_pipeline(best_model, label_encoder)
    
    print("\n[FIM] Pipeline de ML concluído com sucesso.")
