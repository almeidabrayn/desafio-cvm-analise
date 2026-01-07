import pandas as pd
import numpy as np
import requests
import zipfile
import io
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, confusion_matrix, classification_report
import xgboost as xgb
import os

# Configurar estilo dos gráficos
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAÇÕES
# ============================================================================

# Período de análise
DATA_INICIO = '2025-01-01'
DATA_FIM = '2025-12-31'
MIN_FUNDOS_ATIVOS = 50

# Parâmetros do target
DIAS_FUTURO = 21
MIN_PL = 1000000

# Features windows
WINDOWS_RETORNO = [21, 63]
WINDOWS_VOLATILIDADE = [21, 63]

# Configurações do modelo
RANDOM_SEED = 42
TEST_SIZE = 0.2

# ============================================================================
# FUNÇÕES UTILITÁRIAS
# ============================================================================

def baixar_dados_cvm(mes_ano: str, tipo: str = 'inf_diario'):
    """Baixa dados do portal da CVM"""
    try:
        if tipo == 'inf_diario':
            url = f"https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_{mes_ano}.zip"
            print(f"  Baixando: {mes_ano}...", end=" ")
            
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                csv_file = z.namelist()[0]
                with z.open(csv_file) as f:
                    df = pd.read_csv(f, sep=';', encoding='utf-8', low_memory=False)
            
            print(f"OK ({len(df):,} registros)")
            
            # Renomear colunas
            mapeamento_colunas = {
                'TP_FUNDO_CLASSE': 'TP_FUNDO',
                'CNPJ_FUNDO_CLASSE': 'CNPJ_FUNDO',
                'DT_COMPTC': 'DT_COMPTC',
                'VL_TOTAL': 'VL_TOTAL',
                'VL_QUOTA': 'VL_QUOTA',
                'VL_PATRIM_LIQ': 'VL_PATRIM_LIQ',
                'CAPTC_DIA': 'CAPTC_DIA',
                'RESG_DIA': 'RESG_DIA',
                'NR_COTST': 'NR_COTST'
            }
            
            df = df.rename(columns={k: v for k, v in mapeamento_colunas.items() if k in df.columns})
            
            # Converter tipos
            if 'DT_COMPTC' in df.columns:
                df['DT_COMPTC'] = pd.to_datetime(df['DT_COMPTC'], errors='coerce')
            
            colunas_numericas = ['VL_PATRIM_LIQ', 'VL_QUOTA', 'CAPTC_DIA', 'RESG_DIA', 'NR_COTST', 'VL_TOTAL']
            for col in colunas_numericas:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.', regex=False), errors='coerce')
            
            if 'CNPJ_FUNDO' in df.columns:
                df['CNPJ_FUNDO'] = df['CNPJ_FUNDO'].astype(str).str.strip()
            
            return df
        
        elif tipo == 'cadastro':
            print("  Baixando cadastro...", end=" ")
            url = "https://dados.cvm.gov.br/dados/FI/CAD/DADOS/cad_fi.csv"
            df = pd.read_csv(url, sep=';', encoding='latin-1')
            print(f"OK ({len(df):,} registros)")
            
            if 'CNPJ_FUNDO' in df.columns:
                df['CNPJ_FUNDO'] = df['CNPJ_FUNDO'].astype(str).str.strip()
            
            return df
    
    except Exception as e:
        print(f"ERRO: {str(e)[:50]}...")
        return pd.DataFrame()

# ============================================================================
# ETL - EXTRAÇÃO E TRANSFORMAÇÃO
# ============================================================================

def executar_etl():
    """Executa pipeline completo de ETL"""
    print("=" * 70)
    print("ETL - EXTRAÇÃO E TRANSFORMAÇÃO DE DADOS")
    print("=" * 70)
    
    # Baixar dados
    print("\nBaixando dados diários...")
    
    data_inicio = pd.Timestamp(DATA_INICIO)
    data_fim = pd.Timestamp(DATA_FIM)
    meses = pd.date_range(start=data_inicio, end=data_fim, freq='MS')
    
    dados_list = []
    
    for data in meses:
        mes_ano = data.strftime('%Y%m')
        df_mes = baixar_dados_cvm(mes_ano, 'inf_diario')
        
        if not df_mes.empty and 'CNPJ_FUNDO' in df_mes.columns and 'DT_COMPTC' in df_mes.columns:
            dados_list.append(df_mes)
    
    if not dados_list:
        print("Nenhum dado baixado!")
        return None, None
    
    dados_diarios = pd.concat(dados_list, ignore_index=True)
    
    # Baixar cadastro
    print("\nBaixando cadastro...")
    cadastro = baixar_dados_cvm('', 'cadastro')
    
    # Processar dados
    print("\nProcessando dados...")
    
    # Limpeza básica
    dados_diarios = dados_diarios.dropna(subset=['CNPJ_FUNDO', 'DT_COMPTC'])
    if 'VL_PATRIM_LIQ' in dados_diarios.columns:
        dados_diarios = dados_diarios.dropna(subset=['VL_PATRIM_LIQ'])
        dados_diarios = dados_diarios[dados_diarios['VL_PATRIM_LIQ'] >= MIN_PL]
    
    dados_diarios = dados_diarios.sort_values(['CNPJ_FUNDO', 'DT_COMPTC'])
    dados_diarios = dados_diarios.drop_duplicates(subset=['CNPJ_FUNDO', 'DT_COMPTC'])
    
    print(f"\nDADOS PROCESSADOS:")
    print(f"   • Registros: {len(dados_diarios):,}")
    print(f"   • Fundos únicos: {dados_diarios['CNPJ_FUNDO'].nunique()}")
    print(f"   • Período: {dados_diarios['DT_COMPTC'].min().date()} a {dados_diarios['DT_COMPTC'].max().date()}")
    
    # Gerar gráfico de distribuição temporal
    gerar_grafico_distribuicao_temporal(dados_diarios)
    
    return dados_diarios, cadastro

def gerar_grafico_distribuicao_temporal(dados_diarios):
    """Gera gráfico da distribuição temporal dos dados"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Volume de dados por mês
    dados_diarios['mes'] = dados_diarios['DT_COMPTC'].dt.to_period('M')
    volume_mensal = dados_diarios.groupby('mes').size()
    
    axes[0, 0].bar(range(len(volume_mensal)), volume_mensal.values)
    axes[0, 0].set_title('Volume de Registros por Mês', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Mês')
    axes[0, 0].set_ylabel('Número de Registros')
    axes[0, 0].set_xticks(range(len(volume_mensal)))
    axes[0, 0].set_xticklabels([str(m) for m in volume_mensal.index], rotation=45)
    
    # 2. Distribuição do Patrimônio Líquido
    if 'VL_PATRIM_LIQ' in dados_diarios.columns:
        pl_log = np.log10(dados_diarios['VL_PATRIM_LIQ'].replace(0, np.nan).dropna())
        axes[0, 1].hist(pl_log, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Distribuição do Patrimônio Líquido (log10)', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Log10(PL)')
        axes[0, 1].set_ylabel('Frequência')
    
    # 3. Número de fundos ativos por dia
    fundos_ativos = dados_diarios.groupby('DT_COMPTC')['CNPJ_FUNDO'].nunique()
    axes[1, 0].plot(fundos_ativos.index, fundos_ativos.values, linewidth=2)
    axes[1, 0].set_title('Número de Fundos Ativos por Dia', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Data')
    axes[1, 0].set_ylabel('Número de Fundos')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Distribuição por dia da semana
    dados_diarios['dia_semana'] = dados_diarios['DT_COMPTC'].dt.dayofweek
    dias_nomes = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex']
    contagem_dias = dados_diarios['dia_semana'].value_counts().sort_index()
    axes[1, 1].bar(dias_nomes, contagem_dias.values[:5])
    axes[1, 1].set_title('Distribuição por Dia da Semana', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Dia da Semana')
    axes[1, 1].set_ylabel('Número de Registros')
    
    plt.tight_layout()
    plt.savefig('graficos_distribuicao_dados.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Gráfico de distribuição salvo: 'graficos_distribuicao_dados.png'")

# ============================================================================
# FEATURES
# ============================================================================

def calcular_features(dados_diarios):
    """Calcula features para o modelo"""
    print("\n" + "=" * 70)
    print("FEATURES")
    print("=" * 70)
    
    df = dados_diarios.copy().sort_values(['CNPJ_FUNDO', 'DT_COMPTC'])
    
    print(f"Processando {df['CNPJ_FUNDO'].nunique()} fundos...")
    
    # Calcular fluxo e retorno
    if 'CAPTC_DIA' in df.columns and 'RESG_DIA' in df.columns:
        df['flow_diario'] = df['CAPTC_DIA'] - df['RESG_DIA']
    
    if 'VL_QUOTA' in df.columns:
        df['ret_diario'] = df.groupby('CNPJ_FUNDO')['VL_QUOTA'].pct_change()
    
    features_list = []
    
    for cnpj in df['CNPJ_FUNDO'].unique()[:100]:  # Limitar para processamento
        fundo_df = df[df['CNPJ_FUNDO'] == cnpj].copy()
        
        if len(fundo_df) < 50:
            continue
        
        # Features básicas
        fundo_features = {
            'CNPJ_FUNDO': cnpj,
            'DT_COMPTC': fundo_df['DT_COMPTC'],
            'VL_PATRIM_LIQ': fundo_df['VL_PATRIM_LIQ'] if 'VL_PATRIM_LIQ' in fundo_df.columns else np.nan
        }
        
        # Fluxo futuro (simplificado)
        if 'flow_diario' in fundo_df.columns:
            fundo_df['flow_futuro'] = fundo_df['flow_diario'].rolling(10, min_periods=1).mean().shift(-5)
            if 'VL_PATRIM_LIQ' in fundo_df.columns:
                fundo_df['flow_futuro_pct'] = fundo_df['flow_futuro'] / fundo_df['VL_PATRIM_LIQ'].shift(1)
                fundo_features['flow_futuro_pct'] = fundo_df['flow_futuro_pct']
        
        # Features de retorno
        if 'VL_QUOTA' in fundo_df.columns:
            for window in WINDOWS_RETORNO:
                if window < len(fundo_df):
                    fundo_features[f'ret_{window}d'] = fundo_df['VL_QUOTA'].pct_change(window)
        
        # Features de tamanho
        if 'VL_PATRIM_LIQ' in fundo_df.columns:
            fundo_features['log_pl'] = np.log1p(fundo_df['VL_PATRIM_LIQ'])
        
        # Features temporais
        fundo_features['dia_semana'] = fundo_df['DT_COMPTC'].dt.dayofweek
        fundo_features['mes'] = fundo_df['DT_COMPTC'].dt.month
        fundo_features['fim_mes'] = fundo_df['DT_COMPTC'].dt.is_month_end.astype(int)
        fundo_features['fim_trimestre'] = fundo_df['DT_COMPTC'].dt.is_quarter_end.astype(int)
        fundo_features['efeito_janeiro'] = (fundo_df['DT_COMPTC'].dt.month == 1).astype(int)
        
        features_list.append(pd.DataFrame(fundo_features))
    
    if not features_list:
        return None
    
    features = pd.concat(features_list, ignore_index=True).dropna(subset=['flow_futuro_pct'])
    
    # Criar target
    if len(features) > 20:
        features['target_class'] = pd.qcut(features['flow_futuro_pct'].rank(method='first'), 
                                          q=min(10, len(features)//10), labels=False, duplicates='drop')
        features['target_top_decile'] = (features['target_class'] == features['target_class'].max()).astype(int)
    else:
        features['target_top_decile'] = (features['flow_futuro_pct'] > features['flow_futuro_pct'].median()).astype(int)
    
    print(f"Features calculadas: {len(features):,} registros")
    
    # Gerar gráficos das features
    gerar_graficos_features(features)
    
    return features

def gerar_graficos_features(features):
    """Gera gráficos das features calculadas"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Distribuição do target
    axes[0, 0].hist(features['flow_futuro_pct'].clip(-0.1, 0.1), bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_title('Distribuição do Fluxo Futuro (%)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Fluxo Futuro / PL')
    axes[0, 0].set_ylabel('Frequência')
    
    # 2. Balanceamento das classes
    if 'target_top_decile' in features.columns:
        class_counts = features['target_top_decile'].value_counts()
        axes[0, 1].pie(class_counts.values, labels=['Não Top Decile', 'Top Decile'], 
                      autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'lightgreen'])
        axes[0, 1].set_title('Balanceamento das Classes', fontsize=14, fontweight='bold')
    
    # 3. Retorno vs Fluxo Futuro
    if 'ret_21d' in features.columns:
        sample = features.sample(min(1000, len(features)))
        axes[0, 2].scatter(sample['ret_21d'], sample['flow_futuro_pct'], alpha=0.6, s=20)
        axes[0, 2].set_title('Retorno 21d vs Fluxo Futuro', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Retorno 21 dias')
        axes[0, 2].set_ylabel('Fluxo Futuro / PL')
        axes[0, 2].axhline(y=0, color='red', linestyle='--', linewidth=1)
        axes[0, 2].axvline(x=0, color='red', linestyle='--', linewidth=1)
    
    # 4. Tamanho (PL) vs Fluxo Futuro
    if 'log_pl' in features.columns:
        sample = features.sample(min(1000, len(features)))
        axes[1, 0].scatter(sample['log_pl'], sample['flow_futuro_pct'], alpha=0.6, s=20)
        axes[1, 0].set_title('Tamanho (log PL) vs Fluxo Futuro', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Log(Patrimônio Líquido)')
        axes[1, 0].set_ylabel('Fluxo Futuro / PL')
        axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=1)
    
    # 5. Sazonalidade - fluxo por dia da semana
    if 'dia_semana' in features.columns:
        fluxo_por_dia = features.groupby('dia_semana')['flow_futuro_pct'].mean()
        dias_nomes = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex']
        axes[1, 1].bar(dias_nomes, fluxo_por_dia.values[:5] if len(fluxo_por_dia) >= 5 else fluxo_por_dia.values)
        axes[1, 1].set_title('Fluxo Médio por Dia da Semana', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Dia da Semana')
        axes[1, 1].set_ylabel('Fluxo Futuro Médio / PL')
        axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=1)
    
    # 6. Sazonalidade - fluxo por fim de mês
    if 'fim_mes' in features.columns:
        fluxo_fim_mes = features.groupby('fim_mes')['flow_futuro_pct'].mean()
        axes[1, 2].bar(['Dia Comum', 'Fim de Mês'], fluxo_fim_mes.values)
        axes[1, 2].set_title('Fluxo: Dia Comum vs Fim de Mês', fontsize=14, fontweight='bold')
        axes[1, 2].set_ylabel('Fluxo Futuro Médio / PL')
        axes[1, 2].axhline(y=0, color='red', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    plt.savefig('graficos_features.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Gráficos das features salvos: 'graficos_features.png'")

# ============================================================================
# MODELAGEM
# ============================================================================

def executar_modelagem(features):
    """Executa modelagem preditiva"""
    print("\n" + "=" * 70)
    print("MODELAGEM PREDITIVA")
    print("=" * 70)
    
    if features is None or len(features) < 100:
        print("Dados insuficientes")
        return None
    
    # Preparar dados
    exclude_cols = ['CNPJ_FUNDO', 'DT_COMPTC', 'target_top_decile', 'target_class', 'flow_futuro_pct']
    if 'VL_PATRIM_LIQ' in features.columns:
        exclude_cols.append('VL_PATRIM_LIQ')
    
    feature_cols = [col for col in features.columns if col not in exclude_cols]
    X = features[feature_cols].select_dtypes(include=[np.number])
    
    if X.empty:
        # Criar features básicas
        X = pd.DataFrame()
        if 'log_pl' in features.columns:
            X['log_pl'] = features['log_pl']
        if 'dia_semana' in features.columns:
            X['dia_semana'] = features['dia_semana']
    
    X = X.fillna(X.median())
    y = features['target_top_decile']
    
    print(f"📐 Dimensões: X={X.shape}, y={y.shape}")
    print(f"⚖️  Balanceamento: {y.mean():.2%} positivos")
    
    # Split temporal
    features_sorted = features.sort_values('DT_COMPTC')
    X_sorted = X.loc[features_sorted.index]
    y_sorted = y.loc[features_sorted.index]
    
    split_idx = int(len(X_sorted) * 0.7)
    X_train, X_test = X_sorted.iloc[:split_idx], X_sorted.iloc[split_idx:]
    y_train, y_test = y_sorted.iloc[:split_idx], y_sorted.iloc[split_idx:]
    
    print(f"Treino: {len(X_train)} amostras")
    print(f"Teste: {len(X_test)} amostras")
    
    # Treinar modelo
    print("\nTreinando modelo...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5, # Poderia ser maior, entretanto acredito que esse valor esteja bom.
        random_state=RANDOM_SEED,
        class_weight='balanced',
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Avaliar
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    print(f"\nDESEMPENHO:")
    print(f"   • Acurácia: {accuracy:.3f}")
    print(f"   • ROC-AUC: {roc_auc:.3f}")
    print(f"   • PR-AUC: {pr_auc:.3f}")
    
    # Importância das features
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🏆 TOP 5 FEATURES:")
    for idx, row in importance_df.head(5).iterrows():
        print(f"   • {row['feature']}: {row['importance']:.3f}")
    
    resultados = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'importance': importance_df,
        'model': model,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    # Gerar gráficos do modelo
    gerar_graficos_modelo(resultados, X_train, y_test)
    
    return resultados

def gerar_graficos_modelo(resultados, X_train, y_test):
    """Gera gráficos de avaliação do modelo"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Importância das features (Top 10)
    importance_df = resultados['importance']
    top_features = importance_df.head(10)
    
    axes[0, 0].barh(range(len(top_features)), top_features['importance'].values)
    axes[0, 0].set_yticks(range(len(top_features)))
    axes[0, 0].set_yticklabels(top_features['feature'].values)
    axes[0, 0].set_xlabel('Importância')
    axes[0, 0].set_title('Top 10 Features Mais Importantes', fontsize=14, fontweight='bold')
    axes[0, 0].invert_yaxis()
    
    # 2. Matriz de confusão
    cm = confusion_matrix(resultados['y_test'], resultados['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
    axes[0, 1].set_xlabel('Predito')
    axes[0, 1].set_ylabel('Real')
    axes[0, 1].set_title('Matriz de Confusão', fontsize=14, fontweight='bold')
    
    # 3. Distribuição das probabilidades
    axes[0, 2].hist(resultados['y_pred_proba'][resultados['y_test'] == 0], 
                   bins=30, alpha=0.5, label='Classe 0', color='red')
    axes[0, 2].hist(resultados['y_pred_proba'][resultados['y_test'] == 1], 
                   bins=30, alpha=0.5, label='Classe 1', color='green')
    axes[0, 2].set_xlabel('Probabilidade Prevista')
    axes[0, 2].set_ylabel('Frequência')
    axes[0, 2].set_title('Distribuição das Probabilidades', fontsize=14, fontweight='bold')
    axes[0, 2].legend()
    axes[0, 2].axvline(x=0.5, color='black', linestyle='--', linewidth=1)
    
    # 4. Curva ROC
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(resultados['y_test'], resultados['y_pred_proba'])
    axes[1, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {resultados["roc_auc"]:.3f})')
    axes[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Aleatório')
    axes[1, 0].set_xlabel('Taxa de Falsos Positivos')
    axes[1, 0].set_ylabel('Taxa de Verdadeiros Positivos')
    axes[1, 0].set_title('Curva ROC', fontsize=14, fontweight='bold')
    axes[1, 0].legend(loc="lower right")
    
    # 5. Curva Precision-Recall
    from sklearn.metrics import precision_recall_curve
    precision, recall, _ = precision_recall_curve(resultados['y_test'], resultados['y_pred_proba'])
    axes[1, 1].plot(recall, precision, color='blue', lw=2, label=f'PR (AUC = {resultados["pr_auc"]:.3f})')
    axes[1, 1].set_xlabel('Recall')
    axes[1, 1].set_ylabel('Precision')
    axes[1, 1].set_title('Curva Precision-Recall', fontsize=14, fontweight='bold')
    axes[1, 1].legend(loc="lower left")
    
    # 6. Desempenho por quantis de features importantes
    if len(importance_df) > 0:
        top_feature = importance_df.iloc[0]['feature']
        if top_feature in X_train.columns:
            test_features = resultados['X_test'].copy()
            test_features['y_pred_proba'] = resultados['y_pred_proba']
            test_features['y_test'] = resultados['y_test']
            
            # Dividir em quintis da feature mais importante
            test_features['quintil'] = pd.qcut(test_features[top_feature], q=5, labels=False)
            
            # Calcular métricas por quintil
            metrics_by_quintil = []
            for quintil in range(5):
                subset = test_features[test_features['quintil'] == quintil]
                if len(subset) > 0:
                    auc = roc_auc_score(subset['y_test'], subset['y_pred_proba'])
                    metrics_by_quintil.append(auc)
            
            axes[1, 2].bar(range(1, 6), metrics_by_quintil)
            axes[1, 2].set_xlabel(f'Quintil de {top_feature}')
            axes[1, 2].set_ylabel('ROC-AUC')
            axes[1, 2].set_title(f'Desempenho por Quintil de {top_feature}', fontsize=14, fontweight='bold')
            axes[1, 2].axhline(y=0.5, color='red', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    plt.savefig('graficos_modelo.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Gráficos do modelo salvos: 'graficos_modelo.png'")

# ============================================================================
# RELATÓRIO FINAL SIMPLIFICADO
# ============================================================================

def gerar_relatorio_final_simplificado(features, resultados):
    """Gera relatório final simplificado"""
    print("\n" + "=" * 70)
    print("RELATÓRIO FINAL - DESAFIO BI CVM")
    print("=" * 70)
    
    if features is not None:
        print(f"\nDADOS PROCESSADOS:")
        print(f"   • Período: {DATA_INICIO} a {DATA_FIM}")
        print(f"   • Amostra: {len(features):,} observações")
        print(f"   • Fundos analisados: {features['CNPJ_FUNDO'].nunique()}")
        
        if 'target_top_decile' in features.columns:
            print(f"   • Top decile: {features['target_top_decile'].mean():.2%}")
    
    if isinstance(resultados, dict):
        print(f"\nDESEMPENHO DO MODELO:")
        print(f"   • Acurácia: {resultados.get('accuracy', 0):.3f}")
        print(f"   • ROC-AUC: {resultados.get('roc_auc', 0):.3f}")
        print(f"   • PR-AUC: {resultados.get('pr_auc', 0):.3f}")
        
        if 'importance' in resultados:
            print(f"\n🏆 TOP 5 FEATURES:")
            importance_df = resultados['importance']
            for idx, row in importance_df.head(5).iterrows():
                print(f"   • {row['feature']}: {row['importance']:.3f}")
    
    print("\nARQUIVOS GERADOS:")
    print("   • graficos_distribuicao_dados.png - Análise exploratória dos dados")
    print("   • graficos_features.png - Análise das features")
    print("   • graficos_modelo.png - Avaliação do modelo")
    
    # Salvar dados
    if features is not None:
        features.to_csv('dados_processados.csv', index=False)
        print("   • dados_processados.csv - Dados processados")
    
    if isinstance(resultados, dict) and 'importance' in resultados:
        resultados['importance'].to_csv('importancia_features.csv', index=False)
        print("   • importancia_features.csv - Importância das variáveis")
    
    print("\nANÁLISE COMPLETA COM GRÁFICOS GERADA COM SUCESSO!")

# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def main():
    """Função principal"""
    print("=" * 70)
    print("DESAFIO TÉCNICO BI - ANÁLISE DE FUNDOS CVM")
    print("Versão com Gráficos e Visualizações")
    print("=" * 70)
    
    try:
        # 1. ETL
        dados_diarios, cadastro = executar_etl()
        
        if dados_diarios is None:
            print("\nNão foi possível continuar.")
            return
        
        # 2. Engenharia de Features
        features = calcular_features(dados_diarios)
        
        # 3. Modelagem
        if features is not None:
            resultados = executar_modelagem(features)
        else:
            resultados = None
            print("Não foi possível calcular features suficientes")
        
        # 4. Relatório Final Simplificado
        gerar_relatorio_final_simplificado(features, resultados)
        
        print("\n" + "=" * 70)
        print("TODOS OS GRÁFICOS FORAM GERADOS COM SUCESSO!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nERRO: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# EXECUÇÃO
# ============================================================================

if __name__ == "__main__":
    main()