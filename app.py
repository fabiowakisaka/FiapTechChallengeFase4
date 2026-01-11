import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def parse_vol(val):
    try:
        if isinstance(val, str):
            val = val.replace(',', '.')
            if 'M' in val: return float(val.replace('M', '')) * 1_000_000
            if 'K' in val: return float(val.replace('K', '')) * 1_000
            if 'B' in val: return float(val.replace('B', '')) * 1_000_000_000
        return float(val)
    except:
        return 0.0

def compute_rsi(data, window=14):
    diff = data.diff(1)
    gain = diff.where(diff > 0, 0)
    loss = -diff.where(diff < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

st.set_page_config(page_title="IBOV Predictor - Fase 4", layout="wide")

@st.cache_resource
def load_assets():
    try:
        model = joblib.load("modelo/xgb.joblib")
        threshold = joblib.load("modelo/threshold.joblib")
        features_list = joblib.load("modelo/features.joblib")
        return model, threshold, features_list
    except Exception as e:
        st.error(f"Erro ao carregar arquivos do modelo: {e}")
        st.stop()

@st.cache_data
def load_and_process_data():
    try:
        path = "data/dados_historicos_ibovespa230722230725.csv"
        df = pd.read_csv(path, parse_dates=["Date"], dayfirst=True)
        df.sort_values("Date", inplace=True)
        
        for col in ["Price", "Open", "High", "Low", "Change %"]:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace("%", "").str.replace(".", "").str.replace(",", ".").astype(float)
        
        df["Vol."] = df["Vol."].apply(parse_vol)
        
        df["return_1d"] = df["Price"].pct_change()
        df["return_5d"] = df["Price"].pct_change(5)
        df["return_10d"] = df["Price"].pct_change(10)
        df["vol_chg_5d"] = df["Vol."].pct_change(5)
        df["high_low_spread"] = (df["High"] - df["Low"]) / df["Low"]
        df["ma_5"] = df["Price"].rolling(5).mean()
        df["ma_10"] = df["Price"].rolling(10).mean()
        df["ma_20"] = df["Price"].rolling(20).mean()
        df["ma_diff_5_20"] = df["ma_5"] - df["ma_20"]
        df["rsi_14"] = compute_rsi(df["Price"], 14)
        df["volatility_20"] = df["Price"].pct_change().rolling(20).std()
        df["target_real"] = (df["Price"].shift(-1) > df["Price"]).astype(int)
        
        return df.fillna(0)
    except Exception as e:
        st.error(f"Erro ao carregar dados CSV: {e}")
        st.stop()

model, best_threshold, features_list = load_assets()
df_full = load_and_process_data()

st.title("IBOVESPA: Inteligência Preditiva")
st.markdown("---")

st.header("Simulador de Tendência")
col_in, col_out = st.columns([1, 2])

with col_in:
    datas_lista = df_full["Date"].dt.date.unique()
    data_sel = st.selectbox("Selecione a Data Base:", options=reversed(datas_lista))
    user_guess = st.radio("Seu palpite para o dia seguinte:", ["Alta", "Baixa"])
    btn = st.button("Executar Previsão")

if btn:
    X_input = df_full[df_full["Date"].dt.date == data_sel][features_list].astype(float)
    if not X_input.empty:
        prob = model.predict_proba(X_input)[0, 1]
        res = "Alta" if prob >= best_threshold else "Baixa"
        
        with col_out:
            st.metric("Probabilidade de Alta", f"{prob*100:.2f}%")
            st.write(f"Previsão da IA: **{res.upper()}**")
            
            # Log de uso
            log_file = "logs_uso.csv"
            log_entry = pd.DataFrame([[datetime.now().strftime("%Y-%m-%d %H:%M"), data_sel, user_guess, res]], 
                                     columns=["Timestamp", "Data_Mercado", "Palpite_Usuario", "Previsao_IA"])
            log_entry.to_csv(log_file, mode='a', header=not os.path.exists(log_file), index=False)

st.markdown("---")
st.header(f"Análise de Movimentação ao redor de {data_sel}")
st.write("O gráfico abaixo exibe o comportamento do IBOVESPA. A linha laranja marca o dia da análise. Os pontos verdes e vermelhos indicam onde a IA acertou ou errou historicamente.")

data_dt = pd.to_datetime(data_sel)
df_zoom = df_full[(df_full["Date"] >= data_dt - pd.Timedelta(days=20)) & 
                  (df_full["Date"] <= data_dt + pd.Timedelta(days=10))].copy()

if not df_zoom.empty:
    X_zoom = df_zoom[features_list].astype(float)
    y_probs_zoom = model.predict_proba(X_zoom)[:, 1]
    y_pred_zoom = (y_probs_zoom >= best_threshold).astype(int)
    df_zoom["Acerto"] = (y_pred_zoom == df_zoom["target_real"].values)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df_zoom["Date"], df_zoom["Price"], color="silver", label="Preço Fechamento")
    ax.scatter(df_zoom[df_zoom["Acerto"]]["Date"], df_zoom[df_zoom["Acerto"]]["Price"], color="green", label="Acerto IA", s=30)
    ax.scatter(df_zoom[~df_zoom["Acerto"]]["Date"], df_zoom[~df_zoom["Acerto"]]["Price"], color="red", label="Erro IA", s=30)
    ax.axvline(data_dt, color="orange", linestyle="--", label="Dia da Simulação")
    ax.legend()
    st.pyplot(fig)

st.markdown("---")
st.header("Painel de Performance e Validação")

col_g1, col_g2 = st.columns(2)

with col_g1:
    st.subheader("Matriz de Confusão")
    st.write("Validação técnica comparando as previsões da IA com os resultados reais do mercado no período do gráfico.")
    cm = confusion_matrix(df_zoom["target_real"], y_pred_zoom)
    fig_cm, ax_cm = plt.subplots()
    ConfusionMatrixDisplay(cm).plot(ax=ax_cm, cmap="Greys")
    st.pyplot(fig_cm)

with col_g2:
    st.subheader("Importância das Variáveis")
    st.write("Principais indicadores técnicos que mais influenciaram as decisões do modelo estatístico.")
    importances = pd.Series(model.feature_importances_, index=features_list).sort_values().tail(8)
    st.bar_chart(importances)

st.markdown("---")
if os.path.exists("logs_uso.csv"):
    st.subheader("Registro de Consultas Recentes")
    st.dataframe(pd.read_csv("logs_uso.csv").tail(5), use_container_width=True)

st.caption("Fábio Wakisaka - Tech Challenge Fase 4")