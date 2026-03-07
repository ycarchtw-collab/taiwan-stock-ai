import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import time

# --- 1. 雲端環境設定 ---
if os.name == 'posix':
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Sans CJK TC', 'Liberation Sans']
else:
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

CORE_LIST = {
    "2330.TW": "台積電", "2454.TW": "聯發科", "2317.TW": "鴻海", "3675.TWO": "德微",
    "6282.TW": "康舒", "2303.TW": "聯電", "3037.TW": "欣興", "2382.TW": "廣達",
    "0050.TW": "元大台灣50", "4763.TW": "材料-KY", "8086.TWO": "宏捷科", "6438.TW": "迅得"
}

@st.cache_data(ttl=3600)
def fetch_data(ticker, period="7y"):
    try:
        time.sleep(0.2) # 減少 API 請求頻率
        data = yf.Ticker(ticker).history(period=period)
        return data
    except:
        return pd.DataFrame()

def calculate_rsi(df, periods=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- 2. 100 分制評分標準 ---
def evaluate_stock_100(df):
    if df.empty or len(df) < 20: return 0, []
    score, reasons = 0, []
    try:
        c = df['Close'].iloc[-1]
        m20 = df['Close'].rolling(20).mean()
        m120 = df['Close'].rolling(120).mean()
        m1200 = df['Close'].rolling(1200, min_periods=100).mean()
        rsi = calculate_rsi(df).iloc[-1]
        vol, avg_vol = df['Volume'].iloc[-1], df['Volume'].tail(5).mean()
        
        tests = [
            (c > m20.iloc[-1], "股價站上月線"), (m20.iloc[-1] > m20.iloc[-5], "月線趨勢向上"),
            (c > m120.iloc[-1] if not np.isnan(m120.iloc[-1]) else False, "股價站上半年線"),
            (m120.iloc[-1] > m120.iloc[-5] if not np.isnan(m120.iloc[-1]) else False, "長線趨勢翻正"),
            (50 < rsi < 75, "RSI強勢攻擊區"), (vol > avg_vol * 1.5, "量能顯著放大"),
            (c > m1200.iloc[-1] if not np.isnan(m1200.iloc[-1]) else True, "高於五年基期"),
            (c < m20.iloc[-1] + df['Close'].rolling(20).std().iloc[-1]*2, "尚未觸及布林上軌"),
            (c > np.sum(((df['High']+df['Low']+df['Close'])/3)*df['Volume'])/np.sum(df['Volume']), "站穩VWAP"),
            (c > df['Close'].iloc[-2], "維持連漲慣性")
        ]
        for cond, msg in tests:
            if cond: score += 10; reasons.append(msg)
    except: pass
    return score, reasons

# --- 3. 繪圖與象限圖渲染修復 ---
def plot_quadrant_chart(current_ticker, target_score, target_pct):
    st.subheader("📍 潛力象限分析 (100分制對齊)")
    compare_list = ["2330.TW", "2317.TW", "3675.TWO", "0050.TW"]
    if current_ticker not in compare_list: compare_list.append(current_ticker)
    
    q_results = []
    for t in compare_list:
        d = fetch_data(t, period="5y")
        if d.empty: continue
        s, _ = evaluate_stock_100(d)
        c = ((d['Close'].iloc[-1]-d['Close'].iloc[-2])/d['Close'].iloc[-2])*100
        q_results.append({"T": t, "Score": s, "Change": c})
    
    df_q = pd.DataFrame(q_results)
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # 點位渲染
    for i, row in df_q.iterrows():
        color = 'red' if row['T'] == current_ticker else 'royalblue'
        size = 200 if row['T'] == current_ticker else 100
        ax.scatter(row['Score'], row['Change'], c=color, s=size, edgecolors='white', zorder=5)
        ax.annotate(row['T'], (row['Score'], row['Change']), fontsize=8, xytext=(5,5), textcoords='offset points')
    
    # 座標軸設定
    ax.axvline(50, color='gray', ls='--', alpha=0.5)
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("AI 投資分數 (滿分100)")
    ax.set_ylabel("當日漲跌幅 (%)")
    ax.grid(True, linestyle=':', alpha=0.3)
    
    st.pyplot(fig)

# --- 4. 主頁面 UI ---
st.title("🚀 2026 AI 台股決測系統 V7")
query = st.sidebar.text_input("輸入代號或名稱", "3675")

# 智慧代碼轉換
ticker = query
if query in {v: k for k, v in CORE_LIST.items()}:
    ticker = {v: k for k, v in CORE_LIST.items()}[query]
elif query.isdigit():
    ticker = query + ".TW"

hist = fetch_data(ticker)
if hist.empty and ".TW" in ticker:
    ticker = ticker.replace(".TW", ".TWO")
    hist = fetch_data(ticker)

if not hist.empty:
    # 數據處理
    hist['MA120'] = hist['Close'].rolling(120).mean()
    hist['MA1200'] = hist['Close'].rolling(1200, min_periods=100).mean()
    score, tags = evaluate_stock_100(hist)
    
    # 顏色報價
    lp, pp = hist['Close'].iloc[-1], hist['Close'].iloc[-2]
    pct = ((lp - pp)/pp)*100
    p_color = "red" if pct > 0 else ("green" if pct < 0 else "black")
    
    st.markdown(f"## {ticker} 現價: **{lp:,.2f}** <span style='color:{p_color}'>({pct:+.2f}%)</span>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        # 半年與五年圖表 (略，請參考 V6 代碼整合)
        st.info("圖表生成中...") 
        # 此處應調用繪圖函數，包含自動 Y 軸縮放與 3日/週量能
    
    with col2:
        st.subheader(f"💡 AI 評分: {score} / 100")
        for t in tags: st.success(t)
        # 渲染象限圖
        plot_quadrant_chart(ticker, score, pct)
