import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import time

# --- 1. 雲端環境與字體設定 ---
if os.name == 'posix':
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Sans CJK TC', 'Liberation Sans']
else:
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 強化版名稱加載邏輯 ---
@st.cache_data
def load_stock_names():
    # 預設核心清單 (保底用)
    names = {
        "2330.TW": "台積電", "2317.TW": "鴻海", "3675.TWO": "德微", 
        "6282.TW": "康舒", "2454.TW": "聯發科", "0050.TW": "元大台灣50"
    }
    # 嘗試從本地 JSON 檔案讀取完整清單
    try:
        if os.path.exists('tw_stock_names.json'):
            with open('tw_stock_names.json', 'r', encoding='utf-8') as f:
                external_names = json.load(f)
                names.update(external_names)
    except Exception as e:
        st.error(f"無法加載名稱資料庫: {e}")
    return names

STOCK_DB = load_stock_names()

def get_company_name(ticker):
    """從資料庫獲取名稱，完全不調用 API 以確保速度"""
    return STOCK_DB.get(ticker, ticker.split('.')[0])

# --- 3. 核心功能函數 ---
@st.cache_data(ttl=3600)
def fetch_stock_data(ticker, period="7y"):
    try:
        time.sleep(0.3)
        return yf.Ticker(ticker).history(period=period)
    except:
        return pd.DataFrame()

def calculate_rsi(df, periods=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def evaluate_stock_100(df):
    if df.empty or len(df) < 20: return 0, []
    score, reasons = 0, []
    try:
        c = df['Close'].iloc[-1]
        m20 = df['Close'].rolling(20).mean()
        m120 = df['Close'].rolling(120).mean()
        m1200 = df['Close'].rolling(1200, min_periods=100).mean()
        std20 = df['Close'].rolling(20).std()
        rsi = calculate_rsi(df).iloc[-1]
        vol, avg_vol = df['Volume'].iloc[-1], df['Volume'].tail(5).mean()
        
        tests = [
            (c > m20.iloc[-1], "股價站上月線"), (m20.iloc[-1] > m20.iloc[-5] if len(m20)>5 else False, "月線趨勢向上"),
            (c > m120.iloc[-1] if not np.isnan(m120.iloc[-1]) else False, "股價站上半年線"),
            (m120.iloc[-1] > m120.iloc[-5] if not np.isnan(m120.iloc[-1]) else False, "長線趨勢翻正"),
            (50 < rsi < 75, "RSI強勢攻擊區"), (vol > avg_vol * 1.5, "量能顯著放大"),
            (c > m1200.iloc[-1] if not np.isnan(m1200.iloc[-1]) else True, "高於五年基期"),
            (c < m20.iloc[-1] + std20.iloc[-1]*2, "尚未觸及布林上軌"),
            (c > np.sum(((df['High']+df['Low']+df['Close'])/3)*df['Volume'])/np.sum(df['Volume']), "站穩VWAP均價"),
            (c > df['Close'].iloc[-2], "維持連漲慣性")
        ]
        for cond, msg in tests:
            if cond: score += 10; reasons.append(msg)
    except: return 0, []
    return score, reasons

def plot_v6_pro(df, title, days, resample_rule):
    df_slice = df.tail(days).copy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw={'height_ratios':[3, 1]}, sharex=True)
    ma20 = df_slice['Close'].rolling(20).mean()
    std20 = df_slice['Close'].rolling(20).std()
    ax1.fill_between(df_slice.index, ma20+std20*2, ma20-std20*2, color='blue', alpha=0.07, label='布林通道')
    ax1.plot(df_slice.index, df_slice['Close'], color='black', linewidth=1.8, label='收盤價')
    if 'MA120' in df_slice: ax1.plot(df_slice.index, df_slice['MA120'], label='半年線', color='red', ls='--')
    if 'MA1200' in df_slice: ax1.plot(df_slice.index, df_slice['MA1200'], label='五年線', color='green', ls='-.')
    ax1.set_ylim(df_slice['Low'].min()*0.98, df_slice['High'].max()*1.02)
    ax1.set_title(title); ax1.legend(loc='lower left'); ax1.grid(True, alpha=0.3)
    
    df_res = df_slice.resample(resample_rule).agg({'Open':'first', 'Close':'last', 'Volume':'sum'})
    colors = ['red' if df_res['Close'].iloc[i] >= df_res['Open'].iloc[i] else 'green' for i in range(len(df_res))]
    ax2.bar(df_res.index, df_res['Volume'], color=colors, width=(2.5 if resample_rule=='3D' else 5), alpha=0.8)
    return fig

# --- 4. 網頁 UI (紅灰配色) ---
st.sidebar.markdown("""
    <style>
    .potential-item { padding: 10px; border-radius: 5px; margin-bottom: 5px; background-color: #f0f2f6; border-left: 5px solid #ff4b4b; color: #31333F; }
    .potential-score { color: #ff4b4b; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("🚩 潛力參考名單")

@st.cache_data(ttl=3600)
def scan_potential():
    p_list = []
    # 僅對資料庫中的核心個股進行掃描以節省效能
    test_list = ["2330.TW", "2454.TW", "2317.TW", "3675.TWO", "6282.TW", "2303.TW", "3037.TW", "2382.TW", "6669.TW", "3231.TW", "2376.TW", "1513.TW", "1519.TW"]
    for t in test_list:
        d = fetch_stock_data(t, period="1y")
        s, _ = evaluate_stock_100(d)
        if s >= 75: p_list.append((get_company_name(t), t.split('.')[0], s))
    return sorted(p_list, key=lambda x: x[2], reverse=True)

for name, code, sc in scan_potential():
    st.sidebar.markdown(f'<div class="potential-item"><b>{name}</b> ({code})<br>AI 評分: <span class="potential-score">{sc}</span></div>', unsafe_allow_html=True)

st.title("🚀 2026 AI 台股決測系統 V6 Pro")
query_in = st.sidebar.text_input("輸入代號或名稱", "3675")

# 智慧轉換與查詢
ticker = query_in
name_to_ticker = {v: k for k, v in STOCK_DB.items()}
if query_in in name_to_ticker:
    ticker = name_to_ticker[query_in]
elif query_in.isdigit():
    ticker = query_in + ".TW"

if ticker:
    hist = fetch_stock_data(ticker, period="7y")
    if hist.empty and ".TW" in ticker:
        ticker = ticker.replace(".TW", ".TWO")
        hist = fetch_stock_data(ticker, period="7y")
        
    if not hist.empty:
        c_name = get_company_name(ticker)
        score, tags = evaluate_stock_100(hist)
        last_date = hist.index[-1].strftime('%Y-%m-%d')
        lp, pp = hist['Close'].iloc[-1], hist['Close'].iloc[-2]
        pct = ((lp - pp)/pp)*100
        p_color = "red" if pct > 0 else ("green" if pct < 0 else "black")
        
        st.markdown(f"### 📋 查詢標的：{ticker} - {c_name}")
        st.markdown(f"🕒 **最後收盤日**：{last_date}")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"## 現價: **{lp:,.2f}** <span style='color:{p_color}'>({pct:+.2f}%)</span>", unsafe_allow_html=True)
            st.pyplot(plot_v6_pro(hist, "半年趨勢 (三日量能)", 130, '3D'))
            st.pyplot(plot_v6_pro(hist, "五年長線 (每週量能)", 1250, 'W'))
        
        with col2:
            st.subheader(f"💡 AI 評分 (滿分100): {score}")
            for t in tags: st.success(t)
            
            st.subheader("📍 潛力象限分析")
            compare = ["2330.TW", "2317.TW", "3675.TWO", "6282.TW", "0050.TW"]
            if ticker not in compare: compare.append(ticker)
            q_list = []
            for t_item in compare:
                d_q = fetch_stock_data(t_item, period="5y")
                if d_q.empty: continue
                s_q, _ = evaluate_stock_100(d_q)
                c_q = ((d_q['Close'].iloc[-1]-d_q['Close'].iloc[-2])/d_q['Close'].iloc[-2])*100
                q_list.append({"T": t_item, "N": get_company_name(t_item), "S": s_q, "C": c_q})
            
            if q_list:
                q_df = pd.DataFrame(q_list)
                fig_q, ax_q = plt.subplots(figsize=(5, 5))
                colors = ['red' if r == ticker else 'royalblue' for r in q_df['T']]
                ax_q.scatter(q_df['S'], q_df['C'], c=colors, s=150, edgecolors='white', zorder=5)
                for i, txt in enumerate(q_df['N']):
                    ax_q.annotate(txt, (q_df['S'][i], q_df['C'][i]), fontsize=8, xytext=(5,5), textcoords='offset points')
                ax_q.axvline(50, color='gray', ls='--', alpha=0.5); ax_q.axhline(0, color='gray', ls='--', alpha=0.5)
                ax_q.set_xlim(0, 105); ax_q.set_xlabel("評分 (滿分100)"); ax_q.set_ylabel("漲跌幅 %")
                st.pyplot(fig_q)
