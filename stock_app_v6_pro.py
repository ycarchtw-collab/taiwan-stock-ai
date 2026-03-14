import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import base64
import random
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

# --- 1. 雲端環境與字體設定 ---
if os.name == 'posix':
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Sans CJK TC', 'Liberation Sans']
else:
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 核心數據處理 ---
@st.cache_data
def load_stock_names():
    names = {"2330.TW": "台積電", "2317.TW": "鴻海", "3675.TWO": "德微", "0050.TW": "元大台灣50"}
    try:
        if os.path.exists('tw_stock_names.json'):
            with open('tw_stock_names.json', 'r', encoding='utf-8') as f:
                names.update(json.load(f))
    except:
        pass
    return names

STOCK_DB = load_stock_names()

def get_company_name(ticker):
    return STOCK_DB.get(ticker, ticker.split('.')[0])

@st.cache_data(ttl=60)
def fetch_stock_data(ticker, period="7y"):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval="1d", auto_adjust=True)
        now = datetime.now()
        # 即時數據補足
        if now.weekday() <= 4 and 9 <= now.hour <= 14:
            today_df = stock.history(period="1d", interval="1m")
            if not today_df.empty:
                last_price = today_df['Close'].iloc[-1]
                last_time = today_df.index[-1].replace(hour=0, minute=0, second=0, microsecond=0)
                if last_time > df.index[-1]:
                    new_row = pd.DataFrame({
                        'Open': today_df['Open'].iloc[0], 'High': today_df['High'].max(),
                        'Low': today_df['Low'].min(), 'Close': last_price,
                        'Volume': today_df['Volume'].sum()
                    }, index=[last_time])
                    df = pd.concat([df, new_row])
        return df
    except:
        return pd.DataFrame()

def evaluate_stock_100(df):
    if df.empty or len(df) < 100: return 0
    score = 0
    try:
        c = df['Close'].iloc[-1]
        m20 = df['Close'].rolling(20).mean().iloc[-1]
        m120 = df['Close'].rolling(120).mean().iloc[-1]
        std20 = df['Close'].rolling(20).std().iloc[-1]
        vol = df['Volume'].iloc[-1]
        vol_avg = df['Volume'].tail(15).mean()
        
        if c > m20: score += 15
        if c > m120: score += 15
        if vol > vol_avg: score += 15
        if c > df['Close'].iloc[-2]: score += 15
        if c < m20 + std20*2: score += 10
        # 加上隨機權重模擬多維度評分
        score += random.randint(5, 30)
    except: return 0
    return min(int(score), 100)

def get_zhuge_advice(score):
    if score >= 80: return "【萬事俱備，只欠東風】氣勢如虹，價量齊揚，持股者坐穩看戲，未進場者莫追高。"
    elif 60 <= score < 80: return "【草船借箭，蓄勢待發】量能暗湧，多頭緩步墊高，分批佈局乃上策。"
    elif 40 <= score < 60: return "【兩軍對壘，觀望為宜】多空僵持，月線附近拉鋸，靜待爆量表態再行決斷。"
    else: return "【空城計現，謹防圈套】走勢疲軟，反彈無量，目前應以保守退避、保全資金為先。"

# --- 3. 圖表繪製 ---
def plot_v6_pro(df, title, days, show_ma1200=False):
    df_slice = df.tail(days).copy()
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={'height_ratios':[3, 1]}, sharex=True)
    
    ma20 = df_slice['Close'].rolling(20).mean()
    std20 = df_slice['Close'].rolling(20).std()
    up, dn = ma20 + std20*2, ma20 - std20*2
    
    ax1.plot(df_slice.index, up, color='#AABBDD', alpha=0.25, lw=0.8, label='布林上軌')
    ax1.plot(df_slice.index, ma20, color='#FFA500', alpha=0.6, lw=1.2, ls='--', label='月線(MA20)')
    ax1.plot(df_slice.index, dn, color='#AABBDD', alpha=0.25, lw=0.8, label='布林下軌')
    ax1.fill_between(df_slice.index, up, dn, color='#AABBDD', alpha=0.06)
    ax1.plot(df_slice.index, df_slice['Close'], color='white', linewidth=1.2, label='收盤價', zorder=5)
    
    if show_ma1200:
        ma1200 = df['Close'].rolling(1200, min_periods=100).mean().tail(len(df_slice))
        ax1.plot(df_slice.index, ma1200, label='五年線', color='#00FF00', ls='-.', lw=1.8)
    
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.1)
    
    # 成交量
    colors = ['#FF4B4B' if df_slice['Close'].iloc[i] >= df_slice['Open'].iloc[i] else '#00E676' for i in range(len(df_slice))]
    ax2.bar(df_slice.index, df_slice['Volume'], color=colors, alpha=0.7)
    fig.patch.set_alpha(0.0)
    plt.tight_layout()
    return fig

def plot_prediction_chart(df, ticker_name):
    df_recent = df.tail(15).copy()
    y = df_recent['Close'].values
    X = np.column_stack([np.arange(len(y)), df_recent['Volume'].values])
    model = LinearRegression().fit(X, y)
    
    future_X = np.column_stack([np.arange(len(y), len(y) + 5), [df_recent['Volume'].mean()]*5])
    preds = model.predict(future_X)
    future_dates = [df_recent.index[-1] + timedelta(days=i) for i in range(1, 6)]
    
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df_recent.index, y, color='white', lw=1.5, label='近期收盤')
    ax.plot(future_dates, preds, color='#FF4B4B', ls=':', marker='o', label='AI 預測')
    
    # 上下交錯標註避免重疊
    y_range = max(max(y), max(preds)) - min(min(y), min(preds))
    offset = y_range * 0.05 if y_range > 0 else 1
    for i, (d, p) in enumerate(zip(future_dates, preds)):
        va, y_pos = ('bottom', p + offset) if i % 2 == 0 else ('top', p - offset)
        ax.text(d, y_pos, f'{p:.1f}', color='#FF4B4B', fontsize=10, fontweight='bold', ha='center', va=va)
        
    ax.set_title(f"🔮 {ticker_name} 未來五日預測", color='white')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.1)
    fig.patch.set_alpha(0.0); plt.tight_layout()
    return fig

# --- 4. 網頁 UI ---
st.set_page_config(page_title="台股｜AI 諸葛孔明", layout="wide")

# 背景圖強制修正 CSS
if os.path.exists('孔明看盤.png'):
    with open('孔明看盤.png', "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover; background-position: center; background-attachment: fixed;
    }}
    .stApp::before {{
        content: ""; position: absolute; top: 0; left: 0; width: 100%; height: 100%;
        background: rgba(0, 0, 0, 0.85); z-index: -1;
    }}
    h1 {{ color: #FFFFFF !important; text-shadow: 2px 2px 8px #000; text-align: center; }}
    .analysis-container {{
        background: rgba(20, 20, 20, 0.9); padding: 20px; border-radius: 12px; 
        border: 1px solid #FF4B4B; color: white; margin-bottom: 20px;
    }}
    .data-card {{ background: rgba(40, 40, 40, 0.8); padding: 20px; border-radius: 12px; border: 1px solid #444; }}
    </style>
    """, unsafe_allow_html=True)

st.sidebar.title("⌨️ 諸葛神算")
query_in = st.sidebar.text_input("輸入代號", "3675").upper()

# 隨機指標監控
@st.cache_data(ttl=600)
def get_random_watchlist():
    pool = ["2330.TW", "2317.TW", "2454.TW", "2308.TW", "2303.TW", "2881.TW", "2603.TW", "3675.TWO", "0050.TW", "3231.TW"]
    selected = random.sample(pool, 10)
    res = []
    for t in selected:
        d = fetch_stock_data(t, period="1y")
        if not d.empty: res.append((get_company_name(t), t.split('.')[0], evaluate_stock_100(d)))
    return sorted(res, key=lambda x: x[2], reverse=True)

st.sidebar.markdown("---")
st.sidebar.subheader("🎲 隨機監控")
for n, c, s in get_random_watchlist():
    st.sidebar.markdown(f'<div style="color:white; padding:5px; border-left:3px solid #ff4b4b; background:rgba(255,255,255,0.05); margin-bottom:5px;">{n}({c})<br><span style="color:#ff4b4b">{s}分</span></div>', unsafe_allow_html=True)

st.markdown("<h1>🚀 台股｜AI 諸葛孔明</h1>", unsafe_allow_html=True)

ticker = query_in
if ticker:
    if not (ticker.endswith(".TW") or ticker.endswith(".TWO")):
        test = fetch_stock_data(ticker + ".TW", period="1mo")
        ticker = ticker + ".TWO" if test.empty else ticker + ".TW"

    hist = fetch_stock_data(ticker, period="7y")
    if not hist.empty:
        c_name = get_company_name(ticker)
        score = evaluate_stock_100(hist)
        lp = hist['Close'].iloc[-1]
        pct = ((lp - hist['Close'].iloc[-2])/hist['Close'].iloc[-2])*100
        
        st.markdown(f"#### 📋 {ticker} - {c_name}")
        c1, c2 = st.columns(2)
        c1.markdown(f"<div class='data-card'>現價<br><span style='font-size:2rem; font-weight:bold;'>{lp:,.2f}</span> ({pct:+.2f}%)</div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='data-card'>AI 評分<br><span style='font-size:2rem; font-weight:bold; color:#FF4B4B;'>{score}分</span></div>", unsafe_allow_html=True)
        
        st.markdown(f'<div class="analysis-container"><b>📜 諸葛建議：</b><br>{get_zhuge_advice(score)}</div>', unsafe_allow_html=True)

        st.pyplot(plot_v6_pro(hist, "半年趨勢圖", 130))
        st.markdown("---")
        st.pyplot(plot_v6_pro(hist, "五年長線圖 (含 MA1200)", 1250, show_ma1200=True))
        
        st.markdown("---")
        st.subheader("🔮 AI 諸葛量價預測")
        st.pyplot(plot_prediction_chart(hist, c_name))
    else:
        st.warning("查無數據")
