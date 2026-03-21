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

# --- 1. 雲端環境與字體設定 (解決亂碼) ---
if os.name == 'posix':
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC', 'Noto Sans CJK JP', 'Liberation Sans', 'DejaVu Sans']
else:
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 核心數據處理 ---
@st.cache_data
def load_stock_names():
    names = {"2330.TW": "台積電", "2317.TW": "鴻海", "3675.TWO": "德微", "0050.TW": "元大台灣50", "2454.TW": "聯發科"}
    try:
        if os.path.exists('tw_stock_names.json'):
            with open('tw_stock_names.json', 'r', encoding='utf-8') as f:
                names.update(json.load(f))
    except: pass
    return names

STOCK_DB = load_stock_names()

def get_company_name(ticker):
    return STOCK_DB.get(ticker, ticker.split('.')[0])

@st.cache_data(ttl=60)
def fetch_stock_data(ticker, period="7y"):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval="1d", auto_adjust=True)
        return df
    except: return pd.DataFrame()

def evaluate_stock_100(df):
    if df.empty or len(df) < 240: return 0, ["數據不足以計算年線"]
    score, reasons = 0, []
    c = df['Close'].iloc[-1]
    m20 = df['Close'].rolling(20).mean().iloc[-1]
    m120 = df['Close'].rolling(120).mean().iloc[-1]
    m240 = df['Close'].rolling(240).mean().iloc[-1]
    vol = df['Volume'].iloc[-1]; vol_avg = df['Volume'].tail(15).mean()
    
    tests = [
        (c > m20, "股價站上月線 (短線支撐)"),
        (c > m120, "股價站上半年線 (中期趨勢)"),
        (c > m240, "股價站上年線 (長期生命線)"),
        (vol > vol_avg * 1.1, "今日量能增溫 (資金介入)"),
        (c > df['Close'].iloc[-2], "維持連漲慣性"),
        (m20 > df['Close'].rolling(20).mean().iloc[-5], "月線斜率向上")
    ]
    for cond, msg in tests:
        if cond: score += 15; reasons.append(msg)
    return min(int(score + random.randint(0, 10)), 100), reasons

def get_zhuge_advice(score):
    if score >= 85: return "【赤壁火攻】極強勢格局。噴火龍正在噴火，持股者抱緊，空手者切勿盲目追高。"
    elif score >= 65: return "【萬事俱備】趨勢偏多，年線以上無重大壓力。量縮回測均線即是佈局良機。"
    elif score >= 45: return "【兩軍對壘】均線糾結。此乃混水摸魚盤，短線震盪洗盤頻繁，建議觀望。"
    else: return "【火燒連環船】趨勢走空，空頭排列沉重。撤退保全資金為上策，切勿隨意攤平。"

# --- 3. 繪圖模組 ---
def plot_v6_pro(df, title, days, show_long_term=False):
    df_slice = df.tail(days).copy()
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), gridspec_kw={'height_ratios':[3, 1]}, sharex=True)
    
    # 布林通道與均線
    ma20 = df_slice['Close'].rolling(20).mean()
    std20 = df_slice['Close'].rolling(20).std()
    ax1.plot(df_slice.index, ma20, color='#FFA500', alpha=0.8, lw=1.2, ls='--', label='月線')
    ax1.plot(df_slice.index, df_slice['Close'], color='white', lw=1.8, label='收盤價', zorder=5)
    
    # 年線與五年線
    ax1.plot(df_slice.index, df['Close'].rolling(240).mean().tail(days), label='年線(MA240)', color='#A020F0', lw=1.5)
    if show_long_term:
        ax1.plot(df_slice.index, df['Close'].rolling(1200, min_periods=100).mean().tail(days), label='五年線', color='#00FF00', ls='-.', lw=1.8)
    
    ax1.set_title(title, fontsize=14); ax1.legend(loc='upper left', fontsize=8); ax1.grid(True, alpha=0.1)
    
    # 成交量
    colors = ['#FF4B4B' if df_slice['Close'].iloc[i] >= df_slice['Open'].iloc[i] else '#00E676' for i in range(len(df_slice))]
    ax2.bar(df_slice.index, df_slice['Volume'], color=colors, alpha=0.7)
    fig.patch.set_alpha(0.0); plt.tight_layout()
    return fig

def plot_prediction_chart(df, ticker_name):
    df_recent = df.tail(15).copy()
    y = df_recent['Close'].values
    X = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    future_X = np.arange(len(y), len(y) + 5).reshape(-1, 1)
    preds = model.predict(future_X)
    future_dates = [df_recent.index[-1] + timedelta(days=i) for i in range(1, 6)]
    
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df_recent.index, y, color='white', lw=2, label='近期收盤', marker='o', markersize=4)
    ax.plot(future_dates, preds, color='#FF4B4B', ls=':', marker='s', label='AI 模擬預測')
    
    # 防數字重疊：上下位移標註
    y_range = max(max(y), max(preds)) - min(min(y), min(preds))
    offset = y_range * 0.08
    for i, (d, p) in enumerate(zip(future_dates, preds)):
        va = 'bottom' if i % 2 == 0 else 'top'
        y_pos = p + offset if i % 2 == 0 else p - offset
        ax.text(d, y_pos, f'{p:.1f}', color='#FF4B4B', fontsize=10, fontweight='bold', ha='center', va=va)
    
    ax.set_title(f"🔮 {ticker_name} 未來五日 AI 預測", color='white'); ax.grid(True, alpha=0.1)
    fig.patch.set_alpha(0.0); plt.tight_layout()
    return fig

# --- 4. 網頁 UI 與 強制深色模式 CSS ---
st.set_page_config(page_title="台股｜AI 諸葛孔明", layout="wide")

st.markdown("""
    <style>
    /* 強制主背景與側邊欄黑底 */
    [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #0E1117 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #161920 !important;
        border-right: 1px solid #333;
    }
    /* 側邊欄文字強制白色 */
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
        color: #FFFFFF !important;
    }
    /* 卡片式設計 */
    .zhuge-advice { background: rgba(255, 75, 75, 0.12); padding: 20px; border-radius: 10px; border: 1px solid #FF4B4B; color: white; margin-bottom: 20px; }
    .data-card { background: rgba(255, 255, 255, 0.04); padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #444; }
    h1, h2, h3 { color: #FFFFFF !important; }
    </style>
""", unsafe_allow_html=True)

# 背景圖片設定
if os.path.exists('孔明看盤.png'):
    with open('孔明看盤.png', "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(f"""<style>[data-testid="stAppViewContainer"]::before {{ content: ""; position: absolute; top: 0; left: 0; width: 100%; height: 100%; background-image: url("data:image/png;base64,{encoded}"); background-size: cover; background-attachment: fixed; opacity: 0.15; z-index: -1; }}</style>""", unsafe_allow_html=True)

# 側邊欄配置
st.sidebar.title("⌨️ 諸葛神算")
query_in = st.sidebar.text_input("輸入代號 (例: 2330)", "3675").upper()

st.sidebar.markdown("---")
st.sidebar.subheader("🎲 即時監控清單")
watchlist = ["2330.TW", "2317.TW", "2454.TW", "2308.TW", "2603.TW", "3675.TWO", "0050.TW"]
for t in random.sample(watchlist, 5):
    d_w = fetch_stock_data(t, period="5d")
    if not d_w.empty:
        st.sidebar.markdown(f"🔹 {get_company_name(t)} ({t.split('.')[0]}) : `{d_w['Close'].iloc[-1]:.1f}`")

# 主畫面內容
st.markdown("<h1 style='text-align: center;'>🚀 台股｜AI 諸葛孔明分析系統</h1>", unsafe_allow_html=True)

ticker = query_in
if ticker:
    # 自動補完代號
    if not (ticker.endswith(".TW") or ticker.endswith(".TWO")):
        test = fetch_stock_data(ticker + ".TW", period="5d")
        ticker = ticker + ".TWO" if test.empty else ticker + ".TW"

    hist = fetch_stock_data(ticker, period="7y")
    if not hist.empty:
        c_name = get_company_name(ticker)
        score, tags = evaluate_stock_100(hist)
        lp = hist['Close'].iloc[-1]; pct = ((lp / hist['Close'].iloc[-2]) - 1) * 100
        
        st.subheader(f"📊 {ticker} - {c_name} 診斷報告")
        col1, col2 = st.columns(2)
        col1.markdown(f"<div class='data-card'>當前市價<br><span style='font-size:1.8rem; color:#00E676;'>{lp:,.2f}</span> ({pct:+.2f}%)</div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='data-card'>AI 諸葛評分<br><span style='font-size:1.8rem; color:#FF4B4B;'>{score} 分</span></div>", unsafe_allow_html=True)
        
        with st.expander("🔍 檢視詳細評分依據"):
            for t in tags: st.write(f"✅ {t}")
        
        st.markdown(f'<div class="zhuge-advice"><b>📜 諸葛錦囊：</b><br>{get_zhuge_advice(score)}</div>', unsafe_allow_html=True)

        # 趨勢圖表
        st.pyplot(plot_v6_pro(hist, "半年趨勢 (含年線 MA240)", 130))
        st.pyplot(plot_v6_pro(hist, "五年長線 (含五年生命線)", 1250, show_long_term=True))

        # 象限與預測
        st.markdown("---")
        st.subheader("🔮 AI 量價預測與象限分析")
        p_col1, p_col2 = st.columns([2, 1])
        with p_col1:
            st.pyplot(plot_prediction_chart(hist, c_name))
        with p_col2:
            st.write("📈 **個股定位**")
            st.info(f"該股目前在 AI 評分系統中屬於「{ '強勢領漲' if score > 70 else '區間震盪' if score > 40 else '弱勢整理' }」區域。")
            
    else:
        st.error("無法取得數據，請確認代號是否輸入正確（例如：2330）。")

st.markdown("<div style='color: #666; font-size: 0.8em; text-align: center; margin-top: 50px;'>⚠️ 免責聲明：本系統僅供技術研究參考。投資一定有風險，請審慎評估。</div>", unsafe_allow_html=True)
