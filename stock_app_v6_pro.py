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
    "6669.TW": "緯穎", "3231.TW": "緯創", "2376.TW": "技嘉", "1513.TW": "中興電",
    "1519.TW": "華城", "4763.TW": "材料-KY", "8086.TWO": "宏捷科", "6438.TW": "迅得",
    "0050.TW": "元大台灣50", "00981A.TW": "統一台股增長", "2395.TW": "研華", "3034.TW": "聯詠"
}

# --- 2. 緩存數據抓取函數 (最重要：減少 API 調用) ---
@st.cache_data(ttl=3600)
def fetch_stock_data(ticker, period="7y"):
    """使用快取存儲股票數據，避免觸發 Rate Limit"""
    try:
        # 增加一個極短的隨機延遲，避免併發
        time.sleep(0.1) 
        data = yf.Ticker(ticker).history(period=period)
        return data
    except Exception:
        return pd.DataFrame()

def get_ticker_info(query):
    reverse_map = {v: k for k, v in CORE_LIST.items()}
    if query in reverse_map:
        return reverse_map[query], query
    if query.isdigit():
        # 為了節省次數，先預設 .TW，若失敗在主程式處理
        return query + ".TW", "自訂標的"
    return query, "自訂標的"

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
            (50 < rsi < 75, "RSI處於攻擊區"), (vol > avg_vol * 1.5, "量能顯著放大"),
            (c > m1200.iloc[-1] if not np.isnan(m1200.iloc[-1]) else True, "高於五年基期"),
            (c < m20.iloc[-1] + std20.iloc[-1]*2, "尚未觸及布林上軌"),
            (c > np.sum(((df['High']+df['Low']+df['Close'])/3)*df['Volume'])/np.sum(df['Volume']), "站穩VWAP均價"),
            (c > df['Close'].iloc[-2], "維持連漲慣性")
        ]
        for cond, msg in tests:
            if cond: score += 10; reasons.append(msg)
    except: return 0, []
    return score, reasons

# --- 3. 繪圖與 UI ---
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

# 側邊欄：75分潛力標的
st.sidebar.title("🔍 75分潛力標的")
@st.cache_data(ttl=3600)
def scan_potential():
    p_list = []
    # 這裡建議減少掃描數量，或增加延遲
    for t, n in CORE_LIST.items():
        d = fetch_stock_data(t, period="1y")
        s, _ = evaluate_stock_100(d)
        if s >= 75: p_list.append((n, t.split('.')[0], s))
        time.sleep(0.2) # 關鍵：掃描每檔標的間隔 0.2 秒
    return sorted(p_list, key=lambda x: x[2], reverse=True)

for name, code, sc in scan_potential():
    st.sidebar.success(f"🔥 {name} ({code}) : {sc}分")

# 主頁面
st.title("🚀 2026 AI 台股決策系統 V6 Pro")
query_input = st.text_input("輸入股票代號或名稱", "3675")
ticker_code, chinese_name = get_ticker_info(query_input)

if ticker_code:
    hist = fetch_stock_data(ticker_code, period="7y")
    # 如果 .TW 失敗，嘗試 .TWO (德微等上櫃股)
    if hist.empty and ".TW" in ticker_code:
        ticker_code = ticker_code.replace(".TW", ".TWO")
        hist = fetch_stock_data(ticker_code, period="7y")
        
    if not hist.empty:
        score, tags = evaluate_stock_100(hist)
        last_date = hist.index[-1].strftime('%Y-%m-%d')
        lp, pp = hist['Close'].iloc[-1], hist['Close'].iloc[-2]
        pct = ((lp - pp)/pp)*100
        p_color = "red" if pct > 0 else ("green" if pct < 0 else "black")
        
        st.markdown(f"### 📋 查詢：{ticker_code} - {chinese_name}")
        st.markdown(f"📅 **收盤日期**：{last_date}")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"## 現價: **{lp:,.2f}** <span style='color:{p_color}'>({pct:+.2f}%)</span>", unsafe_allow_html=True)
            st.pyplot(plot_v6_pro(hist, "半年趨勢 (三日量能)", 130, '3D'))
            st.pyplot(plot_v6_pro(hist, "五年長線 (每週量能)", 1250, 'W'))
        with col2:
            st.subheader(f"💡 AI 評分: {score}")
            for t in tags: st.success(t)
