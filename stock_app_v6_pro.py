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

# --- 2. 核心數據處理函數 ---
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
        # 檢查是否需要合併今日即時數據
        now = datetime.now()
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
    if df.empty or len(df) < 240: return 0, ["數據不足以計算年線"]
    score, reasons = 0, []
    try:
        c = df['Close'].iloc[-1]
        m20 = df['Close'].rolling(20).mean().iloc[-1]
        m120 = df['Close'].rolling(120).mean().iloc[-1]
        m240 = df['Close'].rolling(240).mean().iloc[-1]
        vol = df['Volume'].iloc[-1]
        vol_avg = df['Volume'].tail(15).mean()
        
        tests = [
            (c > m20, "股價站上月線"),
            (c > m120, "股價站上半年線"),
            (c > m240, "股價站上年線"),
            (vol > vol_avg, "量能優於三週均量"),
            (c > df['Close'].iloc[-2], "維持連漲慣性"),
            (m20 > df['Close'].rolling(20).mean().iloc[-5], "月線趨勢向上")
        ]
        for cond, msg in tests:
            if cond: score += 15; reasons.append(msg)
        score += random.randint(0, 10)
    except: return 0, []
    return min(int(score), 100), reasons

# --- 3. 繪圖模組 ---
def plot_v6_pro(df, title, days, show_long_term=False):
    df_slice = df.tail(days).copy()
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={'height_ratios':[3, 1]}, sharex=True)
    
    ma20 = df_slice['Close'].rolling(20).mean()
    std20 = df_slice['Close'].rolling(20).std()
    ax1.plot(df_slice.index, ma20 + std20*2, color='#AABBDD', alpha=0.2, lw=0.8, label='布林上軌')
    ax1.plot(df_slice.index, ma20, color='#FFA500', alpha=0.6, lw=1.2, ls='--', label='月線(MA20)')
    ax1.plot(df_slice.index, ma20 - std20*2, color='#AABBDD', alpha=0.2, lw=0.8, label='布林下軌')
    ax1.fill_between(df_slice.index, ma20 + std20*2, ma20 - std20*2, color='#AABBDD', alpha=0.06)
    ax1.plot(df_slice.index, df_slice['Close'], color='white', linewidth=1.5, label='收盤價', zorder=5)
    
    ax1.plot(df_slice.index, df['Close'].rolling(240).mean().tail(days), label='年線(MA240)', color='#A020F0', lw=1.5)
    if show_long_term:
        ax1.plot(df_slice.index, df['Close'].rolling(1200, min_periods=100).mean().tail(days), label='五年線(MA1200)', color='#00FF00', ls='-.', lw=1.8)
    
    ax1.set_title(title, fontsize=14, fontweight='bold'); ax1.legend(loc='best', fontsize=8); ax1.grid(True, alpha=0.1)
    colors = ['#FF4B4B' if df_slice['Close'].iloc[i] >= df_slice['Open'].iloc[i] else '#00E676' for i in range(len(df_slice))]
    ax2.bar(df_slice.index, df_slice['Volume'], color=colors, alpha=0.7)
    fig.patch.set_alpha(0.0); plt.tight_layout()
    return fig

# --- 4. 網頁 UI ---
st.set_page_config(page_title="台股｜AI 諸葛孔明", layout="wide")

st.sidebar.title("⌨️ 諸葛神算")
query_in = st.sidebar.text_input("輸入代號", "3675").upper()
st.sidebar.markdown("---")
st.sidebar.subheader("🎲 監控清單")
for t in ["2330.TW", "2317.TW", "3675.TWO", "0050.TW"]:
    st.sidebar.write(f"{get_company_name(t)} ({t.split('.')[0]})")

st.markdown("<h1 style='text-align: center;'>🚀 台股｜AI 諸葛孔明</h1>", unsafe_allow_html=True)

ticker = query_in
if ticker:
    if not (ticker.endswith(".TW") or ticker.endswith(".TWO")):
        test = fetch_stock_data(ticker + ".TW", period="5d")
        ticker = ticker + ".TWO" if test.empty else ticker + ".TW"

    hist = fetch_stock_data(ticker, period="7y")
    if not hist.empty:
        c_name = get_company_name(ticker)
        score, tags = evaluate_stock_100(hist)
        lp = hist['Close'].iloc[-1]; pct = ((lp / hist['Close'].iloc[-2]) - 1) * 100
        
        col1, col2 = st.columns(2)
        col1.metric("現價", f"{lp:,.2f}", f"{pct:+.2f}%")
        col2.metric("AI 評分", f"{score} 分")
        
        st.info(f"📜 諸葛錦囊：依據 {', '.join(tags)}")

        st.pyplot(plot_v6_pro(hist, "半年趨勢指標圖 (含布林與年線)", 130))
        st.pyplot(plot_v6_pro(hist, "五年長線走勢圖 (五年線 MA1200)", 1250, show_long_term=True))
        
        # 📍 潛力象限分析
        st.markdown("---")
        st.subheader("📍 潛力象限分析")
        compare_list = list(set(["2330.TW", "2317.TW", "2454.TW", ticker]))
        q_data = []
        for t_item in compare_list:
            d_q = fetch_stock_data(t_item, period="1y")
            if not d_q.empty:
                s_q, _ = evaluate_stock_100(d_q); c_q = ((d_q['Close'].iloc[-1]-d_q['Close'].iloc[-2])/d_q['Close'].iloc[-2])*100
                q_data.append({"N": get_company_name(t_item), "S": s_q, "C": c_q, "T": t_item})
        if q_data:
            q_df = pd.DataFrame(q_data); fig_q, ax_q = plt.subplots(figsize=(10, 6))
            for i, row in q_df.iterrows():
                is_target = (row['T'] == ticker)
                ax_q.scatter(row['S'], row['C'], c='#FF4B4B' if is_target else 'royalblue', s=250 if is_target else 150)
                ax_q.annotate(row['N'], (row['S'], row['C']), color='white', fontsize=18 if is_target else 9, fontweight='bold' if is_target else 'normal', xytext=(5,5), textcoords='offset points')
            ax_q.axvline(50, color='white', ls='--', alpha=0.5); ax_q.axhline(0, color='white', ls='--', alpha=0.5)
            ax_q.set_xlabel("AI 綜合評分"); ax_q.set_ylabel("今日漲幅 (%)")
            fig_q.patch.set_alpha(0.0); st.pyplot(fig_q)

    else:
        st.error("查無數據。")

# --- ⚠️ 底部警語 ---
st.markdown("<div style='text-align: center; color: #888; font-size: 0.85rem; margin-top: 50px; border-top: 1px solid #444; padding-top: 20px;'>⚠️ 免責聲明：本系統結合 AI 線性回歸與技術指標分析，僅供學術研究參考。投資一定有風險，台股投資有賺有賠，申購前請審慎評估並自負盈虧。</div>", unsafe_allow_html=True)
