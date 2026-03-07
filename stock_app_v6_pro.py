import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# --- 頁面配置 ---
st.set_page_config(page_title="2026 AI 台股決策系統 V6 Pro", layout="wide")
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 核心追蹤名單 (用於側邊欄潛力股篩選)
CORE_LIST = {
    "2330.TW": "台積電", "2454.TW": "聯發科", "2317.TW": "鴻海", "3675.TWO": "德微",
    "6282.TW": "康舒", "2303.TW": "聯電", "3037.TW": "欣興", "2382.TW": "廣達",
    "6669.TW": "緯穎", "3231.TW": "緯創", "2376.TW": "技嘉", "1513.TW": "中興電",
    "1519.TW": "華城", "4763.TW": "材料-KY", "8086.TWO": "宏捷科", "6438.TW": "迅得",
    "0050.TW": "元大台灣50", "00981A.TW": "統一台股增長", "2395.TW": "研華", "3034.TW": "聯詠"
}

# --- 核心邏輯函數 ---

def get_ticker(query):
    mapping = {v: k for k, v in CORE_LIST.items()}
    code = mapping.get(query, query)
    if code.isdigit():
        for sfx in [".TW", ".TWO"]:
            t = yf.Ticker(code + sfx)
            if not t.history(period="1d").empty: return code + sfx, query
    return code, query

def calculate_rsi(df, periods=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def evaluate_stock_100(df):
    if len(df) < 20: return 0, []
    score = 0
    reasons = []
    try:
        c = df['Close'].iloc[-1]
        m20 = df['Close'].rolling(20).mean()
        m120 = df['Close'].rolling(120).mean()
        m1200 = df['Close'].rolling(1200).mean()
        std20 = df['Close'].rolling(20).std()
        rsi = calculate_rsi(df).iloc[-1]
        vol, avg_vol = df['Volume'].iloc[-1], df['Volume'].tail(5).mean()
        
        tests = [
            (c > m20.iloc[-1], "站上月線"), (m20.iloc[-1] > m20.iloc[-5], "月線向上"),
            (c > m120.iloc[-1], "站上半年線"), (m120.iloc[-1] > m120.iloc[-5], "趨勢翻正"),
            (50 < rsi < 75, "RSI強勢區"), (vol > avg_vol * 1.5, "量能放大"),
            (c > m1200.iloc[-1] if not np.isnan(m1200.iloc[-1]) else True, "高於五年基期"),
            (c < m20.iloc[-1] + std20.iloc[-1]*2, "尚未過熱"),
            (c > np.sum(((df['High']+df['Low']+df['Close'])/3)*df['Volume'])/np.sum(df['Volume']), "站穩VWAP"),
            (c > df['Close'].iloc[-2], "連漲慣性")
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
    ax2.set_ylabel("成交量加總")
    return fig

# --- 側邊欄：75分以上潛力名單 ---
st.sidebar.title("🔍 潛力參考名單")
st.sidebar.caption("AI 評分 75 分以上之強勢股")

with st.sidebar:
    potential_list = []
    # 進行背景快速篩選
    for t, n in CORE_LIST.items():
        d = yf.Ticker(t).history(period="1y")
        s, _ = evaluate_stock_100(d)
        if s >= 75:
            potential_list.append(f"🔥 {n} ({t.split('.')[0]}) : {s}分")
    
    if potential_list:
        for item in potential_list[:20]:
            st.write(item)
    else:
        st.write("暫無符合標的")

# --- 主頁面 UI ---
st.title("🚀 2026 AI 台股決策系統 V6 Pro")
query_input = st.text_input("輸入代號或名稱", "3675")
ticker, c_name = get_ticker(query_input)

if ticker:
    stock_obj = yf.Ticker(ticker)
    hist = stock_obj.history(period="7y")
    
    if not hist.empty:
        # 獲取中文名稱 (若 query 是代號，嘗試抓取其資訊)
        full_name = CORE_LIST.get(ticker, stock_obj.info.get('longName', c_name))
        last_trade_date = hist.index[-1].strftime('%Y-%m-%d')
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M')
        
        hist['MA120'], hist['MA1200'] = hist['Close'].rolling(120).mean(), hist['Close'].rolling(1200).mean()
        score, tags = evaluate_stock_100(hist)
        
        # 報價顯示
        last_p, prev_p = hist['Close'].iloc[-1], hist['Close'].iloc[-2]
        pct = ((last_p - prev_p)/prev_p)*100
        p_color = "red" if pct > 0 else ("green" if pct < 0 else "black")
        
        st.markdown(f"### 📋 查詢標的：{ticker} - {full_name}")
        st.markdown(f"📅 **查詢日期**：{current_time} | 🕒 **最後收盤日**：{last_trade_date}")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"## 現價: **{last_p:,.2f}** <span style='color:{p_color}'>({pct:+.2f}%)</span>", unsafe_allow_html=True)
            st.pyplot(plot_v6_pro(hist, "半年趨勢 (3日量能)", 130, '3D'))
            st.pyplot(plot_v6_pro(hist, "五年趨勢 (每週量能)", 1250, 'W'))
        
        with col2:
            st.subheader(f"💡 AI 評分: {score} / 100")
            for t in tags: st.success(t)
            
            # 象限圖
            st.subheader("📍 潛力象限分析")
            compare = ["2330.TW", "2317.TW", "3675.TWO", "6282.TW", "0050.TW"]
            if ticker not in compare: compare.append(ticker)
            q_list = []
            for t_item in compare:
                d = yf.Ticker(t_item).history(period="5y")
                s, _ = evaluate_stock_100(d)
                c = ((d['Close'].iloc[-1]-d['Close'].iloc[-2])/d['Close'].iloc[-2])*100
                q_list.append({"T": t_item, "S": s, "C": c})
            
            q_df = pd.DataFrame(q_list)
            fig_q, ax_q = plt.subplots(figsize=(5, 5))
            colors = ['red' if r == ticker else 'royalblue' for r in q_df['T']]
            ax_q.scatter(q_df['S'], q_df['C'], c=colors, s=180, edgecolors='white', zorder=5)
            for i, txt in enumerate(q_df['T']):
                ax_q.annotate(txt, (q_df['S'][i], q_df['C'][i]), fontsize=8, xytext=(5,5), textcoords='offset points')
            ax_q.axvline(50, color='gray', ls='--', alpha=0.5); ax_q.axhline(0, color='gray', ls='--', alpha=0.5)
            st.pyplot(fig_q)
