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

# --- 3. 核心運算與指標計算 ---
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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={'height_ratios':[3, 1]}, sharex=True)
    
    # 計算指標線
    ma20 = df_slice['Close'].rolling(20).mean()
    std20 = df_slice['Close'].rolling(20).std()
    up, dn = ma20 + std20*2, ma20 - std20*2
    
    # 價格與指標走勢線
    ax1.plot(df_slice.index, up, color='blue', alpha=0.3, lw=1, label='布林上軌')
    ax1.plot(df_slice.index, dn, color='blue', alpha=0.3, lw=1, label='布林下軌')
    ax1.fill_between(df_slice.index, up, dn, color='blue', alpha=0.05)
    
    ax1.plot(df_slice.index, df_slice['Close'], color='black', linewidth=2, label='收盤價')
    
    # 加入均線走勢
    ma120 = df['Close'].rolling(120).mean().tail(days)
    ma1200 = df['Close'].rolling(1200, min_periods=100).mean().tail(days)
    
    ax1.plot(df_slice.index, ma120, label='半年線 (MA120)', color='red', ls='--', lw=1.5)
    ax1.plot(df_slice.index, ma1200, label='五年線 (MA1200)', color='green', ls='-.', lw=1.5)
    
    ax1.set_ylim(df_slice['Low'].min()*0.97, df_slice['High'].max()*1.03)
    ax1.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='best', fontsize=8); ax1.grid(True, alpha=0.2)
    
    # 成交量柱狀圖
    df_res = df_slice.resample(resample_rule).agg({'Open':'first', 'Close':'last', 'Volume':'sum'})
    colors = ['#ff4b4b' if df_res['Close'].iloc[i] >= df_res['Open'].iloc[i] else '#008000' for i in range(len(df_res))]
    ax2.bar(df_res.index, df_res['Volume'], color=colors, width=(2.5 if resample_rule=='3D' else 5), alpha=0.7)
    ax2.set_ylabel("成交量")
    
    plt.tight_layout()
    return fig

# --- 4. 網頁 UI 佈局 ---
st.sidebar.markdown("""
    <style>
    .potential-item { padding: 10px; border-radius: 5px; margin-bottom: 5px; background-color: #f0f2f6; border-left: 5px solid #ff4b4b; color: #31333F; }
    .potential-score { color: #ff4b4b; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# 需求 1：置頂輸入框
st.sidebar.title("⌨️ 查詢系統")
query_in = st.sidebar.text_input("輸入股票代號或公司中文", "3675")

st.sidebar.markdown("---")
st.sidebar.subheader("🚩 潛力參考名單")
@st.cache_data(ttl=3600)
def scan_potential():
    p_list = []
    test_list = ["2330.TW", "2454.TW", "2317.TW", "3675.TWO", "6282.TW", "2303.TW", "3037.TW", "2382.TW", "6669.TW"]
    for t in test_list:
        d = fetch_stock_data(t, period="1y")
        s, _ = evaluate_stock_100(d)
        if s >= 75: p_list.append((get_company_name(t), t.split('.')[0], s))
    return sorted(p_list, key=lambda x: x[2], reverse=True)

for name, code, sc in scan_potential():
    st.sidebar.markdown(f'<div class="potential-item"><b>{name}</b> ({code})<br>AI 評分: <span class="potential-score">{sc} 分</span></div>', unsafe_allow_html=True)

# 主畫面
st.title("🚀 2026 AI 台股決策系統 V6 Pro")

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
        
        st.markdown(f"### 📋 查詢標的：{ticker} - {c_name} | 🕒 最後收盤日：{last_date}")
        
        info_col, score_col = st.columns([1.5, 1])
        with info_col:
            st.markdown(f"## 現價: **{lp:,.2f}** <span style='color:{p_color}'>({pct:+.2f}%)</span>", unsafe_allow_html=True)
        with score_col:
            # 需求 4：分數後加單位「分」
            st.markdown(f"### 💡 AI 評分: <span style='color:#ff4b4b'>{score} 分</span>", unsafe_allow_html=True)

        # 需求 3：圖表抬頭
        st.markdown("---")
        
        st.pyplot(plot_v6_pro(hist, f"【{c_name}】半年波段指標圖 (含布林通道/半年線)", 130, '3D'))
        
        st.markdown("---")
        
        st.pyplot(plot_v6_pro(hist, f"【{c_name}】五年長線循環圖 (含五年均線)", 1250, 'W'))

        # 需求 2：象限圖放大並移至底部
        st.markdown("---")
        st.subheader("📍 潛力象限分析 (市場位階對比)")
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
            fig_q, ax_q = plt.subplots(figsize=(12, 6))
            colors = ['#ff4b4b' if r == ticker else 'royalblue' for r in q_df['T']]
            ax_q.scatter(q_df['S'], q_df['C'], c=colors, s=250, edgecolors='white', zorder=5, alpha=0.8)
            
            for i, txt in enumerate(q_df['N']):
                ax_q.annotate(txt, (q_df['S'][i], q_df['C'][i]), fontsize=10, xytext=(5,5), textcoords='offset points', fontweight='bold')
            
            ax_q.axvline(50, color='gray', ls='--', alpha=0.5)
            ax_q.axhline(0, color='gray', ls='--', alpha=0.5)
            ax_q.set_xlim(0, 105); ax_q.set_xlabel("AI 評分 (分)"); ax_q.set_ylabel("今日漲跌幅 (%)")
            ax_q.set_title("右上：強勢進攻區 / 左下：弱勢修正區", fontsize=12, pad=10)
            
            st.pyplot(fig_q)
