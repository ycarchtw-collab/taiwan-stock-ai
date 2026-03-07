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

# --- 2. 名稱加載邏輯 ---
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

# --- 3. 核心運算與指標 ---
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
    if df.empty or len(df) < 100: 
        return 0, []

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
            (c > m20.iloc[-1], "股價站上月線"), 
            (m20.iloc[-1] > m20.iloc[-5] if len(m20)>5 else False, "月線趨勢向上"),
            (c > m120.iloc[-1] if not np.isnan(m120.iloc[-1]) else False, "股價站上半年線"),
            (m120.iloc[-1] > m120.iloc[-5] if not np.isnan(m120.iloc[-1]) else False, "長線趨勢翻正"),
            (50 < rsi < 75, "RSI 強勢攻擊區"), 
            (vol > avg_vol * 1.5, "量能顯著放大"),
            (c > m1200.iloc[-1] if not np.isnan(m1200.iloc[-1]) else True, "高於五年基期"),
            (c < m20.iloc[-1] + std20.iloc[-1]*2, "尚未觸及布林上軌"),
            (c > np.sum(((df['High']+df['Low']+df['Close'])/3)*df['Volume'])/np.sum(df['Volume']), "站穩 VWAP 均價"),
            (c > df['Close'].iloc[-2], "維持連漲慣性")
        ]
        for cond, msg in tests:
            if cond: score += 10; reasons.append(msg)
    except: return 0, []
    return score, reasons

def plot_v6_pro(df, title, days, resample_rule):
    df_slice = df.tail(days).copy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={'height_ratios':[3, 1]}, sharex=True)
    
    ma20 = df_slice['Close'].rolling(20).mean()
    std20 = df_slice['Close'].rolling(20).std()
    up, dn = ma20 + std20*2, ma20 - std20*2
    
    ax1.plot(df_slice.index, up, color='blue', alpha=0.2, lw=0.8, label='布林上軌')
    ax1.plot(df_slice.index, ma20, color='orange', alpha=0.5, lw=1, ls='--', label='布林中軸(月線)')
    ax1.plot(df_slice.index, dn, color='blue', alpha=0.2, lw=0.8, label='布林下軌')
    ax1.fill_between(df_slice.index, up, dn, color='blue', alpha=0.03)
    
    ax1.plot(df_slice.index, df_slice['Close'], color='black', linewidth=2, label='收盤價')
    
    ma120 = df['Close'].rolling(120).mean().tail(len(df_slice))
    ma1200 = df['Close'].rolling(1200, min_periods=100).mean().tail(len(df_slice))
    ax1.plot(df_slice.index, ma120, label='半年線 (MA120)', color='red', ls='--', lw=1.2)
    ax1.plot(df_slice.index, ma1200, label='五年線 (MA1200)', color='green', ls='-.', lw=1.2)
    
    ax1.set_ylim(df_slice['Low'].min()*0.97, df_slice['High'].max()*1.03)
    ax1.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='best', fontsize=8); ax1.grid(True, alpha=0.2)
    
    df_res = df_slice.resample(resample_rule).agg({'Open':'first', 'Close':'last', 'Volume':'sum'})
    colors = ['#ff4b4b' if df_res['Close'].iloc[i] >= df_res['Open'].iloc[i] else '#008000' for i in range(len(df_res))]
    ax2.bar(df_res.index, df_res['Volume'], color=colors, width=(2.5 if resample_rule=='3D' else 5), alpha=0.7)
    
    plt.tight_layout()
    return fig

# --- 4. 網頁 UI ---
st.sidebar.markdown("""
    <style>
    .potential-item { padding: 10px; border-radius: 5px; margin-bottom: 5px; background-color: #f0f2f6; border-left: 5px solid #ff4b4b; color: #31333F; }
    .potential-score { color: #ff4b4b; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("⌨️ 查詢系統")
query_in = st.sidebar.text_input("輸入股票代號或公司中文", "3675")

st.sidebar.markdown("---")
st.sidebar.subheader("🚩 潛力參考名單")

@st.cache_data(ttl=3600)
def scan_potential():
    p_list = []
    # 擴大掃描池以選出 10 家
    test_list = [
        "2330.TW", "2454.TW", "2317.TW", "3675.TWO", "6282.TW", "2303.TW", 
        "3037.TW", "2382.TW", "6669.TW", "3231.TW", "2376.TW", "1513.TW", 
        "1519.TW", "3034.TW", "2408.TW", "0050.TW", "2357.TW", "2308.TW"
    ]
    for t in test_list:
        d = fetch_stock_data(t, period="7y") 
        s, _ = evaluate_stock_100(d)
        if s >= 75: 
            p_list.append((get_company_name(t), t.split('.')[0], s))
    # 取評分前 10 名
    return sorted(p_list, key=lambda x: x[2], reverse=True)[:10]

for name, code, sc in scan_potential():
    st.sidebar.markdown(f'<div class="potential-item"><b>{name}</b> ({code})<br>AI 評分: <span class="potential-score">{sc} 分</span></div>', unsafe_allow_html=True)

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
        
        twii = fetch_stock_data("^TWII", period="7y")
        t_lp, t_pp = twii['Close'].iloc[-1], twii['Close'].iloc[-2]
        t_pct = ((t_lp - t_pp)/t_pp)*100
        t_color = "red" if t_pct > 0 else "green"

        st.markdown(f"### 📋 查詢標的：{ticker} - {c_name}")
        st.markdown(f"🕒 最後收盤日：{last_date}")
        
        col1, col2 = st.columns([1.5, 1])
        with col1:
            st.markdown(f"## 現價: **{lp:,.2f}** <span style='color:{p_color}'>({pct:+.2f}%)</span>", unsafe_allow_html=True)
            st.markdown(f"🔴 大盤指數: **{t_lp:,.2f}** <span style='color:{t_color}'>({t_pct:+.2f}%)</span>", unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"### 💡 AI 評分: <span style='color:#ff4b4b'>{score} 分</span>", unsafe_allow_html=True)
            with st.expander("🔍 評分明細"):
                for t in tags:
                    st.write(f"✅ {t}")

        st.markdown("---")
        st.pyplot(plot_v6_pro(hist, f"【{c_name}】半年波段指標圖 (含布林中軸/半年線)", 130, '3D'))
        st.markdown("---")
        st.pyplot(plot_v6_pro(hist, f"【{c_name}】五年長線循環圖 (含五年均線)", 1250, 'W'))

        st.markdown("---")
        st.subheader("📍 潛力象限分析")
        compare = ["2330.TW", "2317.TW", "3675.TWO", "6282.TW", "0050.TW"]
        if ticker not in compare: compare.append(ticker)
        q_list = []
        for t_item in compare:
            d_q = fetch_stock_data(t_item, period="7y")
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
            st.pyplot(fig_q)

st.markdown("---")
st.markdown("<p style='color:red; font-size: 0.8em; text-align: center;'>投資一定有風險，基金投資有賺有賠，申購前應詳閱公開說明書</p>", unsafe_allow_html=True)
