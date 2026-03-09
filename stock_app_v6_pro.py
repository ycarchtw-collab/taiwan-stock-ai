import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import time
import base64
from datetime import datetime, timedelta

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

# --- 3. 核心運算 (修正數據抓取邏輯) ---
@st.cache_data(ttl=60) # 修正：縮短快取至 60 秒確保即時性
def fetch_stock_data(ticker, period="7y"):
    try:
        # 強制抓取包含今天的數據
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval="1d", auto_adjust=True)
        
        # 若當前為開盤時間，嘗試抓取 1m 資料補足今日即時點位
        now = datetime.now()
        if now.weekday() <= 4 and 9 <= now.hour <= 14:
            today_df = stock.history(period="1d", interval="1m")
            if not today_df.empty:
                # 取得最新一筆 1m 資料作為今日日線更新
                last_price = today_df['Close'].iloc[-1]
                last_time = today_df.index[-1].replace(hour=0, minute=0, second=0, microsecond=0)
                if last_time > df.index[-1]:
                    new_row = pd.DataFrame({'Open': today_df['Open'].iloc[0], 'High': today_df['High'].max(), 
                                            'Low': today_df['Low'].min(), 'Close': last_price, 
                                            'Volume': today_df['Volume'].sum()}, index=[last_time])
                    df = pd.concat([df, new_row])
        return df
    except:
        return pd.DataFrame()

def calculate_rsi(df, periods=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def evaluate_stock_100(df):
    if df.empty or len(df) < 100: return 0, []
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
            (50 < rsi < 75, "RSI 強勢攻擊區"), (vol > avg_vol * 1.5, "量能顯著放大"),
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
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={'height_ratios':[3, 1]}, sharex=True)
    
    ma20 = df_slice['Close'].rolling(20).mean()
    std20 = df_slice['Close'].rolling(20).std()
    up, dn = ma20 + std20*2, ma20 - std20*2
    
    ax1.plot(df_slice.index, up, color='#AABBDD', alpha=0.5, lw=0.8, label='布林上軌')
    ax1.plot(df_slice.index, ma20, color='#FFA500', alpha=0.7, lw=1.2, ls='--', label='月線(中軸)')
    ax1.plot(df_slice.index, dn, color='#AABBDD', alpha=0.5, lw=0.8, label='布林下軌')
    ax1.fill_between(df_slice.index, up, dn, color='#AABBDD', alpha=0.12)
    
    ax1.plot(df_slice.index, df_slice['Close'], color='white', linewidth=1.75, label='收盤價', zorder=5)
    
    ma120 = df['Close'].rolling(120).mean().tail(len(df_slice))
    ma1200 = df['Close'].rolling(1200, min_periods=100).mean().tail(len(df_slice))
    ax1.plot(df_slice.index, ma120, label='半年線', color='#FF3E3E', ls='--', lw=1.5)
    ax1.plot(df_slice.index, ma1200, label='五年線', color='#00FF00', ls='-.', lw=1.5)
    
    ax1.set_ylim(df_slice['Low'].min()*0.96, df_slice['High'].max()*1.04)
    ax1.set_title(title, fontsize=14, fontweight='bold', color='#FFFFFF')
    ax1.legend(loc='best', fontsize=9, facecolor='#111', edgecolor='#444')
    ax1.grid(True, alpha=0.1)
    
    df_res = df_slice.resample(resample_rule).agg({'Open':'first', 'Close':'last', 'Volume':'sum'})
    colors = ['#FF4B4B' if df_res['Close'].iloc[i] >= df_res['Open'].iloc[i] else '#00E676' for i in range(len(df_res))]
    ax2.bar(df_res.index, df_res['Volume'], color=colors, width=(2.5 if resample_rule=='3D' else 5), alpha=0.8)
    fig.patch.set_alpha(0.0) 
    plt.tight_layout()
    return fig

# --- 4. 網頁 UI 佈局 ---
st.set_page_config(page_title="台股｜AI 諸葛孔明", layout="wide")

if os.path.exists('孔明看盤.png'):
    with open('孔明看盤.png', "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    bg_style = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/png;base64,{encoded_string}");
        background-size: cover; background-position: center; background-attachment: fixed;
    }}
    [data-testid="stAppViewContainer"]::before {{
        content: ""; position: absolute; top: 0; left: 0; width: 100%; height: 100%;
        background: linear-gradient(rgba(0,0,0,0.85), rgba(0,0,0,0.6), rgba(0,0,0,0.85));
        backdrop-filter: blur(6px); z-index: -1;
    }}
    [data-testid="stSidebar"] {{ background-color: rgba(20, 20, 20, 0.95) !important; }}
    h1 {{ font-size: clamp(1.8rem, 6vw, 3rem) !important; color: #FFFFFF !important; text-shadow: 2px 2px 6px #000; white-space: nowrap; font-weight: 800 !important; }}
    .stMarkdown, .stMetric, .stExpander {{ background-color: rgba(0, 0, 0, 0.5) !important; backdrop-filter: blur(10px); padding: 10px; border-radius: 8px; margin-bottom: 10px; }}
    .analysis-container {{
        background-color: rgba(0, 0, 0, 0.9) !important;
        backdrop-filter: blur(15px);
        padding: 18px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.3);
        margin-bottom: 20px; color: #FFFFFF !important; box-shadow: 0 4px 20px rgba(0,0,0,0.6);
    }}
    .data-card {{ background-color: rgba(0, 0, 0, 0.7); padding: 20px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); margin-bottom: 15px; }}
    </style>
    """
else:
    bg_style = "<style>[data-testid='stAppViewContainer'] { background-color: #121212; }</style>"

st.markdown(bg_style, unsafe_allow_html=True)

st.sidebar.title("⌨️ 諸葛神算")
query_in = st.sidebar.text_input("輸入代號或名稱", "3675")

st.sidebar.markdown("---")
@st.cache_data(ttl=3600)
def scan_potential():
    p_list = []
    test_list = ["2330.TW", "2454.TW", "2317.TW", "3675.TWO", "1513.TW", "1519.TW"]
    for t in test_list:
        d = fetch_stock_data(t, period="7y") 
        s, _ = evaluate_stock_100(d)
        p_list.append((get_company_name(t), t.split('.')[0], s))
    return sorted(p_list, key=lambda x: x[2], reverse=True)[:10]

for name, code, sc in scan_potential():
    st.sidebar.markdown(f'<div style="color:white; padding:8px; border-left:4px solid #ff4b4b; background:rgba(255,255,255,0.05); margin-bottom:5px;">{name} ({code})<br><span style="color:#ff4b4b">{sc}分</span></div>', unsafe_allow_html=True)

st.markdown("<h1>🚀 台股｜AI 諸葛孔明</h1>", unsafe_allow_html=True)

ticker = query_in
name_to_ticker = {v: k for k, v in STOCK_DB.items()}
if query_in in name_to_ticker: ticker = name_to_ticker[query_in]
elif query_in.isdigit(): ticker = query_in + ".TW"

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
        pct_color = "#FF4B4B" if pct >= 0 else "#00FF7F"
        
        twii = fetch_stock_data("^TWII", period="7y")
        t_lp, t_pp = twii['Close'].iloc[-1], twii['Close'].iloc[-2]
        t_pct = ((t_lp - t_pp)/t_pp)*100
        t_pct_color = "#FF4B4B" if t_pct >= 0 else "#00FF7F"
        
        st.markdown(f"#### 📋 {ticker} - {c_name} | {last_date}")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(f"""
            <div class='data-card'>
                <span style='color: #AAA;'>現價</span><br>
                <span style='font-size: 2.2rem; font-weight: bold; color: white;'>{lp:,.2f}</span> 
                <span style='color:{pct_color}; font-size: 1.3rem; font-weight: bold;'>({pct:+.2f}%)</span><br>
                <div style='color:white; background:rgba(0,0,0,0.5); padding:5px 10px; border-radius:5px; display:inline-block; border:1px solid #444; margin-top:10px;'>🔴 大盤: {t_lp:,.2f} <span style='color:{t_pct_color};'>({t_pct:+.2f}%)</span></div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            score_color = "#FF4B4B" if score >= 50 else "#00FF7F"
            st.markdown(f"""
            <div class='data-card'>
                <span style='color: #AAA;'>AI 評分</span><br>
                <span style='font-size: 2.2rem; font-weight: bold; color: {score_color};'>{score} 分</span>
            </div>
            """, unsafe_allow_html=True)
            with st.expander("🔍 決策依據"):
                for t in tags: st.write(f"✅ {t}")

        st.markdown("---")
        
        st.pyplot(plot_v6_pro(hist, f"【{c_name}】半年波段指標圖", 130, '3D'))
        st.markdown("---")
        
        st.pyplot(plot_v6_pro(hist, f"【{c_name}】五年波段指標圖", 1250, 'W'))

        st.markdown("---")
        st.subheader("📍 潛力象限分析")
        
        st.markdown(f"""
        <div class="analysis-container">
            <b style="color: #FF4B4B; font-size: 1.15rem;">📊 落點解析說明：</b><br>
            <span style="font-size: 1rem; line-height: 1.6;">
            • <b>右上 (強勢攻擊區)：</b> AI 評分高且漲勢強。標的多頭動能極強。<br>
            • <b>右下 (蓄勢待發區)：</b> AI 評分高但今日壓回。具備補漲潛力。<br>
            • <b>左上 (過熱投機區)：</b> 評分低但今日漲幅大。留意短線回檔風險。<br>
            • <b>左下 (弱勢觀望區)：</b> 評分與趨勢皆疲弱。標的目前處於冷灶期。
            </span>
        </div>
        """, unsafe_allow_html=True)

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
            fig_q, ax_q = plt.subplots(figsize=(10, 6))
            colors = ['#FF4B4B' if r == ticker else 'royalblue' for r in q_df['T']]
            ax_q.scatter(q_df['S'], q_df['C'], c=colors, s=250, edgecolors='white', zorder=5)
            for i, txt in enumerate(q_df['N']):
                is_target = (q_df['T'].iloc[i] == ticker)
                font_size = 18 if is_target else 9
                color_val = 'white' if is_target else '#CCCCCC'
                ax_q.annotate(txt, (q_df['S'].iloc[i], q_df['C'].iloc[i]), fontsize=font_size, xytext=(5,5), textcoords='offset points', fontweight='bold', color=color_val)
            
            ax_q.axvline(50, color='white', ls='--', alpha=0.6, lw=1.2)
            ax_q.axhline(0, color='white', ls='--', alpha=0.6, lw=1.2)
            ax_q.set_xlabel("AI 評分 (分)", color='white'); ax_q.set_ylabel("漲跌幅 (%)", color='white')
            fig_q.patch.set_alpha(0.0)
            
            st.pyplot(fig_q)

st.markdown("---")
st.markdown("<p style='color:#FF9999; font-size: 0.8em; text-align: center; font-weight: bold;'>投資一定有風險，投資有賺有賠，申購前應詳閱公開說明書</p>", unsafe_allow_html=True)
