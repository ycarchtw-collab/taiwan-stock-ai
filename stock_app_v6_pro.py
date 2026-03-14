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

# --- 2. 核心運算函數 ---
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

def calculate_rsi(df, periods=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    if rs.empty or (1 + rs.iloc[-1]) == 0: return 50
    return 100 - (100 / (1 + rs.iloc[-1]))

def evaluate_stock_100(df):
    if df.empty or len(df) < 100: return 0, []
    score, reasons = 0, []
    try:
        c = df['Close'].iloc[-1]
        m20 = df['Close'].rolling(20).mean()
        m120 = df['Close'].rolling(120).mean()
        std20 = df['Close'].rolling(20).std()
        rsi = calculate_rsi(df)
        vol = df['Volume'].iloc[-1]
        recent_vol_avg = df['Volume'].tail(15).mean()
        six_month_vol_avg = df['Volume'].tail(120).mean()
        
        tests = [
            (c > m20.iloc[-1], "股價站上月線"), 
            (m20.iloc[-1] > m20.iloc[-5] if len(m20)>5 else False, "月線趨勢向上"),
            (c > m120.iloc[-1], "股價站上半年線"),
            (50 < rsi < 75, "RSI 強勢攻擊區"), 
            (recent_vol_avg > six_month_vol_avg, "三週均量增溫"),
            (vol > recent_vol_avg * 1.1, "今日量能放大"),
            (c < m20.iloc[-1] + std20.iloc[-1]*2, "尚未觸及布林上軌"),
            (c > df['Close'].iloc[-2], "收盤價連漲慣性")
        ]
        for cond, msg in tests:
            if cond: score += 12.5; reasons.append(msg)
    except: return 0, []
    return int(score), reasons

def get_zhuge_advice(score, df):
    """生成諸葛白話建議"""
    c = df['Close'].iloc[-1]
    m20 = df['Close'].rolling(20).mean().iloc[-1]
    vol_recent = df['Volume'].tail(15).mean()
    vol_today = df['Volume'].iloc[-1]
    
    if score >= 80:
        advice = "【萬事俱備，只欠東風】目前格局極佳，價量配合完美。若是已經在車上的朋友，建議扶好坐穩，讓獲利持續奔跑；若是想進場，可待量縮拉回不破月線時伺機切入。"
    elif 60 <= score < 80:
        advice = "【草船借箭，順勢而為】股價正處於蓄勢期，雖然還沒到大爆發，但量能已經在偷偷增溫。這種盤急不得，適合分批佈局，耐心等候多頭表態。"
    elif 40 <= score < 60:
        advice = "【兩軍對壘，觀望為宜】目前多空交戰激烈，股價就在月線附近糾纏。量能忽大忽小，沒有明顯方向感。建議先看戲，等這幾天大盤穩一點或是個股爆量突圍再動手。"
    elif 20 <= score < 40:
        advice = "【空城計現，謹防圈套】雖然看似跌深，但反彈無力且沒成交量。這時候進去很容易變成幫人墊背。建議先保住資金，等底部信號明確、月線走平後再說。"
    else:
        advice = "【火燒連環船，先撤為快】各項技術指標都在警示，走勢疲軟且有量縮下跌的跡象。目前不是攤平的好時機，若手中有持股應注意風險控制，切莫逆勢而行。"
    
    # 加入關於成交量的白話提醒
    if vol_today > vol_recent * 1.5:
        advice += " 另外，今日爆出大量，這可能是主力換手或是表態信號，要特別留意明日能否守住今日低點。"
    
    return advice

def plot_v6_pro(df, title, days, resample_rule):
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
    
    ma120 = df['Close'].rolling(120).mean().tail(len(df_slice))
    ax1.plot(df_slice.index, ma120, label='半年線', color='#FF3E3E', ls='--', lw=1.5)
    
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

def plot_prediction_chart(df, ticker_name):
    df_recent = df.tail(15).copy()
    y = df_recent['Close'].values
    X = np.column_stack([np.arange(len(y)), df_recent['Volume'].values])
    model = LinearRegression().fit(X, y)
    
    avg_vol = df_recent['Volume'].mean()
    future_X = np.column_stack([np.arange(len(y), len(y) + 5), [avg_vol]*5])
    future_preds = model.predict(future_X)
    
    ma20 = df['Close'].rolling(20).mean().tail(15)
    std20 = df['Close'].rolling(20).std().tail(15)
    up, dn = ma20 + std20*2, ma20 - std20*2
    
    last_date = df_recent.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, 6)]
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df_recent.index, ma20, color='#FFA500', ls='--', alpha=0.4, label='月線 (MA20)')
    ax.plot(df_recent.index, up, color='#AABBDD', alpha=0.2, label='布林上軌')
    ax.plot(df_recent.index, dn, color='#AABBDD', alpha=0.2, label='布林下軌')
    ax.plot(df_recent.index, y, color='white', label='近期收盤', lw=1.5)
    ax.plot(future_dates, future_preds, color='#FF4B4B', linestyle=':', marker='o', markersize=6, label='AI 預測')
    
    y_range = max(max(y), max(future_preds)) - min(min(y), min(future_preds))
    offset = y_range * 0.05 if y_range > 0 else 2
    for i, (d, p) in enumerate(zip(future_dates, future_preds)):
        va_val, y_pos = ('bottom', p + offset*0.5) if i % 2 == 0 else ('top', p - offset*0.5)
        ax.text(d, y_pos, f'{p:.1f}', color='#FF4B4B', fontsize=10, fontweight='bold', ha='center', va=va_val,
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))
    
    ax.set_title(f"🔮 {ticker_name} 未來五日 AI 量價預測", fontsize=12, color='white')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.1)
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
    h1 {{ font-size: clamp(1.4rem, 5vw, 2.5rem) !important; color: #FFFFFF !important; text-shadow: 2px 2px 6px #000; font-weight: 800 !important; text-align: center; }}
    .analysis-container {{
        background-color: rgba(0, 0, 0, 0.9) !important;
        backdrop-filter: blur(15px); padding: 20px; border-radius: 12px; border: 1px solid #FF4B4B;
        margin-bottom: 20px; color: #FFFFFF !important; line-height: 1.6;
    }}
    .data-card {{ background-color: rgba(0, 0, 0, 0.7); padding: 20px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); margin-bottom: 15px; }}
    </style>
    """
else:
    bg_style = "<style>[data-testid='stAppViewContainer'] { background-color: #121212; }</style>"

st.markdown(bg_style, unsafe_allow_html=True)

st.sidebar.title("⌨️ 諸葛神算")
query_in = st.sidebar.text_input("輸入代號", "3675").upper()
st.sidebar.markdown("---")
st.sidebar.subheader("🎲 隨機指標監控")

@st.cache_data(ttl=600)
def get_random_watchlist():
    pool = ["2330.TW", "2317.TW", "2454.TW", "2308.TW", "2382.TW", "2303.TW", "2881.TW", "2603.TW", "3675.TWO", "0050.TW", "2327.TW", "3231.TW", "2618.TW", "1513.TW"]
    selected = random.sample(pool, min(10, len(pool)))
    results = []
    for t in selected:
        d = fetch_stock_data(t, period="1y")
        if not d.empty:
            s, _ = evaluate_stock_100(d)
            results.append((get_company_name(t), t.split('.')[0], s))
    return sorted(results, key=lambda x: x[2], reverse=True)

for name, code, sc in get_random_watchlist():
    st.sidebar.markdown(f'<div style="color:white; padding:8px; border-left:4px solid #ff4b4b; background:rgba(255,255,255,0.05); margin-bottom:5px;">{name} ({code})<br><span style="color:#ff4b4b">{sc}分</span></div>', unsafe_allow_html=True)

st.markdown("<h1>🚀 台股｜AI 諸葛孔明</h1>", unsafe_allow_html=True)

ticker = query_in
if ticker:
    if not (ticker.endswith(".TW") or ticker.endswith(".TWO")):
        test_ticker = ticker + ".TW"
        hist_test = fetch_stock_data(test_ticker, period="1mo")
        ticker = ticker + ".TWO" if hist_test.empty else test_ticker

    hist = fetch_stock_data(ticker, period="7y")
    if not hist.empty:
        c_name = get_company_name(ticker)
        score, tags = evaluate_stock_100(hist)
        lp, last_date = hist['Close'].iloc[-1], hist.index[-1].strftime('%Y-%m-%d')
        pct = ((lp - hist['Close'].iloc[-2])/hist['Close'].iloc[-2])*100
        pct_color = "#FF4B4B" if pct >= 0 else "#00FF7F"
        
        st.markdown(f"#### 📋 {ticker} - {c_name} | {last_date}")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(f"<div class='data-card'><span style='color: #AAA;'>現價</span><br><span style='font-size: 2.2rem; font-weight: bold; color: white;'>{lp:,.2f}</span> <span style='color:{pct_color}; font-size: 1.3rem;'>({pct:+.2f}%)</span></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='data-card'><span style='color: #AAA;'>AI 評分</span><br><span style='font-size: 2.2rem; font-weight: bold; color: {'#FF4B4B' if score>=50 else '#00FF7F'};'>{score} 分</span></div>", unsafe_allow_html=True)

        # 諸葛神算白話建議區
        st.markdown(f"""
        <div class="analysis-container">
            <b style="color: #FF4B4B; font-size: 1.2rem;">📜 諸葛分析建議：</b><br>
            {get_zhuge_advice(score, hist)}
        </div>
        """, unsafe_allow_html=True)

        st.pyplot(plot_v6_pro(hist, f"【{c_name}】趨勢指標圖", 130, '3D'))
        
        st.markdown("---")
        st.subheader("📍 潛力象限分析")
        compare_list = list(set(["2330.TW", "2317.TW", "2454.TW", "0050.TW", ticker]))
        q_data = []
        for t_item in compare_list:
            d_q = fetch_stock_data(t_item, period="1y")
            if not d_q.empty:
                s_q, _ = evaluate_stock_100(d_q)
                c_q = ((d_q['Close'].iloc[-1]-d_q['Close'].iloc[-2])/d_q['Close'].iloc[-2])*100
                q_data.append({"N": get_company_name(t_item), "S": s_q, "C": c_q, "T": t_item})
        
        if q_data:
            q_df = pd.DataFrame(q_data)
            fig_q, ax_q = plt.subplots(figsize=(10, 6))
            for i, row in q_df.iterrows():
                is_target = (row['T'] == ticker)
                ax_q.scatter(row['S'], row['C'], c='#FF4B4B' if is_target else 'royalblue', s=250 if is_target else 150)
                ax_q.annotate(row['N'], (row['S'], row['C']), color='white', fontsize=18 if is_target else 9, fontweight='bold', xytext=(5,5), textcoords='offset points')
            ax_q.axvline(50, color='white', ls='--', alpha=0.5, lw=1.5)
            ax_q.axhline(0, color='white', ls='--', alpha=0.5, lw=1.5)
            fig_q.patch.set_alpha(0.0); st.pyplot(fig_q)

        st.markdown("---")
        st.subheader("🔮 AI 諸葛量價預測")
        st.pyplot(plot_prediction_chart(hist, c_name))
    else:
        st.warning("⚠️ 無法獲取數據。")
