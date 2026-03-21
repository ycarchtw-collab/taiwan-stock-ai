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
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC', 'Noto Sans CJK JP', 'Liberation Sans']
else:
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 核心數據處理函數 ---
@st.cache_data
def load_stock_names():
    # 預設清單
    names = {"2330.TW": "台積電", "2317.TW": "鴻海", "3675.TWO": "德微", "0050.TW": "元大台灣50", "2454.TW": "聯發科"}
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
        # 即時數據合併邏輯
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
        
        # 評分標準
        tests = [
            (c > m20, "股價站上月線"),
            (c > m120, "股價站上半年線"),
            (c > m240, "股價站上年線"),
            (vol > vol_avg * 1.1, "量能優於三週均量"),
            (c > df['Close'].iloc[-2], "維持連漲慣性"),
            (m20 > df['Close'].rolling(20).mean().iloc[-5], "月線趨勢向上")
        ]
        for cond, msg in tests:
            if cond: score += 15; reasons.append(msg)
        score += random.randint(0, 10)
    except: return 0, []
    return min(int(score), 100), reasons

def get_zhuge_advice(score):
    # 完整諸葛評語邏輯
    if score >= 90: return "【赤壁火攻，勢不可擋】極強勢格局。噴火龍正在噴火，持股者抱緊，空手者等量縮回踩再動。"
    elif 75 <= score < 90: return "【萬事俱備，只欠東風】技術面翻多，年線以上無壓力。爆量突破近期高點即是再度發動訊號。"
    elif 60 <= score < 75: return "【草船借箭，蓄勢待發】底部墊高且站穩長線均線，主力偷偷吃貨中，適合分批佈局。"
    elif 40 <= score < 60: return "【兩軍對壘，糾纏不清】月線洗盤，均線糾結。此乃混水摸魚盤，等明確表態再說。"
    elif 20 <= score < 40: return "【空城計現，外強中乾】反彈無力，年線反壓沉重。底在哪還不知道，保命要緊。"
    else: return "【火燒連環船，兵敗如山倒】趨勢走空，不宜攤平。撤退保資金為上策。"

# --- 3. 繪圖模組 ---
def plot_v6_pro(df, title, days, show_long_term=False):
    df_slice = df.tail(days).copy()
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={'height_ratios':[3, 1]}, sharex=True)
    
    ma20 = df_slice['Close'].rolling(20).mean()
    std20 = df_slice['Close'].rolling(20).std()
    up, dn = ma20 + std20*2, ma20 - std20*2
    
    # 布林通道與均線
    ax1.plot(df_slice.index, up, color='#AABBDD', alpha=0.2, lw=0.8, label='布林上軌')
    ax1.plot(df_slice.index, ma20, color='#FFA500', alpha=0.6, lw=1.2, ls='--', label='月線(MA20)')
    ax1.plot(df_slice.index, dn, color='#AABBDD', alpha=0.2, lw=0.8, label='布林下軌')
    ax1.fill_between(df_slice.index, up, dn, color='#AABBDD', alpha=0.06)
    ax1.plot(df_slice.index, df_slice['Close'], color='white', linewidth=1.5, label='收盤價', zorder=5)
    
    # 年線 MA240
    ax1.plot(df_slice.index, df['Close'].rolling(240).mean().tail(days), label='年線(MA240)', color='#A020F0', lw=1.5)
    if show_long_term:
        # 五年線 MA1200
        ax1.plot(df_slice.index, df['Close'].rolling(1200, min_periods=100).mean().tail(days), label='五年線(MA1200)', color='#00FF00', ls='-.', lw=1.8)
    
    ax1.set_title(title, fontsize=14, fontweight='bold'); ax1.legend(loc='best', fontsize=8); ax1.grid(True, alpha=0.1)
    
    # 成交量
    colors = ['#FF4B4B' if df_slice['Close'].iloc[i] >= df_slice['Open'].iloc[i] else '#00E676' for i in range(len(df_slice))]
    ax2.bar(df_slice.index, df_slice['Volume'], color=colors, alpha=0.7)
    fig.patch.set_alpha(0.0); plt.tight_layout()
    return fig

def plot_prediction_chart(df, ticker_name):
    # AI 預測邏輯
    df_recent = df.tail(15).copy()
    y = df_recent['Close'].values
    X = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    preds = model.predict(np.arange(len(y), len(y) + 5).reshape(-1, 1))
    future_dates = [df_recent.index[-1] + timedelta(days=i) for i in range(1, 6)]
    
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df_recent.index, y, color='white', lw=2, marker='o', label='近期收盤')
    ax.plot(future_dates, preds, color='#FF4B4B', ls=':', marker='s', label='AI 預測')
    
    # 防數字重疊之位移標註
    y_range = max(max(y), max(preds)) - min(min(y), min(preds))
    offset = y_range * 0.08
    for i, (d, p) in enumerate(zip(future_dates, preds)):
        va, y_pos = ('bottom', p + offset) if i % 2 == 0 else ('top', p - offset)
        ax.text(d, y_pos, f'{p:.1f}', color='#FF4B4B', fontsize=10, fontweight='bold', ha='center', va=va)
    
    ax.set_title(f"🔮 {ticker_name} 未來五日 AI 預測走勢", color='white'); ax.grid(True, alpha=0.1)
    fig.patch.set_alpha(0.0); plt.tight_layout()
    return fig

# --- 4. 網頁 UI 與視覺修正 ---
st.set_page_config(page_title="台股｜AI 諸葛孔明", layout="wide")

# 背景圖 Base64 轉換與 CSS 注入
bg_img_base64 = ""
if os.path.exists('孔明看盤.png'):
    with open('孔明看盤.png', "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
        bg_img_base64 = f"data:image/png;base64,{encoded}"

st.markdown(f"""
    <style>
    /* 修正底圖顯示問題 */
    [data-testid="stAppViewContainer"] {{
        background-color: transparent !important;
    }}
    [data-testid="stAppViewContainer"]::before {{
        content: "";
        position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
        background-image: url("{bg_img_base64}");
        background-size: cover; background-position: center;
        opacity: 0.22 !important; z-index: -1;
    }}
    
    /* 鎖定側邊欄黑底 */
    [data-testid="stSidebar"] {{
        background-color: #111111 !important;
        border-right: 1px solid #333;
    }}
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {{
        color: #FFFFFF !important;
    }}

    /* 裝飾樣式 */
    .zhuge-advice {{ background: rgba(30, 0, 0, 0.88); padding: 25px; border-radius: 12px; border: 2px solid #FF4B4B; color: white; margin-bottom: 20px; }}
    .data-card {{ background: rgba(45, 45, 45, 0.88); padding: 20px; border-radius: 12px; border: 1px solid #555; text-align: center; }}
    h1, h2, h3 {{ color: #FFFFFF !important; text-shadow: 2px 2px 4px #000; }}
    </style>
""", unsafe_allow_html=True)

# 側邊欄配置
st.sidebar.title("⌨️ 諸葛神算")
query_in = st.sidebar.text_input("輸入代號", "3675").upper()

st.sidebar.markdown("---")
st.sidebar.subheader("🎲 監控清單")
for t in ["2330.TW", "2317.TW", "3675.TWO", "0050.TW"]:
    st.sidebar.write(f"🔹 {get_company_name(t)} ({t.split('.')[0]})")

# 主頁面內容
st.markdown("<h1 style='text-align: center;'>🚀 台股｜AI 諸葛孔明</h1>", unsafe_allow_html=True)

ticker = query_in
if ticker:
    # 代號補完
    if not (ticker.endswith(".TW") or ticker.endswith(".TWO")):
        test = fetch_stock_data(ticker + ".TW", period="5d")
        ticker = ticker + ".TWO" if test.empty else ticker + ".TW"

    hist = fetch_stock_data(ticker, period="7y")
    if not hist.empty:
        c_name = get_company_name(ticker)
        score, tags = evaluate_stock_100(hist)
        lp = hist['Close'].iloc[-1]; pct = ((lp / hist['Close'].iloc[-2]) - 1) * 100
        
        col1, col2 = st.columns(2)
        col1.markdown(f"<div class='data-card'>現價<br><span style='font-size:2.2rem; font-weight:bold; color:#00E676;'>{lp:,.2f}</span> ({pct:+.2f}%)</div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='data-card'>AI 評分<br><span style='font-size:2.2rem; font-weight:bold; color:#FF4B4B;'>{score}分</span></div>", unsafe_allow_html=True)
        
        st.markdown(f'<div class="zhuge-advice"><b style="color:#FF4B4B; font-size:1.3rem;">📜 諸葛錦囊：</b><br>{get_zhuge_advice(score)}<br><small style="color:#aaa;">依據：{", ".join(tags)}</small></div>', unsafe_allow_html=True)

        st.pyplot(plot_v6_pro(hist, "半年趨勢 (布林與年線)", 130))
        st.pyplot(plot_v6_pro(hist, "五年長線走勢 (五年線)", 1250, show_long_term=True))
        
        # 📍 潛力象限分析
        st.markdown("---")
        st.subheader("📍 潛力象限分析")
        compare_list = list(set(["2330.TW", "2317.TW", "2454.TW", ticker]))
        q_data = []
        for t_item in compare_list:
            d_q = fetch_stock_data(t_item, period="1y")
            if not d_q.empty:
                s_q, _ = evaluate_stock_100(d_q)
                c_q = ((d_q['Close'].iloc[-1] - d_q['Close'].iloc[-2]) / d_q['Close'].iloc[-2]) * 100
                q_data.append({"N": get_company_name(t_item), "S": s_q, "C": c_q, "T": t_item})
        if q_data:
            fig_q, ax_q = plt.subplots(figsize=(10, 5))
            for row in q_data:
                is_target = (row['T'] == ticker)
                ax_q.scatter(row['S'], row['C'], c='#FF4B4B' if is_target else 'royalblue', s=200 if is_target else 100)
                ax_q.text(row['S'], row['C']+0.1, row['N'], color='white', fontsize=12 if is_target else 9, ha='center')
            ax_q.axvline(50, color='white', ls='--', alpha=0.3); ax_q.axhline(0, color='white', ls='--', alpha=0.3)
            ax_q.set_xlabel("AI 評分"); ax_q.set_ylabel("今日漲幅 (%)")
            fig_q.patch.set_alpha(0.0); st.pyplot(fig_q)

        st.pyplot(plot_prediction_chart(hist, c_name))
    else:
        st.error("查無股票數據。")

# --- ⚠️ 底部警語 (補回完整版) ---
st.markdown("<div style='text-align: center; color: #888; font-size: 0.85rem; margin-top: 50px; border-top: 1px solid #444; padding-top: 20px;'>⚠️ 免責聲明：本系統結合 AI 線性回歸與技術指標分析，僅供學術研究參考。投資一定有風險，台股投資有賺有賠，申購前請審慎評估並自負盈虧。</div>", unsafe_allow_html=True)
