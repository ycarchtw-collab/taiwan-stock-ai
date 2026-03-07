import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

# --- 1. 雲端環境中文字體修正 ---
if os.name == 'posix':
    # 雲端 Linux 環境使用系統字體
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Sans CJK TC', 'Liberation Sans']
else:
    # Mac 本機環境使用 Arial Unicode MS
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'AppleGothic']
plt.rcParams['axes.unicode_minus'] = False

# 核心監控與名稱映射表
CORE_LIST = {
    "2330.TW": "台積電", "2454.TW": "聯發科", "2317.TW": "鴻海", "3675.TWO": "德微",
    "6282.TW": "康舒", "2303.TW": "聯電", "3037.TW": "欣興", "2382.TW": "廣達",
    "6669.TW": "緯穎", "3231.TW": "緯創", "2376.TW": "技嘉", "1513.TW": "中興電",
    "1519.TW": "華城", "4763.TW": "材料-KY", "8086.TWO": "宏捷科", "6438.TW": "迅得",
    "0050.TW": "元大台灣50", "00981A.TW": "統一台股增長", "2395.TW": "研華", "3034.TW": "聯詠"
}

# --- 2. 核心邏輯函數 ---

def get_ticker_info(query):
    """支援代號與中文，自動補足上市櫃後綴"""
    # 檢查是否為中文名稱
    reverse_map = {v: k for k, v in CORE_LIST.items()}
    if query in reverse_map:
        return reverse_map[query], query
    
    # 檢查是否為純數字代號
    if query.isdigit():
        for sfx in [".TW", ".TWO"]:
            t_code = query + sfx
            t_obj = yf.Ticker(t_code)
            if not t_obj.history(period="1d").empty:
                name = CORE_LIST.get(t_code, t_obj.info.get('shortName', '未知個股'))
                return t_code, name
    return query, "自訂查詢"

def calculate_rsi(df, periods=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def evaluate_stock_100(df):
    """100 分制 AI 評分系統"""
    if len(df) < 20: return 0, []
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
    except: return 0, ["數據計算中"]
    return score, reasons

def plot_advanced_charts(df, title, days, resample_rule):
    """繪製專業趨勢圖 (Y軸縮放 + 布林通道 + 分層量能)"""
    df_slice = df.tail(days).copy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw={'height_ratios':[3, 1]}, sharex=True)
    
    # 布林通道計算
    ma20 = df_slice['Close'].rolling(20).mean()
    std20 = df_slice['Close'].rolling(20).std()
    up, dn = ma20 + std20*2, ma20 - std20*2
    
    # 上圖：價格與布林
    ax1.fill_between(df_slice.index, up, dn, color='blue', alpha=0.07, label='布林通道')
    ax1.plot(df_slice.index, df_slice['Close'], color='black', linewidth=1.8, label='收盤價')
    if 'MA120' in df_slice: ax1.plot(df_slice.index, df_slice['MA120'], color='red', ls='--', label='半年線')
    if 'MA1200' in df_slice: ax1.plot(df_slice.index, df_slice['MA1200'], color='green', ls='-.', label='五年線')
    
    # Y軸縮放至區間內最高最低
    ax1.set_ylim(df_slice['Low'].min()*0.98, df_slice['High'].max()*1.02)
    ax1.set_title(title); ax1.legend(loc='lower left'); ax1.grid(True, alpha=0.3)
    
    # 下圖：分層量能 (3D或W)
    df_res = df_slice.resample(resample_rule).agg({'Open':'first', 'Close':'last', 'Volume':'sum'})
    colors = ['red' if df_res['Close'].iloc[i] >= df_res['Open'].iloc[i] else 'green' for i in range(len(df_res))]
    ax2.bar(df_res.index, df_res['Volume'], color=colors, width=(2.5 if resample_rule=='3D' else 5), alpha=0.8)
    ax2.set_ylabel("成交量加總")
    
    plt.tight_layout()
    return fig

# --- 3. 網頁 UI ---

# 左側側邊欄：75分潛力股
st.sidebar.title("🔍 75分潛力標的")
st.sidebar.caption("即時掃描核心追蹤名單")

# 使用緩存避免重複抓取
@st.cache_data(ttl=3600)
def scan_potential():
    p_list = []
    for t, n in CORE_LIST.items():
        d = yf.Ticker(t).history(period="1y")
        s, _ = evaluate_stock_100(d)
        if s >= 75: p_list.append((n, t.split('.')[0], s))
    return sorted(p_list, key=lambda x: x[2], reverse=True)

for name, code, sc in scan_potential():
    st.sidebar.success(f"🔥 {name} ({code}) : {sc}分")

# 主頁面內容
st.title("🏛️ 2026 AI 台股投資決策系統 V6 Pro")
query_input = st.text_input("輸入股票代號或名稱 (如: 2330, 德微)", "3675")
ticker_code, chinese_name = get_ticker_info(query_input)

if ticker_code:
    with st.spinner('數據計算中...'):
        hist = yf.Ticker(ticker_code).history(period="7y")
        if not hist.empty:
            # 獲取時間資訊
            last_trade_date = hist.index[-1].strftime('%Y-%m-%d')
            now_time = datetime.now().strftime('%Y-%m-%d %H:%M')
            
            hist['MA120'] = hist['Close'].rolling(120).mean()
            hist['MA1200'] = hist['Close'].rolling(1200, min_periods=100).mean()
            score, tags = evaluate_stock_100(hist)
            
            # 即時報價
            lp, pp = hist['Close'].iloc[-1], hist['Close'].iloc[-2]
            pct = ((lp - pp)/pp)*100
            p_color = "red" if pct > 0 else ("green" if pct < 0 else "black")
            
            st.markdown(f"### 📋 查詢標的：{ticker_code} - {chinese_name}")
            st.markdown(f"📅 **查詢日期**：{now_time} | 🕒 **最後收盤日**：{last_trade_date}")

            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"## 現價: **{lp:,.2f}** <span style='color:{p_color}'>({pct:+.2f}%)</span>", unsafe_allow_html=True)
                
                st.pyplot(plot_advanced_charts(hist, "半年趨勢 (自動縮放 + 三日量能)", 130, '3D'))
                
                st.pyplot(plot_advanced_charts(hist, "五年長線 (長線循環 + 每週量能)", 1250, 'W'))
            
            with col2:
                st.subheader(f"💡 AI 評分: {score} / 100")
                for t in tags: st.success(t)
                
                # 象限圖
                st.subheader("📍 潛力象限分析")
                compare = ["2330.TW", "2317.TW", "3675.TWO", "6282.TW", "0050.TW"]
                if ticker_code not in compare: compare.append(ticker_code)
                q_list = []
                for t_item in compare:
                    # 強制對齊長度以確保分數一致
                    d_comp = yf.Ticker(t_item).history(period="5y")
                    s_comp, _ = evaluate_stock_100(d_comp)
                    c_comp = ((d_comp['Close'].iloc[-1]-d_comp['Close'].iloc[-2])/d_comp['Close'].iloc[-2])*100
                    q_list.append({"T": t_item, "S": s_comp, "C": c_comp})
                
                q_df = pd.DataFrame(q_list)
                fig_q, ax_q = plt.subplots(figsize=(5, 5))
                # 查詢標的顯示紅點
                colors = ['red' if r == ticker_code else 'royalblue' for r in q_df['T']]
                ax_q.scatter(q_df['S'], q_df['C'], c=colors, s=180, edgecolors='white', zorder=5)
                for i, txt in enumerate(q_df['T']):
                    ax_q.annotate(txt, (q_df['S'][i], q_df['C'][i]), fontsize=8, xytext=(5,5), textcoords='offset points')
                ax_q.axvline(50, color='gray', ls='--', alpha=0.5); ax_q.axhline(0, color='gray', ls='--', alpha=0.5)
                ax_q.set_xlim(0, 100); ax_q.set_xlabel("AI 分數"); ax_q.set_ylabel("當日漲跌幅 %")
                
                st.pyplot(fig_q)
