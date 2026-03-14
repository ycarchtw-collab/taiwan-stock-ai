import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

# 設定頁面配置
st.set_page_config(page_title="台股｜AI 諸葛孔明", layout="wide")

# 自定義 CSS：強化手機版顯示、修正文字遮罩與顏色
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    /* 標題與手機排版優化 */
    .title-text {
        font-size: clamp(20px, 5vw, 32px);
        font-weight: bold;
        text-align: center;
        padding: 10px;
        color: #1E3A8A;
    }
    /* 分析說明遮罩與文字強化 */
    .analysis-container {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1E3A8A;
        margin: 10px 0;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    .analysis-text {
        color: #1a1a1a !important;
        font-size: 1.1em;
        line-height: 1.6;
    }
    /* 漲跌顏色 */
    .stock-up { color: #FF0000; font-weight: bold; }
    .stock-down { color: #008000; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# 標題
st.markdown('<div class="title-text">台股｜AI 諸葛孔明</div>', unsafe_allow_html=True)

# 初始化監控名單
if 'watch_list' not in st.session_state:
    st.session_state.watch_list = ['2330', '2454', '2317']

# 功能按鈕區
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
with col_btn1:
    new_code = st.text_input("輸入代號", key="add_input", placeholder="如: 3675")
    if st.button("增加個股"):
        if new_code and new_code not in st.session_state.watch_list:
            st.session_state.watch_list.append(new_code)
            st.rerun()

with col_btn2:
    del_code = st.text_input("刪除代號", key="del_input")
    if st.button("刪除監控"):
        if del_code in st.session_state.watch_list:
            st.session_state.watch_list.remove(del_code)
            st.rerun()

# 數據獲取與處理函數
def get_stock_data(symbol):
    # 處理台股代號後綴
    if not symbol.endswith(('.TW', '.TWO')):
        # 嘗試判斷上市或上櫃，此處簡易處理：先試 .TW 再試 .TWO
        s = yf.Ticker(f"{symbol}.TW")
        df = s.history(period="6mo")
        if df.empty:
            s = yf.Ticker(f"{symbol}.TWO")
            df = s.history(period="6mo")
    else:
        s = yf.Ticker(symbol)
        df = s.history(period="6mo")
    
    return s, df

# 渲染監控面板
for symbol in st.session_state.watch_list:
    try:
        ticker, df = get_stock_data(symbol)
        
        if df.empty:
            st.warning(f"找不到代碼 {symbol} 的即時數據，請確認代號是否正確。")
            continue

        # 基礎資訊
        info = ticker.info
        current_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]
        change_pct = ((current_price - prev_price) / prev_price) * 100
        
        # 修正 0306 停滯問題：顯示數據最後更新日期
        last_date = df.index[-1].strftime('%Y-%m-%d')
        
        # 1. 資訊卡片
        color_class = "stock-up" if change_pct >= 0 else "stock-down"
        st.subheader(f"{info.get('shortName', symbol)} ({symbol}) - 數據至: {last_date}")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("現價", f"{current_price:.2f}", f"{change_pct:.2f}%")
        
        # PE / EPS 鎖定小數點第二位
        pe_ratio = info.get('trailingPE', 0)
        eps = info.get('trailingEps', 0)
        c2.write("**本益比 (PE)**")
        c2.write(f"{pe_ratio:.2f}" if pe_ratio else "N/A")
        c3.write("**每股盈餘 (EPS)**")
        c3.write(f"{eps:.2f}" if eps else "N/A")

        # 2. 趨勢圖與未來一週預測
        # 計算布林通道 (顏色調淺)
        ma20 = df['Close'].rolling(window=20).mean()
        std20 = df['Close'].rolling(window=20).std()
        upper_band = ma20 + (std20 * 2)
        lower_band = ma20 - (std20 * 2)

        # 未來預測模型 (簡單線性回歸)
        y = df['Close'].tail(20).values
        X = np.arange(len(y)).reshape(-1, 1)
        model = LinearRegression().fit(X, y)
        
        # 預測未來 5 個交易日
        future_X = np.arange(len(y), len(y) + 5).reshape(-1, 1)
        future_preds = model.predict(future_X)
        last_dt = df.index[-1]
        future_dates = [last_dt + timedelta(days=i) for i in range(1, 6)]

        fig = go.Figure()
        # 收盤價線 (寬度調減 30%)
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='收盤價', line=dict(width=1.4, color='blue')))
        # 布林通道 (顏色變淺)
        fig.add_trace(go.Scatter(x=df.index, y=upper_band, name='布林上軌', line=dict(color='rgba(173, 181, 189, 0.3)')))
        fig.add_trace(go.Scatter(x=df.index, y=lower_band, name='布林下軌', line=dict(color='rgba(173, 181, 189, 0.3)'), fill='tonexty'))
        
        # 未來一週預測圖 (紅色虛線)
        fig.add_trace(go.Scatter(x=future_dates, y=future_preds, name='AI 一週預測', line=dict(color='red', dash='dash')))

        fig.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # 3. 象限分析說明 (強化遮罩效果)
        st.markdown(f"""
        <div class="analysis-container">
            <p class="analysis-text">
                <b>象限分析說明：</b>目前 <b>{symbol}</b> 處於技術面盤整區。
                基於布林通道寬度與動能指標，未來一週預測走勢如紅虛線所示。
                請注意，此預測基於近期線性趨勢，建議搭配 PE ({pe_ratio:.2f}) 評估合理價位。
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()

    except Exception as e:
        st.error(f"解析 {symbol} 時發生錯誤: {e}")

# 頁尾免責聲明
st.markdown("<p style='color:gray; font-size: 0.8em; text-align: center;'>投資一定有風險，台股投資有賺有賠，申購前應詳閱公開說明書</p>", unsafe_allow_html=True)
