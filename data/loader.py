# data/loader.py
import pandas as pd
from vnstock import Vnstock
from vnstock.explorer.vci import Company
import streamlit as st
import datetime
import plotly.graph_objects as go
import numpy as np


@st.cache_data
def fetch_and_prepare_data(symbol, start_date, end_date, interval='1D', source='VCI'):
    try:
        stock = Vnstock().stock(symbol=symbol, source=source)
        df = stock.quote.history(start=start_date, end=end_date, interval=interval)
        
        # Kiểm tra nếu DataFrame rỗng
        if df.empty:
            print(f"⚠️ Không có dữ liệu cho {symbol} từ {start_date} đến {end_date}")
            return None
        
        # Chuyển 'time' về dạng datetime để dễ sử dụng
        df['time'] = pd.to_datetime(df['time'])
        
        return df
    except Exception as e:
        print(f"❌ Lỗi khi lấy dữ liệu {symbol}: {e}")
        return None
    



@st.cache_data
def get_ratios(stock_code, source='VCI'):
    """
    Lấy dữ liệu tài chính theo từng cổ phiếu từ Vnstock.
    Trả về DataFrame chứa các chỉ tiêu tài chính theo từng năm.
    """
    try:
        stock = Vnstock().stock(symbol=stock_code, source=source)
        df = stock.finance.ratio(period='year', lang='vi', dropna=True)

        if df.empty:
            st.warning(f"Không có dữ liệu tài chính cho {stock_code}.")
            return None

        # Định dạng lại cột MultiIndex
        df.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in df.columns]
        df.rename(columns={'Meta_Năm': 'Năm', 'Meta_CP': 'CP','Meta_Kỳ':'Kỳ'}, inplace=True)

        # Chuyển đổi 'Năm' thành dạng số
        df['Năm'] = pd.to_numeric(df['Năm'], errors='coerce')

        return df
    except Exception as e:
        st.error(f"Lỗi khi lấy dữ liệu cho {stock_code}: {e}")
        return None

    

@st.cache_resource
def get_financial_ratios(symbols, period='year', lang='vi', dropna=True):
    """Get financial ratios for a list of symbols."""
    vnstock_instance = Vnstock()
    all_data = []
    if isinstance(symbols, str):
        symbols = [symbols]
    for symbol in symbols:
        try:
            stock = vnstock_instance.stock(symbol=symbol, source='VCI')
            finance_data = stock.finance.ratio(period=period, lang=lang, dropna=dropna)
            if not finance_data.empty:
                finance_data['Symbol'] = symbol
                all_data.append(finance_data)
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


@st.cache_resource
# Hàm lấy dữ liệu từ Vnstock
def get_cash_flow(symbol, period='year'):

    stock = Vnstock().stock(symbol=symbol, source='VCI')
    
    # Lấy dữ liệu dòng tiền
    df = stock.finance.cash_flow(period=period, lang='vi', dropna=True)

    # Kiểm tra xem DataFrame có dữ liệu không
    if df.empty:
        print("Không có dữ liệu cho mã cổ phiếu này.")
        return None

    # Lọc dữ liệu từ năm 2014 trở đi
    df = df[df['Năm'] >= 2020]

    # Kiểm tra cột trùng lặp
    if df.columns.duplicated().any():
        print("Có cột trùng lặp trong DataFrame.")
        return None

    # Trả về DataFrame với chỉ mục là năm
    return df.set_index("Năm").sort_index(ascending=False)


@st.cache_resource
def get_income_statement(symbol, period='year'):
    stock = Vnstock().stock(symbol=symbol, source='VCI')
    df = stock.finance.income_statement(period=period, lang='vi', dropna=True)
    df = df[df['Năm'] >= 2020]
    return df.set_index("Năm").sort_index(ascending=False) if not df.empty else None

@st.cache_resource
def get_balance_sheet(symbol, period='year'):
    stock = Vnstock().stock(symbol=symbol, source='VCI')
    df = stock.finance.balance_sheet(period=period, lang='vi', dropna=True)
    df = df[df['Năm'] >= 2020]
    return df.set_index("Năm").sort_index(ascending=False) if not df.empty else None


@st.cache_resource
def get_company_table(symbol, source='TCBS'):
    """Get company information for a symbol."""
    try:
        company = Vnstock().stock(symbol, source).company
        overview, profile = company.overview(), company.profile()
        fields = ["exchange", "industry", "stock_rating", "website"]
        data = {f: overview.get(f, pd.Series(["N/A"])).iloc[0] for f in fields}
        data["company_name"], data["symbol"] = profile.get("company_name", "N/A"), symbol
        return pd.DataFrame([data])
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()
    
@st.cache_resource    
# Hàm lấy thông tin ban lãnh đạo
def get_officers_info(code):
    company = Vnstock().stock(symbol=code, source='TCBS').company
    officers = company.officers()
    officers['officer_own_percent'] *= 100
    return officers.rename(columns={'officer_name': 'Ban lãnh đạo', 'officer_position': 'Vị trí', 'officer_own_percent': 'Tỷ lệ(%) sở hữu'})


@st.cache_resource
def get_subsidiaries_info(code):
    company = Vnstock().stock(symbol=code, source='VCI').company
    subsidiaries = company.subsidiaries()

    if subsidiaries is None or subsidiaries.empty:
        print("⚠️ Không có dữ liệu về công ty con!")
        return pd.DataFrame()

    # Kiểm tra nếu cột 'organ_code' tồn tại trong DataFrame
    if 'organ_code' in subsidiaries.columns:
        subsidiaries = subsidiaries.drop(columns=['organ_code'])

    # Kiểm tra và xóa các cột không cần thiết nếu tồn tại
    for column in ['id', 'type']:
        if column in subsidiaries.columns:
            subsidiaries = subsidiaries.drop(columns=[column])

    # Đổi tên các cột theo yêu cầu
    subsidiaries.rename(
        columns={'sub_organ_code': 'Mã', 'ownership_percent': 'Tỷ lệ(%) sở hữu', 'organ_name': 'Tên công ty'},
        inplace=True
    )


    return subsidiaries

@st.cache_resource
# Hàm lấy thông tin cổ đông
def get_shareholders_info(code):
    company = Vnstock().stock(symbol=code, source='TCBS').company
    shareholders = company.shareholders()
    return shareholders.rename(columns={'share_holder': 'Cổ đông', 'share_own_percent': 'Tỷ lệ(%) sở hữu'})


def get_all_symbols(source='VCI'):
    """
    Lấy danh sách tất cả các mã cổ phiếu trên thị trường từ nguồn dữ liệu.
    Sử dụng hàm `stock.listing.all_symbols()` từ thư viện vnstock.
    """
    # Sử dụng một mã cổ phiếu bất kỳ (ở đây dùng 'ACB') để truy cập đối tượng listing
    stock = Vnstock().stock(symbol='ACB', source=source)
    return stock.listing.all_symbols()


def calculate_rsi(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series, short_window=12, long_window=26, signal_window=9):
    short_ema = series.ewm(span=short_window, adjust=False).mean()
    long_ema = series.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

def calculate_atr(high, low, close, period=14):
    high_low = high - low
    high_close = np.abs(high - close.shift(1))
    low_close = np.abs(low - close.shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()
    return 

def calculate_sma(series, window=50):
    return series.rolling(window=window).mean()

def calculate_ema(series, window=20):
    return series.ewm(span=window, adjust=False).mean()


def trade_signal_analysis(df_stock):
    if df_stock is None or df_stock.empty:
        st.warning("Không có dữ liệu để phân tích.")
        return
    
    df_stock['RSI'] = calculate_rsi(df_stock['close'])
    df_stock['MACD'], df_stock['Signal'] = calculate_macd(df_stock['close'])
    df_stock['ATR'] = calculate_atr(df_stock['high'], df_stock['low'], df_stock['close'])
    df_stock['SMA_50'] = calculate_sma(df_stock['close'], 50)
    df_stock['EMA_20'] = calculate_ema(df_stock['close'], 20)
    
    df_stock['Buy_Signal'] = (df_stock['RSI'] < 30) & (df_stock['MACD'] > df_stock['Signal'])
    df_stock['Sell_Signal'] = (df_stock['RSI'] > 70) & (df_stock['MACD'] < df_stock['Signal'])
    
    df_stock['Signal_Type'] = np.where(df_stock['Buy_Signal'], 'Buy',
                                       np.where(df_stock['Sell_Signal'], 'Sell', 'Neutral'))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_stock['time'], y=df_stock['close'],
                              mode='lines', name='Giá Đóng Cửa',
                              line=dict(color='royalblue', width=2)))
    
    buy_signals = df_stock[df_stock['Buy_Signal']]
    sell_signals = df_stock[df_stock['Sell_Signal']]
    
    fig.add_trace(go.Scatter(x=buy_signals['time'], y=buy_signals['close'],
                              mode='markers', name='Mua',
                              marker=dict(symbol='triangle-up', size=10, color='green')))
    
    fig.add_trace(go.Scatter(x=sell_signals['time'], y=sell_signals['close'],
                              mode='markers', name='Bán',
                              marker=dict(symbol='triangle-down', size=10, color='red')))
    
    fig.update_xaxes(
        title_text='Date', rangeslider_visible=False,
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(count=5, label="5y", step="year", stepmode="backward"),
                dict(step="all")
            ]
        )
    )
    
    fig.update_layout(title="Tín Hiệu Mua/Bán Dựa Trên Chỉ Báo Kỹ Thuật",
                      xaxis_title="Thời Gian", yaxis_title="Giá Đóng Cửa",
                      template="plotly_dark", height=600, width=900)
    
    st.plotly_chart(fig, use_container_width=True)
