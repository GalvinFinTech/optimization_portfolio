import pandas as pd
from vnstock import Vnstock, Screener
import streamlit as st

@st.cache_data
def fetch_multiple_stock_prices(symbols, start_date, end_date, interval='1D', source='VCI'):
    """
    Tải dữ liệu giá đóng cửa của các mã cổ phiếu và trả về DataFrame với index là ngày.
    """
    close_prices = pd.DataFrame()
    for symbol in symbols:
        try:
            print(f"📥 Đang tải dữ liệu cho {symbol} từ {source}...")
            stock = Vnstock().stock(symbol=symbol, source=source)
            df = stock.quote.history(start=start_date, end=end_date, interval=interval)
            if df.empty:
                print(f"⚠️ Không có dữ liệu cho {symbol} từ {start_date} đến {end_date}")
                continue
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            close_prices[symbol] = df['close']
        except Exception as e:
            print(f"❌ Lỗi khi tải dữ liệu {symbol}: {e}")
    if close_prices.empty:
        raise ValueError("❌ Không có dữ liệu nào được tải thành công!")
    
    close_prices.fillna(method='ffill', inplace=True)  # Điền NaN bằng giá gần nhất

    close_prices.index.name = "date"
    return close_prices

@st.cache_resource
def load_mkt_caps(symbols):
    """
    Lấy dữ liệu vốn hóa thị trường của các mã cổ phiếu.
    """
    params = {"exchangeName": "HOSE,HNX,UPCOM"}
    screener = Screener()
    df = screener.stock(params=params, limit=1700)
    df_filtered = df[df["ticker"].isin(symbols)]
    result = df_filtered[["ticker", "market_cap"]].copy()
    result["market_cap"] = pd.to_numeric(result["market_cap"], errors="coerce")
    return result.reset_index(drop=True)

@st.cache_data
def fetch_vnindex_data(start_date, end_date, interval='1D', source='VCI'):
    """
    Tải dữ liệu VNINDEX từ VnStock.
    """
    try:
        stock = Vnstock().stock('VNINDEX', source)
        df = stock.quote.history(start=start_date, end=end_date, interval=interval)
        df['time'] = pd.to_datetime(df['time'])
        return df
    except Exception as e:
        print(f"❌ Lỗi khi tải dữ liệu VNINDEX: {e}")
        return pd.DataFrame()
