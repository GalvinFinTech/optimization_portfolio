import pandas as pd
import numpy as np
from datetime import datetime
from vnstock import Vnstock

YEARS = 5  # Sử dụng 5 năm làm khoảng thời gian cơ bản

def calculate_cagr(price_series):
    """
    Tính CAGR (Compound Annual Growth Rate) – Lãi kép trung bình hàng năm.
    """
    if len(price_series) < 250:
        return np.nan
    start_price = price_series.iloc[0]
    end_price = price_series.iloc[-1]
    if start_price <= 0 or np.isnan(start_price):
        return np.nan
    return ((end_price / start_price) ** (1 / YEARS) - 1) * 100

def compute_beta(data, stock, market):
    """
    Tính Beta của cổ phiếu so với thị trường.
    """
    log_returns = np.log(data / data.shift(1))
    cov = log_returns.cov() * 250
    cov_with_market = cov.loc[stock, market]
    market_var = log_returns[market].var() * 250
    return cov_with_market / market_var if market_var > 0 else np.nan

def calculate_vnindex_annual_return(vnindex_prices):
    """
    Tính tỷ suất sinh lời trung bình hàng năm của VNINDEX.
    """
    if len(vnindex_prices) < 250:
        return np.nan
    current_price = vnindex_prices.iloc[-1]
    past_price = vnindex_prices.iloc[0]
    annual_return = ((current_price / past_price) ** (1 / YEARS) - 1) * 100
    return annual_return

def compute_capm(data, stock, market, riskfree=0.03):
    """
    Tính tỷ suất sinh lời theo mô hình CAPM.
    """
    beta = compute_beta(data, stock, market)
    market_annual_return = calculate_vnindex_annual_return(data[market])
    risk_premium = market_annual_return - riskfree
    return riskfree + beta * risk_premium if beta is not np.nan else np.nan

def generate_stock_analysis(symbols, start_date, end_date, view_dict, confidence_dict):
    """
    Lấy dữ liệu giá cổ phiếu, tính toán các chỉ số tài chính (CAGR, CAPM, Beta)
    và kết hợp với dữ liệu lợi nhuận kỳ vọng, mức độ tự tin.
    
    Trả về DataFrame chứa các thông số cho từng mã cổ phiếu.
    """
    market_symbol = 'VNINDEX'
    close_prices = pd.DataFrame()

    for symbol in symbols + [market_symbol]:
        try:
            print(f"📥 Đang tải dữ liệu cho {symbol}...")
            stock = Vnstock().stock(symbol=symbol, source="VCI")
            df = stock.quote.history(start=start_date, end=end_date, interval='1D')
            if df.empty:
                print(f"⚠️ Không có dữ liệu cho {symbol}.")
                continue
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            close_prices[symbol] = df['close']
        except Exception as e:
            print(f"❌ Lỗi khi lấy dữ liệu {symbol}: {e}")
    
    if close_prices.empty or market_symbol not in close_prices.columns:
        raise ValueError("❌ Không có dữ liệu tải về thành công hoặc dữ liệu VNINDEX không khả dụng!")
    
    result_df = pd.DataFrame(index=symbols, columns=[
        "CAGR (%)", "CAPM(%)", "Beta", "Lợi nhuận kỳ vọng (%)", "Mức độ tự tin (%)"
    ])
    
    for symbol in symbols:
        if symbol not in close_prices.columns:
            continue
        cagr = calculate_cagr(close_prices[symbol])
        stock_annual_return = compute_capm(close_prices, symbol, market_symbol)
        beta = compute_beta(close_prices, symbol, market_symbol)
        expected_return = view_dict.get(symbol, np.nan) * 100
        confidence = confidence_dict.get(symbol, np.nan) * 100
        result_df.loc[symbol] = [cagr, stock_annual_return, beta, expected_return, confidence]
    
    return result_df
