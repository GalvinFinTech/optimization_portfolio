import pandas as pd
from vnstock import Vnstock, Screener
import streamlit as st
from vnstock.explorer.vci import Company

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


# Hàm lấy tin tức công ty
def get_company_news(code):
    company = Company(code)
    news_df = company.news()
    
    # Lọc và đổi tên các cột
    news_df_filtered = news_df[['news_title', 'news_source_link']]
    news_df_filtered.columns = ['Tiêu đề Tin Tức', 'Liên kết Nguồn']
    
    # Tạo hyperlink cho các liên kết trong cột Tiêu đề Tin Tức
    news_df_filtered['Tiêu đề Tin Tức'] = news_df_filtered.apply(
        lambda row: f"<b><a href='{row['Liên kết Nguồn']}' target='_blank'>{row['Tiêu đề Tin Tức']}</a></b>", axis=1
    )
    
    return news_df_filtered

# Hàm lấy sự kiện công ty
def get_company_events(code):
    company = Company(code)
    events_df = company.events()
    
    # Lọc và đổi tên các cột
    events_df_filtered = events_df[['event_title', 'public_date', 'source_url']]
    events_df_filtered.columns = ['Tiêu đề Sự Kiện', 'Ngày Công Bố', 'Liên kết Nguồn']
    
    # Tạo hyperlink cho các liên kết trong cột Tiêu đề Sự Kiện
    events_df_filtered['Tiêu đề Sự Kiện'] = events_df_filtered.apply(
        lambda row: f"<b><a href='{row['Liên kết Nguồn']}' target='_blank'>{row['Tiêu đề Sự Kiện']}</a></b>", axis=1
    )
    
    return events_df_filtered

import re
import pandas as pd
import streamlit as st
from vnstock.explorer.vci import Company

import re
import pandas as pd
import streamlit as st

# Hàm lấy báo cáo phân tích công ty
def reports(code):
    # Tạo đối tượng công ty từ mã cổ phiếu
    company = Company(code)
    
    # Lấy dữ liệu báo cáo từ công ty
    reports_df = company.reports()
    
    # Kiểm tra nếu không có dữ liệu báo cáo
    if reports_df.empty:
        return pd.DataFrame()  # Trả về DataFrame trống nếu không có báo cáo
    
    # Lọc và đổi tên các cột
    reports_df_filtered = reports_df[['date', 'name', 'description', 'link']]
    reports_df_filtered.columns = ['Ngày', 'Tên Báo Cáo', 'Mô Tả', 'Liên kết']
    
    # Chuyển định dạng Ngày từ ISO 8601 thành DD-MM-YYYY
    reports_df_filtered['Ngày'] = pd.to_datetime(reports_df_filtered['Ngày']).dt.strftime('%d-%m-%Y')
    
    # Tạo hyperlink cho các liên kết trong cột Tên Báo Cáo
    reports_df_filtered['Tên Báo Cáo'] = reports_df_filtered.apply(
        lambda row: f"<b><a href='{row['Liên kết']}' target='_blank'>{row['Tên Báo Cáo']}</a></b>", axis=1
    )
    
    return reports_df_filtered

# Hàm lọc và lấy các báo cáo có chứa từ khóa nhất định (Khuyến nghị, Giá mục tiêu điều chỉnh)
def get_reports(code):
    reports_df_filtered = reports(code)
    
    # Kiểm tra nếu không có báo cáo
    if reports_df_filtered.empty:
        return "Chưa có báo cáo"  # Nếu không có dữ liệu thì trả về thông báo
    
    # Hàm tìm Khuyến nghị trong Mô tả
    def extract_recommendation(description):
        recommendations = ['MUA', 'BÁN', 'KHÔNG ĐÁNH GIÁ', 'KÉM KHẢ QUAN', 'PHÙ HỢP THỊ TRƯỜNG']
        for rec in recommendations:
            if rec in description:
                return rec
        return 'Chưa xác định'  # Nếu không có khuyến nghị rõ ràng

    # Hàm tìm Giá mục tiêu điều chỉnh trong Mô tả (dưới dạng số)
    def extract_target_price(description):
        match = re.search(r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:VNĐ|đ)', description)
        if match:
            return match.group(1)
        return 'Chưa xác định'

    # Áp dụng hàm để lấy Khuyến nghị và Giá mục tiêu điều chỉnh từ Mô tả
    reports_df_filtered['Khuyến nghị'] = reports_df_filtered['Mô Tả'].apply(extract_recommendation)
    reports_df_filtered['Giá mục tiêu điều chỉnh'] = reports_df_filtered['Mô Tả'].apply(extract_target_price)
    
    # Tạo cột "Nguồn" từ tên báo cáo
    reports_df_filtered['Nguồn'] = reports_df_filtered['Tên Báo Cáo']
    
    # Tạo cột "Tải về"
    reports_df_filtered['Tải về'] = reports_df_filtered['Liên kết'].apply(lambda x: f"<a href='{x}' target='_blank'>Tải về</a>")
    
    # Chỉ giữ lại các cột cần thiết
    reports_df_filtered = reports_df_filtered[['Ngày', 'Nguồn', 'Khuyến nghị', 'Giá mục tiêu điều chỉnh', 'Tải về']]
    
    return reports_df_filtered




