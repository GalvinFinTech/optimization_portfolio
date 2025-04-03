# data/loader.py
import pandas as pd
from vnstock import Vnstock
from vnstock.explorer.vci import Company
import streamlit as st
import datetime
# data/loader.py


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

@st.cache_resource
def get_all_symbols(source='VCI'):
    """
    Lấy danh sách tất cả các mã cổ phiếu trên thị trường từ nguồn dữ liệu.
    Sử dụng hàm `stock.listing.all_symbols()` từ thư viện vnstock.
    """
    # Sử dụng một mã cổ phiếu bất kỳ (ở đây dùng 'ACB') để truy cập đối tượng listing
    stock = Vnstock().stock(symbol='ACB', source=source)
    return stock.listing.all_symbols()

