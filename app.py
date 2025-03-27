# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import ta
from datetime import datetime
from vnstock import Vnstock
import numpy as np

from pypfopt import risk_models, black_litterman
from plotly.subplots import make_subplots



from portfolio.stock_analysis import generate_stock_analysis
from portfolio.optimize import optimize_portfolio, display_results
from portfolio.utils import fetch_vnindex_data

from data.loader import (
    get_financial_ratios, get_company_table, fetch_and_prepare_data,get_cash_flow,get_income_statement,
    get_balance_sheet, get_officers_info, get_subsidiaries_info, get_shareholders_info, get_all_symbols)
from charts.plots import (
    plot_price_volume,plot_accounting_balance,plot_business_results,plot_cash_flow,plot_capital_structure,
    plot_asset_structure,plot_profit_structure,plot_financial_ratios,plot_operating_efficiency,plot_leverage_ratios,plot_pe_ratio,
    plot_pb_ratio,dupont_analysis_plot,plot_combined_charts,plot_stock_vs_vnindex)


# ------------------------
# Các hàm hiển thị trang của ứng dụng
# ------------------------
    # Thêm logo vào sidebar


def main():
    st.set_page_config(page_title="Stock Dashboard", page_icon="📈", layout="wide")
    # Thêm CSS tùy chỉnh cho trang và sidebar
    st.markdown(
        """
        <style>
            body {
                background-color: #f0f2f5;  /* Màu nền sáng xám */
            }
            .header {
                text-align: center;
                background: linear-gradient(135deg, #1e1e1e, #333333); 
                padding: 20px; 
                border-radius: 12px;
                color: white;
            }
            .sidebar .sidebar-content {
                background-color: #4e73df;  /* Màu nền cho sidebar */
                color: white;  /* Màu chữ trong sidebar */
            }
            .sidebar .sidebar-content .st-selectbox, .sidebar .sidebar-content .st-button {
                color: #ffffff;  /* Màu chữ cho các nút và selectbox */
            }
            .sidebar .sidebar-content .st-selectbox select {
                background-color: #007bff;  /* Màu nền cho selectbox */
                color: white;
            }
            .stock-info {
                font-size: 18px;
                font-weight: bold;
                color: #2c3e50; /* Một màu tối cho thông tin */
            }
            .column {
                border: 1px solid #ddd;  /* Đường viền nhẹ */
                border-radius: 5px;
                padding: 15px;
                margin: 10px;
                background-color: white;  /* Nền trắng cho cột */
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);  /* Đổ bóng cho cột */
            }
            .highlight {
                color: green;
            }
            .alert {
                color: red;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    logo_path = "LOGO.png"  # Điền đúng đường dẫn đến logo của bạn
    st.sidebar.image(logo_path, use_column_width=True)  # Hiển thị logo trong sidebar

     # Thêm banner/header
    banner_path = "banner.png"  # Điền đúng đường dẫn đến ảnh header của bạn
    st.image(banner_path, use_column_width=True)  # Hiển thị banner ở header
    # Thêm tiêu đề cho ứng dụng






    # Lấy danh sách mã cổ phiếu
    symbols = get_all_symbols()  # symbols là list các mã cổ phiếu
    # Hiển thị selectbox với danh sách mã cổ phiếu, mặc định chọn 'VCI' nếu có
    #default_index = symbols.index('FPT') if 'FPT' in symbols else 0
    default_index = next((i for i, symbol in enumerate(symbols) if symbol.strip().upper() == "FPT"), 0)

    code = st.selectbox("Chọn mã cổ phiếu", options=symbols, index=default_index)
    

       
    # Xác định khoảng thời gian (5 năm gần đây)
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - pd.DateOffset(years=5)).strftime('%Y-%m-%d')
    with st.spinner("Đang tải dữ liệu cổ phiếu..."):
        # Tải dữ liệu từ các module loader
        company_df = get_company_table(code)
        df_vnindex = fetch_vnindex_data(start_date, end_date)
        df_stock = fetch_and_prepare_data(code, start_date, end_date)
        cstc = get_financial_ratios(code)

        
        selected_columns = [
            ( 'Meta', 'Năm'),
            ('Chỉ tiêu định giá', 'Vốn hóa (Tỷ đồng)'),
            ('Chỉ tiêu định giá', 'Số CP lưu hành (Triệu CP)'),
            ('Chỉ tiêu định giá', 'P/E'),
            ('Chỉ tiêu định giá', 'P/B'),
            ('Chỉ tiêu định giá', 'EPS (VND)'),
            ('Chỉ tiêu định giá', 'EV/EBITDA'),
            ('Chỉ tiêu khả năng sinh lợi', 'ROA (%)'),
            ('Chỉ tiêu khả năng sinh lợi', 'ROE (%)'),
            ('Chỉ tiêu cơ cấu nguồn vốn','Nợ/VCSH'),
            ('Chỉ tiêu cơ cấu nguồn vốn','TSCĐ / Vốn CSH'),
            ('Chỉ tiêu hiệu quả hoạt động','Vòng quay tài sản'),
            ( 'Chỉ tiêu khả năng sinh lợi','Biên lợi nhuận ròng (%)'),
            ( 'Chỉ tiêu thanh khoản', 'Đòn bẩy tài chính'), 
            ( 'Chỉ tiêu định giá','BVPS (VND)'),
            ('Chỉ tiêu hiệu quả hoạt động',   'Số ngày thu tiền bình quân'),
                ('Chỉ tiêu hiệu quả hoạt động',    'Số ngày tồn kho bình quân'),
                ('Chỉ tiêu hiệu quả hoạt động', 'Số ngày thanh toán bình quân'),
        ]
        
        df_insights = cstc.loc[:, selected_columns]
        df_insights.columns = ['Năm','Vốn hóa (Tỷ đồng)','Số CP lưu hành (Triệu CP)', 'P/E', 'P/B', 'EPS', 'EV/EBITDA','ROA','ROE','Nợ/VCSH','TSCĐ/VSCH','Vòng quay tài sản',
                            'Biên lợi nhuận ròng','Đòn bẩy tài chính','BVPS','Số ngày thu tiền bình quân','Số ngày tồn kho bình quân','Số ngày thanh toán bình quân']
        
        ebitda = df_insights['EV/EBITDA'].iloc[0]
        pe = df_insights['P/E'].iloc[0]
        pb = df_insights['P/B'].iloc[0]
        eps = df_insights['EPS'].iloc[0]
        mar = df_insights['Vốn hóa (Tỷ đồng)'].iloc[0]
        cp = df_insights['Số CP lưu hành (Triệu CP)'].iloc[0]

    

    
    # Hiển thị thông tin tổng quát
    col1, col2 = st.columns((3, 7))

    with col1: 
        # Lấy dữ liệu của ngày hiện tại
        latest_data = df_stock.iloc[-1]
        current_price = latest_data['close']  # Giá hiện tại
        current_date = latest_data.name  # Giả định rằng cột index là ngày

        # Tạo dropdown và lấy thông tin chọn
        time_period = st.selectbox("Chọn Khoảng Thời Gian:", ["24h", "7 ngày", "1 tháng"], index=0)

        # Khởi tạo biến cho giá tham chiếu
        reference_price = current_price

        # Tính toán giá tham chiếu dựa trên khoảng thời gian đã chọn
        if time_period == "24h":
            reference_data = df_stock.iloc[-2]  # Lấy dữ liệu của ngày hôm trước
        elif time_period == "7 ngày":
            reference_data = df_stock.iloc[-8]  # Lấy dữ liệu 7 ngày trước (nếu có đủ dữ liệu)
        elif time_period == "1 tháng":
            reference_data = df_stock.iloc[-30]  # Lấy dữ liệu 30 ngày trước (nếu có đủ dữ liệu)

        # Lấy giá tham chiếu ở thời điểm đã chọn
        reference_price = reference_data['close']

        # Tính toán chênh lệch và phần trăm tăng giảm
        change = current_price - reference_price
        percent = round((change / reference_price) * 100, 2)

        # Bố cục hiển thị trên Streamlit với kích thước nhỏ và căn giữa
        # Hiển thị thông tin giá hôm nay với bố cục đẹp hơn
        # Hiển thị thông tin giá hiện tại, thay đổi và phần trăm tăng giảm mà không có tiêu đề
        # Hiển thị thông tin giá hiện tại, thay đổi và phần trăm tăng/giảm
        # Tính toán màu sắc cho change
        if change > 0:
            color_change = "green"  # Màu xanh cho thay đổi dương
            change_display = f"+{change:,.2f}"  # Thêm dấu '+' nếu tăng
        else:
            color_change = "red"  # Màu đỏ cho thay đổi âm
            change_display = f"{change:,.2f}"  # Dấu '-' tự động sẽ có nếu giảm

        # Hiển thị thông tin với bố cục đẹp hơn
        st.markdown(
            f"""
            <div style="text-align: center; background: linear-gradient(135deg, #1e1e1e, #333333); padding: 20px; border-radius: 12px; max-width: 400px; margin: auto; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.5);">
                <h2 style="color: white; font-size: 36px; margin: 0;"><strong>{current_price:,.2f}</strong></h2>
                <div style="display: flex; justify-content: center; align-items: center; margin: 10px 0;">
                    <span style="font-size: 24px; color: {color_change}; font-weight: bold; margin-right: 15px;">{change_display}</span>
                    <span style="font-size: 20px; color: white; padding: 5px 10px; border: 2px solid {color_change}; border-radius: 5px;">
                        {percent}%
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        # Thiết lập 3 cột đầu tiên
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("<h6 style='margin: 0;'>Vốn Hóa (Đồng)</h6>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: 18px; font-weight: bold;'>{mar:,.0f}</p>", unsafe_allow_html=True)

        with col2:
            st.markdown("<h6 style='margin: 0;'>Số Cổ Phiếu Lưu Hành</h6>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: 18px; font-weight: bold;'>{cp:,.0f}</p>", unsafe_allow_html=True)

        with col3:
            st.markdown("<h6 style='margin: 0;'>EPS</h6>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: 18px; font-weight: bold;'>{eps:,.0f}</p>", unsafe_allow_html=True)

        # Thiết lập 3 cột thứ hai
        col4, col5, col6 = st.columns(3)

        with col4:
            st.markdown("<h6 style='margin: 0;'>EV/EBITDA</h6>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: 18px; font-weight: bold;'>{ebitda:.2f}</p>", unsafe_allow_html=True)

        with col5:
            st.markdown("<h6 style='margin: 0;'>P/E</h6>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: 18px; font-weight: bold;'>{pe:.2f}</p>", unsafe_allow_html=True)

        with col6:
            st.markdown("<h6 style='margin: 0;'>P/B</h6>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: 18px; font-weight: bold;'>{pb:.2f}</p>", unsafe_allow_html=True)
    # 🔹 Hiển thị biểu đồ
    # Điều hướng trang

    st.sidebar.title("Trợ lý tài chính")
    options = st.sidebar.selectbox("Chức năng", 
                                   ["Phân tích cổ phiếu", "Công cụ tối ưu danh mục đầu tư"])

    if options == 'Phân tích cổ phiếu':
        phan_tich_cp(code, df_stock, df_vnindex, df_insights, company_df)

    elif options == 'Công cụ tối ưu danh mục đầu tư':
        symbols, viewdict, confidences, investment_goal, target_return, total_value = get_user_inputs()
        if st.button("Chạy tối ưu danh mục"):
            portfolio_optimization_tool(symbols, viewdict, confidences, investment_goal, target_return, total_value)



# hàm xử lý nhập liệu chính
def get_user_inputs():
    st.title("Công cụ tối ưu danh mục đầu tư")
    
    # Nhập danh sách mã cổ phiếu (ví dụ: "AAA,BBB,CCC")
    symbols_input = st.text_input("Nhập danh sách mã cổ phiếu (cách nhau bằng dấu phẩy):", "FPT,VCI,HPG")
    symbols = [s.strip() for s in symbols_input.split(",") if s.strip()]
    
    # Nhập tổng vốn đầu tư
    total_value = st.number_input("Nhập tổng vốn đầu tư (VNĐ):", min_value=1000000, value=1000000, step=100000)
    
    # Nhập kỳ vọng lợi nhuận và mức độ tự tin cho từng mã
    st.subheader("Thông số cho từng mã")
    viewdict = {}
    confidences = {}
    for symbol in symbols:
        col1, col2 = st.columns(2)
        with col1:
            view = st.number_input(f"Tỷ suất lợi nhuận kỳ vọng của {symbol} (%)", value=10, key=f"view_{symbol}")
        with col2:
            conf = st.number_input(
                f"Mức độ tự tin của {symbol} (%)",
                min_value=0, 
                max_value=100, 
                value=65, 
                key=f"conf_{symbol}"
            )

        viewdict[symbol] = view / 100
        confidences[symbol] = conf / 100
    
    # Lựa chọn mục tiêu đầu tư
    investment_goal = st.selectbox("Chọn mục tiêu đầu tư:", 
                                   ["Tối đa hoá tỷ lệ Sharpe", 
                                    "Đạt mức lợi nhuận mục tiêu và tối thiểu rủi ro phát sinh",
                                    "KIỂM SOÁT CVaR VÀ TỐI ĐA LỢI NHUẬN KỲ VỌNG",
                                    "KIỂM SOÁT CDaR VÀ TỐI ĐA LỢI NHUẬN KỲ VỌNG"])
    
    target_return = None
    if investment_goal == "Đạt mức lợi nhuận mục tiêu và tối thiểu rủi ro phát sinh":
        target_return = st.number_input("Nhập mức lợi nhuận mục tiêu (%):", value=5.0)
    
    return symbols, viewdict, confidences, investment_goal, target_return, total_value


def portfolio_optimization_tool(symbols, viewdict, confidences, investment_goal, target_return, total_value):
    # Xác định khoảng thời gian lấy dữ liệu (5 năm gần nhất)
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - pd.DateOffset(years=5)).strftime('%Y-%m-%d')
    
    # Phân tích dữ liệu cổ phiếu (tạo bảng thông số tài chính)
    result_df = generate_stock_analysis(symbols, start_date, end_date, viewdict, confidences)
    
    # Tối ưu danh mục – nếu có target_return được nhập thì truyền vào optimize_portfolio
    if investment_goal == "Đạt mức lợi nhuận mục tiêu và tối thiểu rủi ro phát sinh":
        ef, weights, allocation, leftover_cash, details_df = optimize_portfolio(
            symbols, total_value, viewdict, confidences, investment_goal, target_return=target_return / 100
        )
    else:
        ef, weights, allocation, leftover_cash, details_df = optimize_portfolio(
            symbols, total_value, viewdict, confidences, investment_goal
        )
    
    # Hiển thị kết quả qua Streamlit với các hiệu ứng trực quan
    display_results(result_df, ef, weights, allocation, leftover_cash, details_df, investment_goal)
    

# ------------------------
# Trang phân tích cổ phiếu
# ------------------------


def phan_tich_cp(code, df_stock, df_vnindex, df_insights, company_df):

    
    df_cash_flow = get_cash_flow(code)
    df_income_statement = get_income_statement(code)
    df_balance = get_balance_sheet(code)
    
    left_column, right_column = st.columns((7, 3))
    # Cột bên phải
    with right_column:
        with st.expander("**Xem chi tiết dữ liệu tài chính**", expanded=True):
            df_stock_reversed = df_stock.iloc[::-1]
            st.dataframe(df_stock_reversed)  

    # Cột bên trái
    with left_column:
        plot_price_volume(df_stock)

    # Tạo các tab trong trang "Phân tích cổ phiếu"
    t2, t3, t4, t5, t6= st.tabs([
        "Phân tích 360", "Phân tích kĩ thuật",
        "Tài chính", "Hồ sơ", "Dữ liệu"])

    
    with t2:
        # 🔹 Hiển thị biểu đồ chứng khoán so với VN-Index
        plot_stock_vs_vnindex(df_stock, df_vnindex, code)

        # Tạo các tab cho các chỉ số
        tab1, tab2, tab3 = st.tabs(["Chỉ Số Hiệu Quả Hoạt Động", "Chỉ Số Sức Khỏe Tài Chính", "Chỉ Số Định Giá"])

        # Tab 1: Chỉ Số Hiệu Quả Hoạt Động
        with tab1:
       
            plot_operating_efficiency(df_insights)
            st.markdown("Biểu đồ này cho thấy khả năng sinh lời và quản lý tài sản của công ty.")

        # Tab 2: Chỉ Số Sức Khỏe Tài Chính
        with tab2:
  
            plot_leverage_ratios(df_insights)
            st.markdown("Biểu đồ này cho thấy mức độ sử dụng nợ trong cấu trúc vốn công ty.")

            plot_financial_ratios(df_insights)
            st.markdown("Biểu đồ này thể hiện các tỷ suất lợi nhuận khác nhau của công ty.")

        # Tab 3: Chỉ Số Định Giá
        with tab3:            
            plot_pe_ratio(df_insights)
            st.markdown("Biểu đồ này giúp đánh giá cổ phiếu dựa trên thu nhập của nó.")

            plot_pb_ratio(df_insights)
            st.markdown("Biểu đồ này cho thấy giá trị tài sản so với tài sản thực tế.")

            dupont_analysis_plot(df_insights)
            st.markdown("Phân tích DuPont giúp phân tích lợi nhuận và đòn bẩy tài chính của công ty.")
            
    with t3:
        st.subheader("Chọn Các Thông Số Kỹ Thuật")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Chọn Cửa Sổ SMA")
            available_sma_windows = ['10', '14', '20', '50', '100']
            sma_windows = st.multiselect('Lựa chọn cửa sổ SMA (chu kỳ)', available_sma_windows)
        with col2:
            st.markdown("### Chọn Cửa Sổ EMA")
            available_ema_windows = ['10', '14', '20', '50', '100', '200']
            ema_windows = st.multiselect('Lựa chọn cửa sổ EMA (chu kỳ)', available_ema_windows)

        st.markdown("<hr>", unsafe_allow_html=True)

        # Giả sử df_stock là DataFrame chứa dữ liệu thị trường với các cột 'time', 'open', 'close', 'volume'
        plot_combined_charts(df_stock, sma_windows, ema_windows)
            
    with t4:
        st.subheader("Phân Tích Kết Quả Tài Chính")

        # Tạo các expander để phát hiện và hiển thị từng biểu đồ
        with st.expander("Cấu Trúc Vốn"):
            plot_capital_structure(df_balance)

        with st.expander("Cấu Trúc Tài Sản"):
            plot_asset_structure(df_balance)

        with st.expander("Bảng Cân Đối Kế Toán"):
            plot_accounting_balance(df_balance)

        with st.expander("Kết Quả Kinh Doanh"):
            plot_business_results(df_income_statement)

        with st.expander("Lưu Chuyển Tiền Tệ"):
            plot_cash_flow(df_cash_flow)

        with st.expander("Cấu Trúc Lợi Nhuận"):
            plot_profit_structure(df_income_statement)

        # Thêm một số thông báo hỗ trợ, nhắc nhở người dùng về nội dung
        st.markdown("Bạn có thể mở các phần để xem biểu đồ chi tiết hơn. Di chuyển chuột qua các phần để xem thông tin rõ hơn.")
    
        
    with t5:
    # Tạo khung cho mỗi khối thông tin
        st.markdown("<h2 style='text-align: center;'>Thông Tin Công Ty</h2>", unsafe_allow_html=True)

        # Ban lãnh đạo
        with st.container():
            st.markdown("<h3 style='color: #4B0082;'>Ban Lãnh Đạo</h3>", unsafe_allow_html=True)
            officers_info = get_officers_info(code)
            if officers_info is None or officers_info.empty:
                st.warning("Không có thông tin ban lãnh đạo.")
            else:
                # Tạm bỏ styling để kiểm tra
                st.dataframe(officers_info)  # Thử mà không có highlighting
                # Nếu muốn giữ highlighting:
                # st.dataframe(officers_info.style.highlight_max(axis=0))

        # Công ty con
        with st.container():
            st.markdown("<h3 style='color: #4B0082;'>Công Ty Con</h3>", unsafe_allow_html=True)
            subsidiaries_info = get_subsidiaries_info(code)
            if subsidiaries_info is None or subsidiaries_info.empty:
                st.warning("Không có thông tin về công ty con.")
            else:
                # Tạm bỏ styling để kiểm tra
                st.dataframe(subsidiaries_info)  # Thử mà không có highlighting

        # Cổ đông lớn
        with st.container():
            st.markdown("<h3 style='color: #4B0082;'>Cổ Đông Lớn</h3>", unsafe_allow_html=True)
            shareholders_info = get_shareholders_info(code)
            if shareholders_info is None or shareholders_info.empty:
                st.warning("Không có thông tin về cổ đông lớn.")
            else:
                # Tạm bỏ styling để kiểm tra
                st.dataframe(shareholders_info)  # Thử mà không có highlighting
  
    with t6:
        cdkt, kqkd, lctt = st.tabs(["Bảng Cân Đối Kế Toán", "Báo Cáo Kết Quả Kinh Doanh", "Báo Cáo Lưu Chuyển Tiền Tệ"])

        with cdkt:
            # In đậm tên cột
            styled_balance = df_balance.style.set_properties(**{'font-weight': 'bold'}, subset=df_balance.columns)
            st.dataframe(styled_balance.highlight_max(axis=0))  # Highlight maximum values

            # Cung cấp tùy chọn tải xuống
            csv_balance = df_balance.to_csv(index=False).encode('utf-8')
            st.download_button("Tải Bảng Cân Đối Kế Toán", csv_balance, "balance_sheet.csv", "text/csv")

        with kqkd:
             # In đậm tên cột
            styled_income = df_income_statement.style.set_properties(**{'font-weight': 'bold'}, subset=df_income_statement.columns)
            st.dataframe(styled_income.highlight_max(axis=0))

            # Cung cấp tùy chọn tải xuống
            csv_income = df_income_statement.to_csv(index=False).encode('utf-8')
            st.download_button("Tải Báo Cáo Kết Quả Kinh Doanh", csv_income, "income_statement.csv", "text/csv")

        with lctt:
            # In đậm tên cột
            styled_cash_flow = df_cash_flow.style.set_properties(**{'font-weight': 'bold'}, subset=df_cash_flow.columns)
            st.dataframe(styled_cash_flow.highlight_max(axis=0))

            # Cung cấp tùy chọn tải xuống
            csv_cash_flow = df_cash_flow.to_csv(index=False).encode('utf-8')
            st.download_button("Tải Báo Cáo Lưu Chuyển Tiền Tệ", csv_cash_flow, "cash_flow_statement.csv", "text/csv")
        
if __name__ == "__main__":
    main()
