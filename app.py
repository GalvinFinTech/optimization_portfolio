# app.py
import os
if not os.path.isdir('data'):
    os.system('git clone https://github.com/robertmartin8/PyPortfolioOpt.git')
    os.chdir('PyPortfolioOpt/cookbook')
    
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

from vnstock import Screener, Vnstock
from vnstock.explorer.vci import Company
import concurrent.futures
import plotly.graph_objects as go



from portfolio.stock_analysis import generate_stock_analysis
from portfolio.optimize import optimize_portfolio, display_results
from portfolio.utils import fetch_vnindex_data, get_company_news, get_company_events,get_reports

from data.loader import (trade_signal_analysis,
    get_financial_ratios, get_company_table, fetch_and_prepare_data,get_cash_flow,get_income_statement,get_ratios,
    get_balance_sheet, get_officers_info, get_subsidiaries_info, get_shareholders_info, get_all_symbols)
from charts.plots import (
    plot_price_volume,plot_accounting_balance,plot_business_results,plot_cash_flow,plot_capital_structure,plot_metric,
    plot_asset_structure,plot_profit_structure,plot_financial_ratios,plot_operating_efficiency,plot_leverage_ratios,plot_pe_ratio,
    plot_pb_ratio,dupont_analysis_plot,plot_combined_charts,plot_stock_vs_vnindex, visualize_analysis)





# Hàm tải dữ liệu cổ phiếu với xử lý lỗi và caching
@st.cache_data
def load_stock_data(code, start_date, end_date):
    """Tải dữ liệu cổ phiếu với xử lý lỗi."""
    try:
        df_stock = fetch_and_prepare_data(code, start_date, end_date)
        if df_stock.empty:
            st.warning(f"Dữ liệu cổ phiếu {code} trống.")
            return None
        return df_stock
    except Exception as e:
        st.error(f"Lỗi khi tải dữ liệu cho {code}: {e}")
        return None

# Hàm tải dữ liệu tài chính với xử lý lỗi và caching
@st.cache_data
def load_insights(code):
    """Tải dữ liệu tài chính với xử lý lỗi, để NaN cho các cột không có dữ liệu."""
    try:
        cstc = get_financial_ratios(code)
        
        # Danh sách các cột mong muốn
        selected_columns = [
            ('Meta', 'Năm'),
            ('Chỉ tiêu định giá', 'Vốn hóa (Tỷ đồng)'),
            ('Chỉ tiêu định giá', 'Số CP lưu hành (Triệu CP)'),
            ('Chỉ tiêu định giá', 'P/E'),
            ('Chỉ tiêu định giá', 'P/B'),
            ('Chỉ tiêu định giá', 'EPS (VND)'),
            ('Chỉ tiêu định giá', 'EV/EBITDA'),
            ('Chỉ tiêu khả năng sinh lợi', 'ROA (%)'),
            ('Chỉ tiêu khả năng sinh lợi', 'ROE (%)'),
            ('Chỉ tiêu cơ cấu nguồn vốn', 'Nợ/VCSH'),
            ('Chỉ tiêu cơ cấu nguồn vốn', 'TSCĐ / Vốn CSH'),
            ('Chỉ tiêu hiệu quả hoạt động', 'Vòng quay tài sản'),
            ('Chỉ tiêu khả năng sinh lợi', 'Biên lợi nhuận ròng (%)'),
            ('Chỉ tiêu thanh khoản', 'Đòn bẩy tài chính'),
            ('Chỉ tiêu định giá', 'BVPS (VND)'),
            ('Chỉ tiêu hiệu quả hoạt động', 'Số ngày thu tiền bình quân'),
            ('Chỉ tiêu hiệu quả hoạt động', 'Số ngày tồn kho bình quân'),
            ('Chỉ tiêu hiệu quả hoạt động', 'Số ngày thanh toán bình quân'),
        ]
        
        # Tạo DataFrame rỗng với các cột mong muốn
        df_insights = pd.DataFrame(columns=selected_columns)
        
        # Điền dữ liệu từ cstc vào các cột có sẵn
        for col in selected_columns:
            if col in cstc.columns:
                df_insights[col] = cstc[col]
        
        # Đổi tên cột
        df_insights.columns = [
            'Năm', 'Vốn hóa (Tỷ đồng)', 'Số CP lưu hành (Triệu CP)', 'P/E', 'P/B', 'EPS', 'EV/EBITDA',
            'ROA', 'ROE', 'Nợ/VCSH', 'TSCĐ/VSCH', 'Vòng quay tài sản', 'Biên lợi nhuận ròng',
            'Đòn bẩy tài chính', 'BVPS', 'Số ngày thu tiền bình quân', 'Số ngày tồn kho bình quân',
            'Số ngày thanh toán bình quân'
        ]
        
        return df_insights
    except Exception as e:
        st.error(f"Lỗi khi tải dữ liệu tài chính cho {code}: {e}")
        return pd.DataFrame()

# Hàm hiển thị thông tin tổng quát
# Hàm hỗ trợ lấy giá trị an toàn từ DataFrame
def get_safe_value(df, column):
    if column in df.columns and not df[column].isna().all():
        return df[column].iloc[0]  # Lấy giá trị đầu tiên
    return 'N/A'  # Trả về 'N/A' nếu cột không tồn tại hoặc không có dữ liệu


def display_general_info(df_stock, df_insights):
    # Chia layout thành 2 cột: giá cổ phiếu (trái), chỉ số tài chính (phải)
    col1, col2 = st.columns((3, 7))

    # Cột 1: Hiển thị giá cổ phiếu và thay đổi
    with col1:
        if df_stock is not None and not df_stock.empty:
            # Lấy dữ liệu mới nhất
            latest_data = df_stock.iloc[-1]
            current_price = latest_data['close']

            # Dropdown chọn khoảng thời gian
            time_period = st.selectbox("Chọn Khoảng Thời Gian:", ["24h", "7 ngày", "1 tháng"], index=0)

            # Xác định giá tham chiếu dựa trên khoảng thời gian
            if time_period == "24h":
                reference_data = df_stock.iloc[-2] if len(df_stock) > 1 else df_stock.iloc[-1]
            elif time_period == "7 ngày":
                reference_data = df_stock.iloc[-8] if len(df_stock) > 7 else df_stock.iloc[-1]
            elif time_period == "1 tháng":
                reference_data = df_stock.iloc[-30] if len(df_stock) > 29 else df_stock.iloc[-1]
            reference_price = reference_data['close']

            # Tính toán thay đổi giá và phần trăm
            change = current_price - reference_price
            percent = round((change / reference_price) * 100, 2) if reference_price != 0 else 0
            color_change = "green" if change > 0 else "red"
            change_display = f"+{change:,.2f}" if change > 0 else f"{change:,.2f}"

            # Hiển thị giá và thay đổi bằng HTML/CSS
            st.markdown(
                f"""
                <div style="text-align: center; background: linear-gradient(135deg, #1e1e1e, #333333); padding: 20px; border-radius: 12px; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.5);">
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
        else:
            st.warning("Không có dữ liệu giá cổ phiếu.")

    # Cột 2: Hiển thị các chỉ số tài chính
    with col2:
        if not df_insights.empty:
            # Danh sách chỉ số tài chính và nhãn
            metrics = {
                'Vốn Hóa (Tỷ đồng)': 'Vốn hóa (Tỷ đồng)', 
                'Số CP lưu hành (Triệu CP)': 'Số CP lưu hành (Triệu CP)', 
                'EPS': 'EPS', 
                'EV/EBITDA': 'EV/EBITDA', 
                'P/E': 'P/E', 
                'P/B': 'P/B'
            }

            # Chia thành 3 cột nhỏ để hiển thị các chỉ số
            cols = st.columns(3)
            for i, (label, col_name) in enumerate(metrics.items()):
                with cols[i % 3]:
                    value = get_safe_value(df_insights, col_name)
                    st.markdown(f"<h6>{label}</h6>", unsafe_allow_html=True)
                    if value == 'N/A':
                        st.markdown("<p style='font-size: 18px; font-weight: bold; color: gray;'>Không có dữ liệu</p>", unsafe_allow_html=True)
                    else:
                        # Định dạng cho các giá trị lớn (vốn hóa, số cổ phiếu)
                        if label in ['Vốn Hóa (Đồng)', 'Số CP lưu hành (Triệu CP)']:
                            st.markdown(f"<p style='font-size: 18px; font-weight: bold;'>{value:,.0f}</p>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<p style='font-size: 18px; font-weight: bold;'>{value:.2f}</p>", unsafe_allow_html=True)
        else:
            st.warning("Không có dữ liệu tài chính để hiển thị.")

         

def get_user_inputs():
    st.title("Công cụ Tối ưu Danh mục Đầu tư")


    # Nhập danh sách mã cổ phiếu (ví dụ: "aaa,bbb,ccc")
    symbols_input = st.text_input("Nhập danh sách mã cổ phiếu (cách nhau bằng dấu phẩy):", "FPT,VNM,HPG,SSI")
    symbols = [s.strip() for s in symbols_input.split(",") if s.strip()]

    # Nhập tổng vốn đầu tư
    total_value = st.number_input("Nhập tổng vốn đầu tư (VNĐ):", min_value=1000000, value=1000000, step=100000)



    # Nhập kỳ vọng lợi nhuận và mức độ tự tin cho từng mã
    st.subheader("Thông số cho từng mã")
    viewdict = {}
    confidences = {}

    with st.expander("Nhập thông số cho từng mã cổ phiếu", expanded=True):
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
    st.subheader("Lựa chọn mục tiêu đầu tư")
    investment_goal = st.selectbox("Chọn mục tiêu đầu tư:", 
                                   ["Tối đa hoá tỷ lệ Sharpe", 
                                    "Đạt mức lợi nhuận mục tiêu và tối thiểu rủi ro phát sinh",
                                 ])

    target_return = None
    if investment_goal == "Đạt mức lợi nhuận mục tiêu và tối thiểu rủi ro phát sinh":
        target_return = st.number_input("Nhập mức lợi nhuận mục tiêu (%):", value=5.0, step=0.1)

    # Nút hoàn thành
    if st.button("Xác nhận"):
        st.success("Thông tin đã được xác nhận!")

    return symbols, viewdict, confidences, investment_goal, target_return, total_value

def portfolio_optimization_tool(symbols, viewdict, confidences, investment_goal, target_return, total_value):
    # Xác định khoảng thời gian lấy dữ liệu (5 năm gần nhất)
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - pd.DateOffset(years=5)).strftime('%Y-%m-%d')
    
    # Phân tích dữ liệu cổ phiếu (tạo bảng thông số tài chính)
    result_df = generate_stock_analysis(symbols, start_date, end_date, viewdict, confidences)
    
    # Tối ưu danh mục
    try:
        if investment_goal == "Đạt mức lợi nhuận mục tiêu và tối thiểu rủi ro phát sinh" and target_return is not None:
            ef, weights, allocation, leftover_cash, details_df = optimize_portfolio(
                symbols, total_value, viewdict, confidences, investment_goal, target_return=target_return / 100
            )
        else:
            ef, weights, allocation, leftover_cash, details_df = optimize_portfolio(
                symbols, total_value, viewdict, confidences, investment_goal
            )
        
        # Hiển thị kết quả qua Streamlit với các hiệu ứng trực quan
        display_results(result_df, ef, weights, allocation, leftover_cash, details_df, investment_goal)
    
    except Exception as e:
        st.error(f"Lỗi khi tối ưu danh mục: {e}")




@st.cache_data
def get_financial_data(tickers):
    stock_data = []
    # Danh sách các cột cần thiết
    required_columns = [
        ('Chỉ tiêu định giá', 'Vốn hóa (Tỷ đồng)'),
        ('Chỉ tiêu định giá', 'P/B'),
        ('Chỉ tiêu khả năng sinh lợi', 'ROE (%)'),
        ('Chỉ tiêu định giá', 'P/E'),
        ('Chỉ tiêu khả năng sinh lợi', 'ROA (%)')
    ]
    
    for ticker in tickers:
        stock = Vnstock().stock(symbol=ticker, source='VCI')
        finance_ratios = stock.finance.ratio(period='year', lang='vi', dropna=True).head(1)
        
        # Kiểm tra xem cổ phiếu có đầy đủ các cột cần thiết không
        if all(col in finance_ratios.columns for col in required_columns):
            data = {
                'CP': ticker,  # Sử dụng ticker thay vì code
                'Vốn hóa (Tỷ đồng)': finance_ratios[('Chỉ tiêu định giá', 'Vốn hóa (Tỷ đồng)')].values[0] / 1e3,  # Chuyển đổi từ tỷ đồng
                'P/B': finance_ratios[('Chỉ tiêu định giá', 'P/B')].values[0],
                'ROE': finance_ratios[('Chỉ tiêu khả năng sinh lợi', 'ROE (%)')].values[0],
                'P/E': finance_ratios[('Chỉ tiêu định giá', 'P/E')].values[0],
                'ROA': finance_ratios[('Chỉ tiêu khả năng sinh lợi', 'ROA (%)')].values[0]
            }
            stock_data.append(data)
        else:
            # In ra các cổ phiếu thiếu cột cần thiết
            missing_columns = [col for col in required_columns if col not in finance_ratios.columns]
            print(f"Cổ phiếu {ticker} thiếu các cột: {missing_columns}. Bỏ qua cổ phiếu này.")
    
    return pd.DataFrame(stock_data)




@st.cache_data
def get_same_industry_stocks(code):
    screener_df = Screener().stock(params={"exchangeName": "HOSE,HNX,UPCOM"}, limit=1700)
    fpt_industry = screener_df[screener_df['ticker'] == code]['industry'].values[0]
    return screener_df[screener_df['industry'] == fpt_industry]['ticker'].tolist()



def phan_tich_nganh(code):
    # Lọc dữ liệu cổ phiếu
    screener_df = Screener().stock(params={"exchangeName": "HOSE,HNX,UPCOM"}, limit=1700)

    #chart_type = st.radio('Chọn loại biểu đồ:', ['Treemap', 'Sunburst'])
    chart_type = 'treemap'  # Biểu đồ mặc định
    value_col = 'market_cap'  # Cột mặc định là market_cap
    
    # Sử dụng các giá trị mặc định cho chiều rộng và chiều cao
    width = 1000
    height = 600
    

    # Hiển thị biểu đồ
    fig = create_chart(screener_df, value_col, chart_type.lower(), width, height)
    st.plotly_chart(fig)
    
    same_industry_stocks = get_same_industry_stocks(code)
    df_stocks = get_financial_data(same_industry_stocks)

    # Cho phép người dùng chọn các cổ phiếu hiển thị
    selected_stocks = st.multiselect(
        'Chọn các cổ phiếu để hiển thị:',
        options=df_stocks['CP'].tolist(),
        default=df_stocks['CP'].tolist()  # Mặc định chọn tất cả các cổ phiếu
    )

    # Lọc dữ liệu theo các cổ phiếu đã chọn
    df_filtered = df_stocks[df_stocks['CP'].isin(selected_stocks)]

    # Chọn giá trị cho trục x và y
    selected_x = st.selectbox('Chọn giá trị cho trục x:', ['ROE', 'ROA'])
    selected_y = st.selectbox('Chọn giá trị cho trục y:', ['P/B', 'P/E'])

    # Vẽ biểu đồ scatter cho các cổ phiếu đã chọn
    fig_scatter = px.scatter(
        df_filtered, 
        x=selected_x, 
        y=selected_y, 
        size="Vốn hóa (Tỷ đồng)", 
        text="CP",
        color="Vốn hóa (Tỷ đồng)", 
        color_continuous_scale="Rainbow", 
        size_max=120,
        hover_name="CP", 
        hover_data={selected_x: True, selected_y: True, "Vốn hóa (Tỷ đồng)": True, "CP": False},
    
    )

    fig_scatter.update_layout(
        title=f'So sánh {selected_x} vs {selected_y} của các cổ phiếu cùng ngành',
        xaxis=dict(title=selected_x),
        yaxis=dict(title=selected_y)
    )

    # Hiển thị biểu đồ
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Hiển thị bảng dữ liệu của các cổ phiếu đã chọn
    st.dataframe(df_filtered)

    
    # Biểu đồ thống kê ngành
    sector_counts = screener_df['industry'].value_counts()
    fig_sector = px.bar(
        x=sector_counts.index, y=sector_counts.values, title='Số lượng cổ phiếu theo ngành',
        color=sector_counts.index,  # Sử dụng color cho giá trị ngànhr
        color_continuous_scale=px.colors.qualitative.Light24  # Chọn màu hợp lệ cho ngành
    )
    st.plotly_chart(fig_sector, use_container_width=True)


    

def create_chart(df, value_col, chart_type='treemap', width=1000, height=600):
    if chart_type == 'treemap':
        fig = px.treemap(df, path=['industry', 'ticker'], values=value_col)
    elif chart_type == 'sunburst':
        fig = px.sunburst(df, path=['industry', 'ticker'], values=value_col)
    fig.update_layout(width=width, height=height)
    return fig


def phan_tich_cp(code, df_stock, df_vnindex,df_insights):
    if df_stock is None or df_insights.empty:
        st.warning("Không có dữ liệu để hiển thị.")
        return
    
    df_cash_flow = get_cash_flow(code)
    df_income_statement = get_income_statement(code)
    df_balance = get_balance_sheet(code)

    screener_df = Screener().stock(params={"exchangeName": "HOSE,HNX,UPCOM"}, limit=1700)
    stock_data = screener_df[screener_df['ticker'] == code]

    
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
    t0, t1, t2, t3, t4, t5, t6= st.tabs([
       "Tổng quan","Tin tức & Sự kiện" ,  "Phân tích 360", "Phân tích kĩ thuật",
        "Tài chính","Dữ liệu","Hồ sơ"])


    with t0:
        # 🔹 Hiển thị biểu đồ chứng khoán so với VN-Index
        visualize_analysis(screener_df,code)
    
    with t1:
        # Tạo layout chia cột cho các phần tin tức, sự kiện và báo cáo
        # Tiêu đề và giới thiệu
        st.markdown("""
        <style>
            .title {
                font-size: 24px;
                font-weight: bold;
                color: #2D3748;
            }
            .section {
                margin-top: 20px;
                margin-bottom: 20px;
            }
        </style>
        """, unsafe_allow_html=True)
        col1, col2= st.columns([3,1])
                # Lấy và hiển thị báo cáo phân tích công tyr
        with col1:
   
            # Hiển thị kết quả trên Streamlit
            # Hiển thị kết quả trên Streamlit
            d1 = get_reports(code)

             # Kiểm tra nếu không có dữ liệu báo cáo
            if isinstance(d1, str) and d1 == "Chưa có báo cáo":
                st.markdown("### Báo cáo phân tích công ty")
                st.write("**Chưa có báo cáo**")

            else:
                # Chuyển các cột chứa HTML thành dạng Markdown để hiển thị
                for index, row in d1.iterrows():
                    row['Nguồn'] = f"<a href='{row['Nguồn']}' target='_blank'>{row['Nguồn']}</a>"
                    row['Tải về'] = f"<a href='{row['Tải về']}' target='_blank'>Tải về</a>"

                # Hiển thị tiêu đề
                st.markdown("<h3 style='text-align:center; color: #ffffff;'>Báo cáo phân tích công ty</h3>", unsafe_allow_html=True)

                # Định dạng bảng với CSS cho theme dark
                html_table = d1.to_html(escape=False)

                # Áp dụng CSS để tạo bảng với tông màu đen xám
                custom_css = """
                    <style>
                        table {
                            font-size: 14px; /* Chỉnh kích thước chữ */
                            width: 100%;
                            border-collapse: collapse;
                            background-color: #333; /* Nền bảng tối */
                            color: #ddd; /* Màu chữ sáng */
                        }
                        th, td {
                            padding: 10px;
                            text-align: left;
                            border-bottom: 1px solid #444; /* Viền bảng màu xám đậm */
                        }
                        th {
                            background-color: #555; /* Nền tiêu đề bảng xám đậm */
                            color: #fff; /* Chữ tiêu đề sáng */
                        }
                        tr:nth-child(even) {
                            background-color: #444; /* Nền hàng chẵn xám đậm */
                        }
                        tr:hover {
                            background-color: #666; /* Nền khi hover */
                        }
                        a {
                            color: #1E90FF; /* Màu liên kết sáng */
                            text-decoration: none;
                        }
                        a:hover {
                            text-decoration: underline; /* Hiệu ứng underline khi hover vào liên kết */
                        }
                    </style>
                """

                # Thêm CSS vào bảng HTML và hiển thị
                st.markdown(custom_css, unsafe_allow_html=True)
                st.markdown(html_table, unsafe_allow_html=True)


                    
            #st.markdown("<h3><b>Báo Cáo Phân Tích Công Ty</b></h3>", unsafe_allow_html=True)
            #reports_df = reports(code)
            #for index, row in reports_df.iterrows():
                #st.markdown(f"- {row['Tên Báo Cáo']}", unsafe_allow_html=True)
                #st.markdown(f"  *Ngày: {row['Ngày']}* - {row['Mô Tả']}")


           
        with col2:
            #st.markdown("<h3><b>Báo Cáo Phân Tích Công Ty</b></h3>", unsafe_allow_html=True)
            #reports_df = get_company_reports(code)
            #for index, row in reports_df.iterrows():
                #st.markdown(f"- {row['Tên Báo Cáo']}", unsafe_allow_html=True)
                #st.markdown(f"  *Ngày: {row['Ngày']}* - {row['Mô Tả']}")

            # Lấy và hiển thị tin tức công ty
            st.markdown("<h3><b>Tin Tức Công Ty</b></h3>", unsafe_allow_html=True)
            news_df = get_company_news(code)
            for index, row in news_df.iterrows():
                st.markdown(f"- {row['Tiêu đề Tin Tức']}", unsafe_allow_html=True)

            # Lấy và hiển thị sự kiện công ty
            st.markdown("<h3><b>Sự Kiện Công Ty</b></h3>", unsafe_allow_html=True)
            events_df = get_company_events(code)
            for index, row in events_df.iterrows():
                st.markdown(f"- {row['Tiêu đề Sự Kiện']}", unsafe_allow_html=True)


   
        
        
    
    with t2: 
        plot_stock_vs_vnindex(df_stock, df_vnindex, code)
        with st.expander("CHỉ tiêu tài chính"):
            df_ratio = get_ratios(code)
            if df_ratio is not None:
                st.subheader(f"Dữ liệu tài chính của {code}")
                st.dataframe(df_ratio)
                # Vẽ từng chỉ tiêu riêng lẻ
                plot_metric(df_ratio, code)
        


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

        trade_signal_analysis(df_stock)
        # Hiển thị ghi chú
        st.markdown("""
        🔍 **Cách xây dựng tín hiệu mua/bán**  
        Mình đang sử dụng 5 chỉ báo kỹ thuật chính để xác định tín hiệu giao dịch:
        - **RSI (Relative Strength Index)**: Xác định trạng thái quá mua/quá bán.
        - **MACD (Moving Average Convergence Divergence)**: Đánh giá động lượng xu hướng.
        - **ATR (Average True Range)**: Đo lường độ biến động giá.
        - **SMA (Simple Moving Average)**: Trung bình động đơn giản.
        - **EMA (Exponential Moving Average)**: Trung bình động hàm mũ.

        ✅ **Điều kiện kích hoạt tín hiệu**  
        📈 **Tín hiệu mua (Buy Signal)**:
        - RSI < 30 (Cổ phiếu đang bị bán quá mức).
        - MACD cắt lên Signal Line (Báo hiệu xu hướng tăng).
        - Giá cắt lên EMA 20 (Xác nhận đà tăng giá).
        - ATR tăng cao (Thị trường có biến động mạnh, tăng khả năng đảo chiều).
        👉 Nếu các điều kiện trên thỏa mãn cùng lúc, tạo tín hiệu mua.

        📉 **Tín hiệu bán (Sell Signal)**:
        - RSI > 70 (Cổ phiếu đang bị mua quá mức).
        - MACD cắt xuống Signal Line (Báo hiệu xu hướng giảm).
        - Giá cắt xuống EMA 20 (Xác nhận xu hướng giảm).
        - ATR tăng cao (Biến động lớn có thể dẫn đến đảo chiều).
        👉 Nếu các điều kiện trên thỏa mãn cùng lúc, tạo tín hiệu bán.
                    
        📈 **Kết Hợp Các Chỉ Báo – Giảm Sai Số**:
        - 👉 Không có chỉ báo nào hoàn hảo! Do đó, mình kết hợp RSI + MACD + ATR + EMA để:
        - Giảm tín hiệu nhiễu.
        - Tăng xác suất giao dịch chính xác.
        - Lọc tín hiệu giao dịch đáng tin cậy hơn.          
        """)
        

        # Giả sử df_stock là DataFrame chứa dữ liệu thị trường với các cột 'time', 'open', 'close', 'volume'
        plot_combined_charts(df_stock, sma_windows, ema_windows)
            
    with t4:
    
        
        with st.expander("Dòng tiền"):
            plot_cash_flow(df_cash_flow)

        with st.expander("Bảng Cân Đối Kế Toán"):
            plot_accounting_balance(df_balance)

        with st.expander("Kết Quả Kinh Doanh"):
            plot_business_results(df_income_statement)

  
        # Thêm một số thông báo hỗ trợ, nhắc nhở người dùng về nội dung
        st.markdown("Bạn có thể mở các phần để xem biểu đồ chi tiết hơn. Di chuyển chuột qua các phần để xem thông tin rõ hơn.")
    with t5:
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
        

        
    with t6:
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

        # Cổ đông lớn
        with st.container():
            st.markdown("<h3 style='color: #4B0082;'>Cổ Đông Lớn</h3>", unsafe_allow_html=True)
            shareholders_info = get_shareholders_info(code)
            if shareholders_info is None or shareholders_info.empty:
                st.warning("Không có thông tin về cổ đông lớn.")
            else:
                # Tạm bỏ styling để kiểm tra
                st.dataframe(shareholders_info)  # Thử mà không có highlighting

        # Công ty con
        with st.container():
            # Công ty con (chỉ hiển thị nếu có dữ liệu)
            try:
                subsidiaries_info = get_subsidiaries_info(code)

                # Kiểm tra nếu dữ liệu hợp lệ và không rỗng
                if isinstance(subsidiaries_info, pd.DataFrame) and not subsidiaries_info.empty:
                    with st.container():
                        st.markdown("<h3 style='color: #4B0082;'>Công Ty Con</h3>", unsafe_allow_html=True)
                        st.dataframe(subsidiaries_info)  # Hiển thị dữ liệu
                # Nếu không có dữ liệu, KHÔNG hiển thị gì cả (không có st.info hay st.warning)
                
            except Exception as e:
                st.error(f"Không có dữ liệu công ty con")

          
def main():
    st.set_page_config(page_title="Stock Dashboard", page_icon="📈", layout="wide")
    # Thêm CSS tùy chỉnh cho trang và sidebar
    st.markdown(
        """
        <style>
            body {
                background-color: #f0f2f5;
            }
            .header {
                text-align: center;
                background: linear-gradient(135deg, #1e1e1e, #333333);
                padding: 20px;
                border-radius: 12px;
                color: white;
            }
            .sidebar .sidebar-content {
                background-color: #4e73df;
                color: white;
            }
            .sidebar .sidebar-content .st-selectbox, .sidebar .sidebar-content .st-button {
                color: #ffffff;
            }
            .sidebar .sidebar-content .st-selectbox select {
                background-color: #007bff;
                color: white;
            }
            .stock-info {
                font-size: 18px;
                font-weight: bold;
                color: #2c3e50;
            }
            .column {
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 15px;
                margin: 10px;
                background-color: white;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            }
            .highlight {
                color: green;
            }
            .alert {
                color: red;
            }
            .metric-box {
                background-color: #ffffff;
                padding: 10px;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    logo_path = "LOGO.png"  # Điền đúng đường dẫn đến logo của bạn
    st.sidebar.image(logo_path)  # Hiển thị logo trong sidebar



     # Thêm banner/header
    banner_path = "banner.png"  # Điền đúng đường dẫn đến ảnh header của bạn
    st.image(banner_path)  # Hiển thị banner ở header
    

    symbols = get_all_symbols()

    if "FPT" in symbols:
        default_index = symbols.index("FPT")
    else:
        default_index = 0

    code = st.selectbox("Chọn mã cổ phiếu", options=symbols, index=default_index, key="stock_selector")

       
    # Xác định khoảng thời gian (5 năm gần đây)
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - pd.DateOffset(years=5)).strftime('%Y-%m-%d')

    with st.spinner("Đang tải dữ liệu cổ phiếu..."):
        df_stock = load_stock_data(code, start_date, end_date)
        df_insights = load_insights(code)
        df_vnindex = fetch_vnindex_data(start_date, end_date)
        company_df = get_company_table(code)
    
    if df_stock is not None and not df_insights.empty:
        display_general_info(df_stock, df_insights)
    

    # Sidebar
    st.sidebar.title("Trợ lý tài chính")
    options = st.sidebar.selectbox("Chức năng", 
                                ["Phân tích cổ phiếu", "Phân tích ngành", "Công cụ tối ưu danh mục đầu tư"])

    # Phân tích cổ phiếu
    if options == 'Phân tích cổ phiếu':
        if df_stock is not None and not df_insights.empty:
            phan_tich_cp(code, df_stock, df_vnindex, df_insights)
        else:
            st.warning("Không đủ dữ liệu để phân tích cổ phiếu.")

    # Phân tích ngành
    elif options == 'Phân tích ngành':
        phan_tich_nganh(code)  # Gọi hàm phân tích ngành

    # Công cụ tối ưu danh mục đầu tư
    elif options == 'Công cụ tối ưu danh mục đầu tư':
        symbols, viewdict, confidences, investment_goal, target_return, total_value = get_user_inputs()
        if st.button("Chạy tối ưu danh mục"):
            portfolio_optimization_tool(symbols, viewdict, confidences, investment_goal, target_return, total_value)
    
    st.markdown('<div class="footer">© 2025 Portfolio Dashboard. All Rights Reserved.</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
