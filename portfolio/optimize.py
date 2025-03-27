import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pypfopt import risk_models, BlackLittermanModel, EfficientFrontier, objective_functions, black_litterman
from pypfopt.black_litterman import market_implied_risk_aversion
from portfolio.utils import fetch_multiple_stock_prices, load_mkt_caps, fetch_vnindex_data
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components

# Hàm tối ưu danh mục
def optimize_portfolio(symbols, total_investment, view_dict, confidence_dict, investment_goal, source="VCI"):
    # Xác định khoảng thời gian lấy dữ liệu (5 năm)
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - pd.DateOffset(years=5)).strftime('%Y-%m-%d')
    
    close_prices = fetch_multiple_stock_prices(symbols, start_date, end_date, source=source)
    vnindex = fetch_vnindex_data(start_date, end_date, source=source)
    
    # Tính hệ số rủi ro thị trường
    delta = black_litterman.market_implied_risk_aversion(vnindex['close'])
    S = risk_models.CovarianceShrinkage(close_prices).ledoit_wolf()
    
    # Lấy dữ liệu vốn hóa thị trường
    market_caps_df = load_mkt_caps(symbols)
    mcaps = pd.Series(market_caps_df.market_cap.values, index=market_caps_df.ticker).to_dict()
    market_prior = black_litterman.market_implied_prior_returns(mcaps, delta, S)
    
    # Tạo mô hình Black-Litterman
    bl = BlackLittermanModel(S, pi=market_prior, absolute_views=view_dict, 
                             omega="idzorek", view_confidences=list(confidence_dict.values()))
    ret_bl = bl.bl_returns()
    S_bl = bl.bl_cov()
    
    weights = None
    if investment_goal == "Tối đa hoá tỷ lệ Sharpe":
        ef = EfficientFrontier(ret_bl, S_bl)
        ef.add_objective(objective_functions.L2_reg)
        weights = ef.max_sharpe()
    elif investment_goal == "Đạt mức lợi nhuận mục tiêu và tối thiểu rủi ro phát sinh":
        ef = EfficientFrontier(ret_bl, S_bl)
        # Ở giao diện Streamlit, bạn có thể thu thập target_return từ người dùng
        target_return = float(input("Nhập mức lợi nhuận mục tiêu (%): ")) / 100
        ef.efficient_return(target_return)
    elif investment_goal == "KIỂM SOÁT CVaR VÀ TỐI ĐA LỢI NHUẬN KỲ VỌNG":
        from pypfopt.efficient_cvar import EfficientCVaR
        ef_cvar = EfficientCVaR(expected_returns=ret_bl, returns=close_prices.pct_change().dropna())
        weights = ef_cvar.min_cvar()
        weights = dict(zip(close_prices.columns, weights))
        weights = {k: v for k, v in weights.items() if not np.isnan(v)}
        ef = None
    elif investment_goal == "KIỂM SOÁT CDaR VÀ TỐI ĐA LỢI NHUẬN KỲ VỌNG":
        from pypfopt.efficient_cdar import EfficientCDaR
        ef_cdar = EfficientCDaR(expected_returns=ret_bl, returns=close_prices.pct_change().dropna())
        weights = ef_cdar.min_cdar()
        weights = dict(zip(close_prices.columns, weights))
        weights = {k: v for k, v in weights.items() if not np.isnan(v)}
        ef = None
    else:
        raise ValueError("Mục tiêu đầu tư không hợp lệ!")
    
    # Phân bổ số lượng cổ phiếu
    allocation = {
        k: round((weights[k] * total_investment) / (close_prices.iloc[-1][k] * 1000), 2)
        for k in weights.keys()
    }
    allocation = {k: round(allocation[k] / 100) * 100 for k in allocation.keys()}
    leftover_cash = total_investment - sum(allocation[k] * close_prices.iloc[-1][k] * 1000 for k in allocation)
    
    details_df = pd.DataFrame({
        "Mã cổ phiếu": list(weights.keys()),
        "Số lượng cổ phiếu": [allocation[k] for k in weights.keys()],
        "Tỷ suất sinh lời kỳ vọng đã điều chỉnh": bl.bl_returns().values,
        "Rủi ro đã điều chỉnh": np.diag(bl.bl_cov()),
        "Rủi ro ban đầu": np.sqrt(np.diag(S))
    })
    
    return ef, weights, allocation, leftover_cash, details_df


def display_results(result_df, ef, weights, allocation, leftover_cash, details_df, investment_goal):
    """
    Trình bày kết quả tối ưu danh mục với giao diện 'dark mode' tương tự ảnh tham khảo.
    """

    # Hiển thị bảng kết quả phân tích từng cổ phiếu  ( đầu tiên))
    #st.markdown("### Quan điểm đầu tư & số liệu tham khảo")
    #st.dataframe(result_df)
    # Tạo CSS dark theme
    st.markdown("""
    <style>
    .dark-table {
        width: 100%;
        border-collapse: collapse;
        background-color: #1E1E1E;
        color: #FFF;
        text-align: left;
    }
    .dark-table thead th {
        background-color: #333;
        padding: 8px;
    }
    .dark-table tbody td {
        padding: 8px;
        border: 1px solid #2A2A2A;
    }
    .dark-table tbody tr:nth-child(even) {
        background-color: #2A2A2A;
    }
    </style>
    """, unsafe_allow_html=True)

    # Dựng bảng HTML từ DataFrame
    html_table = "<table class='dark-table'>"
    # Tạo header
    html_table += "<thead><tr><th></th>"  # cột đầu cho index
    for col in result_df.columns:
        html_table += f"<th>{col}</th>"
    html_table += "</tr></thead>"

    # Tạo body
    html_table += "<tbody>"
    for idx, row in result_df.iterrows():
        html_table += f"<tr><td>{idx}</td>"
        for col in result_df.columns:
            val = row[col]
            if isinstance(val, float):
                cell_value = f"{val:.4f}"
            else:
                cell_value = str(val)
            html_table += f"<td>{cell_value}</td>"
        html_table += "</tr>"
    html_table += "</tbody></table>"

    st.markdown("<h3 style='color: black;'>Quan điểm đầu tư & số liệu tham khảo</h3>", unsafe_allow_html=True)

    # Hiển thị bảng HTML
    st.markdown(html_table, unsafe_allow_html=True)
    

    # Phần tính toán lấy hiệu suất từ EfficientFrontier (nếu có)
    expected_return = 0
    volatility = 0
    sharpe_ratio = 0
    
    if ef is not None:
        expected_return, volatility, sharpe_ratio = ef.portfolio_performance()
    
    # Tạo phần tiêu đề
    st.markdown("<h3 style='color: black;'>Danh mục phân bổ tối ưu</h3>", unsafe_allow_html=True)

    
    # Bố trí 3 cột để hiển thị các chỉ số chính
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Tỷ suất sinh lời kỳ vọng", value=f"{expected_return:.2%}")
    col2.metric(label="Rủi ro - Mức độ biến động", value=f"{volatility:.2%}")
    col3.metric(label="Sharpe ratio", value=f"{sharpe_ratio:.2f}")
    
    # Biểu đồ Donut thể hiện phân bổ danh mục
    fig = px.pie(
        names=list(weights.keys()),
        values=list(weights.values()),
        hole=0.4,  # Tạo "lỗ" giữa để thành donut
        title="Phân bổ danh mục đầu tư",
        template="plotly_dark"  # Nền tối
    )
    # Tuỳ biến hiển thị
    fig.update_traces(
        textposition='inside', 
        textinfo='percent+label', 
        hovertemplate='%{label}: %{percent:.2%}'
    )
    fig.update_layout(
        showlegend=True,
        legend_title_text="", 
        margin=dict(l=50, r=50, t=60, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Giải thích ý nghĩa của các chỉ số
    st.markdown("### Ý nghĩa:")
    st.markdown(
        f"- Bạn có thể kỳ vọng danh mục này đem lại lợi nhuận bình quân **{expected_return:.2%}/năm**."
    )
    st.markdown(
        f"- Với mức độ biến động (rủi ro) **{volatility:.2%}**, "
        f"lợi nhuận của danh mục này trong 1 năm có thể biến động trong khoảng "
        f"**{(expected_return - volatility):.2%}** đến **{(expected_return + volatility):.2%}**."
    )
    
    
    # Hiển thị chi tiết danh mục (cuối cùng)

    st.markdown("<h3 style='color: black;'>Chi Tiết Danh Mục</h3>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="font-size:22px; font-weight:bold; margin-bottom:5px; color:white;">
            Chi tiết danh mục
        </div>
        <div style="font-size:14px; color:#aaa; margin-bottom:10px;">
            Mục tiêu phân bổ vốn: <b>{investment_goal}</b>
            <span style="float:right;">
                Tiền mặt còn lại <b>{leftover_cash:,.0f} VNĐ</b>
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )
   
    #st.dataframe(details_df)
    st.markdown("""
    <style>
    .dark-table {
        width: 100%;
        border-collapse: collapse;
        background-color: #1E1E1E;
        color: #FFF;
        text-align: left;
    }
    .dark-table thead th {
        background-color: #333;
        padding: 8px;
    }
    .dark-table tbody td {
        padding: 8px;
        border: 1px solid #2A2A2A;
    }
    .dark-table tbody tr:nth-child(even) {
        background-color: #2A2A2A;
    }
    </style>
    """, unsafe_allow_html=True)

    # Dựng bảng HTML từ DataFrame
    html_table = "<table class='dark-table'>"
    # Tạo header
    html_table += "<thead><tr><th></th>"  # cột đầu cho index
    for col in details_df.columns:
        html_table += f"<th>{col}</th>"
    html_table += "</tr></thead>"

    # Tạo body
    html_table += "<tbody>"
    for idx, row in details_df.iterrows():
        html_table += f"<tr><td>{idx}</td>"
        for col in details_df.columns:
            val = row[col]
            if isinstance(val, float):
                cell_value = f"{val:.4f}"
            else:
                cell_value = str(val)
            html_table += f"<td>{cell_value}</td>"
        html_table += "</tr>"
    html_table += "</tbody></table>"

    # Hiển thị bảng HTML
    st.markdown(html_table, unsafe_allow_html=True)
    
    
    
    
    # Tải xuống CSV
    csv_data = details_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Tải danh mục tối ưu (CSV)",
        data=csv_data,
        file_name='portfolio_details.csv',
        mime='text/csv'
    )



    

