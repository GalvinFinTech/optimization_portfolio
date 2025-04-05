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
import pandas as pd
import numpy as np
from datetime import datetime
from pypfopt import EfficientFrontier, EfficientCVaR, EfficientCDaR, risk_models, expected_returns, black_litterman, objective_functions
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

import pandas as pd
import numpy as np
from datetime import datetime
from pypfopt import EfficientFrontier, EfficientCVaR, EfficientCDaR, risk_models, black_litterman, objective_functions
from pypfopt.discrete_allocation import DiscreteAllocation

import pandas as pd
import numpy as np
from datetime import datetime
from pypfopt import EfficientFrontier, EfficientCVaR, EfficientCDaR, risk_models, black_litterman, objective_functions
from pypfopt.discrete_allocation import DiscreteAllocation

import pandas as pd
import numpy as np
from datetime import datetime
from pypfopt import EfficientFrontier, risk_models, black_litterman, objective_functions
from pypfopt.discrete_allocation import DiscreteAllocation

def optimize_portfolio(symbols, total_investment, view_dict, confidence_dict, investment_goal, target_return=None, source="VCI"):
    # Xác định khoảng thời gian lấy dữ liệu (5 năm)
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - pd.DateOffset(years=5)).strftime('%Y-%m-%d')
    
    # Tải dữ liệu giá cổ phiếu và VN-Index
    close_prices = fetch_multiple_stock_prices(symbols, start_date, end_date, source=source)
    if close_prices.isnull().values.any():
        st.error("Dữ liệu giá cổ phiếu chứa giá trị NaN, hãy kiểm tra và làm sạch dữ liệu trước khi tối ưu hóa.")
        return None
    vnindex = fetch_vnindex_data(start_date, end_date, source=source)
    
    if close_prices.empty or vnindex.empty:
        raise ValueError("Không thể tải dữ liệu giá cổ phiếu hoặc VN-Index.")
    
    # Kiểm tra dữ liệu giá có hợp lệ không
    if close_prices.isna().any().any():
        raise ValueError("Dữ liệu giá cổ phiếu chứa giá trị NaN, không thể tiếp tục tối ưu hóa.")
    
    # Tính hệ số rủi ro thị trường
    delta = black_litterman.market_implied_risk_aversion(vnindex['close'])
    S = risk_models.CovarianceShrinkage(close_prices).ledoit_wolf()
    
    # Lấy dữ liệu vốn hóa thị trường
    market_caps_df = load_mkt_caps(symbols)
    mcaps = pd.Series(market_caps_df.market_cap.values, index=market_caps_df.ticker).to_dict()
    market_prior = black_litterman.market_implied_prior_returns(mcaps, delta, S)
    
    # Tạo mô hình Black-Litterman
    bl = black_litterman.BlackLittermanModel(
        S, 
        pi=market_prior, 
        absolute_views=view_dict, 
        omega="idzorek", 
        view_confidences=list(confidence_dict.values())
    )
    ret_bl = bl.bl_returns()
    S_bl = bl.bl_cov()
    
    # Tính mức lợi nhuận tối đa có thể đạt được
    max_possible_return = ret_bl.max()
    
    # Tối ưu hóa danh mục dựa trên mục tiêu đầu tư
    weights = None
    ef = None
    if investment_goal == "Tối đa hoá tỷ lệ Sharpe":
        ef = EfficientFrontier(ret_bl, S_bl, weight_bounds=(0, 1))
        ef.add_objective(objective_functions.L2_reg)
        ef.max_sharpe()
        weights = ef.clean_weights()
    
    elif investment_goal == "Đạt mức lợi nhuận mục tiêu và tối thiểu rủi ro phát sinh":
        if target_return is None:
            raise ValueError("Mục tiêu đầu tư yêu cầu mức lợi nhuận mục tiêu (target_return).")
        if target_return > max_possible_return:
            raise ValueError(
                f"Mức lợi nhuận mục tiêu ({target_return*100:.2f}%) vượt quá mức lợi nhuận tối đa có thể đạt được ({max_possible_return*100:.2f}%). "
                "Vui lòng giảm mức lợi nhuận mục tiêu."
            )
        ef = EfficientFrontier(ret_bl, S_bl, weight_bounds=(0, 1))
        ef.efficient_return(target_return)
        weights = ef.clean_weights()

    
    else:
        raise ValueError("Mục tiêu đầu tư không hợp lệ!")
    
    # Kiểm tra weights có hợp lệ không
    if not weights or all(w == 0 for w in weights.values()):
        raise ValueError("Không thể tối ưu hóa danh mục: weights không hợp lệ (có thể do dữ liệu không đủ hoặc không hợp lệ).")
    
    # Phân bổ số lượng cổ phiếu
    latest_prices = close_prices.iloc[-1]
    if latest_prices.isna().any():
        raise ValueError("Giá mới nhất (latest_prices) chứa giá trị NaN, không thể phân bổ cổ phiếu.")
    
    try:
        da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=total_investment)
        allocation, leftover_cash = da.greedy_portfolio()
    except Exception as e:
        raise ValueError(f"Lỗi khi phân bổ cổ phiếu: {str(e)}")
    
    # Tạo DataFrame chi tiết danh mục
    details_data = []
    for symbol in symbols:
        details_data.append({
            "Mã cổ phiếu": symbol,
            "Tỷ trọng (%)": weights.get(symbol, 0) * 100,
            "Số lượng cổ phiếu": allocation.get(symbol, 0),
            "Giá mua(VNĐ)": latest_prices[symbol] * 1000,
            "Tỷ suất sinh lời kỳ vọng đã điều chỉnh": ret_bl[symbol] * 100 if symbol in ret_bl else 0,
            "Rủi ro đã điều chỉnh": np.sqrt(S_bl.loc[symbol, symbol]) if symbol in S_bl.index else 0,
            "Rủi ro ban đầu": np.sqrt(S.loc[symbol, symbol]) if symbol in S.index else 0
        })
    details_df = pd.DataFrame(details_data)
    
    return ef, weights, allocation, leftover_cash, details_df


import streamlit as st
import plotly.express as px

def display_results(result_df, ef, weights, allocation, leftover_cash, details_df, investment_goal):
    """
    Trình bày kết quả tối ưu danh mục với giao diện 'dark mode'.
    """

    # Tạo CSS dark theme
    st.markdown("""
    <style>
    .dark-table {
        width: 100%;
        border-collapse: collapse;
        background-color: #1e1e1e;
        color: #fff;
        text-align: left;
    }
    .dark-table thead th {
        background-color: #333;
        padding: 8px;
    }
    .dark-table tbody td {
        padding: 8px;
        border: 1px solid #2a2a2a;
    }
    .dark-table tbody tr:nth-child(even) {
        background-color: #2a2a2a;
    }
    .header {
        font-size: 24px;
        font-weight: bold;
        color: #ffffff;
        margin: 20px 0;
    }
    .subheader {
        font-size: 18px;
        color: #ffffff;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Hiển thị tiêu đề chính
    st.markdown("<div class='header'>Kết quả Tối ưu Danh mục Đầu tư</div>", unsafe_allow_html=True)

    # Đẩy bảng kết quả ra màn hình
    html_table = "<table class='dark-table'>"
    # Tạo header
    html_table += "<thead><tr><th></th>"
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

    st.markdown("### Quan điểm đầu tư & Số liệu tham khảo", unsafe_allow_html=True)
    st.markdown(html_table, unsafe_allow_html=True)

    # Phần tính toán lấy hiệu suất từ efficient frontier (nếu có)
    expected_return = 0
    volatility = 0
    sharpe_ratio = 0

    if ef is not None:
        expected_return, volatility, sharpe_ratio = ef.portfolio_performance()

    # Tiêu đề cho danh mục phân bổ tối ưu
    st.markdown("### Danh mục phân bổ tối ưu:", unsafe_allow_html=True)

    # Bố trí 3 cột để hiển thị các chỉ số chính
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Tỷ suất sinh lời kỳ vọng", value=f"{expected_return:.2%}")
    col2.metric(label="Rủi ro - Mức độ biến động", value=f"{volatility:.2%}")
    col3.metric(label="Sharpe Ratio", value=f"{sharpe_ratio:.2f}")

    # Biểu đồ donut thể hiện phân bổ danh mục
    fig = px.pie(
        names=list(weights.keys()),
        values=list(weights.values()),
        hole=0.2,  # Tạo "lỗ" giữa để thành donut
        title="Phân bổ danh mục đầu tư",
        template="plotly_dark"  # Nền tối
    )

    # Tùy biến hiển thị
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
    st.markdown("### Ý nghĩa:", unsafe_allow_html=True)
    st.markdown(
        f"- Bạn có thể kỳ vọng danh mục này đem lại lợi nhuận bình quân **{expected_return:.2%}/năm**."
    )
    st.markdown(
        f"- Với mức độ biến động (rủi ro) **{volatility:.2%}**, "
        f"lợi nhuận của danh mục này trong 1 năm có thể biến động trong khoảng "
        f"**{(expected_return - volatility):.2%}** đến **{(expected_return + volatility):.2%}**."
    )

    # Hiển thị chi tiết danh mục (cuối cùng)
    st.markdown("<div class='header'>Chi tiết danh mục</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="font-size:16px; margin-bottom:5px; color:white;">
            Mục tiêu phân bổ vốn: <b>{investment_goal}</b>
            <span style="float:right;">
                Tiền mặt còn lại <b>{leftover_cash:,.0f} VNĐ</b>
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Dựng bảng chi tiết danh mục
    html_table_details = "<table class='dark-table'>"
    # Tạo header
    html_table_details += "<thead><tr><th></th>"
    for col in details_df.columns:
        html_table_details += f"<th>{col}</th>"
    html_table_details += "</tr></thead>"

    # Tạo body
    html_table_details += "<tbody>"
    for idx, row in details_df.iterrows():
        html_table_details += f"<tr><td>{idx}</td>"
        for col in details_df.columns:
            val = row[col]
            if isinstance(val, float):
                cell_value = f"{val:.4f}"
            else:
                cell_value = str(val)
            html_table_details += f"<td>{cell_value}</td>"
        html_table_details += "</tr>"
    html_table_details += "</tbody></table>"

    # Hiển thị bảng chi tiết
    st.markdown(html_table_details, unsafe_allow_html=True)

    # Tải xuống CSV
    csv_data = details_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Tải danh mục tối ưu (CSV)",
        data=csv_data,
        file_name='portfolio_details.csv',
        mime='text/csv'
    )



    

