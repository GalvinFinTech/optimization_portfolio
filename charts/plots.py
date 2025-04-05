# charts/plots.py
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import streamlit as st
import inspect
import numpy as np
from plotly.subplots import make_subplots

import plotly.graph_objects as go
import streamlit as st

import plotly.graph_objects as go
import streamlit as st

def plot_metric(df, stock_code):
    """
    Hiển thị biểu đồ cực xịn với giao diện mượt mà, đẹp mắt, hiệu ứng hover cao cấp và thêm HTML để làm nổi bật thông tin.
    """
    if df is None or df.empty:
        st.warning(f"Không có dữ liệu để vẽ biểu đồ cho {stock_code}.")
        return
    
    # Lọc các cột số để vẽ biểu đồ (trừ 'Năm' và 'CP')
    numeric_cols = [col for col in df.columns if col not in ['Năm', 'CP','Kỳ']]

    # Chọn chỉ tiêu cần vẽ
    selected_metrics = st.multiselect("🎯 Chọn các chỉ tiêu để vẽ:", options=numeric_cols, default=numeric_cols[:3])

    for metric in selected_metrics:
        # HTML tiêu đề mô tả
        html_title = f"<h3 style='color: #00C9FF; text-align: center;'>📊 {metric} của {stock_code}</h3>"
        html_subtitle = f"<p style='color: #666; text-align: center;'>Biểu đồ này thể hiện {metric} qua các năm.</p>"

        # Hiển thị HTML tiêu đề và mô tả
        st.markdown(html_title, unsafe_allow_html=True)
        st.markdown(html_subtitle, unsafe_allow_html=True)

        fig = go.Figure()

        # Tạo màu sắc gradient tươi sáng hơn (sử dụng dải màu đỏ - xanh)
        color_scale = ["#FF6347", "#32CD32"]  # Gradient từ đỏ đến xanh lá

        # Vẽ đường chính với hiệu ứng gradient
        fig.add_trace(go.Scatter(
            x=df["Năm"], 
            y=df[metric], 
            mode='lines+markers', 
            name=metric,
            line=dict(width=4, shape='spline', color=color_scale[0]),  # Đường cong mượt mà
            marker=dict(size=10, symbol='circle', line=dict(width=2, color=color_scale[1])),
            hoverinfo='text',  # Hiển thị thông tin theo dạng custom
            hovertemplate=(
                f"<b>{metric}</b><br>"  # In tên chỉ tiêu
                "Năm: %{x}<br>"  # Hiển thị năm
                "Giá trị: %{y:.2f}<br>"  # Hiển thị giá trị với 2 chữ số thập phân
                "<extra></extra>"  # Loại bỏ phần "trace 1:" và những thông tin không cần thiết
            )
        ))

        # Cấu hình layout cho giao diện đẹp
        fig.update_layout(
            title=f"📊 {metric} của {stock_code}",
            xaxis_title="Năm",
            yaxis_title="Giá trị",
            hovermode="x unified",  # Hiển thị giá trị tại các điểm trên cùng một trục x
            margin=dict(l=40, r=40, t=60, b=40),
            plot_bgcolor="rgba(0,0,0,0)",  # Nền trong suốt
            paper_bgcolor="rgba(0,0,0,0)",  # Nền trong suốt
            font=dict(family="Arial, sans-serif", size=14),
            xaxis=dict(showgrid=True, gridwidth=0.5, gridcolor="gray"),
            yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor="gray"),
        )

        # Hiệu ứng bóng mờ cho đường line và tạo động tác mượt mà
        fig.update_traces(line=dict(width=4, dash="solid", shape="spline"))

        # Cập nhật màu sắc background và trục
        fig.update_layout(
            xaxis=dict(showline=True, linewidth=2, linecolor='white', tickangle=45),
            yaxis=dict(showline=True, linewidth=2, linecolor='white'),
        )

        # Hiển thị biểu đồ
        st.plotly_chart(fig, use_container_width=True)






def plot_stock_vs_vnindex(df_stock, df_vnindex, code):
    """
    Vẽ biểu đồ biến động giá cổ phiếu so với VN-Index.

    Tham số:
    - df_stock (pd.DataFrame): Dữ liệu giá cổ phiếu (có cột 'time', 'close').
    - df_vnindex (pd.DataFrame): Dữ liệu VN-Index (có cột 'time', 'close').
    - code (str): Mã cổ phiếu cần so sánh.

    Trả về:
    - Biểu đồ Plotly hiển thị trên Streamlit.
    """

    # 🔹 Chỉ lấy cột 'time' và 'close', đổi tên để tránh trùng lặp
    df_stock_close = df_stock[['time', 'close']].rename(columns={'close': f'{code}_close'})
    df_vnindex_close = df_vnindex[['time', 'close']].rename(columns={'close': 'VN-Index_close'})

    # Merge dữ liệu theo 'time'
    df_combined = df_stock_close.merge(df_vnindex_close, on='time', how='inner')

    # 🔹 Đặt lại index theo 'time'
    df_combined.set_index('time', inplace=True)

    # 🔹 Tính % thay đổi so với ngày đầu tiên
    df_combined = (df_combined / df_combined.iloc[0] - 1) * 100

    # 🔹 Vẽ biểu đồ bằng Plotly
    fig = go.Figure()
    function_name = inspect.currentframe().f_code.co_name  # Lấy tên hàm

    fig.add_trace(go.Scatter(
        x=df_combined.index,
        y=df_combined[f'{code}_close'],
        mode='lines',
        name=f"Cổ phiếu {code}",
        line=dict(color='orange', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=df_combined.index,
        y=df_combined['VN-Index_close'],
        mode='lines',
        name="VN-Index",
        line=dict(color='blue', width=2)
    ))

    fig.update_layout(
        title=f"Biến động giá {code} so với VN-Index",
        xaxis_title="Thời gian",
        yaxis_title="Thay đổi (%)",
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(x=0, y=1)
    )

    st.plotly_chart(fig, use_container_width=True, key=f"chart_{function_name}")


def plot_price_volume(df):
    # Kiểm tra và chuyển đổi cột 'Date' sang dạng datetime nếu chưa đúng định dạng
    if not pd.api.types.is_datetime64_any_dtype(df['time']):
        df['time'] = pd.to_datetime(df['time'])

    # Loại bỏ giá trị NaN trong 'close' và 'volume'
    df = df.dropna(subset=['close', 'volume'])

    # Tạo figure
    fig = go.Figure()

    # Vẽ đường giá đóng cửa (Close Price)
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['close'],
        mode='lines',
        name='giá đóng cửa',
        line=dict(color='blue', width=2)  # Tùy chỉnh màu sắc và độ rộng đường
    ))

    # Xác định màu sắc cho biểu đồ cột (Volume)
    colors = ['red' if df['close'].iloc[i] > df['close'].iloc[i - 1] else 'green' for i in range(1, len(df))]
    colors.insert(0, 'green')  # Mặc định màu xanh cho phiên đầu tiên

    # Vẽ biểu đồ khối lượng giao dịch (Volume)
    fig.add_trace(go.Bar(
        x=df['time'], y=df['volume'],
        name='khối lượng giao dịch',
        yaxis='y2',
        marker=dict(color=colors),
        hovertemplate='%{y}k'
    ))

    # Cấu hình trục và layout
    fig.update_layout(
        #title="Stock Price & Volume",
        #xaxis_title="Date",
        #yaxis=dict(title="Close Price"),
        yaxis2=dict(overlaying="y", side="right"),
        hovermode="x unified",
        showlegend=True
    )

    # Thêm thanh trượt thời gian và nút chọn khoảng thời gian
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
    function_name = inspect.currentframe().f_code.co_name  # Lấy tên hàm
    # Hiển thị biểu đồ trên Streamlit
    st.plotly_chart(fig, use_container_width=True,key=f"chart_{function_name}")


def plot_rsi_chart(data):
    delta = data['close'].diff()
    up = delta.where(delta > 0, 0)
    down = -delta.where(delta < 0, 0)

    avg_gain = up.rolling(window=14, min_periods=1).mean()
    avg_loss = down.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / avg_loss
    data["RSI"] = 100 - (100 / (1 + rs))

    
    fig = go.Figure()
    # Đường RSI
    fig.add_trace(go.Scatter(
        x=data.index, y=data["RSI"],
        mode="lines", name="RSI",
        line=dict(color="purple", width=1)
    ))

    # Overbought & Oversold
    fig.add_trace(go.Scatter(
        x=data.index, y=[70] * len(data),
        mode="lines", name="Overbought (70)",
        line=dict(color="red", width=1, dash="dash")
    ))
    fig.add_trace(go.Scatter(
        x=data.index, y=[30] * len(data),
        mode="lines", name="Oversold (30)",
        line=dict(color="blue", width=1, dash="dash")
    ))
  
    fig.update_layout(
        hovermode='x unified',
        template="plotly_dark"
    )
    fig.update_xaxes(rangeslider_visible=False, rangeselector=dict(
        buttons=[
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=3, label="3m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(count=5, label="5y", step="year", stepmode="backward"),
            dict(step="all")
        ]
    ))
    # Hiển thị biểu đồ trên Streamlit
    function_name = inspect.currentframe().f_code.co_name  # Lấy tên hàm
    # Hiển thị biểu đồ trên Streamlit
    st.plotly_chart(fig, use_container_width=True,key=f"chart_{function_name}")

def plot_macd_chart(data):
    data['ema_12'] = data['close'].ewm(span=12, adjust=False).mean()
    data['ema_26'] = data['close'].ewm(span=26, adjust=False).mean()
    data['macd'] = data['ema_12'] - data['ema_26']
    data['signal'] = data['macd'].ewm(span=9, adjust=False).mean()
    data['histogram'] = data['macd'] - data['signal']
    
    fig = go.Figure()

    # Đường MACD
    fig.add_trace(go.Scatter(
        x=data.index, y=data["macd"],
        mode="lines", name="MACD",
        line=dict(color="blue", width=1.5)
    ))

    # Đường Signal
    fig.add_trace(go.Scatter(
        x=data.index, y=data["signal"],
        mode="lines", name="Signal",
        line=dict(color="orange", width=1.5)
    ))

    # Histogram
    fig.add_trace(go.Bar(
        x=data.index, y=data["histogram"],
        name="Histogram",
        marker=dict(
            color=['green' if v > 0 else 'red' for v in data["histogram"]],
        )
    ))
    
    fig.update_layout(
        hovermode='x unified',
        template="plotly_dark"
    )
    fig.update_xaxes(rangeslider_visible=False, rangeselector=dict(
        buttons=[
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=3, label="3m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(count=5, label="5y", step="year", stepmode="backward"),
            dict(step="all")
        ]
    ))
    # Hiển thị biểu đồ trên Streamlit
    function_name = inspect.currentframe().f_code.co_name  # Lấy tên hàm
    # Hiển thị biểu đồ trên Streamlit
    st.plotly_chart(fig, use_container_width=True,key=f"chart_{function_name}")


def plot_combined_charts(df, sma_windows, ema_windows):
    """
    Vẽ biểu đồ kết hợp gồm 3 hàng:
      - Hàng 1: Giá đóng cửa với Bollinger Bands, SMA, EMA và Volume.
      - Hàng 2: MACD chart.
      - Hàng 3: RSI chart.
      
    Chiều cao: row1 chiếm 60%, row2 và row3 chiếm 20% mỗi.
    
    Yêu cầu DataFrame df có các cột:
      - time: thời gian (datetime)
      - open: giá mở cửa (dùng để xác định màu Volume)
      - close: giá đóng cửa
      - volume: khối lượng giao dịch
    Parameters:
      - sma_windows: danh sách chu kỳ SMA (dạng string, ví dụ: ['10', '20'])
      - ema_windows: danh sách chu kỳ EMA (dạng string, ví dụ: ['10', '20'])
    """
    # Chuyển đổi cột time sang datetime nếu cần và sắp xếp theo time
    df = df.sort_values(by='time').copy()
    if not np.issubdtype(df['time'].dtype, np.datetime64):
        df['time'] = pd.to_datetime(df['time'])
    
    # ---------------------------
    # Tính toán các chỉ báo:
    # --- Bollinger Bands (sử dụng cửa sổ 20, độ lệch chuẩn 2)
    bb_window = 20
    bb_std = 2
    df['BB_Middle'] = df['close'].rolling(window=bb_window, min_periods=1).mean()
    df['BB_std'] = df['close'].rolling(window=bb_window, min_periods=1).std()
    df['BB_Upper'] = df['BB_Middle'] + bb_std * df['BB_std']
    df['BB_Lower'] = df['BB_Middle'] - bb_std * df['BB_std']
    
    # --- MACD (EMA12, EMA26, Signal EMA9)
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Histogram'] = df['MACD'] - df['Signal']
    
    # --- RSI (chu kỳ 14)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # ---------------------------
    # Tạo figure với 3 hàng, row_heights: row1=60%, row2=20%, row3=20%
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        specs=[[{"secondary_y": True}],
               [{}],
               [{}]]
    )
    
    # --- HÀNG 1: Giá và chỉ báo, cùng Volume trên trục phụ ---
    # Giá đóng cửa
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['close'],
        mode="lines", name="Giá đóng cửa",
        line=dict(color="blue", width=2)
    ), row=1, col=1, secondary_y=False)
    
    # SMA theo chu kỳ người dùng chọn
    for window in sma_windows:
        window_int = int(window)
        sma = df['close'].rolling(window=window_int, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=df['time'], y=sma,
            mode="lines", name=f"SMA {window}",
            line=dict(width=1.5, dash="dot")
        ), row=1, col=1, secondary_y=False)
    
    # EMA theo chu kỳ người dùng chọn
    for window in ema_windows:
        window_int = int(window)
        ema = df['close'].ewm(span=window_int, adjust=False).mean()
        fig.add_trace(go.Scatter(
            x=df['time'], y=ema,
            mode="lines", name=f"EMA {window}",
            line=dict(width=1.5, dash="dot")
        ), row=1, col=1, secondary_y=False)
    
    # Bollinger Bands (Upper & Lower, đường liên nét)
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['BB_Upper'],
        mode="lines", name="Bollinger Upper",
        line=dict(color="red", width=1, dash="solid")
    ), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['BB_Lower'],
        mode="lines", name="Bollinger Lower",
        line=dict(color="green", width=1, dash="solid")
    ), row=1, col=1, secondary_y=False)
    
    # Volume: xác định màu cho Volume (nếu close >= open -> xanh, ngược lại -> đỏ)
    vol_colors = ['green' if row['close'] >= row['open'] else 'red' for _, row in df.iterrows()]
    fig.add_trace(go.Bar(
        x=df['time'], y=df['volume'],
        name="Volume",
        marker=dict(color=vol_colors),
        opacity=0.5
    ), row=1, col=1, secondary_y=True)
    
    # --- HÀNG 2: MACD ---
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['MACD'],
        mode="lines", name="MACD",
        line=dict(color="white", width=1)
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['Signal'],
        mode="lines", name="Signal",
        line=dict(color="orange", width=1)
    ), row=2, col=1)
    hist_colors = ['green' if val >= 0 else 'red' for val in df['Histogram']]
    fig.add_trace(go.Bar(
        x=df['time'], y=df['Histogram'],
        name="Histogram",
        marker=dict(color=hist_colors),
        opacity=0.5
    ), row=2, col=1)
    
    # --- HÀNG 3: RSI ---
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['RSI'],
        mode="lines", name="RSI",
        line=dict(color="cyan", width=1.5)
    ), row=3, col=1)
    # Thêm đường tham chiếu 70 và 30 cho RSI
    fig.add_trace(go.Scatter(
        x=df['time'], y=[70]*len(df),
        mode="lines", name="Overbought (70)",
        line=dict(color="red", width=1, dash="dash")
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df['time'], y=[30]*len(df),
        mode="lines", name="Oversold (30)",
        line=dict(color="green", width=1, dash="dash")
    ), row=3, col=1)
    
    # --- Cấu hình layout tổng thể ---
    fig.update_layout(
        template="plotly_dark",
        title="Biểu đồ kết hợp: Giá, MACD và RSI",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1)
    )
    
    # Cấu hình trục x (chỉ hiển thị một rangeslider và rangeselector dùng chung cho tất cả)
    fig.update_xaxes(
        title_text="Date",
        rangeslider=dict(visible=False),
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
    
    # Cấu hình trục y cho từng hàng
    fig.update_yaxes(title_text="Price", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Volume", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    
    st.plotly_chart(fig, use_container_width=True, key=f"chart_{inspect.currentframe().f_code.co_name}")


# Biểu đồ cân đối kế toán
def plot_accounting_balance(df):
    fig = go.Figure()

    fig.add_trace(go.Bar(x=df.index, y=df['TỔNG CỘNG TÀI SẢN (đồng)'], name='Tổng tài sản', marker_color=px.colors.qualitative.Plotly[6]))
    fig.add_trace(go.Bar(x=df.index, y=df['VỐN CHỦ SỞ HỮU (đồng)'], name='Vốn chủ sở hữu', marker_color=px.colors.qualitative.Plotly[2]))
    fig.add_trace(go.Scatter(x=df.index, y=df['NỢ PHẢI TRẢ (đồng)'] / df['TỔNG CỘNG TÀI SẢN (đồng)'],
                             mode='lines+markers', name='Tỉ lệ nợ', yaxis='y2', marker_color=px.colors.qualitative.Plotly[9]))
    fig.update_layout(title='Cân đối kế toán', barmode='group', yaxis2=dict(overlaying='y', side='right'))
    # Hiển thị biểu đồ trên Streamlit
    function_name = inspect.currentframe().f_code.co_name  # Lấy tên hàm
    # Hiển thị biểu đồ trên Streamlit
    st.plotly_chart(fig, use_container_width=True,key=f"chart_{function_name}")


# Biểu đồ kết quả kinh doanh
def plot_business_results(df):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df['Doanh thu (đồng)'], name='Doanh thu', marker_color='rgb(250,50,50)'))
    fig.add_trace(go.Bar(x=df.index, y=df['Lợi nhuận thuần'], name='Lợi nhuận sau thuế', marker_color='rgb(0,200,0)'))
    fig.update_layout(title='Kết quả kinh doanh', barmode='group')
    # Hiển thị biểu đồ trên Streamlit
    function_name = inspect.currentframe().f_code.co_name  # Lấy tên hàm
    # Hiển thị biểu đồ trên Streamlit
    st.plotly_chart(fig, use_container_width=True,key=f"chart_{function_name}")

# Biểu đồ dòng tiền
def plot_cash_flow(df):
    colors = ['rgb(250,50, 50)', 'rgb(0, 200,0)', 'rgb(50, 50, 255)']
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df['Lưu chuyển tiền từ hoạt động tài chính'], name='HĐ tài chính', marker_color=colors[0]))
    fig.add_trace(go.Bar(x=df.index, y=df['Lưu chuyển tiền tệ ròng từ các hoạt động SXKD'], name='HĐ kinh doanh', marker_color=colors[1]))
    fig.add_trace(go.Bar(x=df.index, y=df['Lưu chuyển từ hoạt động đầu tư'], name='HĐ đầu tư', marker_color=colors[2]))
    fig.update_layout(title='Dòng tiền', barmode='group')
    # Hiển thị biểu đồ trên Streamlit
    function_name = inspect.currentframe().f_code.co_name  # Lấy tên hàm
    # Hiển thị biểu đồ trên Streamlit
    st.plotly_chart(fig, use_container_width=True,key=f"chart_{function_name}")




def plot_capital_structure(df_balance, stock_code):
    """
    Vẽ biểu đồ cấu trúc nguồn vốn cho từng cổ phiếu. 
    Chỉ hiển thị nếu dữ liệu đầy đủ.
    """
    required_columns = [
        'Năm', 'NỢ PHẢI TRẢ (đồng)', 'Nợ ngắn hạn (đồng)', 'Nợ dài hạn (đồng)',
        'VỐN CHỦ SỞ HỮU (đồng)', 'Vốn góp của chủ sở hữu (đồng)',
        'Vay và nợ thuê tài chính dài hạn (đồng)',
        'Vay và nợ thuê tài chính ngắn hạn (đồng)',
        'TỔNG CỘNG NGUỒN VỐN (đồng)'
    ]
    
    # Kiểm tra xem có đủ cột dữ liệu không
    if not all(col in df_balance.columns for col in required_columns):
        st.warning(f"Dữ liệu cho cổ phiếu {stock_code} không đầy đủ. Bỏ qua!")
        return

    # Lọc dữ liệu theo cổ phiếu đang chọn
    df_filtered = df_balance[df_balance['Mã CK'] == stock_code]
    
    if df_filtered.empty:
        st.warning(f"Không có dữ liệu cho cổ phiếu {stock_code}.")
        return

    # Reset index để tránh lỗi khi vẽ biểu đồ
    df_filtered = df_filtered.reset_index(drop=True)

    # Chuyển đổi dữ liệu để phù hợp với Plotly
    df_melted = pd.melt(df_filtered, id_vars=['Năm'], value_vars=required_columns[1:], 
                         var_name='Loại', value_name='Giá trị')

    # Tính tỷ số Nợ vay trên Tổng nguồn vốn
    df_filtered['Tỷ số Nợ vay trên Tổng nguồn vốn'] = (
        df_filtered['Vay và nợ thuê tài chính ngắn hạn (đồng)'] +
        df_filtered['Vay và nợ thuê tài chính dài hạn (đồng)']
    ) / df_filtered['TỔNG CỘNG NGUỒN VỐN (đồng)']
    
    # Vẽ biểu đồ
    fig = go.Figure()
    
    # Vẽ các cột stacked bar
    for loai in df_melted['Loại'].unique():
        fig.add_trace(go.Bar(
            x=df_melted[df_melted['Loại'] == loai]['Năm'],
            y=df_melted[df_melted['Loại'] == loai]['Giá trị'],
            name=loai
        ))

    # Vẽ đường tỷ lệ Nợ vay / Tổng nguồn vốn
    fig.add_trace(go.Scatter(
        x=df_filtered['Năm'], 
        y=df_filtered['Tỷ số Nợ vay trên Tổng nguồn vốn'], 
        mode='lines+markers',
        name='Tỉ lệ Nợ vay/TTS', 
        yaxis='y2'
    ))

    # Cấu hình layout
    fig.update_layout(
        yaxis2=dict(anchor='x', overlaying='y', side='right'),
        barmode='stack',
        xaxis_tickmode='linear',
        xaxis_title='Năm',
        yaxis_title='Giá trị (tỷ đồng)',
        title=f'NGUỒN VỐN - {stock_code}',
        updatemenus=[{
            'active': 0,
            'buttons': [
                {'label': 'Tăng', 'method': 'relayout', 'args': ['barmode', 'stack']},
                {'label': 'Tăng cường', 'method': 'relayout', 'args': ['barmode', 'group']}
            ],
            'direction': 'down',
            'showactive': True,
            'x': 1.05,
            'xanchor': 'left',
            'y': 1.2,
            'yanchor': 'top'
        }]
    )

    # Hiển thị biểu đồ trên Streamlit
    function_name = inspect.currentframe().f_code.co_name  # Lấy tên hàm
    st.plotly_chart(fig, use_container_width=True, key=f"chart_{function_name}_{stock_code}")



def plot_asset_structure(df_balance):
    df_balance = df_balance.reset_index()

    df_balance['Tiền/TTS'] = df_balance['TÀI SẢN NGẮN HẠN (đồng)'] / df_balance['TỔNG CỘNG TÀI SẢN (đồng)']
    df_melted = pd.melt(df_balance, id_vars=['Năm'], value_vars=[
       'TÀI SẢN NGẮN HẠN (đồng)', 'Tiền và tương đương tiền (đồng)',
        'Các khoản phải thu ngắn hạn (đồng)', 'Hàng tồn kho ròng',
        'TÀI SẢN DÀI HẠN (đồng)', 'Tài sản cố định (đồng)',
        'Đầu tư dài hạn (đồng)', 'TỔNG CỘNG TÀI SẢN (đồng)'
    ], var_name='Loại', value_name='Giá trị')
    df_melted.sort_values(by='Năm', inplace=True)
    fig = go.Figure()
    for loai in df_melted['Loại'].unique():
        fig.add_trace(go.Bar(
            x=df_melted[df_melted['Loại'] == loai]['Năm'],
            y=df_melted[df_melted['Loại'] == loai]['Giá trị'],
            name=loai
        ))
    fig.add_trace(go.Scatter(
        x=df_balance['Năm'],
        y=df_balance['Tiền/TTS'],
        mode='lines+markers',
        name='Tiền/TTS', yaxis='y2'))
    fig.update_layout(yaxis2=dict(anchor='x', overlaying='y', side='right'))
    fig.update_layout(
        barmode='stack',
        xaxis_tickmode='linear',
        xaxis_title='Năm',
        yaxis_title='Giá trị (đồng)',
        title='TÀI SẢN',
        updatemenus=[{
            'active': 0,
            'buttons': [
                {'label': 'Tăng', 'method': 'relayout', 'args': ['barmode', 'stack']},
                {'label': 'Tăng cường', 'method': 'relayout', 'args': ['barmode', 'group']}
            ],
            'direction': 'down',
            'showactive': True,
            'x': 1.05,
            'xanchor': 'left',
            'y': 1.2,
            'yanchor': 'top'
        }]
    )



def plot_profit_structure(df_kqkd):
    df_kqkd = df_kqkd.reset_index()
    fig = go.Figure()

    fig.add_trace(go.Bar(x=df_kqkd['Năm'], y=df_kqkd['Thu nhập tài chính'],
                         name='Lãi/lỗ từ hoạt động tài chính', marker_color=px.colors.qualitative.Plotly[1]))
    
    fig.add_trace(go.Bar(x=df_kqkd['Năm'], y=df_kqkd['Lãi/Lỗ từ hoạt động kinh doanh'],
                         name='Lãi/lỗ từ hoạt động kinh doanh', marker_color=px.colors.qualitative.Plotly[2]))
    
    fig.add_trace(go.Bar(x=df_kqkd['Năm'], y=df_kqkd['Lãi/lỗ từ công ty liên doanh'],
                         name='Lãi/lỗ từ công ty liên doanh', marker_color=px.colors.qualitative.Plotly[2]))
                    
    fig.add_trace(go.Bar(x=df_kqkd['Năm'], y=df_kqkd['Lợi nhuận khác'],
                         name='Lợi nhuận khác', marker_color=px.colors.qualitative.Plotly[4]))
    
    fig.update_layout(
        title='PHÂN TÍCH LỢI NHUẬN',
        xaxis_title='Năm',
        barmode='group',
        hovermode='x',
        yaxis2=dict(overlaying='y', side='right', showgrid=False, showline=False, zeroline=False, showticklabels=True)
    )
    # Hiển thị biểu đồ trên Streamlit
    function_name = inspect.currentframe().f_code.co_name  # Lấy tên hàm
    # Hiển thị biểu đồ trên Streamlit
    st.plotly_chart(fig, use_container_width=True,key=f"chart_{function_name}")

def plot_financial_ratios(df_insights):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_insights['Năm'], y=df_insights['ROA']*100, name='ROA', marker_color=px.colors.qualitative.Plotly[8]))
    fig.add_trace(go.Bar(x=df_insights['Năm'], y=df_insights['ROE']*100, name='ROE', marker_color=px.colors.qualitative.Plotly[2]))
    fig.update_layout(
        title='ROE-ROA',
        xaxis_title='Năm',
        barmode='group',
        hovermode='x'
    )
    # Hiển thị biểu đồ trên Streamlit
    function_name = inspect.currentframe().f_code.co_name  # Lấy tên hàm
    # Hiển thị biểu đồ trên Streamlit
    st.plotly_chart(fig, use_container_width=True,key=f"chart_{function_name}")

def plot_operating_efficiency(cstc):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cstc['Năm'], y=cstc['Số ngày thu tiền bình quân'], mode='lines+markers', name='Phải thu', marker_color=px.colors.qualitative.Plotly[2]))
    fig.add_trace(go.Scatter(x=cstc['Năm'], y=cstc['Số ngày tồn kho bình quân'], mode='lines+markers', name='Tồn kho', marker_color=px.colors.qualitative.Plotly[3]))
    fig.add_trace(go.Scatter(x=cstc['Năm'], y=cstc['Số ngày thanh toán bình quân'], mode='lines+markers', name='Phải trả', marker_color=px.colors.qualitative.Plotly[4]))
    fig.update_layout(
        title='HIỆU QUẢ HOẠT ĐỘNG',
        xaxis_title='Năm',
        yaxis_title='Số ngày',
        legend_title='Chỉ số',
        barmode='group',
        hovermode='x'
    )
    # Hiển thị biểu đồ trên Streamlit
    function_name = inspect.currentframe().f_code.co_name  # Lấy tên hàm
    # Hiển thị biểu đồ trên Streamlit
    st.plotly_chart(fig, use_container_width=True,key=f"chart_{function_name}")

def plot_leverage_ratios(cstc):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=cstc['Năm'], y=cstc['Nợ/VCSH'], name='Nợ trên vốn chủ sở hữu', marker_color=px.colors.qualitative.Plotly[4]))
    fig.add_trace(go.Bar(x=cstc['Năm'], y=cstc['TSCĐ/VSCH'], name='Tài sản cố định trên vốn chủ sở hữu', marker_color=px.colors.qualitative.Plotly[5]))
    fig.update_layout(
        title='HỆ SỐ ĐÒN BẨY',
        xaxis_title='Năm',
        barmode='group',
        hovermode='x',
        yaxis2=dict(overlaying='y', side='right', showgrid=False, showline=False, zeroline=False, showticklabels=True)
    )
    # Hiển thị biểu đồ trên Streamlit
    function_name = inspect.currentframe().f_code.co_name  # Lấy tên hàm
    # Hiển thị biểu đồ trên Streamlit
    st.plotly_chart(fig, use_container_width=True,key=f"chart_{function_name}")

def plot_pe_ratio(cstc):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=cstc['Năm'], y=cstc['EPS'], name='EPS', marker_color=px.colors.qualitative.Plotly[6]))
    fig.add_trace(go.Scatter(x=cstc['Năm'], y=cstc['P/E'], mode='lines+markers', name='P/E', yaxis='y2', marker_color=px.colors.qualitative.Plotly[1]))
    fig.update_layout(
        title='CHỈ SỐ ĐỊNH GIÁ P/E',
        xaxis_title='Năm',
        barmode='group',
        hovermode='x',
        yaxis2=dict(overlaying='y', side='right', showgrid=False, showline=False, zeroline=False, showticklabels=True)
    )
    # Hiển thị biểu đồ trên Streamlit
    function_name = inspect.currentframe().f_code.co_name  # Lấy tên hàm
    # Hiển thị biểu đồ trên Streamlit
    st.plotly_chart(fig, use_container_width=True,key=f"chart_{function_name}")

def plot_pb_ratio(cstc):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=cstc['Năm'], y=cstc['BVPS'], name='BVPS', marker_color=px.colors.qualitative.Plotly[5]))
    fig.add_trace(go.Scatter(x=cstc['Năm'], y=cstc['P/B'], mode='lines+markers', name='P/B', yaxis='y2', marker_color=px.colors.qualitative.Plotly[2]))
    fig.update_layout(
        title='CHỈ SỐ ĐỊNH GIÁ P/B',
        xaxis_title='Năm',
        barmode='group',
        hovermode='x',
        yaxis2=dict(overlaying='y', side='right', showgrid=False, showline=False, zeroline=False, showticklabels=True)
    )
    # Hiển thị biểu đồ trên Streamlit
    function_name = inspect.currentframe().f_code.co_name  # Lấy tên hàm
    # Hiển thị biểu đồ trên Streamlit
    st.plotly_chart(fig, use_container_width=True,key=f"chart_{function_name}")

def dupont_analysis_plot(cstc):
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cstc.index, y=cstc['Biên lợi nhuận ròng'] * 100, mode='lines+markers', yaxis='y2',
                             name='Biên lợi nhuận ròng(%)', marker_color=px.colors.qualitative.Plotly[6]))
    fig.add_trace(go.Scatter(x=cstc.index, y=cstc['Đòn bẩy tài chính'], name='Đòn bẩy tài chính', yaxis='y2',
                             marker_color=px.colors.qualitative.Plotly[9]))
    fig.add_trace(go.Scatter(x=cstc.index, y=cstc['Vòng quay tài sản'], name='Vòng quay tài sản', marker_color=px.colors.qualitative.Plotly[8]))
    fig.add_trace(go.Bar(x=cstc.index, y=cstc['ROE'] * 100, name='ROE(%)', yaxis='y2', marker_color=px.colors.qualitative.Plotly[2]))
    fig.update_layout(
        title='PHÂN TÍCH DUPONT',
        xaxis_title='Năm',
        barmode='group',
        hovermode='x',
        yaxis2=dict(overlaying='y', side='right', showgrid=False, showline=False, zeroline=False, showticklabels=True)
    )
    # Hiển thị biểu đồ trên Streamlit
    function_name = inspect.currentframe().f_code.co_name  # Lấy tên hàm
    # Hiển thị biểu đồ trên Streamlit
    st.plotly_chart(fig, use_container_width=True,key=f"chart_{function_name}")

def plot_portfolio_metrics(allocation_df):
    """
    Vẽ biểu đồ phân bổ danh mục đầu tư.
    """
    if allocation_df.empty:
        st.write("Không có dữ liệu để hiển thị biểu đồ phân bổ.")
        return
    
    fig = px.pie(allocation_df, names="Mã cổ phiếu", values="Tỷ trọng tối ưu",
                 title="Phân bổ danh mục đầu tư")
    st.plotly_chart(fig)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def display_radar_chart(data, title, color):
    categories = data.columns[1:].tolist()
    values = data.iloc[0, 1:].tolist()
    values += values[:1]  # Đóng vòng radar
    categories += categories[:1]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=data.iloc[0, 0],
        line=dict(color=color, width=2)
    ))
    
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), title=title)
    st.plotly_chart(fig, use_container_width=True)

def display_bar_chart(data, x_col, y_col, title):
    fig = px.bar(data, x=x_col, y=y_col, title=title, text_auto=True, color=x_col)
    st.plotly_chart(fig, use_container_width=True)

def display_line_chart(data, x_col, y_col, title):
    fig = px.line(data, x=x_col, y=y_col, title=title, markers=True)
    st.plotly_chart(fig, use_container_width=True)

def visualize_analysis(screener_df, code):
    df_selected = screener_df[screener_df['ticker'] == code]
    if df_selected.empty:
        st.warning(f"Không tìm thấy dữ liệu cho mã cổ phiếu {code}")
        return
    
    # Lấy ngành của cổ phiếu đã chọn
    industry = df_selected['industry'].values[0]
    
    # Lọc các cổ phiếu trong cùng ngành
    df_filtered = screener_df[screener_df['industry'] == industry]
    
    # Radar chart cho cổ phiếu đã chọn
    data1 = df_selected[['ticker', 'business_operation', 'business_model', 'financial_health', 'beta', 'stock_rating']]
    data1.columns = ['Mã', 'Hiệu suất kinh doanh', 'Mô hình kinh doanh', 'Sức khỏe tài chính', 'Rủi ro hệ thống', 'Xếp hạng định giá']
    display_radar_chart(data1, f'Biểu đồ Radar - Đánh giá {code}', 'blue')
    
    # Hiển thị các biểu đồ cho từng cổ phiếu nếu có dữ liệu
    with st.expander("📊 Hiệu Quả Hoạt Động"):
        # Lọc dữ liệu của cổ phiếu đã chọn
        if pd.notnull(df_selected['roe'].values[0]):
            display_bar_chart(df_filtered, 'ticker', 'roe', 'ROE của ngành')
        if pd.notnull(df_selected['gross_margin'].values[0]):
            display_bar_chart(df_filtered, 'ticker', 'gross_margin', 'Biên lợi nhuận gộp của ngành')
        if pd.notnull(df_selected['net_margin'].values[0]):
            display_bar_chart(df_filtered, 'ticker', 'net_margin', 'Biên lợi nhuận ròng của ngành')
        if pd.notnull(df_selected['eps'].values[0]):
            display_bar_chart(df_filtered, 'ticker', 'eps', 'EPS của ngành')
    
    with st.expander("💰 Sức Khỏe Tài Chính"):
        if pd.notnull(df_selected['financial_health'].values[0]):
            display_bar_chart(df_filtered, 'ticker', 'financial_health', 'Sức khỏe tài chính của ngành')
        if pd.notnull(df_selected['doe'].values[0]):
            display_bar_chart(df_filtered, 'ticker', 'doe', 'Tỷ lệ nợ trên vốn chủ sở hữu')
    
    with st.expander("📈 Định Giá"):
        if pd.notnull(df_selected['pe'].values[0]):
            display_bar_chart(df_filtered, 'ticker', 'pe', 'Chỉ số P/E của ngành')
        if pd.notnull(df_selected['pb'].values[0]):
            display_bar_chart(df_filtered, 'ticker', 'pb', 'Chỉ số P/B của ngành')
        if pd.notnull(df_selected['ev_ebitda'].values[0]):
            display_bar_chart(df_filtered, 'ticker', 'ev_ebitda', 'EV/EBITDA của ngành')
    
    with st.expander("🎯 Cổ Tức"):
        if pd.notnull(df_selected['dividend_yield'].values[0]):
            display_bar_chart(df_filtered, 'ticker', 'dividend_yield', 'Tỷ lệ cổ tức của ngành')
    
    with st.expander("🚀 Tăng Trưởng Lợi Nhuận"):
        if pd.notnull(df_selected['revenue_growth_1y'].values[0]):
            display_bar_chart(df_filtered, 'ticker', 'revenue_growth_1y', 'Tăng trưởng doanh thu 1 năm')
        if pd.notnull(df_selected['eps_growth_1y'].values[0]):
            display_bar_chart(df_filtered, 'ticker', 'eps_growth_1y', 'Tăng trưởng EPS 1 năm')

def create_scatter_chart(df_filtered):
    # Chọn giá trị cho trục x và y
    selected_x = st.selectbox('Chọn giá trị cho trục X:', ['roe', 'roa'])
    selected_y = st.selectbox('Chọn giá trị cho trục Y:', ['p/b', 'p/e'])

    # Vẽ biểu đồ scatter cho các cổ phiếu đã chọn
    fig_scatter = px.scatter(
        df_filtered, 
        x=selected_x, 
        y=selected_y, 
        size="vốn hóa (tỷ đồng)", 
        text="cp",
        color="vốn hóa (tỷ đồng)", 
        color_continuous_scale="Viridis",
        size_max=120,
        hover_name="cp", 
        hover_data={selected_x: True, selected_y: True, "vốn hóa (tỷ đồng)": True, "cp": False},
        title=f'So sánh {selected_x} vs {selected_y} của các cổ phiếu cùng ngành',
    )

    # Thêm dòng xu hướng
    fig_scatter.add_trace(px.line(df_filtered, x=selected_x, y=selected_y, line_shape='linear').data[0])

    # Tinh chỉnh bố cục và các thuộc tính khác
    fig_scatter.update_layout(
        title=dict(
            text=f'So sánh {selected_x} vs {selected_y} của các cổ phiếu cùng ngành',
            font=dict(size=24),
            x=0.5,
            y=0.95
        ),
        xaxis=dict(
            title=selected_x,
            gridcolor='LightGrey',
            zerolinecolor='DarkGrey',
            zerolinewidth=2
        ),
        yaxis=dict(
            title=selected_y,
            gridcolor='LightGrey',
            zerolinecolor='DarkGrey',
            zerolinewidth=2
        ),
        legend=dict(title='Vốn hóa (tỷ đồng)', title_font=dict(size=14)),
    )

    # Hiển thị biểu đồ
    st.plotly_chart(fig_scatter, use_container_width=True)