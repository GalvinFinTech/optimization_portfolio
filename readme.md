
stock_dashboard/
├── app.py                    # File chính cho giao diện Streamlit
├── data/
│   ├── __init__.py
│   └── loader.py             # Hàm load, clean, và xử lý dữ liệu trước khi vẽ biểu đồ
├── charts/
│   ├── __init__.py
│   └── plots.py              # Các hàm tạo biểu đồ (RSI, MACD, Bar charts,...)
├── crawler/
│   ├── __init__.py
│   ├── simplize_crawler.py   # Code thu thập dữ liệu từ Simplize.vn
│   └── vnstock_crawler.py    # (Tuỳ chọn) Thu thập dữ liệu từ Vnstock
├── tasks/
│   ├── __init__.py
│   ├── celery_app.py         # Cấu hình Celery
│   ├── tasks.py              # Các tác vụ Celery để crawl và lưu dữ liệu
├── db/
│   ├── __init__.py
│   ├── models.py             # Định nghĩa ORM cho PostgreSQL (SQLAlchemy)
│   ├── database.py           # Kết nối và quản lý PostgreSQL
├── docker/
│   ├── docker-compose.yml    # Cấu hình Docker cho PostgreSQL, Redis, Celery
│   ├── Dockerfile            # Dockerfile chạy ứng dụng
├── scripts/
│   ├── run_celery.sh         # Script khởi động Celery worker
│   ├── run_streamlit.sh      # Script chạy Streamlit
├── requirements.txt          # Danh sách thư viện cần thiết
├── README.md                 # Hướng dẫn sử dụng và cài đặt


python -m venv myenv
myenv\Scripts\activate (Trên Windows)
source myenv/bin/activate (Trên Mac/Linux)

python --version >= 3.10


pip freeze > requirements.txt
streamlit run app.py

# Stock Dashboard

Ứng dụng dashboard chứng khoán sử dụng Streamlit, Plotly, và các hàm phân tích kỹ thuật.

## Cấu trúc dự án

- `app.py`: File chính chạy giao diện Streamlit.
- `data/`: Chứa các hàm load và xử lý dữ liệu.
- `charts/`: Chứa các hàm tạo biểu đồ.
- `analysis/`: Chứa các hàm phân tích kỹ thuật.
- `utils/`: Chứa các hàm hỗ trợ chung.

## Cài đặt

1. Clone repository và chuyển vào thư mục dự án:
   ```bash
   git clone https://github.com/yourusername/stock_dashboard.git
   cd stock_dashboard
2. Cài đặt các thư viện cần thiết:
    pip install -r requirements.txt
3. Chạy ứng dụng
    streamlit run app.py

---




# optimization_portfolio
