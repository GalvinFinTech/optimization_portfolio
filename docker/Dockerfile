FROM python:3.10

# Cài đặt các gói cần thiết
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Sao chép toàn bộ mã nguồn vào container
COPY . /app

# Cài đặt các thư viện Python từ requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Thiết lập biến môi trường để sử dụng Celery
ENV CELERY_BROKER_URL=redis://redis:6379/0
ENV CELERY_RESULT_BACKEND=redis://redis:6379/0
ENV DATABASE_URL=postgresql://user:password@postgres:5432/stock_db

# Chạy lệnh mặc định khi container khởi động
CMD ["streamlit", "run", "app.py"]
