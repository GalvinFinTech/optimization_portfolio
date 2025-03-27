# test_db.py
from db.database import engine

try:
    connection = engine.connect()
    print("Kết nối thành công!")
    connection.close()
except Exception as e:
    print("Lỗi kết nối:", e)
