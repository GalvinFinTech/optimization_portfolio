from sqlalchemy import Column, String, Integer
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class StockReport(Base):
    __tablename__ = "stock_reports"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_code = Column(String, nullable=False)
    date = Column(String, nullable=False)
    title = Column(String, nullable=False)
    source = Column(String, nullable=False)
    recommendation = Column(String, nullable=True)
    target_price = Column(String, nullable=True)
    download_link = Column(String, nullable=True)
