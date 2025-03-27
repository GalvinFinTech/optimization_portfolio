from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "postgresql://user:password@postgres:5432/stock_db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
