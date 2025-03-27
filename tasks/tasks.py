from celery_app import celery_app
from crawler.simplize_crawler import SimplizeCrawler
from db.database import SessionLocal
from db.models import StockReport
import logging

logger = logging.getLogger(__name__)

@celery_app.task
def crawl_stock_data(stock_code):
    session = SessionLocal()
    crawler = SimplizeCrawler()
    crawler.initialize_driver()
    crawler.load_cookies()
    
    data = crawler.crawl_stock(stock_code)
    
    for entry in data:
        report = StockReport(
            stock_code=entry["stock_code"],
            date=entry["date"],
            title=entry["title"],
            source=entry["source"],
            recommendation=entry["recommendation"],
            target_price=entry["target_price"],
            download_link=entry["download_link"]
        )
        session.add(report)
    
    session.commit()
    session.close()
    crawler.close_driver()
    logger.info(f"Completed crawl for {stock_code}")
