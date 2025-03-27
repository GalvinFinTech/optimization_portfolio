# Thông tin đăng nhập
#USERNAME = "vinh21414@st.uel.edu.vn"
#PASSWORD = "funxag-2nubpu-xeFsux"
#COOKIES_FILE = "cookies.pkl"

import os
import pickle
import logging
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from urllib3.exceptions import ReadTimeoutError

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
logger = logging.getLogger(__name__)

COOKIES_PATH = "cookies.pkl"

class SimplizeCrawler:
    def __init__(self, stock_code):
        self.stock_code = stock_code
        self.driver = None

    def initialize_driver(self):
        """Khởi tạo trình duyệt với Selenium và thiết lập các tùy chọn cần thiết."""
        options = Options()
        options.add_argument("--start-maximized")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        # Sử dụng page load strategy "none" để tránh chờ tải toàn bộ trang
        options.page_load_strategy = "none"
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=options)
        logger.info("🖥️  Trình duyệt đã được khởi tạo.")

    def load_cookies(self):
        """Nạp cookies nếu có để bỏ qua quá trình đăng nhập."""
        if os.path.exists(COOKIES_PATH):
            logger.info("🍪 Đang tải cookies...")
            try:
                self.driver.set_page_load_timeout(180)
                self.driver.get("https://simplize.vn/")
            except (TimeoutException, ReadTimeoutError) as e:
                logger.error(f"❌ Quá thời gian chờ tải trang: {e}")
                self.driver.execute_script("window.stop();")
            try:
                WebDriverWait(self.driver, 20).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
            except TimeoutException:
                logger.error("❌ Không thể xác định trang sau khi tải.")
            with open(COOKIES_PATH, "rb") as file:
                cookies = pickle.load(file)
                for cookie in cookies:
                    try:
                        self.driver.add_cookie(cookie)
                    except Exception as e:
                        logger.warning(f"⚠️ Lỗi khi thêm cookie: {e}")
            self.driver.refresh()
            try:
                WebDriverWait(self.driver, 20).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
            except TimeoutException:
                logger.error("❌ Không thể xác định trang sau khi refresh.")
            logger.info("✅ Cookies đã được áp dụng.")

    def save_cookies(self):
        """Lưu cookies sau khi đăng nhập."""
        with open(COOKIES_PATH, "wb") as file:
            pickle.dump(self.driver.get_cookies(), file)
        logger.info("💾 Cookies đã được lưu.")

    def login(self):
        """Yêu cầu người dùng đăng nhập thủ công và xác minh CAPTCHA."""
        url = f"https://simplize.vn/co-phieu/{self.stock_code}/bao-cao"
        logger.info(f"🔄 Mở trang đăng nhập: {url}")
        self.driver.get(url)
        input("🛑 Vui lòng đăng nhập và xác minh CAPTCHA, sau đó nhấn Enter để tiếp tục...")
        self.save_cookies()

    def crawl_data(self, num_rows=10):
        """
        Thu thập dữ liệu báo cáo từ trang.
        Lấy 10 dòng đầu tiên hiển thị trong bảng dữ liệu.
        """
        url = f"https://simplize.vn/co-phieu/{self.stock_code}/bao-cao"
        logger.info(f"📄 Đang thu thập dữ liệu từ: {url}")
        self.driver.get(url)
        try:
            # Chờ tối đa 30 giây cho phần header của bảng xuất hiện
            WebDriverWait(self.driver, 30).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "thead.simplize-table-thead"))
            )
        except TimeoutException:
            logger.error("❌ Không thể tải được header của bảng dữ liệu.")
            return []
        
        try:
            # Chờ tbody chứa dữ liệu xuất hiện
            WebDriverWait(self.driver, 30).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "tbody.simplize-table-tbody"))
            )
        except TimeoutException:
            logger.error("❌ Không thể tải được phần dữ liệu (tbody) của bảng.")
            # In ra một phần mã nguồn để debug
            logger.info("Mã nguồn trang (1000 ký tự đầu):")
            logger.info(self.driver.page_source[:1000])
            return []

        # Lấy các dòng dữ liệu từ tbody
        rows = self.driver.find_elements(By.CSS_SELECTOR, "tbody.simplize-table-tbody tr")
        logger.info(f"📊 Số bài hiển thị: {len(rows)}")
        final_count = min(len(rows), num_rows)
        logger.info(f"✅ Lấy dữ liệu của {final_count} bài.")
        data = []
        for index, row in enumerate(rows[:final_count]):
            tds = row.find_elements(By.TAG_NAME, "td")
            if len(tds) < 6:
                logger.warning(f"⚠️ Dòng {index+1} không có đủ số ô dữ liệu.")
                continue

            # Lấy dữ liệu từ các ô dựa trên thứ tự: Ngày, Tiêu đề, Nguồn, Khuyến nghị, Giá mục tiêu, Tải về
            date = tds[0].text.strip()
            title = tds[1].text.strip()
            source = tds[2].text.strip()
            recommendation = tds[3].text.strip()
            price = tds[4].text.strip()

            pdf_link_download = ""
            try:
                # Tìm thẻ <a> trong ô "Tải về" để lấy link
                pdf_element = tds[5].find_element(By.CSS_SELECTOR, "a.css-z8opiu")
                pdf_link_download = pdf_element.get_attribute("href")
            except NoSuchElementException:
                logger.warning(f"⚠️ Không tìm thấy link tải về ở bài thứ {index + 1}")

            logger.info(f"🔗 [{date}] {title} - {recommendation} - {price} | Link: {pdf_link_download}")
            data.append({
                "Ngày": date,
                "Tiêu đề": title,
                "Nguồn": source,
                "Khuyến nghị": recommendation,
                "Giá mục tiêu": price,
                "Tải về": pdf_link_download
            })
        return data

    def close_driver(self):
        """Đóng trình duyệt."""
        if self.driver:
            self.driver.quit()
            logger.info("🚪 Trình duyệt đã được đóng.")

if __name__ == "__main__":
    stock_code = 'VCB'  # Thay đổi mã cổ phiếu theo nhu cầu
    num_rows = 10       # Lấy 10 bài mới nhất

    crawler = SimplizeCrawler(stock_code)
    try:
        crawler.initialize_driver()
        if os.path.exists(COOKIES_PATH):
            crawler.load_cookies()
        else:
            crawler.login()

        scraped_data = crawler.crawl_data(num_rows=num_rows)
        if scraped_data:
            df = pd.DataFrame(scraped_data)
            df.to_csv("bao_cao.csv", index=False)
            logger.info("💾 Dữ liệu đã được lưu vào 'bao_cao.csv'.")
            print(df.head())
        else:
            logger.warning("⚠️ Không có dữ liệu thu thập được.")
    except Exception as e:
        logger.exception(f"❌ Đã xảy ra lỗi: {e}")
    finally:
        crawler.close_driver()







