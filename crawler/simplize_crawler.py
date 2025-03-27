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
from selenium_stealth import stealth

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Đường dẫn lưu cookies
COOKIES_PATH = "cookies.pkl"

class SimplizeCrawler:
    def __init__(self, stock_code):
        self.stock_code = stock_code
        self.driver = None

    def initialize_driver(self):
        """Khởi tạo trình duyệt với Selenium Stealth"""
        options = Options()
        #options.add_argument("--headless")  # Chạy ẩn nếu cần
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36")
        options.add_argument("--start-maximized")
        options.add_argument("--disable-blink-features=AutomationControlled")  # Tránh bị phát hiện là bot
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("start-maximized")
        options.add_argument("disable-infobars")
        options.add_argument("--disable-extensions")

        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=options)

        # Kích hoạt Selenium Stealth để ẩn bot
        stealth(self.driver,
                languages=["en-US", "en"],
                vendor="Google Inc.",
                platform="Win32",
                webgl_vendor="Intel Inc.",
                renderer="Intel Iris OpenGL Engine",
                fix_hairline=True)

    def load_cookies(self):
        """Nạp cookies nếu có"""
        if os.path.exists(COOKIES_PATH):
            logger.info("🍪 Đang tải cookies...")
            self.driver.get("https://simplize.vn/")  # Mở trang chính trước khi nạp cookies
            time.sleep(3)
            with open(COOKIES_PATH, "rb") as file:
                cookies = pickle.load(file)
                for cookie in cookies:
                    self.driver.add_cookie(cookie)
            self.driver.refresh()
            time.sleep(3)
            logger.info("✅ Đã sử dụng cookies để bỏ qua đăng nhập!")

            # Kiểm tra nếu cookies hết hạn (tránh lỗi bị đá ra ngoài)
            if "Đăng nhập" in self.driver.page_source:
                logger.warning("🔑 Cookies đã hết hạn, cần đăng nhập lại!")
                os.remove(COOKIES_PATH)
                self.login()

    def save_cookies(self):
        """Lưu cookies sau khi đăng nhập"""
        with open(COOKIES_PATH, "wb") as file:
            pickle.dump(self.driver.get_cookies(), file)
        logger.info("💾 Cookies đã được lưu!")

    def login(self):
        """Yêu cầu người dùng tự đăng nhập và xác minh CAPTCHA"""
        url = f"https://simplize.vn/co-phieu/{self.stock_code}/bao-cao"
        logger.info(f"🔄 Mở trang đăng nhập: {url}")
        self.driver.get(url)

        input("🛑 Vui lòng đăng nhập và xác minh CAPTCHA, sau đó nhấn Enter để tiếp tục...")

        # Sau khi người dùng nhập tay xong, lưu cookies để lần sau không cần đăng nhập lại
        self.save_cookies()

    def crawl_data(self, num_rows=30, max_retries=3):
        """Thu thập dữ liệu từ bảng báo cáo"""
        url = f"https://simplize.vn/co-phieu/{self.stock_code}/bao-cao"
        logger.info(f"📄 Đang thu thập dữ liệu từ: {url}")
        self.driver.get(url)
        time.sleep(3)  # Đợi trang load

        data = []
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)

        retry_count = 0
        while retry_count < max_retries:
            try:
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "tbody.simplize-table-tbody"))
                )
                break  # Thoát vòng lặp nếu đã tìm thấy dữ liệu
            except TimeoutException:
                retry_count += 1
                logger.warning(f"⏳ Lần thử {retry_count}/{max_retries} - Chưa tìm thấy bảng dữ liệu, thử lại...")
                time.sleep(3)

        if retry_count == max_retries:
            logger.error("❌ Không tìm thấy dữ liệu trong bảng sau nhiều lần thử!")
            return data

        rows = self.driver.find_elements(By.CSS_SELECTOR, "tbody.simplize-table-tbody tr.simplize-table-row")
        if not rows:
            logger.info("⚠️ Không có dữ liệu trong bảng!")
            return data

        for index, row in enumerate(rows):
            tds = row.find_elements(By.TAG_NAME, "td")
            if len(tds) < 6:
                continue

            date = tds[0].text.strip()
            title = tds[1].text.strip()
            source = tds[2].text.strip()
            recommendation = tds[3].text.strip()
            price = tds[4].text.strip()

            # Lấy link tải về PDF
            pdf_link_download = ""
            try:
                pdf_element = tds[5].find_element(By.CSS_SELECTOR, "a.css-z8opiu")
                pdf_link_download = pdf_element.get_attribute("href") if pdf_element else ""
            except NoSuchElementException:
                logger.warning(f"⚠️ Không tìm thấy link tải về ở dòng {index + 1}")

            logger.info(f"🔗 Link tải về: {pdf_link_download}")

            data.append({
                "Ngày": date,
                "Tiêu đề": title,
                "Nguồn": source,
                "Khuyến nghị": recommendation,
                "Giá mục tiêu": price,
                "Tải về": pdf_link_download
            })

            if len(data) >= num_rows:
                break

        return data

    def close_driver(self):
        """Đóng trình duyệt."""
        if self.driver:
            self.driver.quit()

if __name__ == "__main__":
    stock_code = 'HPG'
    num_rows = 10

    crawler = SimplizeCrawler(stock_code)
    crawler.initialize_driver()

    if os.path.exists(COOKIES_PATH):
        crawler.load_cookies()
    else:
        crawler.login()

    data = crawler.crawl_data(num_rows=num_rows)
    crawler.close_driver()

    # Xuất dữ liệu
    if data:
        df = pd.DataFrame(data)
        df.to_csv("bao_cao.csv", index=False)
        print(df)
    else:
        print("⚠️ Không có dữ liệu để xuất.")
