# vnai/scope/promo.py
# Content display mechanism for various environments

import requests
from datetime import datetime
import random
import threading
import time
import urllib.parse

class ContentManager:
    """Manages content display across different environments"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ContentManager, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance
    
    def _initialize(self):
        """Initialize content manager"""
        self.last_display = 0
        self.display_interval = 24 * 3600  # Once per day by default
        self.content_base_url = "https://vnstock-beam.hf.space/content-delivery"
        self.target_url = "https://vnstocks.com/lp-khoa-hoc-python-chung-khoan"
        self.image_url = "https://vnstocks.com/img/trang-chu-vnstock-python-api-phan-tich-giao-dich-chung-khoan.jpg"
        
        # Start periodic display thread
        self._start_periodic_display()
    
    def _start_periodic_display(self):
        """Start thread for occasional content display"""
        def periodic_display():
            while True:
                # Sleep for a random time between 2-6 hours
                sleep_time = random.randint(2 * 3600, 6 * 3600)
                time.sleep(sleep_time)
                
                # Only show if enough time has passed since last display
                current_time = time.time()
                if current_time - self.last_display >= self.display_interval:
                    self.present_content(context="periodic")
        
        # Start thread as daemon so it doesn't prevent program exit
        thread = threading.Thread(target=periodic_display, daemon=True)
        thread.start()
    
    def fetch_remote_content(self, context="init", html=True):
        """
        Fetch content from remote source
        
        Args:
            context (str): Context of the content request (init, loop, periodic, etc.)
            html (bool): Whether to request HTML content (True) or plain text (False)
        """
        try:
            # Build the URL with query parameters
            params = {
                "context": context,
                "html": "true" if html else "false"
            }
            url = f"{self.content_base_url}?{urllib.parse.urlencode(params)}"
            
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                return response.text
            return None
        except:
            return None
    
    def present_content(self, environment=None, context="init"):
        """
        Present content appropriate for the environment and context
        
        Args:
            environment (str): The display environment (jupyter, terminal, etc.)
            context (str): The context for content (init, loop, periodic)
        """
        # Update last display time
        self.last_display = time.time()
        
        # Get environment if not provided
        if environment is None:
            try:
                from vnai.scope.profile import inspector
                environment = inspector.examine().get("environment", "unknown")
            except:
                environment = "unknown"
        
        # Try to fetch remote content based on context and environment
        if environment == "jupyter":
            remote_content = self.fetch_remote_content(context=context, html=True)
        else:
            remote_content = self.fetch_remote_content(context=context, html=False)
        
        # Generate fallback content based on context
        fallback_content = self._generate_fallback_content(context)
        
        # Display based on environment
        if environment == "jupyter":
            try:
                from IPython.display import display, HTML, Markdown
                if remote_content:
                    # Use remote content if available
                    display(HTML(remote_content))
                else:
                    # Try markdown version first
                    try:
                        display(Markdown(fallback_content["markdown"]))
                    except:
                        # Fall back to HTML if markdown fails
                        display(HTML(fallback_content["html"]))
            except:
                # If IPython display fails, do nothing
                pass
        
        elif environment == "terminal":
            if remote_content:
                print(remote_content)
            else:
                print(fallback_content["terminal"])
        
        else:
            # Simple text version for other environments
            print(fallback_content["simple"])
    
    def _generate_fallback_content(self, context):
        """Generate appropriate fallback content based on context"""
        fallback = {
            "html": "",
            "markdown": "",
            "terminal": "",
            "simple": ""
        }
        
        if context == "loop":
            
                        # Content for rate limit exceeded scenario
            fallback["html"] = f"""
            <div style="border: 1px solid #e74c3c; padding: 15px; border-radius: 5px; margin: 10px 0;">
                <h3 style="color: #e74c3c;">⚠️ Bạn đang sử dụng vòng lặp với quá nhiều requests</h3>
                <p>Để tránh bị giới hạn tốc độ và tối ưu hiệu suất:</p>
                <ul>
                    <li>Thêm thời gian chờ giữa các lần gọi API</li>
                    <li>Sử dụng xử lý theo batch thay vì lặp liên tục</li>
                    <li>Tham gia gói tài trợ <a href="https://vnstocks.com/insiders-program" style="color: #3498db;">Vnstock Insider</a> để tăng 5X giới hạn API</li>
                </ul>
            </div>
            """
            
            fallback["markdown"] = """
## ⚠️ Bạn đang sử dụng vòng lặp với quá nhiều requests

Để tránh bị giới hạn tốc độ và tối ưu hiệu suất:
* Thêm thời gian chờ giữa các lần gọi API
* Sử dụng xử lý theo batch thay vì lặp liên tục
* Tham gia gói tài trợ [Vnstock Insider](https://vnstocks.com/insiders-program) để tăng 5X giới hạn API
            """
            
            fallback["terminal"] = """
╔═════════════════════════════════════════════════════════════════╗
║                                                                 ║
║   🚫 ĐANG BỊ CHẶN BỞI GIỚI HẠN API? GIẢI PHÁP Ở ĐÂY!            ║
║                                                                 ║
║   ✓ Tăng ngay 500% tốc độ gọi API - Không còn lỗi RateLimit     ║
║   ✓ Tiết kiệm 85% thời gian chờ đợi giữa các request            ║
║                                                                 ║
║   ➤ NÂNG CẤP NGAY VỚI GÓI TÀI TRỢ VNSTOCK:                      ║
║     https://vnstocks.com/insiders-program                       ║
║                                                                 ║
╚═════════════════════════════════════════════════════════════════╝
            """
            
            fallback["simple"] = "🚫 Đang bị giới hạn API? Tăng tốc độ gọi API lên 500% với gói Vnstock Insider: https://vnstocks.com/insiders-program"
            
        else:  # init or periodic or default
            # General welcome content
            fallback["html"] = f"""
            <div style="border: 1px solid #3498db; padding: 15px; border-radius: 5px; margin: 10px 0;">
                <h3 style="color: #3498db;">👋 Chào mừng bạn đến với Vnstock!</h3>
                <p>Cảm ơn bạn đã sử dụng thư viện phân tích chứng khoán #1 tại Việt Nam cho Python</p>
                <ul>
                    <li>Tài liệu: <a href="https://vnstocks.com/docs/category/s%E1%BB%95-tay-h%C6%B0%E1%BB%9Bng-d%E1%BA%ABn" style="color: #3498db;">vnstocks.com/docs</a></li>
                    <li>Cộng đồng: <a href="https://www.facebook.com/groups/vnstock.official" style="color: #3498db;">vnstocks.com/community</a></li>
                </ul>
                <p>Khám phá các tính năng mới nhất và tham gia cộng đồng để nhận hỗ trợ.</p>
            </div>
            """
            
            fallback["markdown"] = """
## 👋 Chào mừng bạn đến với Vnstock!

Cảm ơn bạn đã sử dụng package phân tích chứng khoán #1 tại Việt Nam

* Tài liệu: [vnstocks.com/docs](https://vnstocks.com/docs)
* Cộng đồng: [vnstocks.com/community](https://vnstocks.com/community)

Khám phá các tính năng mới nhất và tham gia cộng đồng để nhận hỗ trợ.
            """
            
            fallback["terminal"] = """
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║  👋 Chào mừng bạn đến với Vnstock!                       ║
║                                                          ║
║  Cảm ơn bạn đã sử dụng package phân tích                 ║
║  chứng khoán #1 tại Việt Nam                             ║
║                                                          ║
║  ✓ Tài liệu: https://vnstocks.com/docs                   ║
║  ✓ Cộng đồng: https://vnstocks.com/community             ║
║                                                          ║
║  Khám phá các tính năng mới nhất và tham gia             ║
║  cộng đồng để nhận hỗ trợ.                               ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
            """
            
            fallback["simple"] = "👋 Chào mừng bạn đến với Vnstock! Tài liệu: https://vnstocks.com/docs | Cộng đồng: https://vnstocks.com/community"
            
        return fallback

# Create singleton instance
manager = ContentManager()

def present(context="init"):
    """
    Public API to present content
    
    Args:
        context (str): Context for content presentation (init, loop, periodic)
    """
    return manager.present_content(context=context)
