import threading
from google import genai
import time

class GeminiProcessor:
    def __init__(self, api_keys: list[str]):
        self.api_keys = api_keys
        self.current_key_idx = 0
        self.models = ['gemini-3.1-flash-lite', 'gemini-3.6-flash', 'gemini-1.5-flash', 'gemini-1.5-flash-8b']
        self.current_model_idx = 0
        self.client = None
        self.lock = threading.Lock()
        self.setup_client()

    def setup_client(self):
        if not self.api_keys:
            print("No Gemini API keys provided!")
            return
            
        current_key = self.api_keys[self.current_key_idx]
        self.client = genai.Client(api_key=current_key)
        
    def switch_strategy(self):
        with self.lock:
            if not self.api_keys: return False
            
            # Đổi model trước, nếu hết model thì mới đổi key
            self.current_model_idx += 1
            if self.current_model_idx >= len(self.models):
                self.current_model_idx = 0
                self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
                print(f"Đã thử hết model, chuyển sang API key index: {self.current_key_idx}")
                self.setup_client()
            else:
                print(f"Chuyển sang model: {self.models[self.current_model_idx]} (Vẫn giữ Key hiện tại)")
            return True

    def process_text(self, prompt: str, text: str, max_retries=15) -> tuple[bool, str]:
        if not self.api_keys or not self.client:
            return False, "Lỗi: Không có API key Gemini."
            
        full_prompt = f"{prompt}\n\n[NỘI DUNG TRUYỆN]\n{text}"
        
        retries = 0
        while retries < max_retries:
            try:
                current_model = self.models[self.current_model_idx]
                response = self.client.models.generate_content(
                    model=current_model,
                    contents=full_prompt,
                )
                return True, response.text
            except Exception as e:
                error_str = str(e).lower()
                print(f"Gemini API Error ({current_model}): {error_str}")
                
                # Bắt lỗi 429 Too Many Requests, quota, 503 Unavailable, 500 Internal Error, 404 Not Found, hoặc lỗi kết nối (WinError, Timeout)
                if any(k in error_str for k in ["429", "quota", "exhausted", "503", "500", "unavailable", "404", "not found", "winerror", "connection", "timeout", "aborted", "reset"]):
                    print("Lỗi server/quota (503/429). Đang xoay vòng model/key... Thử lại sau 3s")
                    self.switch_strategy()
                    retries += 1
                    time.sleep(3)
                else:
                    # Các lỗi khác (ví dụ sai định dạng prompt, v.v...)
                    return False, f"Lỗi: Gemini API không thể phục hồi - {str(e)}"
                    
        return False, "Lỗi: Đã thử tất cả các model và key nhưng vẫn thất bại do quá tải."
        
    def cleanup_story(self, prompt: str, raw_text: str) -> tuple[bool, str]:
        """Làm sạch và viết lại truyện dựa trên prompt. Trả về (Trạng_thái_thành_công, Nội_dung)"""
        return self.process_text(prompt, raw_text)
        
    def generate_metadata(self, prompt: str, cleaned_text: str) -> tuple[bool, str]:
        """Tạo Title, Description... dựa trên prompt. Trả về (Trạng_thái_thành_công, Nội_dung)"""
        return self.process_text(prompt, cleaned_text)
