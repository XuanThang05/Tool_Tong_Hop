from PyQt6.QtCore import QThread, pyqtSignal
from core.config_manager import ConfigManager

from modules.gemini_processor.gemini import GeminiProcessor
import time
import os
import re
import concurrent.futures

class PipelineEngine(QThread):
    log_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, config_manager: ConfigManager):
        super().__init__()
        self.config = config_manager
        self.is_running = False
        self.input_files = []

    def run(self):
        self.is_running = True
        self.status_signal.emit("Trạng thái: Đang chạy")
        self.log_signal.emit("Engine started.")
        
        try:
            self.process_workflow()
        except Exception as e:
            self.log_signal.emit(f"Engine Error: {str(e)}")
            
        self.status_signal.emit("Trạng thái: Đang rảnh")
        self.log_signal.emit("Engine stopped.")
        self.finished_signal.emit()

    def stop(self):
        self.is_running = False
        self.log_signal.emit("Đang dừng khẩn cấp các quy trình (Kill All)...")
        if hasattr(self, 'current_tts') and self.current_tts:
            self.current_tts.stop()
        if hasattr(self, 'current_wf') and self.current_wf:
            self.current_wf.stop()
        if hasattr(self, 'current_asm') and self.current_asm:
            self.current_asm.stop()
            
        if hasattr(self, 'stt_executor') and self.stt_executor:
            self.stt_executor.shutdown(wait=False, cancel_futures=True)

        try:
            import psutil
            current_process = psutil.Process(os.getpid())
            children = current_process.children(recursive=True)
            for child in children:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            self.log_signal.emit("✅ Đã dọn dẹp RAM và giải phóng toàn bộ tiến trình ngầm.")
        except Exception as e:
            self.log_signal.emit(f"Lỗi khi dọn dẹp bộ nhớ: {e}")
        
    def process_workflow(self):
        pipeline_cfg = self.config.get("pipeline") or {}
        
        if not self.input_files:
            self.log_signal.emit("❌ Không có file TXT nào được nhập. Vui lòng thêm file trước khi chạy.")
            self.is_running = False
            return
            
        self.log_signal.emit(f"Bắt đầu xử lý {len(self.input_files)} file chương truyện...")
        
        # Đọc nội dung từng file
        chapters = []
        for file_path in self.input_files:
            if not os.path.exists(file_path):
                self.log_signal.emit(f"Lỗi: Không tìm thấy file {file_path}")
                continue
            with open(file_path, 'r', encoding='utf-8') as f:
                chapters.append({
                    "path": file_path,
                    "name": os.path.basename(file_path),
                    "raw_content": f.read()
                })
                
        if not chapters:
            self.is_running = False
            return
            
        # 1. Gộp file để Check trùng lặp
        merged_raw = "\n\n".join([c["raw_content"] for c in chapters])
        
        # Tự động tạo thư mục output dựa trên tên file đầu tiên
        paths_cfg = self.config.get("paths") or {}
        base_output_dir = paths_cfg.get("output_dir", "")
        if not base_output_dir:
            base_output_dir = os.path.join(os.getcwd(), "outputs")
            
        first_file_path = chapters[0]["path"]
        first_file_name = os.path.splitext(chapters[0]["name"])[0]
        
        # Lấy tên thư mục cha chứa file TXT
        parent_dir_name = os.path.basename(os.path.dirname(first_file_path))
        
        if not parent_dir_name or parent_dir_name.endswith(':'):
            prefix_name = first_file_name
        else:
            prefix_name = f"{parent_dir_name}_{first_file_name}"
            
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_title = re.sub(r'[\\/*?:"<>|]', "", f"{prefix_name}_{timestamp}")
        output_dir = os.path.join(base_output_dir, safe_title)
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, "01_raw_merged.txt"), "w", encoding="utf-8") as f:
            f.write(merged_raw)
            
        # 2. Check trùng lặp
        if pipeline_cfg.get("plagiarism_check", True):
            self.log_signal.emit("Đang kiểm tra trùng lặp toàn bộ câu chuyện...")
            plag_cfg = self.config.get("plagiarism") or {}
            warn_threshold = int(plag_cfg.get("similarity_threshold", 80))
            
            import sys
            check_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "modules", "check_trung_lap")
            if check_path not in sys.path:
                sys.path.insert(0, check_path)
            from src.services import AppService
            
            self.app_service = AppService()
            self.log_signal.emit("Đang khởi tạo model AI...")
            self.app_service.warmup()
            
            result = self.app_service.check_text(
                text=merged_raw,
                title=safe_title,
                notebook_id=None,
                warn=warn_threshold,
                block=100
            )
            
            # Lưu lại để hiển thị trong giao diện Cài đặt
            self.last_plag_result = result
            self.last_plag_text = merged_raw
            
            is_plagiarized = result.verdict in ("warn", "block")
            percent = result.overall_score
            self.log_signal.emit(f"Tỉ lệ trùng lặp: {percent}% (Ngưỡng cài đặt: {warn_threshold}%)")
            
            if is_plagiarized:
                self.log_signal.emit("❌ Truyện bị trùng lặp quá mức. TIẾN TRÌNH ĐÃ BỊ HỦY BỎ!")
                self.is_running = False
                return
            else:
                self.log_signal.emit("✅ Vượt qua bài kiểm tra trùng lặp. Tiếp tục quy trình.")
                # Chúng ta sẽ thêm vào thư viện Sổ trùng lặp SAU KHI ĐÃ CLEAN (Bước 3)
                
        # Khởi tạo Gemini Processor
        gemini_cfg = self.config.get("gemini") or {}
        gemini_keys = gemini_cfg.get("api_keys", [])
        if type(gemini_keys) is str:
            gemini_keys = [k.strip() for k in gemini_keys.split('\n') if k.strip()]
        processor = GeminiProcessor(gemini_keys)
        
        # 3. Gemini Clean (Đa luồng cho từng chương)
        if pipeline_cfg.get("gemini_clean", True):
            prompt = gemini_cfg.get("prompt_cleanup", "")
            threads = int(gemini_cfg.get("threads_cleanup", 3))
            
            self.log_signal.emit(f"Bắt đầu làm sạch {len(chapters)} chương bằng Gemini ({threads} luồng)...")
            
            def clean_chapter(idx, chapter):
                if not self.is_running: return None
                self.log_signal.emit(f"  Đang gọi API cho: {chapter['name']}")
                success, cleaned = processor.cleanup_story(prompt, chapter['raw_content'])
                chapter['cleaned_content'] = cleaned
                chapter['api_success'] = success
                self.log_signal.emit(f"  ✅ Xong: {chapter['name']}")
                return chapter
                
            with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
                # Gửi các task
                futures = {executor.submit(clean_chapter, i, c): i for i, c in enumerate(chapters)}
                for future in concurrent.futures.as_completed(futures):
                    if not self.is_running: break
                    try:
                        future.result()
                    except Exception as e:
                        self.log_signal.emit(f"❌ Lỗi khi dọn dẹp chương: {e}")
            
            if not self.is_running: return
            
            # Kiểm tra xem có bất kỳ chương nào gọi API thất bại không
            failed_chapters = [c for c in chapters if not c.get("api_success", True)]
            if failed_chapters:
                self.log_signal.emit(f"❌ API Gemini trả về lỗi ở {len(failed_chapters)} chương. Tiến trình ĐÃ BỊ HỦY BỎ!")
                self.log_signal.emit(f"Chi tiết lỗi: {failed_chapters[0].get('cleaned_content', '')[:200]}")
                self.is_running = False
                return
            
            # Gộp lại sau khi clean thành công toàn bộ
            merged_cleaned = "\n\n".join([(c.get("cleaned_content") or "") for c in chapters])
                
            with open(os.path.join(output_dir, "02_cleaned_merged.txt"), "w", encoding="utf-8") as f:
                f.write(merged_cleaned)
            self.log_signal.emit("✅ Đã làm sạch và gộp toàn bộ các chương.")
        else:
            merged_cleaned = merged_raw
            
        # Khởi tạo biến tracking để stop
        self.current_tts = None
        self.current_wf = None
        self.current_asm = None

        # 4. TTS
        audio_out_path = os.path.join(output_dir, "03_audio.wav")
        if pipeline_cfg.get("tts", True):
            self.log_signal.emit(f"[{safe_title}] Đang tạo âm thanh (TTS Piper)...")
                
            from modules.tts_piper.piper_tts import PiperTTSGenerator
            tts_cfg = self.config.get("tts") or {}
            self.current_tts = PiperTTSGenerator(config=tts_cfg)
            
            def tts_progress(pct, msg):
                self.log_signal.emit(f"[{safe_title}] TTS ({pct}%): {msg}")
                
            success = self.current_tts.generate(merged_cleaned, audio_out_path, progress_callback=tts_progress)
            if success:
                self.log_signal.emit(f"[{safe_title}] ✅ Đã tạo xong Audio.")
            else:
                self.log_signal.emit(f"[{safe_title}] ❌ Lỗi tạo Audio.")
                self.is_running = False
                return

        # 4.5 STT Async (Chạy ngầm)
        stt_future = None
        video_cfg = self.config.get("video") or {}
        sub_enable = video_cfg.get('sub_enable', False)
        ass_path = os.path.join(output_dir, 'subtitle.ass')
        
        if pipeline_cfg.get("video_assemble", True) and sub_enable and os.path.exists(audio_out_path):
            self.log_signal.emit(f"[{safe_title}] Bắt đầu chạy ngầm API STT CapCut...")
            from modules.video_assembler.assembler import VideoAssembler
            temp_assembler = VideoAssembler(output_dir, video_cfg)
            self.stt_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            stt_future = self.stt_executor.submit(temp_assembler.generate_ass_subtitle, audio_out_path, ass_path)

        # 5. Waveform
        waveform_path = os.path.join(output_dir, "04_waveform.mp4")
        if pipeline_cfg.get("waveform", True):
            self.log_signal.emit(f"[{safe_title}] Đang tạo video sóng âm...")
            
            if not os.path.exists(audio_out_path):
                self.log_signal.emit(f"[{safe_title}] ❌ Không tìm thấy file Audio. Bỏ qua tạo sóng âm.")
            else:
                from modules.waveform_gen.generator import WaveformGenerator
                waveform_cfg = self.config.get("waveform") or {}
                waveform_cfg['video_encoder'] = video_cfg.get('video_encoder', 0)
                
                self.current_wf = WaveformGenerator(config=waveform_cfg)
                
                def wf_progress(pct, msg):
                    self.log_signal.emit(f"[{safe_title}] Sóng âm ({pct}%): {msg}")
                    
                success = self.current_wf.generate(audio_out_path, waveform_path, progress_callback=wf_progress)
                if success:
                    self.log_signal.emit(f"[{safe_title}] ✅ Đã tạo xong video sóng âm.")
                else:
                    self.log_signal.emit(f"[{safe_title}] ❌ Lỗi khi tạo video sóng âm.")
                    self.is_running = False
                    return

        # Đợi STT xong nếu chưa xong
        if stt_future is not None:
            if not stt_future.done():
                self.log_signal.emit(f"[{safe_title}] Đang chờ STT CapCut hoàn thành...")
            stt_future.result()
            self.log_signal.emit(f"[{safe_title}] STT CapCut đã xong.")

        # 6. Video Assembler
        if pipeline_cfg.get("video_assemble", True):
            self.log_signal.emit(f"[{safe_title}] Đang tiến hành ghép Video Cuối cùng...")
            
            from modules.video_assembler.assembler import VideoAssembler
            self.current_asm = VideoAssembler(output_dir, video_cfg)
            
            def vid_progress(pct, msg):
                self.log_signal.emit(f"[{safe_title}] Ghép Video ({pct}%): {msg}")
                
            success = self.current_asm.run(audio_out_path, waveform_path, progress_callback=vid_progress)
            if success:
                self.log_signal.emit(f"[{safe_title}] ✅ Đã ghép xong video hoàn chỉnh (05_final.mp4).")
                
                # CHỈ LƯU VÀO SỔ KHI VIDEO HOÀN THÀNH NHƯ CŨ
                if pipeline_cfg.get("plagiarism_check", True) and hasattr(self, 'app_service'):
                    plag_cfg = self.config.get("plagiarism") or {}
                    target_nb_name = plag_cfg.get("target_notebook")
                    if target_nb_name:
                        self.log_signal.emit(f"Đang lưu bản gốc '{safe_title}' vào sổ '{target_nb_name}'...")
                        notebooks = self.app_service.list_notebooks()
                        nb_id = next((nb['id'] for nb in notebooks if nb['name'] == target_nb_name), None)
                        if nb_id is not None:
                            try:
                                self.app_service.add_text(notebook_id=nb_id, title=safe_title, text=merged_cleaned)
                                self.log_signal.emit(f"✅ Đã lưu bản làm sạch vào sổ '{target_nb_name}' thành công.")
                            except Exception as e:
                                self.log_signal.emit(f"Lỗi khi lưu vào sổ: {e}")
                        else:
                            self.log_signal.emit(f"❌ Không tìm thấy sổ '{target_nb_name}'. Vui lòng kiểm tra lại Cài đặt.")
                            
            else:
                self.log_signal.emit(f"[{safe_title}] ❌ Lỗi khi ghép video.")
            
        self.log_signal.emit("🎉 XONG! Quá trình đã hoàn tất hoàn toàn.")
        self.is_running = False
