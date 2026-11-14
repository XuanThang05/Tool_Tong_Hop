import os
import re
import wave
import tempfile
import subprocess
import numpy as np

class PiperTTSGenerator:
    def __init__(self, config=None):
        self.config = config or {}
        
    def _split_text(self, text, max_threads=1):
        total_length = len(text.strip())
        
        # Tính kích thước lý tưởng cho mỗi luồng
        ideal_chunk_size = total_length // max_threads if max_threads > 0 else total_length
        
        # Cân bằng giới hạn: Min 300, Max 3000
        max_chars = max(300, min(ideal_chunk_size, 3000))
        
        sents = re.split(r'(?<=[.!?\n])\s+', text.strip())
        chunks, cur = [], ""
        for s in sents:
            if len(cur) + len(s) + 1 <= max_chars:
                cur = (cur + " " + s).strip()
            else:
                if cur:
                    chunks.append(cur)
                cur = s
        if cur:
            chunks.append(cur)
        return [c for c in chunks if c.strip()]
        
    def _read_wav(self, path):
        with wave.open(path, "rb") as wf:
            sr   = wf.getframerate()
            nch  = wf.getnchannels()
            sw   = wf.getsampwidth()
            data = wf.readframes(wf.getnframes())
        dt = {1: np.int8, 2: np.int16, 4: np.int32}.get(sw, np.int16)
        audio = (np.frombuffer(data, dtype=dt).astype(np.float32)
                 / float(np.iinfo(dt).max))
        if nch > 1:
            audio = audio.reshape(-1, nch)
        return audio, sr

    def generate(self, text: str, output_path: str, progress_callback=None) -> bool:
        self.temp_dir = tempfile.gettempdir()
        
        piper_exe = os.path.join(os.getcwd(), 'bin', 'piper', 'piper.exe')
        if not os.path.exists(piper_exe):
            if progress_callback: progress_callback(0, f"Không tìm thấy piper.exe tại {piper_exe}")
            return False
            
        model_name = self.config.get('model', '')
        if not model_name:
            if progress_callback: progress_callback(0, "Chưa cấu hình giọng đọc TTS.")
            return False
            
        model_path = os.path.join(os.getcwd(), 'data', 'model', 'tts', model_name)
        config_path = model_path + '.json'
        
        if not os.path.exists(model_path):
            if progress_callback: progress_callback(0, f"Không tìm thấy model: {model_path}")
            return False
            
        if not os.path.exists(config_path):
            if progress_callback: progress_callback(0, f"Không tìm thấy config cho model: {config_path}")
            return False
            
        speed = float(self.config.get('speed', 1.0))
        volume = float(self.config.get('volume', 1.0))
        max_threads = int(self.config.get('threads', 3))
        pitch_shift = self.config.get('pitch_shift', False)
        
        chunks = self._split_text(text, max_threads)
        total = len(chunks)
        all_audio = [None] * total
        sr_final = 22050
        completed = 0
        
        if progress_callback: progress_callback(0, f"Bắt đầu tổng hợp {total} đoạn ({max_threads} luồng)...")
        
        self.is_running = True
        self.running_processes = []
        import concurrent.futures

        def process_chunk(idx, chunk):
            if not self.is_running:
                return None, None
                
            f_in = os.path.join(self.temp_dir, f"in_{idx}.txt")
            f_out = os.path.join(self.temp_dir, f"out_{idx}.wav")
            
            # Clean up old files if any
            for f in [f_in, f_out]:
                try: 
                    if os.path.exists(f): os.remove(f)
                except: pass
                
            with open(f_in, "w", encoding="utf-8") as f:
                f.write(chunk)
                
            cmd = [
                piper_exe,
                "--model", model_path,
                "--config", config_path,
                "--output_file", f_out
            ]
            
            kwargs = {}
            if os.name == 'nt':
                kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
                
            with open(f_in, "r", encoding="utf-8") as fin:
                proc = subprocess.Popen(cmd, stdin=fin, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)
                self.running_processes.append(proc)
                proc.wait()
                if proc in self.running_processes:
                    self.running_processes.remove(proc)
                
            try: os.remove(f_in)
            except: pass
            
            if not self.is_running or proc.returncode != 0 or not os.path.exists(f_out):
                return None, None
                
            audio, sr = self._read_wav(f_out)
            try: os.remove(f_out)
            except: pass
                
            audio = (audio * volume).clip(-1.0, 1.0)
            return audio, sr

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            future_to_idx = {executor.submit(process_chunk, i, chunk): i for i, chunk in enumerate(chunks)}
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                audio, sr = future.result()
                if audio is not None:
                    all_audio[idx] = audio
                    sr_final = sr
                completed += 1
                if progress_callback: progress_callback(int((completed/total)*90), f"Đã đọc xong đoạn thứ {idx+1} (Hoàn thành {completed}/{total})")

        if not self.is_running or any(a is None for a in all_audio):
            if progress_callback: progress_callback(0, "Lỗi hoặc đã dừng TTS.")
            return False
            
        if progress_callback: progress_callback(95, "Đang nối file audio...")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined = np.concatenate(all_audio) if all_audio else np.array([])
        i16  = np.clip(combined * 32767, -32768, 32767).astype(np.int16)
        nch  = 1 if i16.ndim == 1 else i16.shape[1]
        
        sweet_voice = self.config.get("sweet_voice", False)
        
        needs_post_process = abs(speed - 1.0) > 0.01 or sweet_voice
        temp_combined = os.path.join(self.temp_dir, "temp_combined.wav")
        target_save_path = temp_combined if needs_post_process else output_path
        
        with wave.open(target_save_path, "wb") as wf:
            wf.setnchannels(nch)
            wf.setsampwidth(2)
            wf.setframerate(sr_final)
            wf.writeframes(i16.tobytes())
            
        if needs_post_process:
            if progress_callback: progress_callback(98, "Đang Hậu kỳ FFmpeg (Chỉnh Tốc độ / Bộ lọc)...")
            ffmpeg_path = os.path.join(os.getcwd(), 'bin', 'ffmpeg.exe')
            
            audio_filter_parts = []
            if abs(speed - 1.0) > 0.01:
                if pitch_shift:
                    new_sr = int(sr_final * speed)
                    audio_filter_parts.append(f"asetrate={new_sr},aresample={sr_final}")
                else:
                    if speed >= 0.5:
                        audio_filter_parts.append(f"atempo={speed}")
                    else:
                        audio_filter_parts.append(f"atempo=0.5,atempo={speed/0.5}")
                        
            if sweet_voice:
                PITCH_RATIO = 1.1289
                new_sr_sweet = int(sr_final * PITCH_RATIO)
                atempo_sweet = 1 / PITCH_RATIO
                
                audio_filter_parts.extend([
                    f"asetrate={new_sr_sweet}",
                    f"aresample={sr_final}",
                    f"atempo={atempo_sweet:.8f}",
                    "equalizer=f=80:t=q:w=0.7:g=-2.3",
                    "equalizer=f=650:t=q:w=0.8:g=0.6",
                    "equalizer=f=1800:t=q:w=0.8:g=0.5",
                    "equalizer=f=7500:t=q:w=0.7:g=-1.8",
                    "equalizer=f=16000:t=q:w=0.7:g=-3.5",
                    "volume=0.9dB"
                ])
                    
            audio_filter = ",".join(audio_filter_parts)
            
            cmd_ff = [
                ffmpeg_path, "-y",
                "-i", temp_combined,
                "-af", audio_filter,
                output_path
            ]
            
            kwargs = {}
            if os.name == 'nt':
                kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
                
            proc = subprocess.run(cmd_ff, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)
            try: os.remove(temp_combined)
            except: pass
            
            if proc.returncode != 0:
                if progress_callback: progress_callback(100, f"Lỗi hậu kỳ FFmpeg: {proc.stderr.decode('utf-8', 'ignore')[-200:]}")
                return False
            
        if progress_callback: progress_callback(100, "Hoàn tất TTS.")
        return True

    def stop(self):
        self.is_running = False
        if hasattr(self, 'running_processes'):
            for p in self.running_processes:
                try: p.kill()
                except: pass
