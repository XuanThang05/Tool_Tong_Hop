import os, sys, time, json, random, glob, shutil, tempfile, re, threading
import subprocess, wave, numpy as np, sqlite3, concurrent.futures
import traceback

try:
    from google import genai
    from google.genai import types
except ImportError:
    pass

# Ensure capcut_tts_api is in path
_libs_path = os.path.join(os.path.dirname(__file__), 'libs')
if _libs_path not in sys.path:
    sys.path.insert(0, _libs_path)

WORK_DIR = "/content/drive/MyDrive/Tool_Tong_Hop_Colab"

# ==================== GEMINI PROCESSOR ====================
class GeminiProcessor:
    def __init__(self, api_keys: list[str]):
        self.api_keys = [k for k in api_keys if k.strip()]
        self.current_key_idx = 0
        self.models = ['gemini-3.1-flash-lite', 'gemini-3.6-flash', 'gemini-1.5-flash', 'gemini-1.5-flash-8b']
        self.current_model_idx = 0
        self.client = None
        self.lock = threading.Lock()
        self._init_client()

    def _init_client(self):
        if not self.api_keys: return
        try:
            self.client = genai.Client(api_key=self.api_keys[self.current_key_idx])
        except Exception as e:
            pass

    def switch_strategy(self):
        with self.lock:
            self.current_model_idx += 1
            if self.current_model_idx >= len(self.models):
                self.current_model_idx = 0
                self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
                self._init_client()

    def process_text(self, prompt: str, text: str, max_retries: int = 15) -> tuple[bool, str]:
        if not self.client:
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
                if any(k in error_str for k in ["429", "quota", "exhausted", "503", "500", "unavailable", "404", "not found", "winerror", "connection", "timeout", "aborted", "reset"]):
                    self.switch_strategy()
                    retries += 1
                    time.sleep(3)
                else:
                    return False, f"Lỗi: Gemini API không thể phục hồi - {str(e)}"
        return False, "Lỗi: Đã thử tất cả các model và key nhưng vẫn thất bại."

    def cleanup_story(self, prompt: str, raw_text: str) -> tuple[bool, str]:
        success, text = self.process_text(prompt, raw_text)
        if success:
            text = text.replace('```', '')
            if text.startswith('json'): text = text[4:]
            text = text.strip()
        return success, text

# ==================== PLAGIARISM CHECKER ====================
class PlagiarismChecker:
    def __init__(self, threshold: int = 80, db_dir: str = None):
        self.threshold = threshold
        self.chunk_similarity_threshold = 0.40 
        self.db_dir = db_dir or os.path.dirname(__file__)
        os.makedirs(self.db_dir, exist_ok=True)
        self.db_path = os.path.join(self.db_dir, "plagiarism.sqlite")
        self.active_collection_id = None
        self.active_collection_name = None
        
        # Cache variables
        self._cached_db_texts = None
        self._cached_vectorizer = None
        self._cached_db_matrix = None
        self._cache_version = -1
        self._current_db_version = 0
        
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('CREATE TABLE IF NOT EXISTS collections (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE)')
            cursor.execute('''CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT, collection_id INTEGER, story_id TEXT, title TEXT, chunk_index INTEGER, text TEXT,
                FOREIGN KEY (collection_id) REFERENCES collections (id))''')
            cursor.execute('SELECT name FROM collections ORDER BY id ASC LIMIT 1')
            row = cursor.fetchone()
            if row: self.set_active_collection(row[0])
            else: self.set_active_collection("stories_collection")

    def get_collections(self) -> list:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT name FROM collections ORDER BY id ASC')
            return [row[0] for row in cursor.fetchall()]

    def set_active_collection(self, name: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('INSERT OR IGNORE INTO collections (name) VALUES (?)', (name,))
            cursor.execute('SELECT id FROM collections WHERE name = ?', (name,))
            self.active_collection_id = cursor.fetchone()[0]
            self.active_collection_name = name
            self._current_db_version += 1

    def delete_collection(self, name: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM collections WHERE name = ?', (name,))
            row = cursor.fetchone()
            if row:
                col_id = row[0]
                cursor.execute('DELETE FROM chunks WHERE collection_id = ?', (col_id,))
                cursor.execute('DELETE FROM collections WHERE id = ?', (col_id,))
        self._current_db_version += 1
        cols = self.get_collections()
        if cols: self.set_active_collection(cols[0])
        else: self.set_active_collection("stories_collection")

    def chunk_text(self, text: str) -> list:
        if not text: return []
        chunks = []
        current_chunk_text = ""
        current_start = 0
        current_word_count = 0
        MIN_WORDS_PER_CHUNK = 200 
        pattern = re.compile(r'[^.!?\\n]+[.!?\\n]*')
        for match in pattern.finditer(text):
            sentence = match.group(0)
            start = match.start()
            end = match.end()
            if not sentence.strip(): continue
            words = sentence.split()
            if current_word_count == 0:
                current_start = start
                current_chunk_text = sentence
            else:
                current_chunk_text = text[current_start:end]
            current_word_count += len(words)
            if current_word_count >= MIN_WORDS_PER_CHUNK:
                chunks.append({"text": current_chunk_text.strip(), "start": current_start, "end": end})
                current_word_count = 0
        if current_word_count > 0:
            if current_word_count < 10 and chunks:
                chunks[-1]["end"] = end
                chunks[-1]["text"] = text[chunks[-1]["start"]:end].strip()
            else:
                chunks.append({"text": current_chunk_text.strip(), "start": current_start, "end": len(text)})
        return chunks

    def check_plagiarism(self, text: str) -> tuple:
        if not self.active_collection_id: return False, 0.0
        chunks = self.chunk_text(text)
        if not chunks: return False, 0.0
        query_texts = [chunk["text"] for chunk in chunks]
        
        try:
            if self._cache_version != self._current_db_version:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('SELECT text FROM chunks WHERE collection_id = ?', (self.active_collection_id,))
                    db_rows = cursor.fetchall()
                if not db_rows: return False, 0.0
                
                db_texts = [row[0] for row in db_rows]
                from sklearn.feature_extraction.text import TfidfVectorizer
                
                self._cached_vectorizer = TfidfVectorizer(ngram_range=(1, 3), analyzer='word')
                self._cached_db_matrix = self._cached_vectorizer.fit_transform(db_texts)
                self._cache_version = self._current_db_version

            from sklearn.metrics.pairwise import cosine_similarity
            query_matrix = self._cached_vectorizer.transform(query_texts)
            similarity_matrix = cosine_similarity(query_matrix, self._cached_db_matrix)
            
            matched_chunks = 0
            for i in range(len(chunks)):
                if np.max(similarity_matrix[i]) >= self.chunk_similarity_threshold:
                    matched_chunks += 1
            percent = (matched_chunks / len(chunks)) * 100
            return percent >= self.threshold, round(percent, 2)
        except Exception as e:
            return False, 0.0

    def get_all_stories(self) -> list:
        if not self.active_collection_id: return []
        stories = {}
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT story_id, title, text FROM chunks WHERE collection_id = ? ORDER BY chunk_index ASC', (self.active_collection_id,))
            rows = cursor.fetchall()
            for row in rows:
                story_id, title, text = row
                if story_id not in stories:
                    preview = text[:100].replace('\\n', ' ') + "..." if text else ""
                    stories[story_id] = {"story_id": story_id, "title": title, "preview": preview, "chunk_count": 1}
                else: stories[story_id]["chunk_count"] += 1
        return list(stories.values())

    def delete_story(self, story_id: str):
        if not self.active_collection_id: return
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM chunks WHERE collection_id = ? AND (story_id = ? OR title = ?)', 
                           (self.active_collection_id, story_id, story_id))
        self._current_db_version += 1

    def get_story_text(self, story_id: str) -> str:
        if not self.active_collection_id: return ""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT text FROM chunks WHERE collection_id = ? AND (story_id = ? OR title = ?) ORDER BY chunk_index ASC', 
                           (self.active_collection_id, story_id, story_id))
            rows = cursor.fetchall()
            if not rows: return ""
            return "\\n\\n".join([row[0] for row in rows])

    def add_story(self, title: str, text: str):
        if not self.active_collection_id: return
        chunks = self.chunk_text(text)
        if not chunks: return
        story_id = str(uuid.uuid4())
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for i, chunk in enumerate(chunks):
                cursor.execute('INSERT INTO chunks (collection_id, story_id, title, chunk_index, text) VALUES (?, ?, ?, ?, ?)', 
                               (self.active_collection_id, story_id, title, i, chunk["text"]))
        self._current_db_version += 1

# ==================== PIPER TTS ====================
class PiperTTSGenerator:
    def __init__(self, config=None, model_dir=None):
        self.config = config or {}
        self.is_running = False
        self.model_dir = model_dir or os.path.join(WORK_DIR, 'models', 'tts')
        os.makedirs(self.model_dir, exist_ok=True)
        
    def _split_text(self, text, max_threads=1):
        total_length = len(text.strip())
        ideal_chunk_size = total_length // max_threads if max_threads > 0 else total_length
        max_chars = max(300, min(ideal_chunk_size, 3000))
        sents = re.split(r'(?<=[.!?\n])\s+', text.strip())
        chunks, cur = [], ""
        for s in sents:
            if len(cur) + len(s) + 1 <= max_chars:
                cur = (cur + " " + s).strip()
            else:
                if cur: chunks.append(cur)
                cur = s
        if cur: chunks.append(cur)
        return [c for c in chunks if c.strip()]
        
    def _read_wav(self, path):
        with wave.open(path, "rb") as wf:
            sr, nch, sw = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
            data = wf.readframes(wf.getnframes())
        dt = {1: np.int8, 2: np.int16, 4: np.int32}.get(sw, np.int16)
        audio = (np.frombuffer(data, dtype=dt).astype(np.float32) / float(np.iinfo(dt).max))
        if nch > 1: audio = audio.reshape(-1, nch)
        return audio, sr

    def generate(self, text: str, output_path: str, progress_callback=None) -> bool:
        self.temp_dir = tempfile.gettempdir()
        model_name = self.config.get('model', '')
        if not model_name: return False
        
        model_path = os.path.join(self.model_dir, model_name)
        config_path = model_path + '.json'
        piper_exe = 'piper' 
        
        speed = float(self.config.get('speed', 1.0))
        volume = float(self.config.get('volume', 1.0))
        max_threads = int(self.config.get('threads', 3))
        pitch_shift = self.config.get('pitch_shift', False)
        sweet_voice = self.config.get('sweet_voice', False)
        
        chunks = self._split_text(text, max_threads)
        total = len(chunks)
        all_audio = [None] * total
        self.is_running = True
        completed = 0

        def process_chunk(idx, chunk):
            if not self.is_running: return None, None
            f_in = os.path.join(self.temp_dir, f"in_{idx}.txt")
            f_out = os.path.join(self.temp_dir, f"out_{idx}.wav")
            for f in [f_in, f_out]:
                if os.path.exists(f): os.remove(f)
            
            with open(f_in, 'w', encoding='utf-8') as f:
                f.write(chunk)
                
            cmd = [piper_exe, '-m', model_path, '-c', config_path, '-f', f_out]
            with open(f_in, 'r', encoding='utf-8') as f:
                try: subprocess.run(cmd, stdin=f, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except: return None, None
                    
            if not os.path.exists(f_out): return None, None
            
            audio, sr = self._read_wav(f_out)
            try: os.remove(f_out); os.remove(f_in)
            except: pass
            
            audio = (audio * volume).clip(-1.0, 1.0)
            return audio, sr

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = {executor.submit(process_chunk, i, c): i for i, c in enumerate(chunks)}
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                audio, sr = future.result()
                if audio is not None:
                    all_audio[idx] = audio
                    sr_final = sr
                completed += 1
                if progress_callback: progress_callback(f"Đã đọc TTS xong {completed}/{total} phần.")

        if not self.is_running or any(a is None for a in all_audio): return False
        
        concat_audio = np.concatenate(all_audio) if all_audio else np.array([])
        i16 = np.clip(concat_audio * 32767, -32768, 32767).astype(np.int16)
        nch = 1 if i16.ndim == 1 else i16.shape[1]
        
        needs_post = abs(speed - 1.0) > 0.01 or sweet_voice
        temp_combined = os.path.join(self.temp_dir, "temp_combined.wav")
        target_save = temp_combined if needs_post else output_path
        
        with wave.open(target_save, "wb") as wf:
            wf.setnchannels(nch)
            wf.setsampwidth(2)
            wf.setframerate(sr_final)
            wf.writeframes(i16.tobytes())
            
        if needs_post:
            if progress_callback: progress_callback("Đang Hậu kỳ FFmpeg (Tốc độ / Tone giọng)...")
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
                    f"asetrate={new_sr_sweet}", f"aresample={sr_final}", f"atempo={atempo_sweet:.8f}",
                    "equalizer=f=80:t=q:w=0.7:g=-2.3", "equalizer=f=650:t=q:w=0.8:g=0.6",
                    "equalizer=f=1800:t=q:w=0.8:g=0.5", "equalizer=f=7500:t=q:w=0.7:g=-1.8",
                    "equalizer=f=16000:t=q:w=0.7:g=-3.5", "volume=0.9dB"
                ])
                    
            cmd_ff = ['ffmpeg', '-y', '-i', temp_combined, '-af', ",".join(audio_filter_parts), output_path]
            proc = subprocess.run(cmd_ff, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try: os.remove(temp_combined)
            except: pass
            if proc.returncode != 0: return False
            
        return True

# ==================== VIDEO ASSEMBLER ====================
class VideoAssembler:
    def __init__(self, output_dir, config):
        self.output_dir = output_dir
        self.config = config
        self.ffmpeg_path = 'ffmpeg'
        self.ffprobe_path = 'ffprobe'

    def get_duration(self, filepath):
        try:
            cmd = [self.ffprobe_path, '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', filepath]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return float(result.stdout.strip())
        except: return 0.0

    def get_bg_videos(self, bg_folder):
        extensions = ('*.mp4', '*.mkv', '*.mov', '*.avi')
        files = []
        for ext in extensions: files.extend(glob.glob(os.path.join(bg_folder, ext)))
        if files: random.shuffle(files)
        return files

    def ms_to_ass_time(self, ms):
        s, ms = divmod(int(ms), 1000)
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        cs = ms // 10
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

    def hex_to_ass_color(self, hex_str):
        hex_str = hex_str.lstrip('#')
        if len(hex_str) == 6:
            r, g, b = hex_str[0:2], hex_str[2:4], hex_str[4:6]
            return f"&H00{b}{g}{r}"
        return "&H00FFFFFF"

    def generate_ass_subtitle(self, audio_path, output_ass):
        try:
            from capcut_tts_api import CapCutClient
            client = CapCutClient()
            query_res = client.transcribe_file(audio_path, language="vi-VN", timeout=3600.0)
            subtitles = client.extract_subtitles(query_res)
            
            font = self.config.get('sub_font', 'Arial')
            size = self.config.get('sub_size', 48)
            color = self.hex_to_ass_color(self.config.get('sub_color', '#FFFFFF'))
            border_color = self.hex_to_ass_color(self.config.get('sub_border_color', '#000000'))
            border_size = self.config.get('sub_border_size', 4)
            if not self.config.get('sub_border_enable', True): border_size = 0
                
            bold = -1 if self.config.get('sub_bold', False) else 0
            italic = -1 if self.config.get('sub_italic', False) else 0
            
            box = self.config.get('sub_box', False)
            box_opacity = self.config.get('sub_box_opacity', 80)
            alpha_val = int(255 * (100 - box_opacity) / 100)
            alpha_hex = f"{alpha_val:02X}"
            box_color = border_color.replace("&H00", f"&H{alpha_hex}")
            border_style = 3 if box else 1
            back_color = box_color if box else "&H00000000"
            
            margin_v = self.config.get('sub_margin_v', 50)
            align_idx = self.config.get('sub_alignment', 0)
            alignment = 2
            if align_val == "Giữa (Middle)": alignment = 5
            elif align_val == "Trên (Top)": alignment = 8
            
            header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{size},{color},&H000000FF,{border_color},{back_color},{bold},{italic},0,0,100,100,0,0,{border_style},{border_size},0,{alignment},10,10,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
            with open(output_ass, 'w', encoding='utf-8') as f:
                f.write(header)
                merged_utterances = []
                current_u = None
                for item in subtitles.utterances:
                    if current_u is None:
                        current_u = {'start_time': item.start_time, 'end_time': item.end_time, 'text': item.text.strip()}
                    else:
                        gap = item.start_time - current_u['end_time']
                        if gap < 500 and len(current_u['text'].split()) + len(item.text.split()) <= 10:
                            current_u['end_time'] = item.end_time
                            current_u['text'] += " " + item.text.strip()
                        else:
                            merged_utterances.append(current_u)
                            current_u = {'start_time': item.start_time, 'end_time': item.end_time, 'text': item.text.strip()}
                if current_u is not None: merged_utterances.append(current_u)

                for item in merged_utterances:
                    start = self.ms_to_ass_time(item['start_time'])
                    end = self.ms_to_ass_time(item['end_time'])
                    text = item['text'].replace('\n', '\\N')
                    f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n")
            return True
        except Exception as e:
            return False

    def run(self, audio_path, progress_callback=None):
        bg_folder = self.config.get('bg_folder', '')
        static_bg_enable = self.config.get('static_bg_enable', False)
        static_bg_file = self.config.get('static_bg_file', '')
        is_static = static_bg_enable and os.path.exists(static_bg_file)
        is_image = False
        if is_static: is_image = static_bg_file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))

        zoom = self.config.get('zoom', 100) / 100.0
        opacity = self.config.get('opacity', 100) / 100.0

        target_duration = self.get_duration(audio_path)
        if target_duration <= 0: return False

        selected = []
        concat_txt_path = os.path.join(self.output_dir, 'bg_concat.txt')
        
        if not is_static:
            bg_files = self.get_bg_videos(bg_folder)
            accumulated = 0.0
            for v in bg_files:
                dur = self.get_duration(v)
                if dur <= 0: continue
                selected.append(v)
                accumulated += dur
                if accumulated >= target_duration: break
            
            with open(concat_txt_path, 'w', encoding='utf-8') as f:
                for v in selected: f.write(f"file '{v.replace('\"', '')}'\n")

        ass_path = os.path.join(self.output_dir, 'subtitle.ass')
        has_sub = self.config.get('sub_enable', False) and os.path.exists(ass_path)
        
        has_effect = self.config.get('effect_enable', False)
        effect_path = self.config.get('effect_video_path', '')
        if has_effect and not os.path.exists(effect_path): has_effect = False
        
        final_output = os.path.join(self.output_dir, '05_final.mp4')
        cmd_final = ['ffmpeg', '-y']
        
        if is_static:
            if is_image: cmd_final.extend(['-loop', '1', '-t', str(target_duration + 2.0), '-i', static_bg_file])
            else: cmd_final.extend(['-i', static_bg_file])
        else:
            cmd_final.extend(['-f', 'concat', '-safe', '0', '-i', concat_txt_path])
            
        curr_idx = 1
        idx_eff = -1
        idx_a = -1
        
        if has_effect:
            cmd_final.extend(['-stream_loop', '-1', '-i', effect_path])
            idx_eff = curr_idx
            curr_idx += 1
            
        cmd_final.extend(['-i', audio_path])
        idx_a = curr_idx
        
        # Build Filter Complex
        bg_filter_parts = []
        if is_static and not is_image: bg_filter_parts.append("[0:v]tpad=stop=-1:stop_mode=clone,fps=30")
        else: bg_filter_parts.append("[0:v]fps=30")
            
        bg_filter_parts.extend(["scale=1920:1080:force_original_aspect_ratio=increase", "crop=1920:1080"])
        if abs(zoom - 1.0) > 0.01:
            bg_filter_parts.append(f"scale=iw*{zoom}:ih*{zoom}")
            if zoom < 1.0: bg_filter_parts.append("pad=1920:1080:(ow-iw)/2:(oh-ih)/2")
            else: bg_filter_parts.append("crop=1920:1080")
                
        bg_filter_parts.append(f"colorchannelmixer=rr={opacity}:gg={opacity}:bb={opacity}[bg_out]")
        bg_filter = ",".join(bg_filter_parts)
        
        filter_parts = [bg_filter]
        last_v = "bg_out"
        
        if has_effect:
            filter_parts.append(f"[{idx_eff}:v]scale=1920:1080[eff_scaled]; [{last_v}][eff_scaled]blend=all_mode='screen':shortest=1[eff_out]")
            last_v = "eff_out"
        
        if has_sub:
            safe_ass = ass_path.replace('\\', '/').replace(':', '\\:').replace("'", "\\'")
            filter_parts.append(f"[{last_v}]subtitles=filename='{safe_ass}'[outv]")
            last_v = "outv"
        else:
            filter_parts.append(f"[{last_v}]copy[outv]")
            
        filter_complex = "; ".join(filter_parts)
        
        cmd_final.extend([
            '-filter_complex', filter_complex,
            '-map', '[outv]', '-map', f'{idx_a}:a',
            '-c:v', 'libx264', '-preset', 'fast', '-pix_fmt', 'yuv420p',
            '-c:a', 'aac', '-b:a', '320k', '-shortest', final_output
        ])
        
        p = subprocess.Popen(cmd_final, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = p.communicate()
        if p.returncode != 0: return False
        return True

# ==================== ENGINE (Generator for Gradio) ====================
def run_pipeline(config):
    output_dir_base = os.path.join(WORK_DIR, "outputs")
    os.makedirs(output_dir_base, exist_ok=True)
    
    yield "🚀 Bắt đầu quá trình..."
    files = config.get('input_files', [])
    
    if not files:
        yield "❌ Lỗi: Bạn chưa tải lên file .txt nào."
        return
        
    chapters = []
    for f in sorted(files):
        with open(f, 'r', encoding='utf-8') as file:
            chapters.append({'name': os.path.basename(f), 'raw_content': file.read()})
            
    merged_raw = "\n\n".join([c["raw_content"] for c in chapters])
    safe_title = os.path.splitext(chapters[0]["name"])[0] + "_" + time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(output_dir_base, safe_title)
    os.makedirs(out_dir, exist_ok=True)
    yield f"✅ Đã tải {len(chapters)} chương."
    
    # Plagiarism
    if config.get("plagiarism_check", True):
        yield "🔍 Đang kiểm tra đạo văn..."
        checker = PlagiarismChecker(threshold=config.get("plag_threshold", 80), db_dir=os.path.join(WORK_DIR, "data"))
        is_plag, pct = checker.check_plagiarism(merged_raw)
        yield f"Tỉ lệ trùng lặp: {pct}%"
        if is_plag:
            yield "❌ Hủy bỏ: Đạo văn vượt ngưỡng!"
            return
            
    # Gemini
    merged_cleaned = merged_raw
    if config.get("gemini_clean", True):
        yield "🧹 Đang dọn dẹp bằng Gemini..."
        keys = config.get("api_keys", "").split('\n')
        processor = GeminiProcessor(keys)
        prompt = config.get("prompt", "")
        
        for c in chapters:
            yield f"  -> Xử lý {c['name']}..."
            succ, txt = processor.cleanup_story(prompt, c['raw_content'])
            if not succ:
                yield f"❌ Lỗi Gemini tại {c['name']}. Hủy bỏ!"
                return
            c['cleaned_content'] = txt
            
        merged_cleaned = "\n\n".join([c["cleaned_content"] for c in chapters])
        yield "✅ Đã dọn dẹp xong."
        
    if config.get("plagiarism_check", True):
        checker.add_story(safe_title, merged_cleaned)
        
    # TTS
    audio_path = os.path.join(out_dir, "audio.wav")
    if config.get("tts", True):
        yield "🎙️ Đang tạo âm thanh TTS (Piper)..."
        tts_cfg = {
            'model': config.get("tts_model", ""),
            'threads': config.get("tts_threads", 3),
            'speed': config.get("tts_speed", 1.0),
            'volume': config.get("tts_volume", 1.0),
            'sweet_voice': config.get("tts_sweet", False),
            'pitch_shift': config.get("tts_pitch", False)
        }
        tts = PiperTTSGenerator(tts_cfg, model_dir=os.path.join(WORK_DIR, "models", "tts"))
        def tts_prog(msg): yield "   - " + msg
        succ = tts.generate(merged_cleaned, audio_path, tts_prog)
        if not succ:
            yield "❌ Lỗi khi tạo TTS."
            return
        yield "✅ TTS Hoàn tất."
        
    # Assembler
    if config.get("video_assemble", True):
        yield "🎥 Bắt đầu ghép Video..."
        asm_cfg = config
        asm = VideoAssembler(out_dir, asm_cfg)
        
        ass_path = os.path.join(out_dir, "subtitle.ass")
        if asm_cfg.get('sub_enable', True):
            yield "📝 Đang trích xuất phụ đề (CapCut STT)..."
            asm.generate_ass_subtitle(audio_path, ass_path)
            
        yield "🎞️ Đang render FFmpeg (Video + Sub + Effects)..."
        succ = asm.run(audio_path)
        if succ:
            yield "✅ Render thành công!"
        else:
            yield "❌ Lỗi Render FFmpeg."
            
    yield "🎉 HOÀN TẤT QUÁ TRÌNH!"


# ==================== GRADIO UI ====================
CONFIG_FILE = os.path.join(WORK_DIR, "colab_settings.json")

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except: return {}
    return {}

def save_config(cfg):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=4)
    except: pass

def install_and_get_fonts():
    import subprocess, shutil
    fonts_dir = os.path.join(WORK_DIR, "fonts")
    if os.path.exists(fonts_dir):
        dest = os.path.expanduser("~/.fonts")
        os.makedirs(dest, exist_ok=True)
        for f in os.listdir(fonts_dir):
            if f.lower().endswith(('.ttf', '.otf')):
                shutil.copy(os.path.join(fonts_dir, f), dest)
        subprocess.run(['fc-cache', '-fv'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
    try:
        out = subprocess.check_output(['fc-list', ':', 'family']).decode('utf-8')
        fonts = set()
        for line in out.split('\n'):
            for f in line.split(','):
                f = f.strip()
                if f and not f.startswith(':'): fonts.add(f)
        fonts = sorted(list(fonts))
        return fonts if fonts else ["Arial"]
    except: return ["Arial"]

def get_files(folder, ext=None):

    full_path = os.path.join(WORK_DIR, folder)
    if not os.path.exists(full_path): os.makedirs(full_path, exist_ok=True); return []
    res = []
    for f in os.listdir(full_path):
        if ext:
            if any(f.endswith(e) for e in ext): res.append(f)
        else: res.append(f)
    return sorted(res)

def get_models(): return get_files("models/tts", [".onnx"])
def get_effects(): return get_files("effects", [".mp4", ".mov", ".webm", ".avi"])


# Global DB for UI management
global_plagiarism_db = PlagiarismChecker(db_dir=os.path.join(WORK_DIR, "data"))

def start_ui():

    import gradio as gr
    import tempfile
    
    cfg_saved = load_config()
    fonts_list = install_and_get_fonts()
    if cfg_saved.get('sub_font', 'Arial') not in fonts_list:
        fonts_list.insert(0, cfg_saved.get('sub_font', 'Arial'))
    
    def generate_preview(font, size, color, border_color, border_enable, border_size, bold, italic, box, box_opacity, align_val, margin_v, static_file):
        try:
            asm = VideoAssembler("", {})
            color_ass = asm.hex_to_ass_color(color)
            border_ass = asm.hex_to_ass_color(border_color)
            if not border_enable: border_size = 0
            
            b = -1 if bold else 0
            i = -1 if italic else 0
            
            alpha_val = int(255 * (100 - box_opacity) / 100)
            alpha_hex = f"{alpha_val:02X}"
            box_color = border_ass.replace("&H00", f"&H{alpha_hex}")
            border_style = 3 if box else 1
            back_color = box_color if box else "&H00000000"
            
            alignment = 2
            if align_val == "Giữa (Middle)": alignment = 5
            elif align_val == "Trên (Top)": alignment = 8
            
            header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{size},{color_ass},&H000000FF,{border_ass},{back_color},{b},{i},0,0,100,100,0,0,{border_style},{border_size},0,{alignment},10,10,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:10.00,Default,,0,0,0,,Đây là dòng Phụ đề\\NXem trước (Preview)
"""
            tdir = tempfile.gettempdir()
            ass_path = os.path.join(tdir, 'preview.ass')
            out_img = os.path.join(tdir, 'preview.jpg')
            with open(ass_path, 'w', encoding='utf-8') as f:
                f.write(header)
                
            safe_ass = ass_path.replace('\\', '/').replace(':', '\\:').replace("'", "\\'")
            
            if static_file and hasattr(static_file, 'name') and os.path.exists(static_file.name):
                bg_input = static_file.name
                vf = f"scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,subtitles=filename='{safe_ass}'"
                cmd = ['ffmpeg', '-y', '-i', bg_input, '-vf', vf, '-vframes', '1', '-q:v', '2', out_img]
            else:
                cmd = ['ffmpeg', '-y', '-f', 'lavfi', '-i', 'color=c=black:s=1920x1080', '-vf', f"subtitles=filename='{safe_ass}'", '-vframes', '1', out_img]
                
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(out_img): return out_img
            return None
        except: return None

    with gr.Blocks(title="Tool Tổng Hợp Truyện (Colab)") as ui:
        gr.Markdown("# 🚀 Tool Tổng Hợp Truyện - Full Edition")
        
        with gr.Tabs():
            with gr.Tab("1. Chung & Gemini"):
                gr.Markdown("### Kiểm tra Đạo văn")
                plag_threshold = gr.Slider(0, 100, value=cfg_saved.get('plag_threshold', 80), label="Ngưỡng bỏ qua Đạo văn (%)")
                
                gr.Markdown("### Input & Xử lý Text")
                input_files = gr.File(label="Tải lên file Truyện (.txt)", file_count="multiple", file_types=[".txt"])
                api_keys = gr.Textbox(label="Gemini API Keys (Mỗi dòng 1 key)", lines=4, value=cfg_saved.get('api_keys', ''))
                prompt = gr.Textbox(label="Prompt Dọn dẹp", lines=3, value=cfg_saved.get('prompt', 'Viết lại câu chuyện sau một cách lôi cuốn.'))
                gemini_threads = gr.Slider(1, 20, value=cfg_saved.get('gemini_threads', 3), step=1, label="Số luồng API Gemini")
                
            with gr.Tab("2. Đọc truyện (TTS)"):
                with gr.Row():
                    tts_model = gr.Dropdown(choices=get_models(), label="Chọn Model Voice (.onnx)", allow_custom_value=True, value=cfg_saved.get('tts_model', ''))
                    btn_ref_tts = gr.Button("🔄 Làm mới Voice", size="sm")
                tts_speed = gr.Slider(0.5, 2.0, value=cfg_saved.get('tts_speed', 1.0), step=0.01, label="Tốc độ (Speed)")
                tts_volume = gr.Slider(0.0, 5.0, value=cfg_saved.get('tts_volume', 1.0), step=0.01, label="Âm lượng (Volume)")
                tts_pitch = gr.Checkbox(label="Đổi tông tự nhiên (Pitch Shift)", value=cfg_saved.get('tts_pitch', False))
                tts_sweet = gr.Checkbox(label="Giọng Ngọt Ngào (Sweet Voice)", value=cfg_saved.get('tts_sweet', False))
                tts_threads = gr.Slider(1, 50, value=cfg_saved.get('tts_threads', 3), step=1, label="Số luồng TTS")

            with gr.Tab("3. Video & Phụ đề"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Cài đặt Nền")
                        bg_folder = gr.Textbox(label="Thư mục Video nền (Trường hợp chạy nối file)", value=cfg_saved.get('bg_folder', ''))
                        static_bg_enable = gr.Checkbox(label="Dùng 1 Ảnh/Video nền tĩnh (Thay vì nối file)", value=cfg_saved.get('static_bg_enable', False))
                        static_bg_file = gr.File(label="Tải lên Ảnh/Video tĩnh", file_count="single", file_types=["image", "video"])
                        
                        effect_enable = gr.Checkbox(label="Bật Hiệu ứng lặp vòng (Overlay)", value=cfg_saved.get('effect_enable', False))
                        with gr.Row():
                            effect_file = gr.Dropdown(choices=get_effects(), label="Chọn file Hiệu ứng", allow_custom_value=True, value=cfg_saved.get('effect_file_name', ''))
                            btn_ref_eff = gr.Button("🔄 Làm mới", size="sm")
                            
                        zoom = gr.Slider(50, 200, value=cfg_saved.get('zoom', 100), step=1, label="Zoom nền (%)")
                        opacity = gr.Slider(10, 100, value=cfg_saved.get('opacity', 100), step=1, label="Độ sáng nền (%)")
                        
                    with gr.Column():
                        gr.Markdown("### Cài đặt Phụ đề (CapCut)")
                        sub_enable = gr.Checkbox(label="Bật Phụ đề", value=cfg_saved.get('sub_enable', True))
                        
                        with gr.Row():
                            sub_font = gr.Dropdown(choices=fonts_list, label="Font chữ", allow_custom_value=True, value=cfg_saved.get('sub_font', 'Arial'))
                            btn_ref_font = gr.Button("🔄 Cập nhật Font từ Drive", size="sm")
                            
                        sub_size = gr.Slider(10, 150, value=cfg_saved.get('sub_size', 48), step=1, label="Cỡ chữ")
                        
                        with gr.Row():
                            sub_color = gr.ColorPicker(label="Màu chữ", value=cfg_saved.get('sub_color', '#FFFFFF'))
                            sub_border_color = gr.ColorPicker(label="Màu viền", value=cfg_saved.get('sub_border_color', '#000000'))
                            
                        sub_border_enable = gr.Checkbox(label="Bật viền chữ", value=cfg_saved.get('sub_border_enable', True))
                        sub_border_size = gr.Slider(1, 10, value=cfg_saved.get('sub_border_size', 4), step=1, label="Độ dày viền")
                        
                        with gr.Row():
                            sub_bold = gr.Checkbox(label="In đậm", value=cfg_saved.get('sub_bold', False))
                            sub_italic = gr.Checkbox(label="In nghiêng", value=cfg_saved.get('sub_italic', False))
                            
                        sub_box = gr.Checkbox(label="Nền hộp chữ (Box)", value=cfg_saved.get('sub_box', False))
                        sub_box_opacity = gr.Slider(0, 100, value=cfg_saved.get('sub_box_opacity', 80), step=1, label="Độ mờ Box (%)")
                        
                        sub_alignment = gr.Radio(choices=["Dưới (Bottom)", "Giữa (Middle)", "Trên (Top)"], label="Vị trí Phụ đề", value=cfg_saved.get('sub_alignment', 'Dưới (Bottom)'))
                        sub_margin_v = gr.Slider(0, 500, value=cfg_saved.get('sub_margin_v', 50), step=1, label="Khoảng cách lề dọc")
                        
                        btn_preview_sub = gr.Button("👀 XEM TRƯỚC PHỤ ĐỀ (PREVIEW)")
                        img_preview = gr.Image(label="Ảnh xem trước Phụ đề", interactive=False)
                        
                        btn_preview_sub.click(
                            generate_preview,
                            inputs=[sub_font, sub_size, sub_color, sub_border_color, sub_border_enable, sub_border_size, sub_bold, sub_italic, sub_box, sub_box_opacity, sub_alignment, sub_margin_v, static_bg_file],
                            outputs=img_preview
                        )


            with gr.Tab("4. Quản lý Đạo văn (DB)"):
                gr.Markdown("### 📚 Kho Dữ liệu Chống Đạo văn")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        btn_refresh_db = gr.Button("🔄 Tải lại Danh sách Truyện")
                        db_table = gr.Dataframe(headers=["ID", "Tiêu đề", "Preview", "Số đoạn"], interactive=False)
                        
                    with gr.Column(scale=1):
                        gr.Markdown("#### Chi tiết Truyện")
                        selected_id = gr.Textbox(label="ID Truyện đang chọn", interactive=False)
                        selected_content = gr.Textbox(label="Nội dung Truyện", lines=10, interactive=False)
                        btn_delete_story = gr.Button("🗑️ Xóa Truyện này", variant="stop")
                        
                gr.Markdown("---")
                gr.Markdown("### ➕ Thêm Truyện Thủ Công")
                with gr.Row():
                    with gr.Column(scale=2):
                        manual_text = gr.Textbox(label="Nội dung Truyện", lines=5)
                    with gr.Column(scale=1):
                        manual_title = gr.Textbox(label="Tiêu đề Truyện")
                        btn_add_story = gr.Button("Thêm vào Kho", variant="primary")
                        add_status = gr.Markdown()
                        
        with gr.Row():
            start_btn = gr.Button("▶️ BẮT ĐẦU CHẠY TOÀN BỘ", variant="primary", size="lg")
        log_box = gr.Textbox(label="Tiến độ (Logs)", lines=12, interactive=False)
        
        # Events
        def refresh_models(): return gr.update(choices=get_models())
        def refresh_effects(): return gr.update(choices=get_effects())
        def refresh_fonts():
            flist = install_and_get_fonts()
            return gr.update(choices=flist)
            
        btn_ref_tts.click(refresh_models, outputs=tts_model)
        btn_ref_eff.click(refresh_effects, outputs=effect_file)
        btn_ref_font.click(refresh_fonts, outputs=sub_font)


        # Events DB
        def load_stories():
            try:
                stories = global_plagiarism_db.get_all_stories()
                return [[s['story_id'], s['title'], s['preview'], s['chunk_count']] for s in stories]
            except Exception as e:
                return []

        def on_select_story(evt: gr.SelectData, data):
            try:
                row_idx = evt.index[0]
                if hasattr(data, 'iloc'): story_id = data.iloc[row_idx, 0]
                else: story_id = data[row_idx][0]
                content = global_plagiarism_db.get_story_text(story_id)
                return story_id, content
            except: return "", ""

        def delete_story_action(story_id):
            if story_id: global_plagiarism_db.delete_story(story_id)
            return load_stories(), "", ""

        def add_manual_action(title, text):
            if not text.strip(): return "Lỗi: Nội dung trống!", load_stories()
            title = title.strip() or f"Manual_{uuid.uuid4().hex[:6]}"
            global_plagiarism_db.add_story(title, text)
            return f"✅ Đã thêm truyện: {title}", load_stories()

        btn_refresh_db.click(load_stories, outputs=db_table)
        db_table.select(on_select_story, inputs=db_table, outputs=[selected_id, selected_content])
        btn_delete_story.click(delete_story_action, inputs=selected_id, outputs=[db_table, selected_id, selected_content])
        btn_add_story.click(add_manual_action, inputs=[manual_title, manual_text], outputs=[add_status, db_table])
        ui.load(load_stories, outputs=db_table)

        def execute(*args):

            cfg = {
                "api_keys": args[1], "prompt": args[2],
                "gemini_threads": args[3], "plag_threshold": args[4],
                "tts_model": args[5], "tts_speed": args[6], "tts_volume": args[7],
                "tts_pitch": args[8], "tts_sweet": args[9], "tts_threads": args[10],
                "bg_folder": args[11], "static_bg_enable": args[12], 
                "effect_enable": args[14], "effect_file_name": args[15],
                "zoom": args[16], "opacity": args[17],
                "sub_enable": args[18], "sub_font": args[19], "sub_size": args[20],
                "sub_color": args[21], "sub_border_color": args[22], "sub_border_enable": args[23],
                "sub_border_size": args[24], "sub_bold": args[25], "sub_italic": args[26],
                "sub_box": args[27], "sub_box_opacity": args[28], "sub_alignment": args[29],
                "sub_margin_v": args[30]
            }
            # Luu truoc khi chay
            save_config(cfg)
            
            # Tiep tuc setup chay
            run_cfg = cfg.copy()
            run_cfg["input_files"] = [f.name for f in args[0]] if args[0] else []
            run_cfg["static_bg_file"] = args[13].name if args[13] else ""
            run_cfg["effect_video_path"] = os.path.join(WORK_DIR, "effects", args[15]) if args[15] else ""
            run_cfg["plagiarism_check"] = True
            run_cfg["gemini_clean"] = True
            run_cfg["tts"] = True
            run_cfg["video_assemble"] = True
            run_cfg["sub_alignment"] = 0 if args[29] == "Dưới (Bottom)" else (1 if args[29] == "Giữa (Middle)" else 2)
            
            logs = ""
            for step in run_pipeline(run_cfg):
                logs += str(step) + "\n"
                yield logs

        start_btn.click(
            execute, 
            inputs=[
                input_files, api_keys, prompt, gemini_threads, plag_threshold,
                tts_model, tts_speed, tts_volume, tts_pitch, tts_sweet, tts_threads,
                bg_folder, static_bg_enable, static_bg_file, effect_enable, effect_file, zoom, opacity,
                sub_enable, sub_font, sub_size, sub_color, sub_border_color, sub_border_enable, sub_border_size,
                sub_bold, sub_italic, sub_box, sub_box_opacity, sub_alignment, sub_margin_v
            ], 
            outputs=log_box
        )
        
    ui.launch()

if __name__ == "__main__":
    start_ui()
