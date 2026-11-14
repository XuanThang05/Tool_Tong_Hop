import os
import subprocess
import json
import random
import glob
import shutil
import sys

# Thêm thư mục libs vào sys.path để import capcut_tts_api
_libs_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'libs')
if _libs_path not in sys.path:
    sys.path.insert(0, _libs_path)

class VideoAssembler:
    def __init__(self, output_dir, config):
        self.output_dir = output_dir
        self.config = config
        self.ffmpeg_path = os.path.join(os.getcwd(), 'bin', 'ffmpeg.exe')
        self.ffprobe_path = os.path.join(os.getcwd(), 'bin', 'ffprobe.exe')

    def get_duration(self, filepath):
        try:
            cmd = [
                self.ffprobe_path, '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', filepath
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
            return float(result.stdout.strip())
        except Exception as e:
            print(f"Error getting duration for {filepath}: {e}")
            return 0.0

    def get_bg_videos(self, bg_folder):
        extensions = ('*.mp4', '*.mkv', '*.mov', '*.avi')
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(bg_folder, ext)))
        
        if files:
            random.shuffle(files)
        return files

    def hex_to_ass_color(self, hex_str):
        hex_str = hex_str.lstrip('#')
        if len(hex_str) == 6:
            r, g, b = hex_str[0:2], hex_str[2:4], hex_str[4:6]
            return f"&H00{b}{g}{r}"
        return "&H00FFFFFF"

    def ms_to_ass_time(self, ms):
        s, ms = divmod(int(ms), 1000)
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        cs = ms // 10
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"
    def generate_ass_subtitle(self, audio_path, output_ass):
        try:
            from capcut_tts_api import CapCutClient
            from capcut_tts_api.models import SubtitleResult
            import math
            
            client = CapCutClient()
            
            total_duration = self.get_duration(audio_path)
            MAX_CHUNK_DUR = 1800.0  # 30 phút (tránh giới hạn 60 phút của CapCut)
            
            subtitles = SubtitleResult()
            
            if total_duration > MAX_CHUNK_DUR:
                import concurrent.futures
                num_chunks = math.ceil(total_duration / MAX_CHUNK_DUR)
                
                def process_chunk(i):
                    start_sec = i * MAX_CHUNK_DUR
                    chunk_dur = min(MAX_CHUNK_DUR, total_duration - start_sec)
                    chunk_path = os.path.join(self.output_dir, f"temp_sub_chunk_{i}.wav")
                    
                    cmd = [
                        self.ffmpeg_path, '-y',
                        '-ss', str(start_sec),
                        '-t', str(chunk_dur),
                        '-i', audio_path,
                        '-c:a', 'copy',
                        chunk_path
                    ]
                    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    
                    chunk_utterances = []
                    if os.path.exists(chunk_path):
                        from capcut_tts_api import CapCutClient
                        local_client = CapCutClient()
                        query_res = local_client.transcribe_file(chunk_path, language="vi-VN", timeout=3600.0)
                        chunk_subs = local_client.extract_subtitles(query_res)
                        
                        offset_ms = int(start_sec * 1000)
                        for u in chunk_subs.utterances:
                            u.start_time += offset_ms
                            u.end_time += offset_ms
                            chunk_utterances.append(u)
                        
                        try:
                            os.remove(chunk_path)
                        except:
                            pass
                    return i, chunk_utterances

                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    futures = [executor.submit(process_chunk, i) for i in range(num_chunks)]
                    results = []
                    for future in concurrent.futures.as_completed(futures):
                        results.append(future.result())
                
                results.sort(key=lambda x: x[0])
                for idx, uts in results:
                    subtitles.utterances.extend(uts)
            else:
                query_res = client.transcribe_file(audio_path, language="vi-VN", timeout=3600.0)
                subtitles = client.extract_subtitles(query_res)

            
            font = self.config.get('sub_font', 'Arial')
            size = self.config.get('sub_size', 24)
            color = self.hex_to_ass_color(self.config.get('sub_color', '#FFFFFF'))
            border_color = self.hex_to_ass_color(self.config.get('sub_border_color', '#000000'))
            border_size = self.config.get('sub_border_size', 2)
            border_enable = self.config.get('sub_border_enable', True)
            if not border_enable:
                border_size = 0
                
            bold = -1 if self.config.get('sub_bold', False) else 0
            italic = -1 if self.config.get('sub_italic', False) else 0
            
            box = self.config.get('sub_box', False)
            box_opacity = self.config.get('sub_box_opacity', 80)
            
            # Alpha tính theo thang 255 (0 = đặc, 255 = trong suốt). 
            # Opacity 100% -> Alpha 00
            # Opacity 10% -> Alpha E5
            alpha_val = int(255 * (100 - box_opacity) / 100)
            alpha_hex = f"{alpha_val:02X}"
            
            # Äá»•i mÃ£ hex (OutlineColour Ä‘ang lÃ  &H00BBGGRR)
            # Ta tạo BoxColor có chứa Alpha
            box_color = border_color.replace("&H00", f"&H{alpha_hex}")
            
            border_style = 3 if box else 1
            back_color = box_color if box else "&H00000000"
            
            # Mới: Lấy MarginV và Alignment từ Config
            margin_v = self.config.get('sub_margin_v', 50)
            align_idx = self.config.get('sub_alignment', 0)
            
            # Map: 0 -> Bottom(2), 1 -> Middle(5), 2 -> Top(8)
            alignment = 2
            if align_idx == 1: alignment = 5
            elif align_idx == 2: alignment = 8
            
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
                max_words = 10 # Số từ tối đa mỗi dòng
                max_gap_ms = 500 # Khoảng cách tối đa (ms) để nối câu
                
                current_u = None
                for item in subtitles.utterances:
                    if current_u is None:
                        current_u = {'start_time': item.start_time, 'end_time': item.end_time, 'text': item.text.strip()}
                    else:
                        gap = item.start_time - current_u['end_time']
                        current_words = len(current_u['text'].split())
                        new_words = len(item.text.strip().split())
                        
                        if gap < max_gap_ms and (current_words + new_words) <= max_words:
                            current_u['end_time'] = item.end_time
                            current_u['text'] += " " + item.text.strip()
                        else:
                            merged_utterances.append(current_u)
                            current_u = {'start_time': item.start_time, 'end_time': item.end_time, 'text': item.text.strip()}
                
                if current_u is not None:
                    merged_utterances.append(current_u)

                for item in merged_utterances:
                    start = self.ms_to_ass_time(item['start_time'])
                    end = self.ms_to_ass_time(item['end_time'])
                    text = item['text'].replace('\n', '\\N')
                    f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n")
            return True
        except Exception as e:
            print(f"Error generating subtitles: {e}")
            return False

    def generate_subtitle_preview_image(self, output_jpg_path):
        """Tạo một file ảnh Preview 1920x1080 chứa Subtitle mẫu để hiển thị trên UI."""
        try:
            # 1. Sinh file .ass mẫu
            temp_ass = os.path.join(self.output_dir, 'preview_sub.ass')
            os.makedirs(self.output_dir, exist_ok=True)
            
            font = self.config.get('sub_font', 'Arial')
            size = self.config.get('sub_size', 24)
            color = self.hex_to_ass_color(self.config.get('sub_color', '#FFFFFF'))
            border_color = self.hex_to_ass_color(self.config.get('sub_border_color', '#000000'))
            border_size = self.config.get('sub_border_size', 2)
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
            if align_idx == 1: alignment = 5
            elif align_idx == 2: alignment = 8
            
            header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{size},{color},&H000000FF,{border_color},{back_color},{bold},{italic},0,0,100,100,0,0,{border_style},{border_size},0,{alignment},10,10,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:10.00,Default,,0,0,0,,Đây là dòng phụ đề xem trước.\\N(Preview Subtitle)
"""
            with open(temp_ass, 'w', encoding='utf-8') as f:
                f.write(header)
                
            # 2. Dùng FFmpeg render 1 frame ảnh đen với phụ đề 
            safe_ass = temp_ass.replace('\\', '/').replace(':', '\\:').replace("'", "\\'")
            cmd = [
                self.ffmpeg_path, '-y',
                '-f', 'lavfi', '-i', 'color=c=black:s=1920x1080',
                '-vframes', '1',
                '-vf', f"subtitles=filename='{safe_ass}'",
                output_jpg_path
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return os.path.exists(output_jpg_path)
            
        except Exception as e:
            print(f"Preview Error: {e}")
            return False

    def run(self, audio_path, waveform_path, progress_callback=None):
        if progress_callback: progress_callback(0, "Bắt đầu lấy video nền...")
        
        bg_folder = self.config.get('bg_folder', '')
        zoom = self.config.get('zoom', 100) / 100.0
        opacity = self.config.get('opacity', 100) / 100.0

        static_bg_enable = self.config.get('static_bg_enable', False)
        static_bg_file = self.config.get('static_bg_file', '')
        is_static = static_bg_enable and os.path.exists(static_bg_file)
        is_image = False
        if is_static:
            is_image = static_bg_file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))

        target_duration = self.get_duration(audio_path)
        if target_duration <= 0:
            if progress_callback: progress_callback(100, "Lỗi: File audio không hợp lệ hoặc lỗi ffprobe.")
            return False

        selected = []
        concat_txt_path = os.path.join(self.output_dir, 'bg_concat.txt')
        
        if not is_static:
            if not os.path.exists(bg_folder):
                if progress_callback: progress_callback(100, "Lỗi: Không tìm thấy thư mục video nền.")
                return False

            bg_files = self.get_bg_videos(bg_folder)
            if not bg_files:
                if progress_callback: progress_callback(100, "Lỗi: Không có video nào trong thư mục nền.")
                return False

            # --- 1. Chọn video nền đủ độ dài ---
            accumulated = 0.0
            for v in bg_files:
                dur = self.get_duration(v)
                if dur <= 0: continue
                selected.append(v)
                accumulated += dur
                if accumulated >= target_duration:
                    break
            
            if accumulated < target_duration:
                if progress_callback: 
                    progress_callback(100, f"Lỗi: Không đủ video nền! (Cần {target_duration:.1f}s nhưng chỉ có {accumulated:.1f}s). Vui lòng thêm video vào thư mục.")
                return False

            if progress_callback: progress_callback(20, f"Đã chọn {len(selected)} video, tổng độ dài {accumulated:.2f}s...")

            # --- 2. Tạo file danh sách nối ---
            with open(concat_txt_path, 'w', encoding='utf-8') as f:
                for v in selected:
                    safe_path = v.replace('\\', '/')
                    f.write(f"file '{safe_path}'\n")

        # KHÔNG TẠO temp_bg.mp4 NỮA ĐỂ TRÁNH LỖI CONTAINER (như VP6 FLV sang MP4)
        # Thay vào đó sẽ đưa trực tiếp concat_txt_path vào filter_complex của FFmpeg ở bước cuối.
        
        # --- 3.5 Kiểm tra Subtitles ---
        sub_enable = self.config.get('sub_enable', False)
        ass_path = os.path.join(self.output_dir, 'subtitle.ass')
        has_sub = False
        
        if sub_enable and os.path.exists(ass_path):
            has_sub = True

        # --- 4. Render Final (Overlay + Effects) ---
        if progress_callback: progress_callback(60, "Đang Overlay sóng âm và hiệu ứng...")
        final_output = os.path.join(self.output_dir, '06_final.mp4')
        
        has_waveform = os.path.exists(waveform_path) if waveform_path else False
        has_effect = self.config.get('effect_enable', False)
        effect_path = self.config.get('effect_video_path', '')
        if has_effect and not os.path.exists(effect_path):
            has_effect = False
            
        cmd_final = [self.ffmpeg_path, '-y']
        
        # 0: Background
        if is_static:
            if is_image:
                cmd_final.extend(['-loop', '1', '-t', str(target_duration + 2.0), '-i', static_bg_file])
            else:
                cmd_final.extend(['-i', static_bg_file])
        else:
            cmd_final.extend(['-f', 'concat', '-safe', '0', '-i', concat_txt_path])
        idx_eff = -1
        idx_wf = -1
        idx_a = -1
        curr_idx = 1
        
        # Effect
        if has_effect:
            cmd_final.extend(['-stream_loop', '-1', '-i', effect_path])
            idx_eff = curr_idx
            curr_idx += 1
            
        # Waveform OR Audio
        if has_waveform:
            cmd_final.extend(['-i', waveform_path])
            idx_wf = curr_idx
            idx_a = curr_idx
            curr_idx += 1
        else:
            cmd_final.extend(['-i', audio_path])
            idx_a = curr_idx
            curr_idx += 1

        # Build Filter Complex
        # 1. Background (Thêm fps=30 để ép Frame Rate cố định, tránh lỗi NVENC -22)
        bg_filter_parts = []
        if is_static and not is_image:
            bg_filter_parts.append("[0:v]tpad=stop=-1:stop_mode=clone,fps=30")
        else:
            bg_filter_parts.append("[0:v]fps=30")
            
        bg_filter_parts.extend([
            "scale=1920:1080:force_original_aspect_ratio=increase",
            "crop=1920:1080"
        ])
        if abs(zoom - 1.0) > 0.01:
            bg_filter_parts.append(f"scale=iw*{zoom}:ih*{zoom}")
            if zoom < 1.0:
                bg_filter_parts.append("pad=1920:1080:(ow-iw)/2:(oh-ih)/2")
            else:
                bg_filter_parts.append("crop=1920:1080")
                
        bg_filter_parts.append(f"colorchannelmixer=rr={opacity}:gg={opacity}:bb={opacity}[bg_out]")
        bg_filter = ",".join(bg_filter_parts)
        filter_parts = [bg_filter]
        last_v = "bg_out"
        
        # 2. Effect Overlay
        if has_effect:
            filter_parts.append(f"[{idx_eff}:v]scale=1920:1080[eff_scaled]; [{last_v}][eff_scaled]blend=all_mode='screen':shortest=1[eff_out]")
            last_v = "eff_out"
            
        # 3. Waveform Overlay
        if has_waveform:
            filter_parts.append(f"[{last_v}][{idx_wf}:v]blend=all_mode='screen':shortest=1[wf_out]")
            last_v = "wf_out"
            
        # 4. Subtitles
        if has_sub and os.path.exists(ass_path):
            safe_ass = ass_path.replace('\\', '/').replace(':', '\\:').replace("'", "\\'")
            filter_parts.append(f"[{last_v}]subtitles=filename='{safe_ass}'[outv]")
            last_v = "outv"
        else:
            filter_parts.append(f"[{last_v}]copy[outv]")
            
        filter_complex = "; ".join(filter_parts)

        vid_encoder_idx = self.config.get('video_encoder', 0)
        c_v_arg = 'libx264'
        extra_args = ['-preset', 'fast']
        
        if vid_encoder_idx == 1:
            c_v_arg = 'h264_nvenc'
            extra_args = ['-preset', 'fast', '-rc', 'vbr']
        elif vid_encoder_idx == 2:
            c_v_arg = 'h264_qsv'
            extra_args = ['-preset', 'fast', '-global_quality', '20']

        audio_codec = ['-c:a', 'copy'] if has_waveform else ['-c:a', 'aac', '-b:a', '320k']
            
        cmd_final.extend([
            '-filter_complex', filter_complex,
            '-map', '[outv]',
            '-map', f'{idx_a}:a',
            '-c:v', c_v_arg,
            *extra_args,
            '-pix_fmt', 'yuv420p',
            *audio_codec,
            '-shortest',
            final_output
        ])
        
        try:
            self.is_running = True
            self.running_processes = []
            
            p = subprocess.Popen(cmd_final, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
            self.running_processes.append(p)
            
            stdout, stderr = p.communicate()
            
            if p in self.running_processes:
                self.running_processes.remove(p)
                
            if not self.is_running:
                return False
                
            if p.returncode != 0:
                if progress_callback: progress_callback(100, f"Lỗi FFmpeg render: {stderr[-500:]}")
                # Cleanup
                if os.path.exists(concat_txt_path): os.remove(concat_txt_path)
                return False
                
            # Xóa các video đã lấy từ thư mục chứa để lần sau không lấy lại (chỉ khi không dùng Nền Tĩnh)
            if not is_static:
                for v in set(selected):
                    try:
                        os.remove(v)
                    except Exception as e:
                        print(f"Không thể xóa {v}: {e}")
                    
            if os.path.exists(concat_txt_path): os.remove(concat_txt_path)
            
        except Exception as e:
            if progress_callback: progress_callback(100, f"FFmpeg error: {e}")
            return False

        if 'concat_txt_path' in locals() and os.path.exists(concat_txt_path): 
            try: os.remove(concat_txt_path)
            except: pass

        if progress_callback: progress_callback(100, "Hoàn tất ghép Video!")
        return True

    def stop(self):
        self.is_running = False
        if hasattr(self, 'running_processes'):
            for p in self.running_processes:
                try: p.kill()
                except: pass
