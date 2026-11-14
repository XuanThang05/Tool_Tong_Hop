import os
import sys
import numpy as np
import librosa
import subprocess
import cv2
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image, ImageDraw, ImageFont
from .utils import extract_audio_fast, load_audio_fast

def render_chunk_worker(args):
    (chunk_idx, start_frame, end_frame, norm_mel, bass_envelope, config, ffmpeg_path, width, height, fps, temp_out) = args
    try:
        # Cấu hình thanh sóng
        bg_color_hex = config.get('bg_color', '#000000')
        enable_bars = config.get('enable_bars', True)
        spectrumScale = config.get('spectrumScale', 1.0)
        barsToDraw = config.get('barsToDraw', 64)
        base_barWidth = config.get('barWidth', 13) * spectrumScale
        base_spacing = config.get('spectrumSpacing', 6) * spectrumScale
        positionX = config.get('positionX', 0.5)
        positionY = config.get('positionY', 0.5)
        barColor = config.get('barColor', '#ffeeee')
        barHeightMult = config.get('barHeightMultiplier', 17)
        shadowBlur = int(config.get('shadowBlur', 64) * spectrumScale)
        shadowAlpha = config.get('shadowAlpha', 0.28)

        # Cấu hình Particles
        enable_particles = config.get('enable_particles', True)
        particle_count = config.get('particleCount', 150)
        baseSpeed = config.get('baseSpeed', 18)
        particleColor = config.get('particleColor', '#ffffff')
        impact_amp = config.get('impact_amplitude', 520)
        particle_dir = config.get('particle_dir', -1) 

        # Cấu hình Text
        enable_text = config.get('enable_text', True)
        artist_text = config.get('artist_text', '')
        song_text = config.get('song_text', '')
        remix_text = config.get('remix_text', '')

        def hex_to_rgb(hex_str):
            hex_str = hex_str.lstrip('#')
            return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
            
        def hex_to_bgr(hex_str):
            r, g, b = hex_to_rgb(hex_str)
            return (b, g, r)

        bg_bgr = hex_to_bgr(bg_color_hex)
        bar_rgb = hex_to_rgb(barColor)
        particle_bgr = hex_to_bgr(particleColor)
        
        bg_pil = Image.new('RGB', (width, height), color=hex_to_rgb(bg_color_hex))
        draw = ImageDraw.Draw(bg_pil)
        
        try:
            font_artist = ImageFont.truetype("arialbd.ttf", 102)
            font_song = ImageFont.truetype("arial.ttf", 44)
        except:
            font_artist = ImageFont.load_default()
            font_song = ImageFont.load_default()

        total_bars_width = barsToDraw * (base_barWidth + base_spacing)
        start_x = width * positionX - total_bars_width / 2.0
        base_y = height * positionY
        
        text_x = int(start_x)
        text_y = int(height * 0.62)
        if enable_text:
            if artist_text: draw.text((text_x, text_y), artist_text, font=font_artist, fill="white")
            if song_text: draw.text((text_x, text_y + 110), song_text, font=font_song, fill="white")
            if remix_text: draw.text((text_x, text_y + 160), remix_text, font=font_song, fill="white")

        bg_cv = cv2.cvtColor(np.array(bg_pil), cv2.COLOR_RGB2BGR)

        particles = []
        for _ in range(particle_count):
            particles.append({
                'x': np.random.uniform(0, width),
                'base_y': np.random.uniform(0, height),
                'speed': np.random.uniform(0.5, 2.0) * (baseSpeed / 5.0),
                'size': np.random.uniform(1.0, 3.5),
                'brightness': int(np.random.uniform(100, 255)),
                'phase': np.random.uniform(0, 2 * np.pi),
                'freq': np.random.uniform(0.01, 0.05),
                'amp': np.random.uniform(10, 60),
                'pulse_freq': np.random.uniform(0.02, 0.08)
            })

        # FORCE libx264 ultrafast for chunks to avoid NVIDIA NVENC concurrent session limits
        c_v_arg = 'libx264'
        extra_args = ['-preset', 'ultrafast', '-crf', '18']

        ffmpeg_cmd = [
            ffmpeg_path, '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', 'bgr24',
            '-r', str(fps),
            '-i', '-',
            '-c:v', c_v_arg,
            *extra_args,
            '-pix_fmt', 'yuv420p',
            temp_out
        ]
        
        kwargs = {}
        if os.name == 'nt':
            kwargs['creationflags'] = 0x08000000

        stderr_file = open(temp_out + ".log", "w")
        process = subprocess.Popen(
            ffmpeg_cmd, stdin=subprocess.PIPE, 
            stdout=subprocess.DEVNULL, stderr=stderr_file, **kwargs
        )

        max_bar_height = barHeightMult * 15 * spectrumScale
        
        for frame_idx in range(start_frame, end_frame):
            current_spectrum = norm_mel[:, frame_idx]
            current_bass = bass_envelope[frame_idx]
            frame = bg_cv.copy()
            
            if enable_particles:
                for p in particles:
                    p['phase'] += p['freq']
                    y = p['base_y'] + math.sin(p['phase']) * p['amp']
                    p['x'] += p['speed'] * (1 + current_bass * (impact_amp/100.0)) * particle_dir
                    
                    if particle_dir == -1 and p['x'] < 0:
                        p['x'] = width
                        p['base_y'] = np.random.uniform(0, height)
                        p['speed'] = np.random.uniform(0.5, 2.0) * (baseSpeed / 5.0)
                    elif particle_dir == 1 and p['x'] > width:
                        p['x'] = 0
                        p['base_y'] = np.random.uniform(0, height)
                        p['speed'] = np.random.uniform(0.5, 2.0) * (baseSpeed / 5.0)
                        
                    pulse = (math.sin(p['phase'] * (p['pulse_freq'] / p['freq'])) + 1) / 2
                    cur_bright = int(p['brightness'] * (0.2 + 0.8 * pulse))
                    
                    b = int(particle_bgr[0] * cur_bright / 255)
                    g = int(particle_bgr[1] * cur_bright / 255)
                    r = int(particle_bgr[2] * cur_bright / 255)
                    color = (b, g, r)
                    
                    trail_length = int(p['speed'] * (1 + current_bass * (impact_amp/100.0)) * 2) * -particle_dir
                    
                    cv2.line(frame, (int(p['x']), int(y)), (int(p['x'] + trail_length), int(y)), color, int(p['size']))
                    cv2.circle(frame, (int(p['x']), int(y)), int(p['size']), color, -1, cv2.LINE_AA)
            
            if enable_bars:
                if shadowBlur > 0:
                    for i in range(barsToDraw):
                        h = max(3.0, current_spectrum[i] * max_bar_height)
                        x_pos_float = start_x + i * (base_barWidth + base_spacing)
                        x_pos = int(x_pos_float)
                        w = max(1, int(x_pos_float + base_barWidth) - x_pos)
                        gw = int(shadowBlur / 4)
                        cv2.rectangle(frame, (x_pos - gw, int(base_y - h - gw)), 
                                      (x_pos + w + gw, int(base_y + gw)), 
                                      (int(bar_rgb[2]*shadowAlpha), int(bar_rgb[1]*shadowAlpha), int(bar_rgb[0]*shadowAlpha)), -1)
                                      
                for i in range(barsToDraw):
                    h = max(3.0, current_spectrum[i] * max_bar_height)
                    x_pos_float = start_x + i * (base_barWidth + base_spacing)
                    x_pos = int(x_pos_float)
                    w = max(1, int(x_pos_float + base_barWidth) - x_pos)
                    cv2.rectangle(frame, (x_pos, int(base_y - h)), (x_pos + w, int(base_y)), (bar_rgb[2], bar_rgb[1], bar_rgb[0]), -1)

            try:
                process.stdin.write(frame.tobytes())
            except Exception as e:
                import traceback
                stderr_file.close()
                with open(temp_out + ".log", "r") as f:
                    stderr_output = f.read()
                return chunk_idx, f"FFmpeg stdin write error: {str(e)}\nFFmpeg Stderr: {stderr_output}\n{traceback.format_exc()}"
                
        process.stdin.close()
        process.wait()
        stderr_file.close()
        
        if process.returncode != 0:
            with open(temp_out + ".log", "r") as f:
                stderr_output = f.read()
            return chunk_idx, f"FFmpeg process exited with code {process.returncode}\nFFmpeg Stderr: {stderr_output}"
            
        try:
            if os.path.exists(temp_out + ".log"):
                os.remove(temp_out + ".log")
        except: pass
            
        return chunk_idx, True
    except Exception as e:
        import traceback
        return chunk_idx, f"Worker exception: {str(e)}\n{traceback.format_exc()}"

class WaveformGenerator:
    def __init__(self, config=None):
        self.config = config or {}
        self.executor = None
        self.is_running = True

    def generate(self, audio_path: str, output_path: str, progress_callback=None) -> bool:
        """
        Tạo video sóng âm từ file audio.
        progress_callback: function(percent: int, message: str)
        """
        try:
            if progress_callback: progress_callback(0, "Đang khởi tạo FFmpeg...")
            ffmpeg_path = os.path.join(os.getcwd(), 'bin', 'ffmpeg.exe')

            if not os.path.exists(ffmpeg_path):
                if progress_callback: progress_callback(0, f"Không tìm thấy ffmpeg tại: {ffmpeg_path}")
                return False

            temp_wav = os.path.join(os.getcwd(), 'temp_audio.wav')
            
            # 1. Trích xuất âm thanh
            if progress_callback: progress_callback(2, "Đang trích xuất và tối ưu hoá âm thanh (Siêu tốc)...")
            success = extract_audio_fast(audio_path, temp_wav, ffmpeg_path)
            if not success:
                if progress_callback: progress_callback(0, "Lỗi khi trích xuất âm thanh bằng FFmpeg. File bị hỏng hoặc FFmpeg lỗi.")
                return False
            
            if progress_callback: progress_callback(5, "Đang nạp âm thanh vào RAM...")
            y, sr = load_audio_fast(temp_wav)
            
            if os.path.exists(temp_wav):
                try: os.remove(temp_wav)
                except: pass

            # 2. Lấy cấu hình
            width = 1920
            height = 1080
            fps = self.config.get('fps', 60)
            bg_color_hex = self.config.get('bg_color', '#000000')

            # Cấu hình thanh sóng
            enable_bars = self.config.get('enable_bars', True)
            spectrumScale = self.config.get('spectrumScale', 1.0)
            barsToDraw = self.config.get('barsToDraw', 64)
            base_barWidth = self.config.get('barWidth', 13) * spectrumScale
            base_spacing = self.config.get('spectrumSpacing', 6) * spectrumScale
            positionX = self.config.get('positionX', 0.5)
            positionY = self.config.get('positionY', 0.5)
            barColor = self.config.get('barColor', '#ffeeee')
            barHeightMult = self.config.get('barHeightMultiplier', 17)
            shadowBlur = int(self.config.get('shadowBlur', 64) * spectrumScale)
            shadowAlpha = self.config.get('shadowAlpha', 0.28)

            # Phổ âm
            minDecibel = self.config.get('minDecibel', -95.0)
            maxDecibel = self.config.get('maxDecibel', -14.0)
            smoothingTime = self.config.get('smoothingTimeConstant', 0.8)
            fallSpeed = self.config.get('fallSpeed', 0.5)

            # Cấu hình Particles
            enable_particles = self.config.get('enable_particles', True)
            particle_count = self.config.get('particleCount', 150)
            baseSpeed = self.config.get('baseSpeed', 18)
            particleColor = self.config.get('particleColor', '#ffffff')
            impact_amp = self.config.get('impact_amplitude', 520)
            particle_dir = self.config.get('particle_dir', -1) 

            # Cấu hình Text
            enable_text = self.config.get('enable_text', True)
            artist_text = self.config.get('artist_text', '')
            song_text = self.config.get('song_text', '')
            remix_text = self.config.get('remix_text', '')

            # --- 3. Tính toán phổ âm thanh (Tối ưu hoá Chunking chống tràn RAM) ---
            if progress_callback: progress_callback(10, "Đang phân tích phổ âm thanh (Tối ưu RAM)...")
            hop_length = int(sr / fps)
            n_fft = 4096
            
            mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=barsToDraw, fmin=50, fmax=12000)
            
            # Chia file thành các block 5 phút để tránh vọt RAM (5 * 60 = 300s)
            chunk_duration = 300 
            chunk_samples = sr * chunk_duration
            
            mel_specs = []
            
            for i in range(0, len(y), chunk_samples):
                y_chunk = y[i:i+chunk_samples]
                stft_chunk = librosa.stft(y_chunk, n_fft=n_fft, hop_length=hop_length)
                power_chunk = np.abs(stft_chunk)**2
                mel_chunk = np.dot(mel_basis, power_chunk)
                mel_specs.append(mel_chunk)
                
            # Giải phóng RAM của mảng Audio gốc
            import gc
            del y
            gc.collect()

            # Nối các phổ âm lại
            mel_spec = np.concatenate(mel_specs, axis=1)
            mel_db = librosa.power_to_db(mel_spec, ref=np.max)

            smooth_mel = np.copy(mel_db)
            for i in range(1, smooth_mel.shape[1]):
                # So sánh frame hiện tại với frame trước đó
                # Nếu âm thanh tăng lên (Attack) -> dùng smoothingTime (Độ vút)
                # Nếu âm thanh giảm đi (Fall/Release) -> dùng fallSpeed (Gia tốc rơi)
                is_rising = mel_db[:, i] > smooth_mel[:, i-1]
                
                # Tính giá trị làm mượt cho cả 2 trường hợp
                rise_val = smooth_mel[:, i-1] * smoothingTime + mel_db[:, i] * (1 - smoothingTime)
                fall_val = smooth_mel[:, i-1] * fallSpeed + mel_db[:, i] * (1 - fallSpeed)
                
                # Áp dụng
                smooth_mel[:, i] = np.where(is_rising, rise_val, fall_val)

            norm_mel = (smooth_mel - minDecibel) / (maxDecibel - minDecibel)
            norm_mel = np.clip(norm_mel, 0, 1)

            taper = np.sin(np.linspace(0, np.pi, barsToDraw)) ** 4.0
            mute_left = max(1, int(barsToDraw * 0.15))
            mute_right = max(1, int(barsToDraw * 0.25))
            taper[:mute_left] *= np.linspace(0, 1, mute_left)
            taper[-mute_right:] *= np.linspace(1, 0, mute_right)
            
            for i in range(norm_mel.shape[1]):
                norm_mel[:, i] = np.power(norm_mel[:, i], 1.2) * taper

            bass_envelope = np.mean(norm_mel[3:15, :], axis=0) 
            bass_min, bass_max = np.min(bass_envelope), np.max(bass_envelope)
            if bass_max > bass_min:
                bass_envelope = (bass_envelope - bass_min) / (bass_max - bass_min)

            total_frames = norm_mel.shape[1]
            
            # --- 4. Render Engine (Multiprocessing) ---
            if progress_callback: progress_callback(25, "Đang chuẩn bị Render Engine Đa tiến trình...")
            
            # Lấy cấu hình số luồng từ config, mặc định là os.cpu_count() - 1 nếu chưa cài đặt
            configured_threads = int(self.config.get('threads', max(1, multiprocessing.cpu_count() - 1)))
            num_workers = max(1, configured_threads)
            chunk_size = math.ceil(total_frames / num_workers)
            
            temp_dir = os.path.dirname(output_path)
            temp_files = []
            
            self.is_running = True
            
            args_list = []
            for i in range(num_workers):
                start_f = i * chunk_size
                end_f = min((i + 1) * chunk_size, total_frames)
                if start_f >= end_f:
                    break
                    
                temp_out = os.path.join(temp_dir, f"temp_wf_{i}.mp4")
                temp_files.append(temp_out)
                
                args = (
                    i, start_f, end_f, norm_mel, bass_envelope, 
                    self.config, ffmpeg_path, width, height, fps, temp_out
                )
                args_list.append(args)

            if progress_callback: progress_callback(30, f"Đang xuất video sóng âm với {len(args_list)} tiến trình...")
            
            self.executor = ProcessPoolExecutor(max_workers=num_workers)
            futures = [self.executor.submit(render_chunk_worker, args) for args in args_list]
            
            completed_chunks = 0
            success = True
            
            for future in as_completed(futures):
                if not self.is_running:
                    success = False
                    break
                try:
                    chunk_idx, res = future.result()
                    if res is not True:
                        success = False
                        if progress_callback: progress_callback(0, f"Lỗi tiến trình {chunk_idx}: {res}")
                        break
                    completed_chunks += 1
                    percent = 30 + int((completed_chunks / len(args_list)) * 60)
                    if progress_callback: progress_callback(percent, f"Đã render xong phần {completed_chunks}/{len(args_list)}")
                except Exception as e:
                    success = False
                    import traceback
                    if progress_callback: progress_callback(0, f"Lỗi catch future: {str(e)}\n{traceback.format_exc()}")
                    break
                    
            if not success or not self.is_running:
                self.executor.shutdown(wait=False, cancel_futures=True)
                for f in temp_files:
                    if os.path.exists(f):
                        try: os.remove(f)
                        except: pass
                    if os.path.exists(f + ".log"):
                        try: os.remove(f + ".log")
                        except: pass
                return False
                
            self.executor.shutdown(wait=True)
            
            # --- 5. Nối các mảnh video và trộn âm thanh (Concat) ---
            if progress_callback: progress_callback(95, "Đang nối các mảnh video sóng âm...")
            
            list_txt = os.path.join(temp_dir, "concat_wf.txt")
            with open(list_txt, "w", encoding="utf-8") as f:
                for temp_file in temp_files:
                    f.write(f"file '{temp_file.replace(chr(92), '/')}'\n")
                    
            vid_encoder_idx = self.config.get('video_encoder', 0)
            c_v_arg = 'libx264'
            extra_args = ['-preset', 'fast']
            if vid_encoder_idx == 1:
                c_v_arg = 'h264_nvenc'
                extra_args = ['-preset', 'p4']
            elif vid_encoder_idx == 2:
                c_v_arg = 'h264_qsv'
                
            concat_cmd = [
                ffmpeg_path, '-y',
                '-f', 'concat', '-safe', '0',
                '-i', list_txt,
                '-i', audio_path,
                '-c:v', 'copy',
                '-c:a', 'aac', '-b:a', '320k',
                '-shortest',
                output_path
            ]
            
            kwargs = {}
            if os.name == 'nt':
                kwargs['creationflags'] = 0x08000000
                
            concat_proc = subprocess.Popen(concat_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kwargs)
            self.running_processes = [concat_proc]
            concat_proc.wait()
            self.running_processes = []
            
            for f in temp_files + [list_txt]:
                if os.path.exists(f):
                    try: os.remove(f)
                    except: pass
                if os.path.exists(f + ".log"):
                    try: os.remove(f + ".log")
                    except: pass
                    
            if concat_proc.returncode != 0:
                if progress_callback: progress_callback(0, "Lỗi nối file video sóng âm bằng FFmpeg.")
                return False

            if progress_callback: progress_callback(100, "Tạo Waveform thành công.")
            return True
            
        except Exception as e:
            if progress_callback: progress_callback(0, f"Lỗi WaveformGenerator: {str(e)}")
            return False

    def stop(self):
        self.is_running = False
        if hasattr(self, 'executor') and self.executor:
            self.executor.shutdown(wait=False, cancel_futures=True)
        if hasattr(self, 'running_processes'):
            for p in self.running_processes:
                try: p.terminate()
                except: pass
