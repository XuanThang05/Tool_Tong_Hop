import os
import subprocess
import soundfile as sf

def extract_audio_fast(input_path, output_path, ffmpeg_path):
    """
    Sử dụng FFmpeg để chuyển đổi mọi loại file âm thanh/video đầu vào thành .wav 22050Hz, mono.
    """
    if os.path.exists(output_path):
        try: os.remove(output_path)
        except: pass
        
    cmd = [
        ffmpeg_path,
        '-i', input_path,
        '-vn', 
        '-ac', '1', 
        '-ar', '22050', 
        '-c:a', 'pcm_s16le', 
        '-y', 
        output_path
    ]
    
    kwargs = {}
    if os.name == 'nt':
        kwargs['creationflags'] = 0x08000000
        
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)
        return True
    except subprocess.CalledProcessError as e:
        print("FFmpeg Error:", e.stderr.decode('utf-8', errors='ignore'))
        return False
    except FileNotFoundError:
        return False

def load_audio_fast(wav_path):
    data, sr = sf.read(wav_path, dtype='float32')
    return data, sr
