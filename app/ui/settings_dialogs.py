from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QWidget, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
                               QLineEdit, QSpinBox, QTextEdit, QPushButton, QHBoxLayout, QLabel, QFileDialog, QComboBox, QInputDialog, QCheckBox)

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QWidget, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
                               QLineEdit, QSpinBox, QTextEdit, QPushButton, QHBoxLayout, QLabel, QFileDialog, QComboBox, QInputDialog, QCheckBox, QTabWidget, QDoubleSpinBox, QSlider, QGroupBox, QColorDialog)



class PlagiarismSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cài đặt Check Trùng Lặp")
        self.resize(1100, 750)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Ngưỡng đạo văn (Phần trên)
        form = QFormLayout()
        self.spin_threshold = QSpinBox()
        self.spin_threshold.setRange(0, 100)
        self.spin_threshold.setSuffix("%")
        form.addRow("Bỏ qua nếu giống nhau >", self.spin_threshold)
        
        # Chọn Sổ lưu
        self.combo_target_notebook = QComboBox()
        # Load danh sách sổ từ app_service
        import sys, os
        from PyQt6.QtCore import Qt
        check_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "modules", "check_trung_lap")
        if check_path not in sys.path:
            sys.path.insert(0, check_path)
        from src.services import AppService
        self.temp_app_service = AppService()
        self.notebooks = self.temp_app_service.list_notebooks()
        self.notebook_map = { nb['name']: nb['id'] for nb in self.notebooks }
        self.combo_target_notebook.addItems(list(self.notebook_map.keys()))
        
        form.addRow("Lưu vào sổ (khi hoàn thành):", self.combo_target_notebook)
        layout.addLayout(form)
        
        # Phần dưới: Nhúng trực tiếp App check_trung_lap
        from src.ui_app import App as CheckTrungLapApp
        self.check_app = CheckTrungLapApp()
        self.check_app.setWindowFlags(Qt.WindowType.Widget)
        layout.addWidget(self.check_app, stretch=1)
        
        # Nút lưu của bảng phụ
        btn_save = QPushButton("Lưu cấu hình ngưỡng & Đóng")
        btn_save.clicked.connect(self.accept)
        layout.addWidget(btn_save)
        
        self.setLayout(layout)

class GeminiSettingsDialog(QDialog):
    def __init__(self, step_name, parent=None):
        super().__init__(parent)
        self.step_name = step_name
        self.setWindowTitle(f"Cài đặt Gemini ({step_name})")
        layout = QVBoxLayout()
        form = QFormLayout()
        
        self.txt_keys = QTextEdit()
        self.txt_keys.setPlaceholderText("API keys (mỗi dòng 1 key)")
        self.txt_keys.setMaximumHeight(60)
        
        # --- Presets UI ---
        preset_layout = QHBoxLayout()
        self.combo_presets = QComboBox()
        self.combo_presets.addItem("-- Chọn mẫu prompt --")
        self.combo_presets.currentIndexChanged.connect(self.on_preset_selected)
        
        btn_save_preset = QPushButton("Lưu mẫu")
        btn_save_preset.clicked.connect(self.save_preset)
        btn_del_preset = QPushButton("Xóa mẫu")
        btn_del_preset.clicked.connect(self.delete_preset)
        
        preset_layout.addWidget(self.combo_presets, stretch=1)
        preset_layout.addWidget(btn_save_preset)
        preset_layout.addWidget(btn_del_preset)
        # ------------------
        
        self.txt_prompt = QTextEdit()
        self.txt_prompt.setPlaceholderText("Nhập prompt ở đây...")
        self.txt_prompt.setMaximumHeight(150)
        
        # Thêm spinbox cho số luồng API
        self.spin_threads = QSpinBox()
        self.spin_threads.setRange(1, 20)
        self.spin_threads.setValue(3)
        
        form.addRow("API Keys:", self.txt_keys)
        form.addRow("Mẫu Prompt:", preset_layout)
        form.addRow("Prompt:", self.txt_prompt)
        form.addRow("Số luồng (Threads):", self.spin_threads)
        layout.addLayout(form)
        
        btn = QPushButton("Lưu cấu hình")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)
        self.setLayout(layout)
        
        import os
        self.presets_file = os.path.join(os.getcwd(), 'data', 'config', 'gemini_presets.json')
        self.presets = {}
        self.load_presets()
        
    def load_presets(self):
        import json, os
        if os.path.exists(self.presets_file):
            try:
                with open(self.presets_file, "r", encoding="utf-8") as f:
                    self.presets = json.load(f)
            except:
                self.presets = {}
        else:
            os.makedirs(os.path.dirname(self.presets_file), exist_ok=True)
            self.presets = {}
            
        self.combo_presets.blockSignals(True)
        self.combo_presets.clear()
        self.combo_presets.addItem("-- Chọn mẫu prompt --")
        for k in self.presets.keys():
            self.combo_presets.addItem(k)
        self.combo_presets.blockSignals(False)

    def save_preset(self):
        import json
        name, ok = QInputDialog.getText(self, "Lưu mẫu", "Nhập tên cho mẫu prompt này:")
        if ok and name.strip():
            name = name.strip()
            self.presets[name] = self.txt_prompt.toPlainText()
            with open(self.presets_file, "w", encoding="utf-8") as f:
                json.dump(self.presets, f, ensure_ascii=False, indent=4)
            self.load_presets()
            self.combo_presets.setCurrentText(name)
            QMessageBox.information(self, "Thành công", f"Đã lưu mẫu: {name}")

    def delete_preset(self):
        import json
        name = self.combo_presets.currentText()
        if name in self.presets:
            reply = QMessageBox.question(self, "Xác nhận", f"Bạn có chắc muốn xóa mẫu '{name}'?")
            if reply == QMessageBox.StandardButton.Yes:
                del self.presets[name]
                with open(self.presets_file, "w", encoding="utf-8") as f:
                    json.dump(self.presets, f, ensure_ascii=False, indent=4)
                self.load_presets()
                self.txt_prompt.clear()

    def on_preset_selected(self):
        name = self.combo_presets.currentText()
        if name in self.presets:
            self.txt_prompt.setPlainText(self.presets[name])

class FolderSelectRow(QWidget):
    def __init__(self, placeholder=""):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        self.line_edit = QLineEdit()
        self.line_edit.setPlaceholderText(placeholder)
        self.btn_browse = QPushButton("Chọn...")
        self.btn_browse.clicked.connect(self.browse)
        layout.addWidget(self.line_edit)
        layout.addWidget(self.btn_browse)
        
    def browse(self):
        folder = QFileDialog.getExistingDirectory(self, "Chọn thư mục")
        if folder:
            self.line_edit.setText(folder)
            
    def text(self):
        return self.line_edit.text()
        
    def setText(self, text):
        self.line_edit.setText(text)

class FileSelectRow(QWidget):
    def __init__(self, placeholder="", filter_str="Video Files (*.mp4 *.avi *.mov *.mkv)"):
        super().__init__()
        self.filter_str = filter_str
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        self.line_edit = QLineEdit()
        self.line_edit.setPlaceholderText(placeholder)
        self.btn_browse = QPushButton("Chọn...")
        self.btn_browse.clicked.connect(self.browse)
        layout.addWidget(self.line_edit)
        layout.addWidget(self.btn_browse)
        
    def browse(self):
        file, _ = QFileDialog.getOpenFileName(self, "Chọn file", "", self.filter_str)
        if file:
            self.line_edit.setText(file)
            
    def text(self):
        return self.line_edit.text()
        
    def setText(self, text):
        self.line_edit.setText(text)

class VideoSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cài đặt Ghép Video & Phụ đề")
        self.resize(600, 600)
        layout = QVBoxLayout()
        form = QFormLayout()
        
        self.inp_bg_folder = FolderSelectRow("Thư mục chứa video nền (vào)")
        
        self.chk_static_bg_enable = QCheckBox("Sử dụng 1 Ảnh/Video tĩnh làm nền (Bỏ qua thư mục)")
        self.inp_static_bg_file = FileSelectRow("Đường dẫn file Ảnh/Video tĩnh", "Media Files (*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png *.webp *.bmp)")
        
        self.chk_effect_enable = QCheckBox("Bật Video Hiệu ứng (Nền đen - Mưa, tuyết, hạt,...)")
        self.inp_effect_file = FileSelectRow("Đường dẫn file video hiệu ứng")
        
        self.spin_zoom = QSpinBox()
        self.spin_zoom.setRange(10, 200)
        self.spin_zoom.setSingleStep(5)
        self.spin_zoom.setValue(100)
        
        self.spin_opacity = QSpinBox()
        self.spin_opacity.setRange(10, 100)
        self.spin_opacity.setSingleStep(5)
        self.spin_opacity.setValue(100)
        
        # --- Cài đặt Phụ đề (STT) ---
        self.chk_sub_enable = QCheckBox("Bật Phụ đề (STT từ âm thanh)")
        self.chk_sub_enable.setChecked(False)
        
        from PyQt6.QtWidgets import QFontComboBox
        self.inp_sub_font = QFontComboBox()
        # Đặt Arial làm mặc định nếu có
        from PyQt6.QtGui import QFont
        self.inp_sub_font.setCurrentFont(QFont("Arial"))
        
        self.spin_sub_size = QSpinBox()
        self.spin_sub_size.setRange(10, 200) # Mở rộng lên 200
        self.spin_sub_size.setValue(48)      # Tăng mặc định lên 48 cho màn 1080p
        
        self.btn_sub_color = QPushButton()
        self.btn_sub_color.clicked.connect(lambda: self.choose_color(self.btn_sub_color))
        self.set_color_button(self.btn_sub_color, "#FFFFFF")
        
        self.chk_sub_border = QCheckBox("Hiển thị viền chữ")
        self.chk_sub_border.setChecked(True)
        
        self.btn_sub_border = QPushButton()
        self.btn_sub_border.clicked.connect(lambda: self.choose_color(self.btn_sub_border))
        self.set_color_button(self.btn_sub_border, "#000000")
        
        self.spin_border_size = QSpinBox()
        self.spin_border_size.setRange(0, 50) # Mở rộng lên 50
        self.spin_border_size.setValue(4)     # Tăng mặc định lên 4
        
        self.chk_sub_bold = QCheckBox("In đậm (Bold)")
        self.chk_sub_italic = QCheckBox("In nghiêng (Italic)")
        
        self.chk_sub_box = QCheckBox("Hiển thị hộp nền (Box)")
        self.spin_box_opacity = QSpinBox()
        self.spin_box_opacity.setRange(10, 100)
        self.spin_box_opacity.setSingleStep(5)
        self.spin_box_opacity.setValue(80)
        
        form.addRow("Video nền (Nguồn):", self.inp_bg_folder)
        
        form.addRow("", self.chk_static_bg_enable)
        form.addRow("File Nền tĩnh:", self.inp_static_bg_file)
        
        form.addRow("Hiệu ứng (Nền đen):", self.chk_effect_enable)
        form.addRow("File Video Hiệu ứng:", self.inp_effect_file)
        form.addRow("Thu phóng Zoom (%):", self.spin_zoom)
        form.addRow("Độ mờ Opacity (%):", self.spin_opacity)
        
        self.combo_video_encoder = QComboBox()
        self.combo_video_encoder.addItems(["CPU (libx264)", "NVIDIA (h264_nvenc)", "Intel (h264_qsv)"])
        form.addRow("Tăng tốc GPU:", self.combo_video_encoder)
        
        # Add a visual separator
        from PyQt6.QtWidgets import QFrame
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        form.addRow(line)
        
        form.addRow("", self.chk_sub_enable)
        form.addRow("Font chữ:", self.inp_sub_font)
        form.addRow("Cỡ chữ:", self.spin_sub_size)
        form.addRow("Màu chữ:", self.btn_sub_color)
        
        form.addRow("", self.chk_sub_border)
        form.addRow("Màu viền (Border):", self.btn_sub_border)
        form.addRow("Độ dày viền:", self.spin_border_size)
        
        style_layout = QHBoxLayout()
        style_layout.addWidget(self.chk_sub_bold)
        style_layout.addWidget(self.chk_sub_italic)
        form.addRow("Kiểu chữ:", style_layout)
        
        box_layout = QHBoxLayout()
        box_layout.addWidget(self.chk_sub_box)
        box_layout.addWidget(QLabel("Độ mờ Box (%):"))
        box_layout.addWidget(self.spin_box_opacity)
        form.addRow("Hộp nền (Box):", box_layout)
        
        # --- Mới: Vị trí & Căn lề ---
        pos_layout = QHBoxLayout()
        self.combo_alignment = QComboBox()
        self.combo_alignment.addItems(["Dưới (Bottom)", "Giữa (Middle)", "Trên (Top)"])
        # Map: Bottom=2, Middle=10 (hoặc 5), Top=8
        
        self.spin_margin_v = QSpinBox()
        self.spin_margin_v.setRange(0, 1000)
        self.spin_margin_v.setValue(50)
        
        pos_layout.addWidget(QLabel("Lề (Alignment):"))
        pos_layout.addWidget(self.combo_alignment)
        pos_layout.addWidget(QLabel("Khoảng cách dọc (MarginV):"))
        pos_layout.addWidget(self.spin_margin_v)
        form.addRow("Vị trí (Position):", pos_layout)
        
        # --- Nút Preview ---
        self.btn_preview = QPushButton("👀 Xem trước (Preview Subtitle)")
        self.btn_preview.setStyleSheet("background-color: #3498db; color: white; font-weight: bold; padding: 5px;")
        self.btn_preview.clicked.connect(self.generate_preview)
        form.addRow("", self.btn_preview)
        
        # Khung hiển thị Preview
        self.lbl_preview = QLabel("Hình ảnh Preview sẽ hiển thị ở đây")
        self.lbl_preview.setMinimumSize(480, 270) # 16:9 ratio
        self.lbl_preview.setStyleSheet("background-color: black; color: white; border: 1px solid #555;")
        self.lbl_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        form.addRow("Preview:", self.lbl_preview)
        
        layout.addLayout(form)
        
        btn = QPushButton("Lưu")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)
        self.setLayout(layout)

    def set_color_button(self, btn, hex_color):
        btn.setStyleSheet(f"background-color: {hex_color}; border: 1px solid #ccc;")
        btn.setProperty("hex_color", hex_color)

    def choose_color(self, btn):
        from PyQt6.QtWidgets import QColorDialog
        from PyQt6.QtGui import QColor
        color = QColorDialog.getColor(QColor(btn.property("hex_color")), self, "Chọn màu")
        if color.isValid():
            self.set_color_button(btn, color.name())

    def get_current_config(self):
        return {
            "sub_font": self.inp_sub_font.currentFont().family(),
            "sub_size": self.spin_sub_size.value(),
            "sub_color": self.btn_sub_color.property("hex_color"),
            "sub_border_color": self.btn_sub_border.property("hex_color"),
            "sub_border_size": self.spin_border_size.value(),
            "sub_border_enable": self.chk_sub_border.isChecked(),
            "sub_bold": self.chk_sub_bold.isChecked(),
            "sub_italic": self.chk_sub_italic.isChecked(),
            "sub_box": self.chk_sub_box.isChecked(),
            "sub_box_opacity": self.spin_box_opacity.value(),
            "sub_alignment": self.combo_alignment.currentIndex(),
            "sub_margin_v": self.spin_margin_v.value(),
        }

    def generate_preview(self):
        from PyQt6.QtGui import QPixmap
        import os
        from modules.video_assembler.assembler import VideoAssembler
        
        self.btn_preview.setText("Đang tạo Preview...")
        self.btn_preview.setEnabled(False)
        self.lbl_preview.setText("Đang tải...")
        
        try:
            # Tạo instance tạm thời
            output_dir = os.path.join(os.getcwd(), 'outputs', 'preview_temp')
            os.makedirs(output_dir, exist_ok=True)
            
            cfg = self.get_current_config()
            assembler = VideoAssembler(output_dir, cfg)
            
            preview_jpg = os.path.join(output_dir, 'preview.jpg')
            if os.path.exists(preview_jpg):
                os.remove(preview_jpg)
                
            success = assembler.generate_subtitle_preview_image(preview_jpg)
            
            if success and os.path.exists(preview_jpg):
                pixmap = QPixmap(preview_jpg)
                # Scale pixmap fit với khung
                scaled = pixmap.scaled(self.lbl_preview.width(), self.lbl_preview.height(), Qt.AspectRatioMode.KeepAspectRatio)
                self.lbl_preview.setPixmap(scaled)
            else:
                self.lbl_preview.setText("Lỗi: Không tạo được Preview.")
        except Exception as e:
            self.lbl_preview.setText(f"Lỗi: {e}")
        finally:
            self.btn_preview.setText("👀 Xem trước (Preview Subtitle)")
            self.btn_preview.setEnabled(True)

class GeneralPathsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cài đặt Đường dẫn Chung")
        layout = QVBoxLayout()
        form = QFormLayout()
        
        self.inp_output = FolderSelectRow("Thư mục lưu Output (Truyện hoàn chỉnh)")
        form.addRow("Thư mục lưu:", self.inp_output)
        
        layout.addLayout(form)
        
        btn = QPushButton("Lưu")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)
        self.setLayout(layout)

class PlaceholderSettingsDialog(QDialog):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Các tùy chọn cho bước này (Scraper/TTS/Waveform) sẽ được bổ sung sau khi có code của bạn."))
        btn = QPushButton("Đóng")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)
        self.setLayout(layout)

class TTSSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cài đặt Đọc truyện (TTS Piper)")
        self.resize(350, 200)
        
        layout = QVBoxLayout(self)
        form = QFormLayout()
        
        self.combo_model = QComboBox()
        import os
        model_dir = os.path.join(os.getcwd(), 'data', 'model', 'tts')
        if os.path.exists(model_dir):
            for file in os.listdir(model_dir):
                if file.endswith('.onnx'):
                    self.combo_model.addItem(file)
                    
        form.addRow("Giọng đọc (Model):", self.combo_model)
        
        self.spin_speed = QDoubleSpinBox()
        self.spin_speed.setRange(0.1, 3.0)
        self.spin_speed.setSingleStep(0.1)
        self.spin_speed.setValue(1.0)
        form.addRow("Tốc độ (Speed):", self.spin_speed)
        
        self.chk_pitch_shift = QCheckBox("Biến đổi tông giọng theo tốc độ (Méo giọng)")
        self.chk_pitch_shift.setChecked(False)
        form.addRow("", self.chk_pitch_shift)
        
        self.chk_sweet_voice = QCheckBox("Bộ lọc Capcut: Giọng Ngọt Ngào (Sweet Voice)")
        self.chk_sweet_voice.setChecked(False)
        form.addRow("", self.chk_sweet_voice)
        
        self.spin_vol = QDoubleSpinBox()
        self.spin_vol.setRange(0.1, 3.0)
        self.spin_vol.setSingleStep(0.1)
        self.spin_vol.setValue(1.0)
        form.addRow("Âm lượng (Volume):", self.spin_vol)
        
        self.spin_threads = QSpinBox()
        self.spin_threads.setRange(1, 10)
        self.spin_threads.setValue(3)
        form.addRow("Số luồng (Threads):", self.spin_threads)
        
        layout.addLayout(form)
        
        btn = QPushButton("Lưu")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QColor
import cv2
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont

class WaveformSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cài đặt Sóng âm & Live Preview")
        self.resize(1050, 650)
        
        # Debounce timer for preview
        self.preview_timer = QTimer()
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self.generate_preview)

        self.init_ui()
        
        import os
        self.presets_file = os.path.join(os.getcwd(), 'data', 'config', 'waveform_presets.json')
        self.presets = {}
        self.load_presets()

        # Preview initial
        self.update_preview_throttled()

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        # --- Left Panel (Settings) ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0,0,0,0)

        # --- Presets UI ---
        preset_layout = QHBoxLayout()
        self.combo_presets = QComboBox()
        self.combo_presets.addItem("-- Chọn mẫu Cài đặt --")
        self.combo_presets.currentIndexChanged.connect(self.on_preset_selected)
        
        btn_save_preset = QPushButton("Lưu mẫu")
        btn_save_preset.clicked.connect(self.save_preset)
        btn_del_preset = QPushButton("Xóa mẫu")
        btn_del_preset.clicked.connect(self.delete_preset)
        
        preset_layout.addWidget(QLabel("Mẫu Sóng Âm:"))
        preset_layout.addWidget(self.combo_presets, stretch=1)
        preset_layout.addWidget(btn_save_preset)
        preset_layout.addWidget(btn_del_preset)
        left_layout.addLayout(preset_layout)
        # ------------------

        self.tabs = QTabWidget()
        
        # 1. Tab Sóng Nhạc (Bars)
        tab_bars = QWidget()
        form_bars = QFormLayout(tab_bars)
        
        self.chk_enable_bars = QCheckBox("Hiển thị Sóng Nhạc")
        self.chk_enable_bars.setChecked(True)
        self.chk_enable_bars.stateChanged.connect(self.update_preview_throttled)
        form_bars.addRow("", self.chk_enable_bars)
        
        self.val_spectrumScale = self.create_slider(10, 300, 100, form_bars, "Tỷ lệ kích thước sóng (%):")
        self.val_barsToDraw = self.create_slider(16, 128, 64, form_bars, "Số lượng cột sóng:")
        self.val_barWidth = self.create_slider(1, 30, 13, form_bars, "Độ rộng cột:")
        self.val_spectrumSpacing = self.create_slider(0, 20, 6, form_bars, "Khoảng cách cột:")
        self.val_positionX = self.create_slider(0, 100, 50, form_bars, "Vị trí ngang X (%):")
        self.val_positionY = self.create_slider(0, 100, 50, form_bars, "Vị trí dọc Y (%):")
        
        self.btn_barColor = QPushButton()
        self.set_color_button(self.btn_barColor, "#ffeeee")
        self.btn_barColor.clicked.connect(lambda: self.choose_color(self.btn_barColor))
        form_bars.addRow("Màu sắc cột:", self.btn_barColor)
        
        self.val_barHeightMult = self.create_slider(1, 50, 17, form_bars, "Chiều cao sóng:")
        self.val_shadowBlur = self.create_slider(0, 100, 64, form_bars, "Độ toả sáng (Glow):")
        self.tabs.addTab(tab_bars, "Sóng Nhạc")

        # 2. Tab Thuật Toán Âm (Spectrum Analyzer)
        tab_analyzer = QWidget()
        form_analyzer = QFormLayout(tab_analyzer)
        self.val_minDecibel = self.create_slider(-120, -40, -96, form_analyzer, "Âm tối thiểu (minDecibel):")
        self.val_maxDecibel = self.create_slider(-40, 0, -15, form_analyzer, "Âm tối đa (maxDecibel):")
        self.val_smoothingTime = self.create_slider(0, 99, 80, form_analyzer, "Độ vút (Attack/Rise):")
        self.val_fallSpeed = self.create_slider(0, 99, 50, form_analyzer, "Gia tốc rơi (Fall Speed):")
        
        import os
        max_cpu = max(1, os.cpu_count() - 1)
        self.val_threads = self.create_slider(1, max_cpu, max_cpu, form_analyzer, "Số luồng Render (Threads):")
        
        self.btn_bg_color = QPushButton()
        self.set_color_button(self.btn_bg_color, "#000000")
        self.btn_bg_color.clicked.connect(lambda: self.choose_color(self.btn_bg_color))
        form_analyzer.addRow("Màu nền Background:", self.btn_bg_color)
        
        self.tabs.addTab(tab_analyzer, "Thuật toán")

        # 3. Tab Hạt Bụi (Particles)
        tab_particles = QWidget()
        form_particles = QFormLayout(tab_particles)
        
        self.chk_enable_particles = QCheckBox("Bật Hạt Bụi")
        self.chk_enable_particles.setChecked(True)
        self.chk_enable_particles.stateChanged.connect(self.update_preview_throttled)
        form_particles.addRow("", self.chk_enable_particles)
        
        self.chk_particle_dir = QCheckBox("Bay từ Trái sang Phải")
        self.chk_particle_dir.setChecked(False) 
        self.chk_particle_dir.stateChanged.connect(self.update_preview_throttled)
        form_particles.addRow("Hướng bay:", self.chk_particle_dir)

        self.val_particleCount = self.create_slider(10, 500, 150, form_particles, "Số lượng hạt:")
        self.val_baseSpeed = self.create_slider(1, 50, 18, form_particles, "Tốc độ bay:")
        
        self.btn_particleColor = QPushButton()
        self.set_color_button(self.btn_particleColor, "#ffffff")
        self.btn_particleColor.clicked.connect(lambda: self.choose_color(self.btn_particleColor))
        form_particles.addRow("Màu hạt bụi:", self.btn_particleColor)
        
        self.val_impact_amp = self.create_slider(0, 1000, 520, form_particles, "Độ giật theo Bass:")
        self.tabs.addTab(tab_particles, "Hạt bụi")

        # 4. Tab Chữ (Text)
        tab_text = QWidget()
        form_text = QFormLayout(tab_text)
        
        self.chk_enable_text = QCheckBox("Hiển thị Chữ")
        self.chk_enable_text.setChecked(True)
        self.chk_enable_text.stateChanged.connect(self.update_preview_throttled)
        form_text.addRow("", self.chk_enable_text)
        
        self.txt_artist = QLineEdit("Artist Name")
        self.txt_artist.textChanged.connect(self.update_preview_throttled)
        form_text.addRow("Tên Ca sĩ:", self.txt_artist)
        
        self.txt_song = QLineEdit("Song Name")
        self.txt_song.textChanged.connect(self.update_preview_throttled)
        form_text.addRow("Tên Bài hát:", self.txt_song)
        
        self.txt_remix = QLineEdit("Remix Text")
        self.txt_remix.textChanged.connect(self.update_preview_throttled)
        form_text.addRow("Chữ phụ:", self.txt_remix)
        self.tabs.addTab(tab_text, "Văn bản")

        left_layout.addWidget(self.tabs)

        btn_save = QPushButton("Lưu Cài Đặt")
        btn_save.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold; padding: 10px;")
        btn_save.clicked.connect(self.accept)
        left_layout.addWidget(btn_save)

        # --- Right Panel (Preview) ---
        right_panel = QGroupBox("Xem trước (Preview Tỷ lệ 16:9)")
        right_layout = QVBoxLayout(right_panel)
        self.lbl_preview = QLabel("Đang tải Preview...")
        self.lbl_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # 800x450 for 16:9 
        self.lbl_preview.setFixedSize(800, 450)
        self.lbl_preview.setStyleSheet("background-color: black;")
        right_layout.addWidget(self.lbl_preview)
        
        main_layout.addWidget(left_panel, stretch=0)
        main_layout.addWidget(right_panel, stretch=0)

    def set_color_button(self, button, hex_color):
        button.setStyleSheet(f"background-color: {hex_color}; border: 1px solid #aaa; height: 25px;")
        button.setProperty("hex_color", hex_color)

    def choose_color(self, button):
        current_hex = button.property("hex_color")
        color = QColorDialog.getColor(QColor(current_hex), self, "Chọn màu")
        if color.isValid():
            self.set_color_button(button, color.name())
            self.update_preview_throttled()

    def create_slider(self, min_val, max_val, default_val, layout, label):
        row = QHBoxLayout()
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default_val)
        
        lbl_val = QLabel(str(default_val))
        slider.valueChanged.connect(lambda v: lbl_val.setText(str(v)))
        slider.valueChanged.connect(self.update_preview_throttled)
        
        row.addWidget(slider)
        row.addWidget(lbl_val)
        layout.addRow(label, row)
        return slider

    def update_preview_throttled(self):
        self.preview_timer.start(100) 

    def get_current_state(self):
        return {
            "enable_bars": self.chk_enable_bars.isChecked(),
            "barColor": self.btn_barColor.property("hex_color"),
            "barsToDraw": self.val_barsToDraw.value(),
            "barWidth": self.val_barWidth.value(),
            "spectrumSpacing": self.val_spectrumSpacing.value(),
            "barHeightMultiplier": self.val_barHeightMult.value(),
            "positionX": self.val_positionX.value(),
            "positionY": self.val_positionY.value(),
            "shadowBlur": self.val_shadowBlur.value(),
            "spectrumScale": self.val_spectrumScale.value(),
            "enable_particles": self.chk_enable_particles.isChecked(),
            "particleColor": self.btn_particleColor.property("hex_color"),
            "particleCount": self.val_particleCount.value(),
            "baseSpeed": self.val_baseSpeed.value(),
            "impact_amplitude": self.val_impact_amp.value(),
            "particle_dir": self.chk_particle_dir.isChecked(),
            "bg_color": self.btn_bg_color.property("hex_color"),
            "minDecibel": self.val_minDecibel.value(),
            "maxDecibel": self.val_maxDecibel.value(),
            "smoothingTimeConstant": self.val_smoothingTime.value(),
            "fallSpeed": self.val_fallSpeed.value(),
            "enable_text": self.chk_enable_text.isChecked(),
            "artist_text": self.txt_artist.text(),
            "song_text": self.txt_song.text(),
            "remix_text": self.txt_remix.text(),
        }

    def apply_state(self, state):
        self.chk_enable_bars.setChecked(state.get("enable_bars", True))
        self.set_color_button(self.btn_barColor, state.get("barColor", "#ffeeee"))
        self.val_barsToDraw.setValue(state.get("barsToDraw", 64))
        self.val_barWidth.setValue(state.get("barWidth", 13))
        self.val_spectrumSpacing.setValue(state.get("spectrumSpacing", 6))
        self.val_barHeightMult.setValue(state.get("barHeightMultiplier", 17))
        self.val_positionX.setValue(state.get("positionX", 50))
        self.val_positionY.setValue(state.get("positionY", 50))
        self.val_shadowBlur.setValue(state.get("shadowBlur", 64))
        self.val_spectrumScale.setValue(state.get("spectrumScale", 100))
        
        self.chk_enable_particles.setChecked(state.get("enable_particles", True))
        self.set_color_button(self.btn_particleColor, state.get("particleColor", "#ffffff"))
        self.val_particleCount.setValue(state.get("particleCount", 150))
        self.val_baseSpeed.setValue(state.get("baseSpeed", 18))
        self.val_impact_amp.setValue(state.get("impact_amplitude", 520))
        self.chk_particle_dir.setChecked(state.get("particle_dir", False))
        
        self.set_color_button(self.btn_bg_color, state.get("bg_color", "#000000"))
        self.val_minDecibel.setValue(state.get("minDecibel", -96))
        self.val_maxDecibel.setValue(state.get("maxDecibel", -15))
        self.val_smoothingTime.setValue(state.get("smoothingTimeConstant", 80))
        
        self.chk_enable_text.setChecked(state.get("enable_text", True))
        
        self.val_fallSpeed.setValue(state.get("fallSpeed", 50))
        self.txt_artist.setText(state.get("artist_text", "Artist Name"))
        self.txt_song.setText(state.get("song_text", "Song Name"))
        self.txt_remix.setText(state.get("remix_text", "Remix Text"))
        
        self.update_preview_throttled()

    def load_presets(self):
        import json, os
        if os.path.exists(self.presets_file):
            try:
                with open(self.presets_file, "r", encoding="utf-8") as f:
                    self.presets = json.load(f)
            except:
                self.presets = {}
        else:
            os.makedirs(os.path.dirname(self.presets_file), exist_ok=True)
            self.presets = {}
            
        self.combo_presets.blockSignals(True)
        self.combo_presets.clear()
        self.combo_presets.addItem("-- Chọn mẫu Cài đặt --")
        for k in self.presets.keys():
            self.combo_presets.addItem(k)
        self.combo_presets.blockSignals(False)

    def save_preset(self):
        import json
        name, ok = QInputDialog.getText(self, "Lưu mẫu", "Nhập tên cho mẫu Sóng âm này:")
        if ok and name.strip():
            name = name.strip()
            self.presets[name] = self.get_current_state()
            with open(self.presets_file, "w", encoding="utf-8") as f:
                json.dump(self.presets, f, ensure_ascii=False, indent=4)
            self.load_presets()
            self.combo_presets.setCurrentText(name)
            QMessageBox.information(self, "Thành công", f"Đã lưu mẫu: {name}")

    def delete_preset(self):
        import json
        name = self.combo_presets.currentText()
        if name in self.presets:
            reply = QMessageBox.question(self, "Xác nhận", f"Bạn có chắc muốn xóa mẫu '{name}'?")
            if reply == QMessageBox.StandardButton.Yes:
                del self.presets[name]
                with open(self.presets_file, "w", encoding="utf-8") as f:
                    json.dump(self.presets, f, ensure_ascii=False, indent=4)
                self.load_presets()

    def on_preset_selected(self):
        name = self.combo_presets.currentText()
        if name in self.presets:
            self.apply_state(self.presets[name])

    def hex_to_rgb(self, hex_str):
        hex_str = hex_str.lstrip('#')
        return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

    def generate_preview(self):
        width, height = 800, 450 
        scale = (width / 1920.0) * (self.val_spectrumScale.value() / 100.0)
        text_scale = width / 1920.0
        
        bg_color_hex = self.btn_bg_color.property("hex_color")
        bg_color = self.hex_to_rgb(bg_color_hex)
        barColor = self.btn_barColor.property("hex_color")
        particleColor = self.btn_particleColor.property("hex_color")
        
        barsToDraw = self.val_barsToDraw.value()
        base_barWidth = self.val_barWidth.value() * scale
        base_spacing = self.val_spectrumSpacing.value() * scale
        positionX = self.val_positionX.value() / 100.0
        positionY = self.val_positionY.value() / 100.0
        
        # 1. Background and Text (Pillow)
        bg_pil = Image.new('RGB', (width, height), color=bg_color)
        draw = ImageDraw.Draw(bg_pil)
        
        try:
            font_artist = ImageFont.truetype("arialbd.ttf", int(102 * text_scale))
            font_song = ImageFont.truetype("arial.ttf", int(44 * text_scale))
        except:
            font_artist = ImageFont.load_default()
            font_song = ImageFont.load_default()

        total_bars_width = barsToDraw * (base_barWidth + base_spacing)
        start_x = width * positionX - total_bars_width / 2.0
        base_y = height * positionY
        
        text_x = start_x
        text_y = int(height * 0.62)
        
        if self.chk_enable_text.isChecked():
            if self.txt_artist.text():
                draw.text((text_x, text_y), self.txt_artist.text(), font=font_artist, fill="white")
            if self.txt_song.text():
                draw.text((text_x, text_y + int(110*text_scale)), self.txt_song.text(), font=font_song, fill="white")
            if self.txt_remix.text():
                draw.text((text_x, text_y + int(160*text_scale)), self.txt_remix.text(), font=font_song, fill="white")

        frame = cv2.cvtColor(np.array(bg_pil), cv2.COLOR_RGB2BGR)
        
        # 2. Dummy Audio Data
        x = np.linspace(-3, 3, barsToDraw)
        current_spectrum = np.exp(-x**2)
        
        taper = np.sin(np.linspace(0, np.pi, barsToDraw)) ** 4.0
        mute_left = max(1, int(barsToDraw * 0.15))
        mute_right = max(1, int(barsToDraw * 0.25))
        taper[:mute_left] *= np.linspace(0, 1, mute_left)
        taper[-mute_right:] *= np.linspace(1, 0, mute_right)
        
        current_spectrum = current_spectrum * taper
        current_bass = 0.8
        
        # 3. Draw Particles (Static dummy)
        if self.chk_enable_particles.isChecked():
            particle_bgr = tuple(reversed(self.hex_to_rgb(particleColor)))
            np.random.seed(42) 
            
            p_count = min(150, self.val_particleCount.value())
            dir_mult = 1 if self.chk_particle_dir.isChecked() else -1
            
            for _ in range(p_count):
                px = np.random.randint(0, width)
                py = np.random.randint(0, height)
                
                pulse = (math.sin(np.random.uniform(0, 2*math.pi)) + 1) / 2
                brightness = int(np.random.uniform(100, 255) * (0.2 + 0.8 * pulse))
                b = int(particle_bgr[0] * brightness / 255)
                g = int(particle_bgr[1] * brightness / 255)
                r = int(particle_bgr[2] * brightness / 255)
                color = (b, g, r)
                
                psize = np.random.uniform(1.0, 3.5) * text_scale
                
                wave_offset = int(math.sin(px * 0.05) * 5)
                
                cv2.circle(frame, (px, py + wave_offset), int(psize), color, -1, cv2.LINE_AA)
                cv2.line(frame, (px, py + wave_offset), (px - int(15*text_scale) * dir_mult, py + wave_offset), color, 1)
        
        # 4. Draw Bars
        if self.chk_enable_bars.isChecked():
            bar_rgb = self.hex_to_rgb(barColor)
            bar_bgr = (bar_rgb[2], bar_rgb[1], bar_rgb[0])
            max_bar_height = self.val_barHeightMult.value() * 15 * scale
            
            shadowBlur = int(self.val_shadowBlur.value() * scale)
            shadowAlpha = 0.28
            
            for i in range(barsToDraw):
                h = max(3.0, current_spectrum[i] * max_bar_height)
                
                x_pos_float = start_x + i * (base_barWidth + base_spacing)
                x_pos = int(x_pos_float)
                w = max(1, int(x_pos_float + base_barWidth) - x_pos)
                
                if shadowBlur > 0:
                    gw = int(shadowBlur / 4)
                    cv2.rectangle(frame, (x_pos - gw, int(base_y - h - gw)), 
                                  (x_pos + w + gw, int(base_y + gw)), 
                                  (int(bar_bgr[0]*shadowAlpha), int(bar_bgr[1]*shadowAlpha), int(bar_bgr[2]*shadowAlpha)), -1)
                cv2.rectangle(frame, (x_pos, int(base_y - h)), (x_pos + w, int(base_y)), bar_bgr, -1)

        # 5. Display
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        qimg = QImage(rgb_img.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.lbl_preview.setPixmap(QPixmap.fromImage(qimg))


