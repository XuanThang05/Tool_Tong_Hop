from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QLabel, QTextEdit, QPushButton, QCheckBox, QGroupBox, QScrollArea, 
                               QListWidget, QAbstractItemView, QFileDialog, QListWidgetItem)
from PyQt6.QtCore import pyqtSlot, Qt
from core.config_manager import ConfigManager
from core.engine import PipelineEngine
from ui.settings_dialogs import (PlagiarismSettingsDialog, 
                                GeminiSettingsDialog, VideoSettingsDialog, GeneralPathsDialog)
import datetime
import os
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Auto Story Video Creator")
        self.resize(1000, 650)
        
        self.config_manager = ConfigManager()
        self.engine = PipelineEngine(self.config_manager)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # --- LEFT PANEL: WORKFLOW ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 10, 0)
        
        # Các bước trong quy trình
        self.steps = {
            "plagiarism_check": ("1. Kiểm tra trùng lặp", True),
            "gemini_clean": ("2. Làm sạch bằng Gemini", True),
            "tts": ("3. Đọc truyện (TTS Piper)", True),
            "waveform": ("4. Tạo sóng âm", True),
            "video_assemble": ("5. Video nền & Ghép Final", True)
        }
        self.checkboxes = {}
        
        for key, (label, has_settings) in self.steps.items():
            step_group = QGroupBox(label)
            step_group.setStyleSheet("QGroupBox { font-weight: bold; margin-top: 5px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
            row = QHBoxLayout()
            row.setContentsMargins(5, 5, 5, 5)
            
            chk = QCheckBox("Kích hoạt")
            # Khôi phục trạng thái từ config
            chk.setChecked(self.config_manager.get("pipeline", key) is True)
            chk.toggled.connect(lambda checked, k=key: self.config_manager.set("pipeline", k, checked))
            self.checkboxes[key] = chk
            row.addWidget(chk)
            
            if has_settings:
                btn = QPushButton("Cài đặt")
                btn.clicked.connect(lambda _, k=key: self.open_settings(k))
                row.addWidget(btn)
            
            step_group.setLayout(row)
            left_layout.addWidget(step_group)
        
        # Import TXT Group
        import_group = QGroupBox("Nhập Truyện (Import TXT)")
        import_layout = QVBoxLayout()
        
        self.list_txt = QListWidget()
        self.list_txt.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list_txt.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        
        btn_import_layout = QHBoxLayout()
        self.btn_add_txt = QPushButton("Thêm File TXT")
        self.btn_add_txt.clicked.connect(self.add_txt_files)
        self.btn_remove_txt = QPushButton("Xóa File Chọn")
        self.btn_remove_txt.clicked.connect(self.remove_txt_files)
        self.btn_clear_txt = QPushButton("Xóa Tất Cả")
        self.btn_clear_txt.clicked.connect(self.list_txt.clear)
        
        btn_import_layout.addWidget(self.btn_add_txt)
        btn_import_layout.addWidget(self.btn_remove_txt)
        btn_import_layout.addWidget(self.btn_clear_txt)
        
        import_layout.addWidget(self.list_txt)
        import_layout.addLayout(btn_import_layout)
        import_group.setLayout(import_layout)
        left_layout.addWidget(import_group)
        
        # Tools
        tools_group = QGroupBox("Cài đặt Chung")
        tools_layout = QVBoxLayout()
        btn_paths = QPushButton("Cài đặt Thư mục Output")
        btn_paths.clicked.connect(lambda: self.open_settings("general_paths"))
        tools_layout.addWidget(btn_paths)
        tools_group.setLayout(tools_layout)
        left_layout.addWidget(tools_group)
        
        left_layout.addStretch()
        
        # Start / Stop
        controls_layout = QHBoxLayout()
        self.btn_start = QPushButton("▶ Bắt đầu chạy")
        self.btn_start.setMinimumHeight(40)
        self.btn_stop = QPushButton("⏹ Dừng lại")
        self.btn_stop.setMinimumHeight(40)
        self.btn_stop.setEnabled(False)
        
        self.btn_start.clicked.connect(self.start_engine)
        self.btn_stop.clicked.connect(self.stop_engine)
        
        controls_layout.addWidget(self.btn_start)
        controls_layout.addWidget(self.btn_stop)
        left_layout.addLayout(controls_layout)
        
        main_layout.addWidget(left_panel, 1) # Left takes 1 part
        
        # --- RIGHT PANEL: LOG & STATUS ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 0, 0, 0)
        
        self.status_label = QLabel("Trạng thái: Đang rảnh")
        self.status_label.setStyleSheet("font-weight: bold; color: #2ecc71;")
        right_layout.addWidget(self.status_label)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4; font-family: Consolas;")
        self.log_text.append("=== Auto Story Video Creator ===")
        self.log_text.append("Hệ thống đã sẵn sàng.")
        right_layout.addWidget(self.log_text)
        
        main_layout.addWidget(right_panel, 2) # Right takes 2 parts
        
        # Connect signals
        self.engine.log_signal.connect(self.append_log)
        self.engine.status_signal.connect(self.update_status)
        self.engine.finished_signal.connect(self.on_engine_finished)

    def open_settings(self, key):
        if key == "telegram":
            dlg = TelegramSettingsDialog(self)
            # Load current
            tg_cfg = self.config_manager.get("telegram")
            if tg_cfg:
                dlg.inp_tg_token.setText(tg_cfg.get("bot_token", ""))
                dlg.inp_tg_chatid.setText(tg_cfg.get("chat_id", ""))
            if dlg.exec():
                self.config_manager.set("telegram", "bot_token", dlg.inp_tg_token.text())
                self.config_manager.set("telegram", "chat_id", dlg.inp_tg_chatid.text())
                
        elif key == "general_paths":
            dlg = GeneralPathsDialog(self)
            paths_cfg = self.config_manager.get("paths") or {}
            dlg.inp_output.setText(paths_cfg.get("output_dir", ""))
            if dlg.exec():
                self.config_manager.set("paths", "output_dir", dlg.inp_output.text())
                
        elif key == "video_assemble":
            dlg = VideoSettingsDialog(self)
            vid_cfg = self.config_manager.get("video") or {}
            
            dlg.inp_bg_folder.line_edit.setText(vid_cfg.get("bg_folder", ""))
            
            dlg.chk_static_bg_enable.setChecked(vid_cfg.get("static_bg_enable", False))
            dlg.inp_static_bg_file.line_edit.setText(vid_cfg.get("static_bg_file", ""))
            
            dlg.chk_effect_enable.setChecked(vid_cfg.get("effect_enable", False))
            dlg.inp_effect_file.line_edit.setText(vid_cfg.get("effect_video_path", ""))
            
            dlg.spin_zoom.setValue(vid_cfg.get("zoom", 100))
            dlg.spin_opacity.setValue(vid_cfg.get("opacity", 100))
            dlg.combo_video_encoder.setCurrentIndex(vid_cfg.get("video_encoder", 0))
            
            dlg.chk_sub_enable.setChecked(vid_cfg.get("sub_enable", False))
            from PyQt6.QtGui import QFont
            dlg.inp_sub_font.setCurrentFont(QFont(vid_cfg.get("sub_font", "Arial")))
            dlg.spin_sub_size.setValue(vid_cfg.get("sub_size", 48))
            
            if "sub_color" in vid_cfg:
                dlg.set_color_button(dlg.btn_sub_color, vid_cfg["sub_color"])
            if "sub_border_color" in vid_cfg:
                dlg.set_color_button(dlg.btn_sub_border, vid_cfg["sub_border_color"])
                
            dlg.spin_border_size.setValue(vid_cfg.get("sub_border_size", 4))
            dlg.chk_sub_border.setChecked(vid_cfg.get("sub_border_enable", True))
            dlg.chk_sub_bold.setChecked(vid_cfg.get("sub_bold", False))
            dlg.chk_sub_italic.setChecked(vid_cfg.get("sub_italic", False))
            dlg.chk_sub_box.setChecked(vid_cfg.get("sub_box", False))
            dlg.spin_box_opacity.setValue(vid_cfg.get("sub_box_opacity", 80))
            
            dlg.combo_alignment.setCurrentIndex(vid_cfg.get("sub_alignment", 0))
            dlg.spin_margin_v.setValue(vid_cfg.get("sub_margin_v", 50))
            
            if dlg.exec():
                self.config_manager.set("video", "bg_folder", dlg.inp_bg_folder.line_edit.text())
                
                self.config_manager.set("video", "static_bg_enable", dlg.chk_static_bg_enable.isChecked())
                self.config_manager.set("video", "static_bg_file", dlg.inp_static_bg_file.line_edit.text())
                
                self.config_manager.set("video", "effect_enable", dlg.chk_effect_enable.isChecked())
                self.config_manager.set("video", "effect_video_path", dlg.inp_effect_file.line_edit.text())
                
                self.config_manager.set("video", "zoom", dlg.spin_zoom.value())
                self.config_manager.set("video", "opacity", dlg.spin_opacity.value())
                self.config_manager.set("video", "video_encoder", dlg.combo_video_encoder.currentIndex())
                
                self.config_manager.set("video", "sub_enable", dlg.chk_sub_enable.isChecked())
                self.config_manager.set("video", "sub_font", dlg.inp_sub_font.currentFont().family())
                self.config_manager.set("video", "sub_size", dlg.spin_sub_size.value())
                self.config_manager.set("video", "sub_color", dlg.btn_sub_color.property("hex_color"))
                self.config_manager.set("video", "sub_border_color", dlg.btn_sub_border.property("hex_color"))
                self.config_manager.set("video", "sub_border_size", dlg.spin_border_size.value())
                self.config_manager.set("video", "sub_border_enable", dlg.chk_sub_border.isChecked())
                self.config_manager.set("video", "sub_bold", dlg.chk_sub_bold.isChecked())
                self.config_manager.set("video", "sub_italic", dlg.chk_sub_italic.isChecked())
                self.config_manager.set("video", "sub_box", dlg.chk_sub_box.isChecked())
                self.config_manager.set("video", "sub_box_opacity", dlg.spin_box_opacity.value())
                self.config_manager.set("video", "sub_alignment", dlg.combo_alignment.currentIndex())
                self.config_manager.set("video", "sub_margin_v", dlg.spin_margin_v.value())


        elif key == "plagiarism_check":
            dlg = PlagiarismSettingsDialog(self)
            dlg.spin_threshold.setValue(int(self.config_manager.get("plagiarism", "similarity_threshold") or 80))
            
            # Khôi phục notebook đã chọn
            saved_nb = self.config_manager.get("plagiarism", "target_notebook")
            if saved_nb:
                dlg.combo_target_notebook.setCurrentText(saved_nb)
                
            # Đổ dữ liệu check gần nhất vào giao diện App (nếu có)
            if hasattr(self, 'engine') and hasattr(self.engine, 'last_plag_result') and self.engine.last_plag_result:
                plag_text = self.engine.last_plag_text or ""
                dlg.check_app._pending_paste = ("Kiểm tra tự động", plag_text)
                dlg.check_app._pending_paths = []
                dlg.check_app.source_lbl.setText(f"Dán: Kiểm tra tự động ({len(plag_text)} ký tự)")
                dlg.check_app._clear_results()
                dlg.check_app._show_results([self.engine.last_plag_result])
                
            if dlg.exec():
                self.config_manager.set("plagiarism", "similarity_threshold", dlg.spin_threshold.value())
                self.config_manager.set("plagiarism", "target_notebook", dlg.combo_target_notebook.currentText())
                
        elif key == "gemini_clean":
            dlg = GeminiSettingsDialog(self.steps[key][0], self)
            gemini_cfg = self.config_manager.get("gemini")
            if gemini_cfg:
                keys = gemini_cfg.get("api_keys", [])
                if isinstance(keys, list): keys = "\n".join(keys)
                dlg.txt_keys.setText(keys)
                
                dlg.txt_prompt.setText(gemini_cfg.get("prompt_cleanup", ""))
                dlg.spin_threads.setValue(int(gemini_cfg.get("threads_cleanup", 3)))
                    
            if dlg.exec():
                keys_list = [k.strip() for k in dlg.txt_keys.toPlainText().split('\n') if k.strip()]
                self.config_manager.set("gemini", "api_keys", keys_list)
                self.config_manager.set("gemini", "prompt_cleanup", dlg.txt_prompt.toPlainText())
                self.config_manager.set("gemini", "threads_cleanup", dlg.spin_threads.value())
        
        elif key == "tts":
            from ui.settings_dialogs import TTSSettingsDialog
            dlg = TTSSettingsDialog(self)
            tts_cfg = self.config_manager.get("tts") or {}
            
            # Load model
            saved_model = tts_cfg.get("model", "")
            index = dlg.combo_model.findText(saved_model)
            if index >= 0:
                dlg.combo_model.setCurrentIndex(index)
                
            dlg.spin_speed.setValue(float(tts_cfg.get("speed", 1.0)))
            dlg.chk_pitch_shift.setChecked(tts_cfg.get("pitch_shift", False))
            
            dlg.chk_sweet_voice.setChecked(tts_cfg.get("sweet_voice", False))
            
            dlg.spin_vol.setValue(float(tts_cfg.get("volume", 1.0)))
            dlg.spin_threads.setValue(int(tts_cfg.get("threads", 3)))
            
            if dlg.exec():
                self.config_manager.set("tts", "model", dlg.combo_model.currentText())
                self.config_manager.set("tts", "speed", dlg.spin_speed.value())
                self.config_manager.set("tts", "pitch_shift", dlg.chk_pitch_shift.isChecked())
                self.config_manager.set("tts", "sweet_voice", dlg.chk_sweet_voice.isChecked())
                
                # Cleanup old keys if any
                if "speed2_enable" in self.config_manager.config["tts"]:
                    del self.config_manager.config["tts"]["speed2_enable"]
                if "speed2" in self.config_manager.config["tts"]:
                    del self.config_manager.config["tts"]["speed2"]
                self.config_manager.set("tts", "volume", dlg.spin_vol.value())
                self.config_manager.set("tts", "threads", dlg.spin_threads.value())
                
        elif key == "waveform":
            from ui.settings_dialogs import WaveformSettingsDialog
            dlg = WaveformSettingsDialog(self)
            wf_cfg = self.config_manager.get("waveform") or {}
            
            dlg.chk_enable_bars.setChecked(wf_cfg.get("enable_bars", True))
            dlg.set_color_button(dlg.btn_barColor, wf_cfg.get("barColor", "#ffffff"))
            dlg.val_barsToDraw.setValue(int(wf_cfg.get("barsToDraw", 64)))
            dlg.val_barWidth.setValue(int(wf_cfg.get("barWidth", 13)))
            dlg.val_spectrumSpacing.setValue(int(wf_cfg.get("spectrumSpacing", 6)))
            dlg.val_barHeightMult.setValue(int(wf_cfg.get("barHeightMultiplier", 17)))
            dlg.val_positionX.setValue(int(wf_cfg.get("positionX", 0.5) * 100))
            dlg.val_positionY.setValue(int(wf_cfg.get("positionY", 0.5) * 100))
            dlg.val_shadowBlur.setValue(int(wf_cfg.get("shadowBlur", 64)))
            dlg.val_spectrumScale.setValue(int(wf_cfg.get("spectrumScale", 1.0) * 100))
            
            dlg.chk_enable_particles.setChecked(wf_cfg.get("enable_particles", True))
            dlg.set_color_button(dlg.btn_particleColor, wf_cfg.get("particleColor", "#ffffff"))
            dlg.val_particleCount.setValue(int(wf_cfg.get("particleCount", 150)))
            dlg.val_baseSpeed.setValue(int(wf_cfg.get("baseSpeed", 18)))
            dlg.val_impact_amp.setValue(int(wf_cfg.get("impact_amplitude", 520)))
            
            if int(wf_cfg.get("particle_dir", -1)) == -1:
                dlg.chk_particle_dir.setChecked(False)
            else:
                dlg.chk_particle_dir.setChecked(True)
                
            dlg.set_color_button(dlg.btn_bg_color, wf_cfg.get("bg_color", "#000000"))
            dlg.val_minDecibel.setValue(int(wf_cfg.get("minDecibel", -96)))
            dlg.val_maxDecibel.setValue(int(wf_cfg.get("maxDecibel", -15)))
            dlg.val_smoothingTime.setValue(int(wf_cfg.get("smoothingTimeConstant", 0.8) * 100))
            dlg.val_fallSpeed.setValue(int(wf_cfg.get("fallSpeed", 0.5) * 100))
            import os
            dlg.val_threads.setValue(int(wf_cfg.get("threads", max(1, os.cpu_count() - 1))))
            
            dlg.chk_enable_text.setChecked(wf_cfg.get("enable_text", True))
            dlg.txt_artist.setText(wf_cfg.get("artist_text", "Artist Name"))
            dlg.txt_song.setText(wf_cfg.get("song_text", "Song Name"))
            dlg.txt_remix.setText(wf_cfg.get("remix_text", "Remix Text"))
            
            dlg.update_preview_throttled()
            
            if dlg.exec():
                self.config_manager.set("waveform", "enable_bars", dlg.chk_enable_bars.isChecked())
                self.config_manager.set("waveform", "barColor", dlg.btn_barColor.property("hex_color"))
                self.config_manager.set("waveform", "barsToDraw", dlg.val_barsToDraw.value())
                self.config_manager.set("waveform", "barWidth", dlg.val_barWidth.value())
                self.config_manager.set("waveform", "spectrumSpacing", dlg.val_spectrumSpacing.value())
                self.config_manager.set("waveform", "barHeightMultiplier", dlg.val_barHeightMult.value())
                self.config_manager.set("waveform", "positionX", dlg.val_positionX.value() / 100.0)
                self.config_manager.set("waveform", "positionY", dlg.val_positionY.value() / 100.0)
                self.config_manager.set("waveform", "shadowBlur", dlg.val_shadowBlur.value())
                self.config_manager.set("waveform", "spectrumScale", dlg.val_spectrumScale.value() / 100.0)
                
                self.config_manager.set("waveform", "enable_particles", dlg.chk_enable_particles.isChecked())
                self.config_manager.set("waveform", "particleColor", dlg.btn_particleColor.property("hex_color"))
                self.config_manager.set("waveform", "particleCount", dlg.val_particleCount.value())
                self.config_manager.set("waveform", "baseSpeed", dlg.val_baseSpeed.value())
                self.config_manager.set("waveform", "impact_amplitude", dlg.val_impact_amp.value())
                self.config_manager.set("waveform", "particle_dir", 1 if dlg.chk_particle_dir.isChecked() else -1)
                
                self.config_manager.set("waveform", "bg_color", dlg.btn_bg_color.property("hex_color"))
                self.config_manager.set("waveform", "minDecibel", dlg.val_minDecibel.value())
                self.config_manager.set("waveform", "maxDecibel", dlg.val_maxDecibel.value())
                self.config_manager.set("waveform", "smoothingTimeConstant", dlg.val_smoothingTime.value() / 100.0)
                self.config_manager.set("waveform", "fallSpeed", dlg.val_fallSpeed.value() / 100.0)
                self.config_manager.set("waveform", "threads", dlg.val_threads.value())
                
                self.config_manager.set("waveform", "enable_text", dlg.chk_enable_text.isChecked())
                self.config_manager.set("waveform", "artist_text", dlg.txt_artist.text())
                self.config_manager.set("waveform", "song_text", dlg.txt_song.text())
                self.config_manager.set("waveform", "remix_text", dlg.txt_remix.text())
        else:
            from ui.settings_dialogs import PlaceholderSettingsDialog
            dlg = PlaceholderSettingsDialog(self.steps.get(key, ("Cài đặt",))[0], self)
            dlg.exec()

    @pyqtSlot(str)
    def append_log(self, text):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {text}")

    @pyqtSlot(str)
    def update_status(self, text):
        self.status_label.setText(text)
        if "Đang chạy" in text:
            self.status_label.setStyleSheet("font-weight: bold; color: #e74c3c;")
        else:
            self.status_label.setStyleSheet("font-weight: bold; color: #2ecc71;")

    def add_txt_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Chọn file TXT", "", "Text Files (*.txt)")
        if files:
            for f in files:
                item = QListWidgetItem(os.path.basename(f))
                item.setToolTip(f)
                item.setData(Qt.ItemDataRole.UserRole, f)
                self.list_txt.addItem(item)
                
    def remove_txt_files(self):
        for item in self.list_txt.selectedItems():
            self.list_txt.takeItem(self.list_txt.row(item))

    def start_engine(self):
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        
        # Get list of files
        files = []
        for i in range(self.list_txt.count()):
            files.append(self.list_txt.item(i).data(Qt.ItemDataRole.UserRole))
        self.engine.input_files = files
            
        self.engine.start()

    def stop_engine(self):
        self.btn_stop.setEnabled(False)
        self.append_log("Đang dừng tiến trình... (chờ tác vụ hiện tại hoàn tất)")
        self.engine.stop()

    def on_engine_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
