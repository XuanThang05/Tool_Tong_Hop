from __future__ import annotations

import sys
import threading
import traceback
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QTabWidget, QRadioButton, QButtonGroup, QProgressBar, QTextEdit, 
    QMessageBox, QFileDialog, QInputDialog, QDialog, QLineEdit, QSplitter
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QThread, pyqtSlot

from src.checker import CheckResult
from src.config import APP_NAME, MODEL_NAME, ROOT, SUPPORTED_EXT
from src.services import AppService
from src.text_extract import ExtractError


def _fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def _verdict_text(code: str) -> str:
    return {"ok": "Ổn — có thể thêm", "warn": "Cảnh báo", "block": "Trùng cao — chặn"}.get(
        code, code
    )


class WorkerSignals(QObject):
    progress = pyqtSignal(str, float)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)


class Worker(QThread):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.is_running = True

    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
            if self.is_running:
                self.signals.finished.emit(result)
        except Exception as exc:
            if self.is_running:
                tb = traceback.format_exc()
                self.signals.error.emit(f"{exc}\n\n{tb[-400:]}")


class PasteDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dán nội dung")
        self.resize(720, 520)
        self.data = None
        
        layout = QVBoxLayout(self)
        
        layout.addWidget(QLabel("Tiêu đề:"))
        self.title_edit = QLineEdit("Nội dung dán")
        layout.addWidget(self.title_edit)
        
        layout.addWidget(QLabel("Nội dung truyện:"))
        self.text_edit = QTextEdit()
        layout.addWidget(self.text_edit)
        
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("Dùng nội dung này")
        cancel_btn = QPushButton("Hủy")
        ok_btn.clicked.connect(self.accept_data)
        cancel_btn.clicked.connect(self.reject)
        
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
    def accept_data(self):
        text = self.text_edit.toPlainText().strip()
        title = self.title_edit.text().strip() or "Nội dung dán"
        if not text:
            QMessageBox.information(self, APP_NAME, "Chưa có nội dung.")
            return
        self.data = (title, text)
        self.accept()


class App(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.service = AppService()
        self.setWindowTitle(APP_NAME)
        self.resize(1240, 760)
        self.setMinimumSize(980, 620)

        self.selected_nb: int | None = None
        self._busy = False
        self._pending_paths: list[Path] = []
        self._pending_paste: tuple[str, str] | None = None
        self._results: list[CheckResult] = []
        self._result_map: dict[int, CheckResult] = {}
        self._doc_map: dict[int, int] = {}
        self._nb_map: dict[int, int] = {}
        self._worker: Worker | None = None

        self._warn, self._block = self.service.thresholds()

        self._build()
        self.refresh_notebooks()
        
        # Warmup delay
        self.status_lbl.setText("Đang khởi động…")
        self._warmup()

    def _build(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # Sidebar
        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)

        title_lbl = QLabel("SỔ / KHO DỮ LIỆU")
        font = title_lbl.font()
        font.setPointSize(12)
        font.setBold(True)
        title_lbl.setFont(font)
        side_layout.addWidget(title_lbl)
        
        desc_lbl = QLabel("Chọn sổ để xem kho và thêm truyện.")
        desc_lbl.setWordWrap(True)
        side_layout.addWidget(desc_lbl)

        self.nb_table = QTableWidget(0, 2)
        self.nb_table.setHorizontalHeaderLabels(["Tên sổ", "Truyện"])
        self.nb_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.nb_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.nb_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.nb_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.nb_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.nb_table.verticalHeader().setVisible(True) # Show row numbers
        self.nb_table.itemSelectionChanged.connect(self._on_nb_select)
        side_layout.addWidget(self.nb_table)

        btn_layout = QHBoxLayout()
        create_btn = QPushButton("Tạo sổ")
        create_btn.clicked.connect(self._create_notebook)
        rename_btn = QPushButton("Đổi tên")
        rename_btn.clicked.connect(self._rename_notebook)
        delete_btn = QPushButton("Xóa sổ")
        delete_btn.clicked.connect(self._delete_notebook)
        
        btn_layout.addWidget(create_btn)
        btn_layout.addWidget(rename_btn)
        btn_layout.addWidget(delete_btn)
        side_layout.addLayout(btn_layout)
        
        splitter.addWidget(side)

        # Right Panel
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(10, 0, 0, 0)

        self.tabs = QTabWidget()
        right_layout.addWidget(self.tabs, stretch=1)

        tab_lib = QWidget()
        tab_chk = QWidget()
        tab_set = QWidget()

        self.tabs.addTab(tab_lib, "Kho truyện")
        self.tabs.addTab(tab_chk, "Check trùng")
        self.tabs.addTab(tab_set, "Cài đặt")

        self._build_library(tab_lib)
        self._build_check(tab_chk)
        self._build_settings(tab_set)

        # Status bar
        self.status_lbl = QLabel("")
        right_layout.addWidget(self.status_lbl)

        splitter.addWidget(right)
        splitter.setSizes([260, 960])

    def _build_library(self, parent: QWidget) -> None:
        layout = QVBoxLayout(parent)
        
        bar = QHBoxLayout()
        btn1 = QPushButton("Thêm file")
        btn1.clicked.connect(self._lib_add_files)
        btn2 = QPushButton("Thêm thư mục")
        btn2.clicked.connect(self._lib_add_folder)
        btn3 = QPushButton("Dán nội dung")
        btn3.clicked.connect(self._lib_paste)
        btn4 = QPushButton("Đọc")
        btn4.clicked.connect(self._lib_read)
        btn5 = QPushButton("Xóa")
        btn5.clicked.connect(self._lib_delete)
        
        bar.addWidget(btn1)
        bar.addWidget(btn2)
        bar.addWidget(btn3)
        bar.addWidget(btn4)
        bar.addWidget(btn5)
        bar.addStretch()
        layout.addLayout(bar)

        self.lib_hint = QLabel("Chưa chọn sổ.")
        layout.addWidget(self.lib_hint)

        self.doc_table = QTableWidget(0, 4)
        self.doc_table.setHorizontalHeaderLabels(["Truyện", "Ký tự", "Nguồn", "Ngày thêm"])
        self.doc_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.doc_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.doc_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.doc_table.verticalHeader().setVisible(True)
        self.doc_table.cellDoubleClicked.connect(self._lib_read)
        layout.addWidget(self.doc_table, stretch=1)

    def _build_check(self, parent: QWidget) -> None:
        layout = QVBoxLayout(parent)

        top = QHBoxLayout()
        top.addWidget(QLabel("<b>Phạm vi:</b>"))
        self.scope_nb = QRadioButton("Trong sổ đang chọn")
        self.scope_nb.setChecked(True)
        self.scope_all = QRadioButton("Tất cả sổ (chéo sổ)")
        self.scope_group = QButtonGroup()
        self.scope_group.addButton(self.scope_nb)
        self.scope_group.addButton(self.scope_all)
        top.addWidget(self.scope_nb)
        top.addWidget(self.scope_all)
        top.addStretch()
        layout.addLayout(top)

        src = QHBoxLayout()
        btn_f = QPushButton("Chọn file")
        btn_f.clicked.connect(self._chk_files)
        btn_d = QPushButton("Chọn thư mục")
        btn_d.clicked.connect(self._chk_folder)
        btn_p = QPushButton("Dán nội dung")
        btn_p.clicked.connect(self._chk_paste)
        src.addWidget(btn_f)
        src.addWidget(btn_d)
        src.addWidget(btn_p)
        self.source_lbl = QLabel("Chưa chọn nội dung check.")
        src.addWidget(self.source_lbl)
        src.addStretch()
        layout.addLayout(src)

        run_lyt = QHBoxLayout()
        self.check_btn = QPushButton("Bắt đầu check")
        self.check_btn.setMinimumWidth(150)
        self.check_btn.clicked.connect(self._run_check)
        self.add_btn = QPushButton("Thêm vào sổ")
        self.add_btn.clicked.connect(self._add_selected_result)
        self.clear_btn = QPushButton("Xóa bảng kết quả")
        self.clear_btn.clicked.connect(self._clear_results)
        
        run_lyt.addWidget(self.check_btn)
        run_lyt.addWidget(self.add_btn)
        run_lyt.addWidget(self.clear_btn)
        run_lyt.addStretch()
        
        self.progress = QProgressBar()
        self.progress.setFixedWidth(180)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        run_lyt.addWidget(self.progress)
        self.prog_lbl = QLabel("")
        run_lyt.addWidget(self.prog_lbl)
        layout.addLayout(run_lyt)

        self.verdict_lbl = QLabel("Chưa check.")
        font = self.verdict_lbl.font()
        font.setPointSize(11)
        font.setBold(True)
        self.verdict_lbl.setFont(font)
        self.verdict_lbl.setStyleSheet("padding: 8px; border: 1px solid #ccc; background-color: #f5f5f5;")
        layout.addWidget(self.verdict_lbl)

        self.res_table = QTableWidget(0, 6)
        self.res_table.setHorizontalHeaderLabels(["Nguồn check", "Sổ khớp", "Truyện khớp", "Điểm", "Kiểu", "Mức"])
        self.res_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.res_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.res_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.res_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.res_table.verticalHeader().setVisible(True)
        self.res_table.itemSelectionChanged.connect(self._fill_spans)
        self.res_table.cellDoubleClicked.connect(self._open_match)
        layout.addWidget(self.res_table, stretch=1)

        layout.addWidget(QLabel("Đoạn giống nhất (điểm đã hiệu chỉnh)"))
        self.span_box = QTextEdit()
        self.span_box.setFixedHeight(140)
        self.span_box.setReadOnly(True)
        self.span_box.setText("Chưa check.")
        layout.addWidget(self.span_box)

    def _build_settings(self, parent: QWidget) -> None:
        layout = QVBoxLayout(parent)
        self.set_model = QLabel(f"Model: {MODEL_NAME}")
        layout.addWidget(self.set_model)
        self.set_device = QLabel("Thiết bị: …")
        layout.addWidget(self.set_device)
        self.set_stats = QLabel("Kho: …")
        layout.addWidget(self.set_stats)
        
        lbl = QLabel(f"Thư mục dữ liệu: {ROOT / 'data'}")
        lbl.setWordWrap(True)
        layout.addWidget(lbl)
        
        lbl2 = QLabel("Định dạng đọc: " + ", ".join(sorted(SUPPORTED_EXT)))
        lbl2.setWordWrap(True)
        layout.addWidget(lbl2)
        
        lbl3 = QLabel("Check so khớp nguyên bản, gần giống, và viết lại kiểu đổi từ/câu.\nKho càng nhiều truyện, lần check sau càng có cơ sở.")
        lbl3.setWordWrap(True)
        layout.addWidget(lbl3)
        layout.addStretch()

    def _set_prog(self, msg: str, frac: float) -> None:
        self.progress.setValue(int(max(0.0, min(1.0, frac)) * 100))
        self.prog_lbl.setText(msg)
        if msg:
            self.status_lbl.setText(msg)

    def _fail(self, msg: str) -> None:
        self._busy = False
        self.check_btn.setEnabled(True)
        self.status_lbl.setText("Lỗi xảy ra")
        QMessageBox.critical(self, APP_NAME, msg)

    def _warmup(self) -> None:
        def work():
            self.service.warmup(lambda msg, frac: self._worker.signals.progress.emit(msg, frac))
            return True
            
        self._run_bg(work, lambda _: self._after_warmup(), check_ready=False)

    def _after_warmup(self) -> None:
        st = self.service.stats()
        self.set_device.setText(f"Thiết bị: {self.service.device_label()}")
        self._refresh_stats()
        self.status_lbl.setText(
            f"Sẵn sàng · {self.service.device_label()} · {st['notebooks']} sổ · {st['documents']} truyện"
        )
        self._set_prog("", 0)

    def _refresh_stats(self) -> None:
        st = self.service.stats()
        self.set_stats.setText(
            f"Kho: {st['notebooks']} sổ, { _fmt_int(st['documents'])} truyện, "
            f"{_fmt_int(st['chunks'])} đoạn, {_fmt_int(st['chars'])} ký tự"
        )

    def refresh_notebooks(self) -> None:
        self.nb_table.setRowCount(0)
        self._nb_map.clear()
        
        notebooks = self.service.list_notebooks()
        if self.selected_nb not in {n["id"] for n in notebooks}:
            self.selected_nb = notebooks[0]["id"] if notebooks else None
            
        for i, nb in enumerate(notebooks):
            self.nb_table.insertRow(i)
            self.nb_table.setItem(i, 0, QTableWidgetItem(nb["name"]))
            self.nb_table.setItem(i, 1, QTableWidgetItem(_fmt_int(nb["doc_count"])))
            self._nb_map[i] = int(nb["id"])
            if self.selected_nb == nb["id"]:
                self.nb_table.selectRow(i)
                
        self.refresh_documents()
        self._refresh_stats()

    def _on_nb_select(self) -> None:
        sel = self.nb_table.selectedItems()
        if sel:
            row = sel[0].row()
            nid = self._nb_map.get(row)
            if nid != self.selected_nb:
                self.selected_nb = nid
                self.refresh_documents()

    def refresh_documents(self) -> None:
        self.doc_table.setRowCount(0)
        self._doc_map.clear()
        if self.selected_nb is None:
            self.lib_hint.setText("Chưa chọn sổ.")
            return
            
        docs = self.service.list_documents(self.selected_nb)
        nb = self.service.db.get_notebook(self.selected_nb)
        name = nb["name"] if nb else ""
        self.lib_hint.setText(f"Sổ «{name}»: {_fmt_int(len(docs))} truyện. Double-click để đọc.")
        
        for i, d in enumerate(docs):
            self.doc_table.insertRow(i)
            src = Path(d["source_path"]).name if d["source_path"] else "—"
            self.doc_table.setItem(i, 0, QTableWidgetItem(d["title"]))
            self.doc_table.setItem(i, 1, QTableWidgetItem(_fmt_int(d["char_count"])))
            self.doc_table.setItem(i, 2, QTableWidgetItem(src))
            self.doc_table.setItem(i, 3, QTableWidgetItem(d["created_at"]))
            self._doc_map[i] = int(d["id"])

    def _create_notebook(self) -> None:
        name, ok = QInputDialog.getText(self, "Tạo sổ", "Tên sổ / kho dữ liệu:")
        if not ok or not name:
            return
        try:
            nid = self.service.create_notebook(name)
            self.selected_nb = nid
            self.refresh_notebooks()
        except Exception as exc:
            QMessageBox.critical(self, APP_NAME, str(exc))

    def _rename_notebook(self) -> None:
        if self.selected_nb is None:
            QMessageBox.information(self, APP_NAME, "Hãy chọn một sổ.")
            return
        nb = self.service.db.get_notebook(self.selected_nb)
        name, ok = QInputDialog.getText(self, "Đổi tên sổ", "Tên mới:", text=str(nb["name"]) if nb else "")
        if not ok or not name:
            return
        try:
            self.service.rename_notebook(self.selected_nb, name)
            self.refresh_notebooks()
        except Exception as exc:
            QMessageBox.critical(self, APP_NAME, str(exc))

    def _delete_notebook(self) -> None:
        if self.selected_nb is None:
            return
        nb = self.service.db.get_notebook(self.selected_nb)
        reply = QMessageBox.question(
            self, APP_NAME, f"Xóa sổ «{nb['name'] if nb else ''}» và toàn bộ truyện trong sổ này?"
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.service.delete_notebook(self.selected_nb)
            self.selected_nb = None
            self.refresh_notebooks()

    def _filetypes(self):
        exts = " ".join(f"*{e}" for e in sorted(SUPPORTED_EXT))
        return f"Truyện ({exts});;Tất cả (*.*)"

    def _lib_add_files(self) -> None:
        if self.selected_nb is None:
            QMessageBox.information(self, APP_NAME, "Hãy chọn hoặc tạo một sổ trước.")
            return
        paths, _ = QFileDialog.getOpenFileNames(self, "Chọn file truyện", filter=self._filetypes())
        if paths:
            self._import_paths([Path(p) for p in paths])

    def _lib_add_folder(self) -> None:
        if self.selected_nb is None:
            QMessageBox.information(self, APP_NAME, "Hãy chọn hoặc tạo một sổ trước.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Chọn thư mục truyện")
        if folder:
            self._import_paths([Path(folder)])

    def _lib_paste(self) -> None:
        if self.selected_nb is None:
            QMessageBox.information(self, APP_NAME, "Hãy chọn hoặc tạo một sổ trước.")
            return
        dlg = PasteDialog(self)
        if dlg.exec():
            title, text = dlg.data
            self._run_bg(
                lambda: self.service.add_text(
                    self.selected_nb, title, text, None, 
                    lambda m, f: self._worker.signals.progress.emit(m, f)
                ),
                lambda _: self._after_import(1, []),
            )

    def _lib_read(self) -> None:
        sel = self.doc_table.selectedItems()
        if not sel:
            return
        doc_id = self._doc_map.get(sel[0].row())
        if not doc_id:
            return
        try:
            title, text = self.service.read_document(doc_id)
            self._open_text(title, text)
        except Exception as exc:
            QMessageBox.critical(self, APP_NAME, f"Lỗi khi đọc truyện: {exc}")

    def _lib_delete(self) -> None:
        sel = self.doc_table.selectedRanges()
        if not sel:
            return
        rows = []
        for r in sel:
            rows.extend(range(r.topRow(), r.bottomRow() + 1))
        rows = list(set(rows))
        
        reply = QMessageBox.question(self, APP_NAME, f"Xóa {len(rows)} truyện khỏi sổ?")
        if reply == QMessageBox.StandardButton.Yes:
            for r in rows:
                doc_id = self._doc_map.get(r)
                if doc_id:
                    self.service.delete_document(doc_id)
            self.refresh_notebooks()

    def _import_paths(self, paths: list[Path]) -> None:
        def work():
            return self.service.import_paths(
                self.selected_nb, paths, 
                lambda m, f: self._worker.signals.progress.emit(m, f)
            )

        self._run_bg(work, lambda pair: self._after_import(*pair))

    def _after_import(self, ok: int, errors: list[str]) -> None:
        self.refresh_notebooks()
        self._set_prog("", 0)
        extra = ""
        if errors:
            extra = "\n\n" + "\n".join(errors[:12])
            if len(errors) > 12:
                extra += f"\n… và {len(errors) - 12} lỗi nữa"
        QMessageBox.information(self, APP_NAME, f"Đã thêm {ok} truyện.{extra}")

    def _chk_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "File cần check", filter=self._filetypes())
        if paths:
            self._pending_paths = [Path(p) for p in paths]
            self._pending_paste = None
            self.source_lbl.setText(f"{len(self._pending_paths)} file")
        elif not self._pending_paths and not self._pending_paste:
            self.source_lbl.setText("Chưa chọn file")

    def _chk_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Thư mục cần check")
        if folder:
            self._pending_paths = [Path(folder)]
            self._pending_paste = None
            self.source_lbl.setText(f"Thư mục: {Path(folder).name}")
        elif not self._pending_paths and not self._pending_paste:
            self.source_lbl.setText("Chưa chọn file")

    def _chk_paste(self) -> None:
        dlg = PasteDialog(self)
        if dlg.exec():
            self._pending_paste = dlg.data
            self._pending_paths = []
            self.source_lbl.setText(f"Dán: {dlg.data[0]} ({_fmt_int(len(dlg.data[1]))} ký tự)")

    def _run_check(self) -> None:
        scope_nb = self.selected_nb if self.scope_nb.isChecked() else None
        if self.scope_nb.isChecked() and self.selected_nb is None:
            QMessageBox.information(self, APP_NAME, "Hãy chọn sổ để check trong sổ.")
            return
        if not self._pending_paths and not self._pending_paste:
            QMessageBox.information(self, APP_NAME, "Hãy chọn file, thư mục, hoặc dán nội dung.")
            return

        def work():
            if self._pending_paste:
                title, text = self._pending_paste
                return [
                    self.service.check_text(
                        text, title, notebook_id=scope_nb,
                        warn=self._warn, block=self._block,
                        progress=lambda m, f: self._worker.signals.progress.emit(m, f)
                    )
                ]
            return self.service.check_paths(
                self._pending_paths, notebook_id=scope_nb,
                warn=self._warn, block=self._block,
                progress=lambda m, f: self._worker.signals.progress.emit(m, f)
            )

        self._run_bg(work, self._show_results)

    def _show_results(self, results: list[CheckResult]) -> None:
        self._results.extend(results)
        
        worst = "ok"
        best_score = 0.0
        best_name = ""
        total_t = 0.0
        
        row_idx = self.res_table.rowCount()
        start_row = row_idx
        for res in results:
            total_t += res.elapsed
            if res.verdict == "block":
                worst = "block"
            elif res.verdict == "warn" and worst != "block":
                worst = "warn"
            if res.overall_score > best_score:
                best_score = res.overall_score
                best_name = res.matches[0].title if res.matches else ""
                
            if not res.matches:
                self.res_table.insertRow(row_idx)
                
                item0 = QTableWidgetItem(res.title)
                item0.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                item0.setCheckState(Qt.CheckState.Unchecked)
                
                self.res_table.setItem(row_idx, 0, item0)
                self.res_table.setItem(row_idx, 1, QTableWidgetItem("—"))
                self.res_table.setItem(row_idx, 2, QTableWidgetItem("—"))
                
                item3 = QTableWidgetItem(f"{res.overall_score:.1f}%")
                item3.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.res_table.setItem(row_idx, 3, item3)
                
                item4 = QTableWidgetItem("Không khớp")
                item4.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.res_table.setItem(row_idx, 4, item4)
                
                item5 = QTableWidgetItem(_verdict_text(res.verdict))
                item5.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.res_table.setItem(row_idx, 5, item5)
                
                self._result_map[row_idx] = res
                row_idx += 1
                continue
                
            for match in res.matches:
                v = res.verdict if match is res.matches[0] else ""
                self.res_table.insertRow(row_idx)
                
                item0 = QTableWidgetItem(res.title)
                item0.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                item0.setCheckState(Qt.CheckState.Unchecked)
                self.res_table.setItem(row_idx, 0, item0)
                
                item5 = QTableWidgetItem(_verdict_text(res.verdict) if v else "")
                item5.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.res_table.setItem(row_idx, 5, item5)
                
                self.res_table.setItem(row_idx, 1, QTableWidgetItem(match.notebook_name))
                self.res_table.setItem(row_idx, 2, QTableWidgetItem(match.title))
                
                item3 = QTableWidgetItem(f"{match.score:.1f}%")
                item3.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.res_table.setItem(row_idx, 3, item3)
                
                item4 = QTableWidgetItem(match.kind)
                item4.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.res_table.setItem(row_idx, 4, item4)
                
                self._result_map[row_idx] = res
                row_idx += 1

        extra = f" Khớp cao nhất {best_score:.1f}% với «{best_name}»." if best_name else ""
        note = results[0].note if len(results) == 1 and results[0].note else ""
        msg = f"Kết luận: {_verdict_text(worst)} · {len(results)} file · {total_t:.1f}s.{extra} {note}"
        
        color = {"ok": "#e8f5e9", "warn": "#fff8e1", "block": "#ffebee"}.get(worst, "#f5f5f5")
        self.verdict_lbl.setStyleSheet(f"padding: 8px; border: 1px solid #ccc; background-color: {color};")
        self.verdict_lbl.setText(msg.strip())
        
        self._set_prog("", 0)
        self.status_lbl.setText(msg.strip())
        
        if results and self.res_table.rowCount() > start_row:
            self.res_table.selectRow(start_row)
            self._fill_spans(results[0])

    def _fill_spans(self, res: CheckResult | None = None) -> None:
        if res is None:
            res = self._selected_result()
            
        if not res or not getattr(res, "top_spans", None):
            self.span_box.setText("Không có đoạn giống để hiện.")
        else:
            lines = [
                (
                    f"{i}. {sp.score:.1f}%\n"
                    f"  File check: {sp.query_excerpt}\n"
                    f"  Trong kho:  {sp.source_excerpt}"
                )
                for i, sp in enumerate(res.top_spans, 1)
            ]
            self.span_box.setText("\n\n".join(lines))

    def _selected_result(self) -> CheckResult | None:
        sel = self.res_table.selectedItems()
        if not sel:
            if len(self._results) == 1:
                return self._results[0]
            return None
        return self._result_map.get(sel[0].row())

    def _add_selected_result(self) -> None:
        if self.selected_nb is None:
            QMessageBox.information(self, APP_NAME, "Hãy chọn hoặc tạo một sổ trước.")
            return

        checked_results = []
        seen = set()
        for row in range(self.res_table.rowCount()):
            item = self.res_table.item(row, 0)
            if item and item.checkState() == Qt.CheckState.Checked:
                res = self._result_map.get(row)
                if res and id(res) not in seen:
                    checked_results.append(res)
                    seen.add(id(res))
                    
        if not checked_results:
            QMessageBox.information(self, APP_NAME, "Hãy tích chọn ít nhất một file trong bảng kết quả.")
            return
            
        def work():
            ok = 0
            errors: list[str] = []
            for i, res in enumerate(checked_results, start=1):
                self._worker.signals.progress.emit(f"Thêm {i}/{len(checked_results)}: {res.title}", i / len(checked_results))
                try:
                    self.service.add_text(
                        self.selected_nb, res.title, res.text, res.source_path
                    )
                    ok += 1
                except Exception as exc:
                    errors.append(f"{res.title}: {exc}")
            return ok, errors

        self._run_bg(work, lambda pair: self._after_import(*pair))

    def _clear_results(self) -> None:
        self._results.clear()
        self._result_map.clear()
        self.res_table.setRowCount(0)
        self.verdict_lbl.setText("Đã xóa bảng.")
        self.verdict_lbl.setStyleSheet("padding: 8px; border: 1px solid #ccc; background-color: #f5f5f5;")
        self.span_box.setText("Chưa check.")

    def _open_match(self) -> None:
        sel = self.res_table.selectedItems()
        if not sel:
            return
        row = sel[0].row()
        res = self._result_map.get(row)
        if not res:
            return
            
        col = self.res_table.currentColumn()
        if col == 0 or not res.matches:
            self._open_text(f"[Văn bản check] {res.title}", res.text)
            return

        match_title_item = self.res_table.item(row, 2)
        match_nb_item = self.res_table.item(row, 1)
        if not match_title_item or not match_nb_item:
            return

        match_title = match_title_item.text()
        match_nb = match_nb_item.text()

        if match_title == "—":
            self._open_text(f"[Văn bản check] {res.title}", res.text)
            return
        
        for m in res.matches:
            if m.title == match_title and m.notebook_name == match_nb:
                try:
                    title, text = self.service.read_document(m.document_id)
                    self._open_text(f"{m.notebook_name} / {title}", text)
                except Exception as exc:
                    QMessageBox.critical(self, APP_NAME, f"Lỗi khi mở nội dung: {exc}")
                return

    def _run_bg(self, fn, on_done, check_ready=True) -> None:
        if self._busy:
            QMessageBox.information(self, APP_NAME, "Đang xử lý, đợi xong đã.")
            return
        if check_ready and not self.service.ready():
            QMessageBox.information(self, APP_NAME, "Model chưa sẵn sàng, đợi vài giây.")
            return
            
        self._busy = True
        self.check_btn.setEnabled(False)

        self._worker = Worker(fn)
        self._worker.signals.progress.connect(self._set_prog)
        self._worker.signals.error.connect(self._fail)
        
        def finish(result):
            try:
                self._busy = False
                self.check_btn.setEnabled(True)
                on_done(result)
            except RuntimeError:
                pass
            
        self._worker.signals.finished.connect(finish)
        self._worker.start()

    def _open_text(self, title: str, text: str) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(820, 640)
        
        layout = QVBoxLayout(dlg)
        box = QTextEdit()
        font = box.font()
        font.setPointSize(11)
        box.setFont(font)
        box.setReadOnly(True)
        box.setText(text)
        layout.addWidget(box)
        dlg.show()

    def closeEvent(self, event) -> None:
        if self._worker:
            self._worker.is_running = False
            self._worker.quit()
            self._worker.wait()
        try:
            self.service.close()
        finally:
            event.accept()


def run() -> None:
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec())
