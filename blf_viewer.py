# blf_viewer.py (modified: pause stops reader, resume from last play_pos)
# ベース: 元の blf_viewer.py を修正（参照: uploaded blf_viewer.py） 
# 必要パッケージ: PySide6, pyqtgraph, python-can, cantools, numpy

import sys
import time
import bisect
import csv
import os
import json
from collections import defaultdict

from PySide6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import numpy as np
import cantools
import can

CONFIG_FILE = "blf_viewer_workspace.json"

def _try_get_attr(obj, names, default=None):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default

def get_signal_meta_from_cantools_signal(s):
    meta = {}
    meta['min'] = _try_get_attr(s, ['minimum', 'min', 'physical_minimum'], None)
    meta['max'] = _try_get_attr(s, ['maximum', 'max', 'physical_maximum'], None)
    meta['unit'] = _try_get_attr(s, ['unit'], None)
    meta['choices'] = _try_get_attr(s, ['choices', 'enum', 'values'], None)
    return meta

class TimeAxisItem(pg.AxisItem):
    def __init__(self, orientation='bottom', *args, **kwargs):
        super().__init__(orientation=orientation, *args, **kwargs)

    def tickStrings(self, values, scale, spacing):
        out = []
        for v in values:
            try:
                t = float(v)
            except Exception:
                out.append(str(v)); continue
            if t >= 3600:
                h = int(t // 3600)
                m = int((t % 3600) // 60)
                out.append(f"{h}:{m:02d}:{t%60:06.3f}")
            elif t >= 60:
                m = int(t // 60)
                out.append(f"{m}:{t%60:06.3f}")
            else:
                out.append(f"{t:.3f}")
        return out

class BLFReaderThread(QtCore.QThread):
    """
    BLF ファイルを逐次読み込み、指定した signal keys に該当するデータをバッチで main thread に渡す。
    変更点:
      - resume_time (秒) を受け取り、ts < resume_time のメッセージは読み捨てることで「指定時刻から再開」を実装。
      - stop() を呼ぶと _running を False にしてループを抜ける（即時停止を試みる）。
    """
    data_batch_ready = QtCore.Signal(dict, float)
    progress = QtCore.Signal(int)
    finished = QtCore.Signal()
    error = QtCore.Signal(str)

    def __init__(self, blf_path, dbs_info, target_keys, resume_time=None, parent=None):
        super().__init__(parent)
        self.blf_path = blf_path
        self.dbs_info = dbs_info
        self.target_keys = set(target_keys)
        self._running = True
        self.resume_time = resume_time  # seconds from start (relative timestamp), None means start from beginning
        self.target_messages = defaultdict(list)
        for db, dbc_name in self.dbs_info:
            for msg in db.messages:
                self.target_messages[msg.frame_id].append((msg, dbc_name))

    def stop(self):
        # main thread が呼ぶ（Pause のとき等）
        self._running = False

    def run(self):
        try:
            reader = can.BLFReader(self.blf_path)
        except Exception as e:
            self.error.emit(f"BLF Open Error: {e}")
            self.finished.emit()
            return

        count = 0
        base_ts = None
        max_ts = 0.0
        batch = defaultdict(lambda: {"t": [], "v": []})

        for msg in reader:
            # 停止要求があればすぐ抜ける
            if not self._running:
                break

            try:
                raw_ts = float(msg.timestamp)
            except Exception:
                # タイムスタンプが取得できないならスキップ
                continue

            if base_ts is None:
                base_ts = raw_ts
            ts = raw_ts - base_ts

            # resume_time が指定されていれば、それより前のデータは読み捨てる（スキップ）
            if self.resume_time is not None and ts < self.resume_time:
                continue

            arb = msg.arbitration_id
            msgs_to_try = self.target_messages.get(arb)
            if not msgs_to_try:
                continue

            count += 1
            if ts > max_ts:
                max_ts = ts

            for message, dbc_name in msgs_to_try:
                try:
                    decoded = message.decode(msg.data)
                    for sname, sval in decoded.items():
                        key = f"{dbc_name}:{arb}:{message.name}:{sname}"
                        if key in self.target_keys:
                            batch[key]["t"].append(ts)
                            batch[key]["v"].append(sval)
                except Exception:
                    # デコードエラー等は無視して先に進める
                    pass

            # 定期的にバッチを送る（負荷を分散）
            if count % 10000 == 0:
                self.data_batch_ready.emit(dict(batch), max_ts)
                batch.clear()
                self.progress.emit(count)

        # 残りを送る
        if batch:
            self.data_batch_ready.emit(dict(batch), max_ts)

        try:
            reader.close()
        except Exception:
            pass

        self.finished.emit()

class CheckableListWidget(QtWidgets.QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.last_clicked_row = -1
        self.itemClicked.connect(self.on_item_clicked)

    def on_item_clicked(self, item):
        current_row = self.row(item)
        modifiers = QtGui.QGuiApplication.keyboardModifiers()

        if modifiers & QtCore.Qt.ShiftModifier and self.last_clicked_row != -1:
            start_row = min(self.last_clicked_row, current_row)
            end_row = max(self.last_clicked_row, current_row)
            target_state = item.checkState()
            for r in range(start_row, end_row + 1):
                target_item = self.item(r)
                if not target_item.isHidden():
                    target_item.setCheckState(target_state)

        self.last_clicked_row = current_row

class SignalSelectionDialog(QtWidgets.QDialog):
    def __init__(self, dbc_name, frame_id, frame_name, signals, selected_keys, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Select Signals - {frame_name} ({hex(frame_id)})")
        self.resize(350, 400)
        self.result_keys = []
        layout = QtWidgets.QVBoxLayout(self)
        self.list_widget = CheckableListWidget()
        layout.addWidget(self.list_widget)

        for sig in signals:
            key = f"{dbc_name}:{frame_id}:{frame_name}:{sig['name']}"
            unit_str = f" [{sig['unit']}]" if sig['unit'] else ""
            item = QtWidgets.QListWidgetItem(f"{sig['name']}{unit_str}")
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            if key in selected_keys:
                item.setCheckState(QtCore.Qt.Checked)
            else:
                item.setCheckState(QtCore.Qt.Unchecked)
            item.setData(QtCore.Qt.UserRole, key)
            self.list_widget.addItem(item)

        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def accept(self):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == QtCore.Qt.Checked:
                self.result_keys.append(item.data(QtCore.Qt.UserRole))
        super().accept()

class GlobalSearchDialog(QtWidgets.QDialog):
    def __init__(self, meta_dict, selected_keys, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Global Search (Frame / Signal)")
        self.resize(500, 500)
        self.meta_dict = meta_dict
        self.result_keys = []
        layout = QtWidgets.QVBoxLayout(self)
        self.search_input = QtWidgets.QLineEdit()
        self.search_input.setPlaceholderText("Search by frame name, ID, or signal name...")
        self.search_input.textChanged.connect(self.filter_list)
        layout.addWidget(self.search_input)
        self.list_widget = CheckableListWidget()
        layout.addWidget(self.list_widget)

        for key, m in self.meta_dict.items():
            fid_hex = hex(m['frame_id'])
            dbc_name = m.get('dbc_name', 'Unknown')
            display_text = f"[{dbc_name}] {fid_hex} : {m['msg']} . {m['sig']}"
            item = QtWidgets.QListWidgetItem(display_text)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            if key in selected_keys:
                item.setCheckState(QtCore.Qt.Checked)
            else:
                item.setCheckState(QtCore.Qt.Unchecked)
            item.setData(QtCore.Qt.UserRole, key)
            self.list_widget.addItem(item)

        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def filter_list(self, text):
        query = text.lower()
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setHidden(query not in item.text().lower())
        self.list_widget.last_clicked_row = -1

    def accept(self):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == QtCore.Qt.Checked:
                self.result_keys.append(item.data(QtCore.Qt.UserRole))
        super().accept()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BLF Viewer")
        self.resize(1400, 850)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        pg.setConfigOptions(antialias=False)

        self.dbs_info = []
        self.dbc_paths = []
        self.meta = {}
        self.data = defaultdict(lambda: {"t": [], "v_raw": [], "v_num": []})
        self.plots = {}
        self.max_time = 0.0
        self.play_pos = 0.0
        self.current_display_start = 0.0
        self.is_workspace_modified = False

        # resume 時刻（秒）を保持。Pause で保存し、Resume で利用する
        self.resume_ts = None

        # initialize active_state for stale hysteresis if needed later
        self.active_state = {}

        self.setup_ui()
        self.load_config()
        self.setup_timers()
        save_shortcut = QtGui.QShortcut(QtGui.QKeySequence.Save, self)
        save_shortcut.activated.connect(self.manual_save)

    def setup_ui(self):
        tb = self.addToolBar("Main")
        tb.setMovable(False)
        act_blf = QtGui.QAction("📂 Open BLF", self); act_blf.triggered.connect(self.open_blf)
        tb.addAction(act_blf)
        act_dbc = QtGui.QAction("➕ Add DBC(s)", self); act_dbc.triggered.connect(self.add_dbc_dialog)
        tb.addAction(act_dbc)
        tb.addSeparator()
        act_search = QtGui.QAction("🔍 Search", self); act_search.triggered.connect(self.open_global_search)
        tb.addAction(act_search)
        tb.addSeparator()
        self.x_mode_combo = QtWidgets.QComboBox()
        self.x_mode_combo.addItems(["Fixed Window", "Follow (Trailing)", "Auto Fit"])
        self.x_mode_combo.currentIndexChanged.connect(self.mark_workspace_modified)
        self.x_mode_combo.currentIndexChanged.connect(lambda: self.update_plots_and_table())
        tb.addWidget(QtWidgets.QLabel(" X-Mode: ")); tb.addWidget(self.x_mode_combo)
        self.window_spin = QtWidgets.QDoubleSpinBox(); self.window_spin.setRange(0.1, 3600.0);
        self.window_spin.setValue(5.0); self.window_spin.setSingleStep(0.5)
        self.window_spin.valueChanged.connect(self.mark_workspace_modified)
        tb.addWidget(QtWidgets.QLabel(" Window(s): ")); tb.addWidget(self.window_spin)
        self.autoy_checkbox = QtWidgets.QCheckBox(" Auto Y-Fit "); self.autoy_checkbox.toggled.connect(self.mark_workspace_modified)
        tb.addWidget(self.autoy_checkbox)
        tb.addSeparator()
        self.stale_spin = QtWidgets.QDoubleSpinBox(); self.stale_spin.setRange(0.1, 3600.0); self.stale_spin.setSingleStep(0.5); self.stale_spin.setValue(2.0)
        self.stale_spin.valueChanged.connect(self.mark_workspace_modified)
        tb.addWidget(QtWidgets.QLabel(" Stale(s): ")); tb.addWidget(self.stale_spin)
        tb.addSeparator()
        self.play_btn = QtWidgets.QPushButton("▶ Play / Load"); self.play_btn.clicked.connect(self.toggle_play)
        tb.addWidget(self.play_btn)
        self.speed_spin = QtWidgets.QDoubleSpinBox(); self.speed_spin.setRange(0.1, 50.0); self.speed_spin.setValue(1.0); self.speed_spin.setSuffix("x")
        self.speed_spin.valueChanged.connect(self.mark_workspace_modified)
        tb.addWidget(QtWidgets.QLabel(" Speed: ")); tb.addWidget(self.speed_spin)
        tb.addSeparator()
        act_export = QtGui.QAction("💾 Export CSV", self); act_export.triggered.connect(self.export_csv); tb.addAction(act_export)

        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central); main_layout.setContentsMargins(4,4,4,4)
        h_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        v_splitter_left = QtWidgets.QSplitter(QtCore.Qt.Vertical)

        # Frames list
        frame_group = QtWidgets.QGroupBox("1. Frames (Double-click)")
        frame_layout = QtWidgets.QVBoxLayout(frame_group)
        self.frame_tree = QtWidgets.QTreeWidget(); self.frame_tree.setHeaderLabels(["DBC","ID","Message"])
        self.frame_tree.setRootIsDecorated(False); self.frame_tree.setUniformRowHeights(True)
        header_tree = self.frame_tree.header()
        header_tree.setSectionResizeMode(0, QtWidgets.QHeaderView.Interactive)
        header_tree.setSectionResizeMode(1, QtWidgets.QHeaderView.Interactive)
        header_tree.setSectionResizeMode(2, QtWidgets.QHeaderView.Interactive)
        header_tree.setStretchLastSection(True)
        self.frame_tree.setColumnWidth(0, 100); self.frame_tree.setColumnWidth(1, 60)
        self.frame_tree.itemDoubleClicked.connect(self.open_signal_popup)
        frame_layout.addWidget(self.frame_tree)
        v_splitter_left.addWidget(frame_group)

        # Active table
        active_group = QtWidgets.QGroupBox("2. Active Signals & Values")
        active_layout = QtWidgets.QVBoxLayout(active_group)
        self.active_table = QtWidgets.QTableWidget(0,5)
        self.active_table.setHorizontalHeaderLabels(["👁","Frame","Signal","Value",""])
        header_tbl = self.active_table.horizontalHeader()
        header_tbl.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header_tbl.setSectionResizeMode(1, QtWidgets.QHeaderView.Interactive)
        header_tbl.setSectionResizeMode(2, QtWidgets.QHeaderView.Interactive)
        header_tbl.setSectionResizeMode(3, QtWidgets.QHeaderView.Interactive)
        header_tbl.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeToContents)
        self.active_table.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.active_table.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.active_table.setColumnWidth(1,100); self.active_table.setColumnWidth(2,120); self.active_table.setColumnWidth(3,80)
        self.active_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.active_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.active_table.itemChanged.connect(self.on_active_table_item_changed)
        active_layout.addWidget(self.active_table)
        v_splitter_left.addWidget(active_group)

        # Plots
        plot_group = QtWidgets.QGroupBox("3. Signal Plots")
        plot_layout = QtWidgets.QVBoxLayout(plot_group)
        self.plot_scroll = QtWidgets.QScrollArea(); self.plot_scroll.setWidgetResizable(True)
        self.plot_container = QtWidgets.QWidget(); self.plot_vbox = QtWidgets.QVBoxLayout(self.plot_container)
        self.plot_vbox.setContentsMargins(0,0,0,0); self.plot_vbox.addStretch()
        self.plot_scroll.setWidget(self.plot_container)
        plot_layout.addWidget(self.plot_scroll)
        slider_layout = QtWidgets.QHBoxLayout()
        self.time_label = QtWidgets.QLabel("Time: 0.000 s")
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.slider.setRange(0,10000); self.slider.sliderMoved.connect(self.on_slider_moved)
        slider_layout.addWidget(self.time_label); slider_layout.addWidget(self.slider)
        plot_layout.addLayout(slider_layout)

        h_splitter.addWidget(v_splitter_left); h_splitter.addWidget(plot_group)
        h_splitter.setSizes([450,950]); main_layout.addWidget(h_splitter)
        self.status = self.statusBar()

    def mark_workspace_modified(self, *args):
        self.is_workspace_modified = True

    def manual_save(self):
        self.save_config()
        self.status.showMessage("Workspace saved successfully.", 3000)

    def load_config(self):
        default_config = {"x_mode":"Fixed Window","window_span":5.0,"auto_y":False,"stale_time":2.0,"speed":1.0,"dbc_paths":[],"blf_path":None,"selected_signals":[]}
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE,"r",encoding="utf-8") as f:
                    user_config = json.load(f); default_config.update(user_config)
            except Exception as e:
                print(f"Failed to load workspace: {e}")
        idx = self.x_mode_combo.findText(default_config["x_mode"])
        if idx >= 0: self.x_mode_combo.setCurrentIndex(idx)
        self.window_spin.setValue(default_config["window_span"]); self.autoy_checkbox.setChecked(default_config["auto_y"])
        self.stale_spin.setValue(default_config["stale_time"]); self.speed_spin.setValue(default_config["speed"])
        valid_dbcs = [p for p in default_config["dbc_paths"] if os.path.exists(p)]
        if valid_dbcs: self._load_dbc_files(valid_dbcs)
        blf_path = default_config["blf_path"]
        if blf_path and os.path.exists(blf_path):
            self.blf_path = blf_path; self.status.showMessage(f"Selected BLF: {self.blf_path}")
        for key in default_config["selected_signals"]:
            if key in self.meta and key not in self.plots:
                self.add_signal_plot(key)
        self.is_workspace_modified = False

    def save_config(self):
        config = {"x_mode":self.x_mode_combo.currentText(),"window_span":self.window_spin.value(),"auto_y":self.autoy_checkbox.isChecked(),"stale_time":self.stale_spin.value(),"speed":self.speed_spin.value(),"dbc_paths":self.dbc_paths,"blf_path":getattr(self,'blf_path',None),"selected_signals":list(self.plots.keys())}
        try:
            with open(CONFIG_FILE,"w",encoding="utf-8") as f: json.dump(config,f,indent=4)
            self.is_workspace_modified = False
        except Exception as e:
            print(f"Failed to save workspace: {e}")

    def closeEvent(self, event):
        # スレッドを停止してから閉じるようにする
        try:
            if hasattr(self, 'reader') and getattr(self, 'reader') is not None and getattr(self, 'reader').isRunning():
                # ユーザーが閉じる際は reader を停止して少し待つ
                self.reader.stop()
                self.reader.wait(1000)
        except Exception:
            pass

        if not self.dbc_paths and not getattr(self,'blf_path',None) and not self.plots:
            event.accept(); return
        if not self.is_workspace_modified: event.accept(); return
        reply = QtWidgets.QMessageBox.question(self,'Save Workspace','You have unsaved changes. Save workspace?', QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No | QtWidgets.QMessageBox.StandardButton.Cancel, QtWidgets.QMessageBox.StandardButton.Yes)
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            self.save_config(); event.accept()
        elif reply == QtWidgets.QMessageBox.StandardButton.No:
            event.accept()
        else:
            event.ignore()

    def setup_timers(self):
        self.is_playing = False
        self.play_timer = QtCore.QTimer(); self.play_timer.timeout.connect(self.advance_playback)
        self.play_timer.setInterval(50)
        self.ui_timer = QtCore.QTimer(); self.ui_timer.setInterval(100); self.ui_timer.timeout.connect(self.update_plots_and_table); self.ui_timer.start()

    def add_dbc_dialog(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self,"Add DBC(s)","","DBC Files (*.dbc)")
        if paths: self._load_dbc_files(paths)

    def _load_dbc_files(self, paths):
        loaded_any = False
        for path in paths:
            if path in self.dbc_paths: continue
            try:
                db = cantools.database.load_file(path); dbc_name = os.path.basename(path)
                self.dbs_info.append((db, dbc_name)); self.dbc_paths.append(path)
                for msg in db.messages:
                    is_first_msg = True
                    display_dbc = dbc_name if is_first_msg else ""
                    is_first_msg = False
                    item = QtWidgets.QTreeWidgetItem(self.frame_tree, [display_dbc, hex(msg.frame_id), msg.name])
                    item.setToolTip(0, dbc_name); item.setToolTip(2, msg.name)
                    item.setData(0, QtCore.Qt.UserRole, msg.frame_id)
                    item.setData(1, QtCore.Qt.UserRole, dbc_name)
                    for s in msg.signals:
                        key = f"{dbc_name}:{msg.frame_id}:{msg.name}:{s.name}"
                        sm = get_signal_meta_from_cantools_signal(s)
                        self.meta[key] = {"dbc_name":dbc_name,"frame_id":msg.frame_id,"msg":msg.name,"sig":s.name,"min":sm.get("min"),"max":sm.get("max"),"unit":sm.get("unit",""),"choices":sm.get("choices")}
                loaded_any = True
                self.status.showMessage(f"Loaded total {len(self.dbs_info)} DBC(s)")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self,"DBC Error",f"Failed to load {path}\n{str(e)}")
        if loaded_any: self.mark_workspace_modified()

    def open_blf(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self,"Open BLF","","BLF Files (*.blf)")
        if path:
            self.blf_path = path; self.status.showMessage(f"Selected BLF: {path}"); self.mark_workspace_modified()

    def open_signal_popup(self, item, column):
        frame_id = item.data(0, QtCore.Qt.UserRole); dbc_name = item.data(1, QtCore.Qt.UserRole); msg_name = item.text(2)
        if frame_id is None or dbc_name is None: return
        signals = [{"name": v["sig"], "unit": v["unit"]} for k,v in self.meta.items() if v["frame_id"]==frame_id and v["dbc_name"]==dbc_name and v["msg"]==msg_name]
        if not signals: return
        dialog = SignalSelectionDialog(dbc_name, frame_id, msg_name, signals, list(self.plots.keys()), self)
        if dialog.exec():
            for key in dialog.result_keys:
                if key not in self.plots: self.add_signal_plot(key)

    def open_global_search(self):
        if not self.meta:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please load a DBC file first."); return
        dialog = GlobalSearchDialog(self.meta, list(self.plots.keys()), self)
        if dialog.exec():
            for key in dialog.result_keys:
                if key not in self.plots: self.add_signal_plot(key)

    def export_csv(self):
        if not self.plots:
            QtWidgets.QMessageBox.warning(self,"No Signals","Please select and plot signals to export."); return
        path,_ = QtWidgets.QFileDialog.getSaveFileName(self,"Save CSV","can_data_export.csv","CSV Files (*.csv')")
        if not path: return
        keys = list(self.plots.keys()); all_times = set()
        for k in keys: all_times.update(self.data[k]["t"])
        sorted_times = sorted(list(all_times))
        try:
            with open(path,"w",newline="") as f:
                writer = csv.writer(f)
                headers = ["Time"] + [f"[{self.meta[k]['dbc_name']}] {hex(self.meta[k]['frame_id'])} {self.meta[k]['msg']}.{self.meta[k]['sig']}" for k in keys]
                writer.writerow(headers)
                for t in sorted_times:
                    row=[t]
                    for k in keys:
                        t_list = self.data[k]["t"]; v_list = self.data[k]["v_raw"]
                        idx = bisect.bisect_right(t_list, t) - 1
                        if idx >= 0: row.append(v_list[idx])
                        else: row.append("")
                    writer.writerow(row)
            self.status.showMessage(f"Exported to {path}"); QtWidgets.QMessageBox.information(self,"Success","CSV Export completed successfully.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self,"Export Error", str(e))

    def add_signal_plot(self, key):
        meta = self.meta[key]
        color = pg.intColor(len(self.plots), hues=15)
        axis = TimeAxisItem(orientation='bottom')
        pw = pg.PlotWidget(axisItems={'bottom': axis}); pw.setFixedHeight(180); pi = pw.getPlotItem()
        pi.showGrid(x=True, y=True, alpha=0.5)
        title = f"[{meta['dbc_name']}] {meta['msg']} . {meta['sig']}"
        if meta['unit']: title += f" [{meta['unit']}]"
        pi.setTitle(title, size="10pt")
        curve = pi.plot([], [], pen=pg.mkPen(color=color, width=2))
        curve.setClipToView(True)
        curve.setDownsampling(ds=True, auto=True, method='peak')
        for other in self.plots.values():
            try:
                pi.setXLink(other['widget'].getPlotItem())
            except Exception:
                pass
        self.plot_vbox.insertWidget(self.plot_vbox.count() - 1, pw)
        # store
        self.plots[key] = {"widget": pw, "curve": curve, "meta": meta, "color": color, "axis_applied": False}
        # apply DBC meta to axis (NEW)
        self.apply_meta_to_axis(key)
        self.add_to_active_table(key)
        self.mark_workspace_modified()

    def apply_meta_to_axis(self, key):
        """
        Apply DBC signal meta (choices/min/max/unit) to the plot's Y axis.
        - If choices (enum) exist: set left axis ticks to the labels and set range.
        - Else if min/max exist: set YRange accordingly with small padding.
        - Else do nothing (auto-y can be applied at draw time).
        """
        if key not in self.plots:
            return
        entry = self.plots[key]
        pi = entry["widget"].getPlotItem()
        meta = entry.get("meta", {}) or {}
        choices = meta.get("choices"); minv = meta.get("min"); maxv = meta.get("max")
        # choices: try to build tick list (numeric value -> label)
        if choices and isinstance(choices, dict) and len(choices) > 0:
            tick_items = []
            numeric_keys = []
            # Cantools choices keys may be ints or strings; normalize
            for kk, lbl in choices.items():
                try:
                    nk = float(kk)
                except Exception:
                    try:
                        nk = float(int(kk))
                    except Exception:
                        continue
                numeric_keys.append(nk)
                tick_items.append((nk, str(lbl)))
            if tick_items:
                tick_items = sorted(tick_items, key=lambda x: x[0])
                try:
                    left_axis = pi.getAxis('left')
                    left_axis.setTicks([tick_items])
                except Exception:
                    pass
                numeric_keys = sorted(set(numeric_keys))
                if numeric_keys:
                    lo = min(numeric_keys); hi = max(numeric_keys)
                    pad = max(0.5, (hi - lo) * 0.05) if hi != lo else 0.5
                    try:
                        pi.setYRange(lo - pad, hi + pad)
                    except Exception:
                        pass
                entry["axis_applied"] = True
                return
        # numeric min/max
        if (isinstance(minv, (int, float)) and isinstance(maxv, (int, float)) and maxv > minv):
            try:
                span = float(maxv) - float(minv)
                pad = span * 0.05 if span > 0 else 0.5
                pi.setYRange(float(minv) - pad, float(maxv) + pad)
            except Exception:
                pass
            entry["axis_applied"] = True
            return
        # else mark as applied = False; auto Y will be used during update_plots_and_table
        entry["axis_applied"] = False

    def add_to_active_table(self, key):
        row = self.active_table.rowCount(); self.active_table.insertRow(row)
        chk_item = QtWidgets.QTableWidgetItem()
        chk_item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
        chk_item.setCheckState(QtCore.Qt.Checked); chk_item.setData(QtCore.Qt.UserRole, key)
        self.active_table.setItem(row, 0, chk_item)
        frame_item = QtWidgets.QTableWidgetItem(self.meta[key]['msg']); frame_item.setToolTip(self.meta[key]['msg'])
        self.active_table.setItem(row, 1, frame_item)
        sig_item = QtWidgets.QTableWidgetItem(self.meta[key]['sig'])
        sig_item.setForeground(QtGui.QBrush(self.plots[key]['color']))
        font = sig_item.font(); font.setBold(True); sig_item.setFont(font)
        sig_item.setToolTip(self.meta[key]['sig']); self.active_table.setItem(row, 2, sig_item)
        val_item = QtWidgets.QTableWidgetItem("-"); val_item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.active_table.setItem(row, 3, val_item)
        del_btn = QtWidgets.QPushButton("✖"); del_btn.setFixedSize(24,24)
        del_btn.clicked.connect(lambda checked, k=key: self.remove_signal(k))
        self.active_table.setCellWidget(row, 4, del_btn)

    def on_active_table_item_changed(self, item):
        if item.column() == 0:
            key = item.data(QtCore.Qt.UserRole)
            if key and key in self.plots:
                is_visible = (item.checkState() == QtCore.Qt.Checked)
                self.plots[key]['widget'].setVisible(is_visible)

    def remove_signal(self, key):
        for r in range(self.active_table.rowCount()):
            item = self.active_table.item(r, 0)
            if item and item.data(QtCore.Qt.UserRole) == key:
                self.active_table.removeRow(r); break
        if key in self.plots:
            pw = self.plots[key]['widget']; self.plot_vbox.removeWidget(pw); pw.deleteLater()
            del self.plots[key]; self.mark_workspace_modified()

    def toggle_play(self):
        """
        Play / Pause のトグル。
        - Play 押下:
            - self.resume_ts が None => 新規読み込み（data をクリアして先頭から読み始める）
            - self.resume_ts がある => Pause 解除（data を保持し resume_ts からスキップして再開）
        - Pause 押下:
            - self.resume_ts = self.play_pos を保存して reader.stop() を呼ぶ（読み取りを止める）
        """
        if not hasattr(self, 'blf_path') or not self.dbs_info:
            QtWidgets.QMessageBox.warning(self,"Warning","Please load at least one DBC and a BLF file first."); return

        if not self.is_playing:
            # Play / Resume
            target_keys = list(self.plots.keys())
            if not target_keys:
                QtWidgets.QMessageBox.information(self,"Info","Please select at least one signal to plot first."); return

            resume_ts = getattr(self, 'resume_ts', None)
            if resume_ts is None:
                # 新規再生: 既存データはクリアして先頭から開始
                self.data.clear(); self.max_time = 0.0; self.play_pos = 0.0
                self.status.showMessage("Starting playback from beginning...")
            else:
                # 再開: data は保持、play_pos は resume_ts のままにする
                self.play_pos = resume_ts
                self.status.showMessage(f"Resuming playback from {resume_ts:.3f} s...")

            # reader に resume_time を渡す（None なら先頭から）
            self.reader = BLFReaderThread(self.blf_path, self.dbs_info, target_keys, resume_time=resume_ts)
            self.reader.data_batch_ready.connect(self.on_data_batch)
            self.reader.progress.connect(lambda n: self.status.showMessage(f"Reading... {n} target frames parsed"))
            # スレッド終了時は _on_reader_finished を呼ぶ
            self.reader.finished.connect(self._on_reader_finished)
            self.reader.error.connect(lambda s: self.status.showMessage(s))
            self.reader.start()
            self.is_playing = True; self.play_btn.setText("⏸ Pause"); self.play_timer.start()
        else:
            # Pause 押下: 読み取りを止め、次回再開時にここから再開する
            try:
                # 現在の再生位置を保存
                self.resume_ts = float(self.play_pos)
            except Exception:
                self.resume_ts = getattr(self, 'play_pos', 0.0)
            try:
                if hasattr(self, 'reader') and self.reader is not None:
                    self.reader.stop()
            except Exception:
                pass
            self.is_playing = False; self.play_btn.setText("▶ Play / Load"); self.play_timer.stop()
            self.status.showMessage(f"Paused at {self.resume_ts:.3f} s")

    def _on_reader_finished(self):
        # スレッド終了後のクリーンアップ
        try:
            # reader が終了したらオブジェクト解放できるようにする
            self.reader = None
        except Exception:
            pass
        # 状態に応じてメッセージを出す
        if getattr(self, 'resume_ts', None) is not None and not self.is_playing:
            self.status.showMessage("Paused")
        else:
            self.status.showMessage("Reading Finished")

    @QtCore.Slot(dict, float)
    def on_data_batch(self, batch, new_max_time):
        # データを受け取って internal buffer に追加。max_time を更新。
        if new_max_time > self.max_time: self.max_time = new_max_time
        for key, new_data in batch.items():
            if key not in self.data: self.data[key] = {"t": [], "v_raw": [], "v_num": []}
            self.data[key]["t"].extend(new_data["t"]); self.data[key]["v_raw"].extend(new_data["v"])
            choices = self.meta.get(key, {}).get("choices")
            nums = []
            for val in new_data["v"]:
                if isinstance(val, (int, float)): nums.append(float(val))
                elif choices:
                    mapped = 0.0
                    for k_c, lbl in choices.items():
                        if str(lbl) == str(val) or k_c == val:
                            try: mapped = float(k_c)
                            except: pass
                            break
                    nums.append(mapped)
                else:
                    try: nums.append(float(val))
                    except: nums.append(0.0)
            self.data[key]["v_num"].extend(nums)
            # After new data arrives, if axis not applied (no DBC min/max/choices), we may still want auto Y next draw.
            # If DBC min/max/choices exist and axis not yet applied, call apply_meta_to_axis to enforce.
            if key in self.plots:
                if not self.plots[key].get("axis_applied", False):
                    self.apply_meta_to_axis(key)

    def advance_playback(self):
        dt = 0.05 * max(0.001, float(self.speed_spin.value()))
        # If reader is running and mode is Follow (Trailing), keep play_pos synced to latest received time
        try:
            mode = self.x_mode_combo.currentText()
        except Exception:
            mode = 'Fixed Window'
        if hasattr(self, 'reader') and getattr(self, 'reader') is not None and getattr(self, 'reader').isRunning() and mode == 'Follow (Trailing)':
            # sync to latest available timestamp for minimal latency
            if self.max_time is not None:
                self.play_pos = self.max_time
        else:
            self.play_pos += dt
            if self.max_time > 0 and self.play_pos > self.max_time:
                self.play_pos = self.max_time
        if self.max_time > 0:
            spos = int((self.play_pos / self.max_time) * 10000) if self.max_time > 0 else 0
            self.slider.blockSignals(True)
            self.slider.setValue(min(10000, spos))
            self.slider.blockSignals(False)

    def on_slider_moved(self, val):
        if self.max_time > 0:
            self.play_pos = (val / 10000.0) * self.max_time
            self.update_plots_and_table()

    def update_plots_and_table(self):
        if not self.plots:
            return
        self.time_label.setText(f"Time: {self.play_pos:.3f} s")
        mode = self.x_mode_combo.currentText()
        win = float(self.window_spin.value())
        stale_threshold = float(self.stale_spin.value())
        auto_y = self.autoy_checkbox.isChecked()

        # --- Follow (Trailing) / Fixed Window / Auto Fit handling ---
        if mode == "Fixed Window":
            start = max(0.0, self.play_pos - win/2)
            end = start + win
        elif mode == "Follow (Trailing)":
            end = max(0.0, self.play_pos)
            start = max(0.0, end - win)
        else:  # Auto Fit
            times = []
            for k in self.plots.keys():
                d = self.data.get(k)
                if d and len(d["t"]) > 0:
                    times.append(min(d["t"])); times.append(max(d["t"]))
            if not times:
                start = 0.0; end = max(1.0, win)
            else:
                start = max(0.0, min(times) - 0.5); end = max(times) + 0.5
                if end - start < 1e-6: end = start + max(0.5, win)

        self.current_display_start = start  # TimeAxisItem will use this for tick labels
        window_span = max(1e-6, end - start)

        for row in range(self.active_table.rowCount()):
            item = self.active_table.item(row, 0)
            if not item:
                continue
            key = item.data(QtCore.Qt.UserRole)
            if not key or key not in self.plots:
                continue
            p_data = self.plots[key]
            if not isinstance(p_data, dict) or 'curve' not in p_data or 'widget' not in p_data:
                continue
            d = self.data.get(key)
            if d and len(d["t"]) > 0:
                t_list = d["t"]
                # choose indices within window
                idx_start = bisect.bisect_left(t_list, start)
                idx_end = bisect.bisect_right(t_list, end)
                if idx_end > idx_start:
                    t_slice = t_list[idx_start:idx_end]
                    v_slice = d["v_num"][idx_start:idx_end]
                    # heavy-data guard
                    if len(t_slice) > 30000:
                        step = max(1, len(t_slice) // 30000)
                        t_slice = t_slice[::step]; v_slice = v_slice[::step]
                    t_plot = np.array(t_slice)
                    v_plot = np.array(v_slice)
                    # Auto Y-scale if enabled or axis not applied
                    if (auto_y or not p_data.get("axis_applied", False)) and len(v_plot) > 0:
                        vmin = np.min(v_plot); vmax = np.max(v_plot)
                        if vmax > vmin:
                            pad = (vmax - vmin) * 0.1
                            try:
                                p_data["widget"].getPlotItem().setYRange(vmin - pad, vmax + pad, padding=0)
                            except Exception:
                                pass
                        else:
                            try:
                                p_data["widget"].getPlotItem().setYRange(vmin - 0.5, vmax + 0.5, padding=0)
                            except Exception:
                                pass
                    # stepify to create hold-like step display
                    if len(t_plot) > 0:
                        t_step = np.repeat(t_plot, 2)[1:]
                        v_step = np.repeat(v_plot, 2)[:-1]
                        p_data["curve"].setData(t_step, v_step)
                else:
                    p_data["curve"].setData([], [])
                try:
                    p_data["widget"].getPlotItem().setXRange(start, end, padding=0)
                except Exception:
                    pass
                # current value display with stale check
                idx_cur = bisect.bisect_right(t_list, self.play_pos) - 1
                if idx_cur >= 0:
                    age = self.play_pos - t_list[idx_cur]
                    is_stale = (age > stale_threshold)
                    raw_val = d["v_raw"][idx_cur]
                    val_str = f"{raw_val:.3f}" if isinstance(raw_val, float) else str(raw_val)
                    unit = p_data['meta'].get('unit', "")
                    if unit:
                        val_str += f" {unit}"
                    if is_stale:
                        val_str += " (stale)"
                        p_data["curve"].setPen(pg.mkPen(color=(160,160,160), width=1.5, style=QtCore.Qt.DashLine))
                    else:
                        p_data["curve"].setPen(pg.mkPen(color=p_data["color"], width=2))
                    val_item = self.active_table.item(row, 3)
                    if val_item:
                        val_item.setText(val_str)

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()