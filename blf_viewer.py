# blf_viewer.py
# Updated: Cleaned up for static data analysis. Allows free sliding/zooming of loaded parts during BLF loading.
# Requirements: PySide6, pyqtgraph, python-can, cantools, numpy

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
MAX_TIMELINE = 500_000  # 解析用に保持サンプル数を拡張

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
                out.append(str(v))
                continue
                
            sign = "-" if t < 0 else ""
            t = abs(t)
            
            if t >= 3600:
                h = int(t // 3600)
                m = int((t % 3600) // 60)
                out.append(f"{sign}{h}:{m:02d}:{t%60:06.3f}")
            elif t >= 60:
                m = int(t // 60)
                out.append(f"{sign}{m}:{t%60:06.3f}")
            else:
                out.append(f"{sign}{t:.3f}")
        return out

class BLFReaderThread(QtCore.QThread):
    data_batch_ready = QtCore.Signal(dict, float)
    progress = QtCore.Signal(int)
    finished = QtCore.Signal()
    error = QtCore.Signal(str)

    def __init__(self, blf_path, dbs_info, target_keys, parent=None):
        super().__init__(parent)
        self.blf_path = blf_path
        self.dbs_info = dbs_info
        self.target_keys = set(target_keys)
        self._running = True
        
        self.target_messages = defaultdict(list)
        for db, dbc_name in self.dbs_info:
            for msg in db.messages:
                self.target_messages[msg.frame_id].append((msg, dbc_name))

    def stop(self):
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
            if not self._running:
                break
            try:
                raw_ts = float(msg.timestamp)
            except Exception:
                continue
                
            if base_ts is None:
                base_ts = raw_ts
            ts = raw_ts - base_ts
            
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
                    pass
                    
            if count % 20000 == 0:
                self.data_batch_ready.emit(dict(batch), max_ts)
                batch.clear()
                self.progress.emit(count)

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
            unit_str = f" [{sig['unit']}]" if sig.get('unit') else ""
            item = QtWidgets.QListWidgetItem(f"{sig['name']}{unit_str}")
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setData(QtCore.Qt.UserRole, key)
            item.setCheckState(QtCore.Qt.Checked if key in selected_keys else QtCore.Qt.Unchecked)
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
            item.setData(QtCore.Qt.UserRole, key)
            item.setCheckState(QtCore.Qt.Checked if key in selected_keys else QtCore.Qt.Unchecked)
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
            if item.checkState() == QtCore.Qt.Checked and not item.isHidden():
                self.result_keys.append(item.data(QtCore.Qt.UserRole))
        super().accept()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pro BLF Signal Viewer (Analysis Edition)")
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
        self.current_window_span = 5.0  
        self.is_workspace_modified = False
        self.reader = None

        self.user_interacting = False    
        self._mouse_interacting = False  

        self.transition_active = False
        self.transition_steps = 8
        self.transition_step = 0
        self.prev_start = 0.0
        self.prev_end = 0.0
        self.target_start = 0.0
        self.target_end = 0.0

        self.setup_ui()
        self.load_config()
        self.setup_timers()

        save_shortcut = QtGui.QShortcut(QtGui.QKeySequence.Save, self)
        save_shortcut.activated.connect(self.manual_save)

    def setup_ui(self):
        tb = self.main_toolbar = self.addToolBar("Main")
        tb.setMovable(False)
        self.act_blf = QtGui.QAction("📂 Open BLF", self); self.act_blf.triggered.connect(self.open_blf)
        tb.addAction(self.act_blf)
        self.act_dbc = QtGui.QAction("➕ Add DBC(s)", self); self.act_dbc.triggered.connect(self.add_dbc_dialog)
        tb.addAction(self.act_dbc)
        tb.addSeparator()
        self.act_search = QtGui.QAction("🔍 Search", self); self.act_search.triggered.connect(self.open_global_search)
        tb.addAction(self.act_search)
        tb.addSeparator()

        self.x_mode_combo = QtWidgets.QComboBox()
        self.x_mode_combo.addItems(["Fixed Window", "Auto Fit"])
        self.x_mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        tb.addWidget(QtWidgets.QLabel(" X-Mode: ")); tb.addWidget(self.x_mode_combo)

        self.window_spin = QtWidgets.QDoubleSpinBox(); self.window_spin.setRange(0.1, 3600.0)
        self.window_spin.setValue(5.0); self.window_spin.setSingleStep(0.5)
        self.window_spin.setKeyboardTracking(False)  
        self.window_spin.valueChanged.connect(self.on_window_spin_changed)
        tb.addWidget(QtWidgets.QLabel(" Window(s): ")); tb.addWidget(self.window_spin)

        self.autoy_checkbox = QtWidgets.QCheckBox(" Auto Y-Fit ")
        self.autoy_checkbox.toggled.connect(self.mark_workspace_modified)
        tb.addWidget(self.autoy_checkbox)
        tb.addSeparator()

        self.load_btn = QtWidgets.QPushButton("📥 Load / Reload Data"); self.load_btn.clicked.connect(self.trigger_data_load)
        tb.addWidget(self.load_btn)
        tb.addSeparator()

        self.ui_interval_spin = QtWidgets.QDoubleSpinBox(); self.ui_interval_spin.setRange(0.01, 10.0)
        self.ui_interval_spin.setValue(1.0); self.ui_interval_spin.setSingleStep(0.1); self.ui_interval_spin.setSuffix(" s")
        self.ui_interval_spin.setToolTip("UI update interval in seconds")
        self.ui_interval_spin.valueChanged.connect(self._on_ui_interval_changed)
        tb.addWidget(QtWidgets.QLabel(" UI Interval: ")); tb.addWidget(self.ui_interval_spin)
        tb.addSeparator()

        self.act_export = QtGui.QAction("💾 Export CSV", self); self.act_export.triggered.connect(self.export_csv); tb.addAction(self.act_export)

        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central); main_layout.setContentsMargins(4,4,4,4)
        h_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        v_splitter_left = QtWidgets.QSplitter(QtCore.Qt.Vertical)

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
        self.active_table.setColumnWidth(1,100); self.active_table.setColumnWidth(2,120); self.active_table.setColumnWidth(3,100)
        self.active_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.active_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.active_table.itemChanged.connect(self.on_active_table_item_changed)
        active_layout.addWidget(self.active_table)
        v_splitter_left.addWidget(active_group)

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

    def set_ui_enabled(self, enabled):
        """ロード中はバックグラウンド処理に干渉するボタンのみ無効化し、スライダーやズーム操作は生かす"""
        self.act_blf.setEnabled(enabled)
        self.act_dbc.setEnabled(enabled)
        self.act_search.setEnabled(enabled)
        self.act_export.setEnabled(enabled)
        self.load_btn.setEnabled(enabled)
        
        # 以下の解析系コントロール、ツリー、スライダー等はロード中も常に利用可能
        self.x_mode_combo.setEnabled(True)
        self.window_spin.setEnabled(True)
        self.autoy_checkbox.setEnabled(True)
        self.ui_interval_spin.setEnabled(True)
        self.frame_tree.setEnabled(True)
        self.active_table.setEnabled(True)
        self.slider.setEnabled(True)

    def mark_workspace_modified(self, *args):
        self.is_workspace_modified = True

    def manual_save(self):
        self.save_config()
        self.status.showMessage("Workspace saved successfully.", 3000)

    def on_window_spin_changed(self, val):
        self.current_window_span = val
        self.mark_workspace_modified()
        self.update_plots_and_table()

    def load_config(self):
        default_config = {"x_mode":"Fixed Window","window_span":5.0,"auto_y":False,"ui_interval":1.0,"dbc_paths":[],"blf_path":None,"selected_signals":[]}
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE,"r",encoding="utf-8") as f:
                    user_config = json.load(f); default_config.update(user_config)
            except Exception as e:
                print(f"Failed to load workspace: {e}")
                
        idx = self.x_mode_combo.findText(default_config["x_mode"])
        if idx >= 0: self.x_mode_combo.setCurrentIndex(idx)
        self.window_spin.setValue(default_config["window_span"])
        self.current_window_span = default_config["window_span"]
        self.autoy_checkbox.setChecked(default_config["auto_y"])
        self.ui_interval_spin.setValue(default_config.get("ui_interval", 1.0))
        
        valid_dbcs = [p for p in default_config["dbc_paths"] if os.path.exists(p)]
        if valid_dbcs: self._load_dbc_files(valid_dbcs)
        
        blf_path = default_config["blf_path"]
        if blf_path and os.path.exists(blf_path):
            self.blf_path = blf_path; self.status.showMessage(f"Selected BLF: {blf_path}")
            
        for key in default_config["selected_signals"]:
            if key in self.meta and key not in self.plots:
                self.add_signal_plot(key)
        self.is_workspace_modified = False

    def save_config(self):
        config = {
            "x_mode":self.x_mode_combo.currentText(),
            "window_span":self.current_window_span,
            "auto_y":self.autoy_checkbox.isChecked(),
            "ui_interval":self.ui_interval_spin.value(),
            "dbc_paths":self.dbc_paths,
            "blf_path":getattr(self,'blf_path',None),
            "selected_signals":list(self.plots.keys())
        }
        try:
            with open(CONFIG_FILE,"w",encoding="utf-8") as f: json.dump(config,f,indent=4)
            self.is_workspace_modified = False
        except Exception as e:
            print(f"Failed to save workspace: {e}")

    def closeEvent(self, event):
        try:
            if hasattr(self, 'reader') and self.reader is not None and self.reader.isRunning():
                self.reader.stop()
                self.reader.wait(1000)
        except Exception:
            pass
            
        if not self.dbc_paths and not getattr(self,'blf_path',None) and not self.plots:
            event.accept(); return
            
        if not self.is_workspace_modified: 
            event.accept(); return
            
        reply = QtWidgets.QMessageBox.question(
            self, 'Save Workspace', 
            'You have unsaved changes. Save workspace?', 
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No | QtWidgets.QMessageBox.StandardButton.Cancel, 
            QtWidgets.QMessageBox.StandardButton.Yes
        )
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            self.save_config(); event.accept()
        elif reply == QtWidgets.QMessageBox.StandardButton.No:
            event.accept()
        else:
            event.ignore()

    def setup_timers(self):
        self.ui_timer = QtCore.QTimer()
        self.ui_timer.timeout.connect(self.update_plots_and_table)
        ms = int(float(self.ui_interval_spin.value()) * 1000)
        self.ui_timer.setInterval(max(10, ms)) 
        self.ui_timer.start()

    def _on_ui_interval_changed(self, val):
        ms = int(float(val) * 1000)
        self.ui_timer.setInterval(max(10, ms))
        self.status.showMessage(f"UI interval set to {val:.2f} s", 1500)
        self.mark_workspace_modified()

    def _load_dbc_files(self, paths):
        loaded_any = False
        for path in paths:
            if path in self.dbc_paths: continue
            try:
                db = cantools.database.load_file(path); dbc_name = os.path.basename(path)
                self.dbs_info.append((db, dbc_name)); self.dbc_paths.append(path)
                
                is_first_msg = True
                for msg in db.messages:
                    display_dbc = dbc_name if is_first_msg else ""
                    is_first_msg = False
                    item = QtWidgets.QTreeWidgetItem(self.frame_tree, [display_dbc, hex(msg.frame_id), msg.name])
                    item.setToolTip(0, dbc_name); item.setToolTip(2, msg.name)
                    item.setData(0, QtCore.Qt.UserRole, msg.frame_id)
                    item.setData(1, QtCore.Qt.UserRole, dbc_name)
                    for s in msg.signals:
                        key = f"{dbc_name}:{msg.frame_id}:{msg.name}:{s.name}"
                        sm = get_signal_meta_from_cantools_signal(s)
                        self.meta[key] = {
                            "dbc_name":dbc_name, "frame_id":msg.frame_id, "msg":msg.name, "sig":s.name, 
                            "min":sm.get("min"), "max":sm.get("max"), "unit":sm.get("unit",""), "choices":sm.get("choices")
                        }
                loaded_any = True
                self.status.showMessage(f"Loaded total {len(self.dbs_info)} DBC(s)")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self,"DBC Error",f"Failed to load {path}\n{str(e)}")
        if loaded_any: self.mark_workspace_modified()

    def add_dbc_dialog(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self,"Add DBC(s)","","DBC Files (*.dbc)")
        if paths: self._load_dbc_files(paths)

    def open_blf(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self,"Open BLF","","BLF Files (*.blf)")
        if path:
            if hasattr(self, 'reader') and self.reader is not None and self.reader.isRunning():
                self.reader.stop()
                self.reader.wait(1000)
                
            self.blf_path = path
            self.status.showMessage(f"Selected BLF: {path}")
            
            self.data.clear()
            self.max_time = 0.0
            self.play_pos = 0.0
            self.user_interacting = False
            self._mouse_interacting = False
            
            for p_data in self.plots.values():
                try:
                    p_data["curve"].setData([], [])
                except Exception:
                    pass
            
            self._compute_and_set_slider_from_play_pos()
            self.time_label.setText("Time: 0.000 s")
            
            for row in range(self.active_table.rowCount()):
                val_item = self.active_table.item(row, 3)
                if val_item:
                    val_item.setText("-")
            
            self.trigger_data_load()

    def trigger_data_load(self):
        if not getattr(self, 'blf_path', None) or not self.dbs_info:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please load at least one DBC and a BLF file first.")
            return

        target_keys = list(self.plots.keys())
        if not target_keys:
            QtWidgets.QMessageBox.information(self, "Info", "Please select at least one signal to plot first.")
            return

        if hasattr(self, 'reader') and self.reader is not None and self.reader.isRunning():
            self.reader.stop()
            self.reader.wait(1000)

        # ロード中にバックグラウンド処理を阻害するアクションのみを無効化（限定グレーアウト）
        self.set_ui_enabled(False)

        self.data.clear()
        self.max_time = 0.0
        self.status.showMessage("Loading all data from BLF...")
        
        self.reader = BLFReaderThread(self.blf_path, self.dbs_info, target_keys)
        self.reader.data_batch_ready.connect(self.on_data_batch)
        self.reader.progress.connect(lambda n: self.status.showMessage(f"Reading... {n} target frames parsed"))
        self.reader.finished.connect(self._on_reader_finished)
        self.reader.error.connect(self._on_reader_error)
        self.reader.start()

    def _on_reader_finished(self):
        try: self.reader = None
        except Exception: pass
        self.status.showMessage(f"Loading Finished. Total Time: {self.max_time:.3f} s")
        self.update_plots_and_table()
        self.set_ui_enabled(True)

    def _on_reader_error(self, s):
        self.status.showMessage(s)
        self.set_ui_enabled(True)

    def on_data_batch(self, batch, new_max_time):
        if new_max_time > self.max_time: 
            self.max_time = new_max_time
            
        for key, new_data in batch.items():
            if key not in self.data: self.data[key] = {"t": [], "v_raw": [], "v_num": []}
            
            self.data[key]["t"].extend(new_data["t"])
            self.data[key]["v_raw"].extend(new_data["v"])
            
            choices = self.meta.get(key, {}).get("choices")
            nums = []
            for val in new_data["v"]:
                if isinstance(val, (int, float)):
                    nums.append(float(val))
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
            
            if len(self.data[key]["t"]) > MAX_TIMELINE:
                self.data[key]["t"] = self.data[key]["t"][-MAX_TIMELINE:]
                self.data[key]["v_raw"] = self.data[key]["v_raw"][-MAX_TIMELINE:]
                self.data[key]["v_num"] = self.data[key]["v_num"][-MAX_TIMELINE:]

            if key in self.plots:
                if not self.plots[key].get("axis_applied", False):
                    self.apply_meta_to_axis(key)

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
        
        try:
            vb = pw.getViewBox()
            vb.sigRangeChanged.connect(self._on_user_interaction)
        except Exception:
            pass

        for other in self.plots.values():
            try:
                pi.setXLink(other['widget'].getPlotItem())
            except Exception:
                pass
                
        self.plot_vbox.insertWidget(self.plot_vbox.count() - 1, pw)
        self.plots[key] = {"widget": pw, "curve": curve, "meta": meta, "color": color, "axis_applied": False}
        self.apply_meta_to_axis(key)
        self.add_to_active_table(key)
        self.mark_workspace_modified()

    def apply_meta_to_axis(self, key):
        if key not in self.plots:
            return
        entry = self.plots[key]
        pi = entry["widget"].getPlotItem()
        meta = entry.get("meta", {}) or {}
        choices = meta.get("choices"); minv = meta.get("min"); maxv = meta.get("max")
        
        if choices and isinstance(choices, dict) and len(choices) > 0:
            tick_items = []
            numeric_keys = []
            for kk, lbl in choices.items():
                try: nk = float(kk)
                except Exception:
                    try: nk = float(int(kk))
                    except Exception: continue
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
                    try: pi.setYRange(lo - pad, hi + pad)
                    except Exception: pass
                entry["axis_applied"] = True
                return
                
        if (isinstance(minv, (int, float)) and isinstance(maxv, (int, float)) and maxv > minv):
            try:
                span = float(maxv) - float(minv)
                pad = span * 0.05 if span > 0 else 0.5
                pi.setYRange(float(minv) - pad, float(maxv) + pad)
            except Exception:
                pass
            entry["axis_applied"] = True
            return
            
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

    def _on_user_interaction(self, *args):
        if getattr(self, '_auto_ranging', False):
            return
        self.user_interacting = True
        self._mouse_interacting = True

    def on_slider_moved(self, val):
        if self.max_time <= 0: return
        self.play_pos = (val / 10000.0) * self.max_time
        self.transition_active = False
        
        self.user_interacting = True
        self._mouse_interacting = False
        self.update_plots_and_table()

    def on_mode_changed(self, *args):
        new_mode = self.x_mode_combo.currentText()
        curr_start, curr_end = self._get_current_display_range_estimate()
        win = self.current_window_span
        
        if new_mode == "Fixed Window":
            t_start = max(0.0, self.play_pos - win / 2.0)
            t_end = t_start + win
        else:
            times = []
            for k in self.plots.keys():
                d = self.data.get(k)
                if d and len(d["t"]) > 0:
                    times.append(d["t"][0]); times.append(d["t"][-1])
            if not times:
                t_start = 0.0; t_end = max(1.0, win)
            else:
                t_start = max(0.0, min(times) - 0.5); t_end = max(times) + 0.5
                if t_end - t_start < 1e-6: t_end = t_start + max(0.5, win)
        
        self.prev_start = curr_start; self.prev_end = curr_end
        self.target_start = t_start; self.target_end = t_end
        self.transition_active = True
        self.transition_step = 0
            
        self.update_plots_and_table()

    def _get_current_display_range_estimate(self, p_data=None):
        if self.user_interacting and getattr(self, '_mouse_interacting', False):
            if p_data is not None:
                try:
                    vb = p_data["widget"].getPlotItem().getViewBox()
                    vr = vb.viewRange()[0]
                    return vr[0], vr[1]
                except Exception:
                    pass
            else:
                for key in self.plots:
                    try:
                        vb = self.plots[key]["widget"].getPlotItem().getViewBox()
                        vr = vb.viewRange()[0]
                        return vr[0], vr[1]
                    except Exception:
                        pass
        
        win = self.current_window_span
        mode = self.x_mode_combo.currentText()
        if self.transition_active:
            frac = (self.transition_step + 1) / float(max(1, self.transition_steps))
            start = self.prev_start + (self.target_start - self.prev_start) * frac
            end = self.prev_end + (self.target_end - self.prev_end) * frac
            return start, end
        else:
            if mode == "Fixed Window":
                start = max(0.0, self.play_pos - win / 2.0)
                end = start + win
            else:
                times = []
                for k in self.plots.keys():
                    d = self.data.get(k)
                    if d and len(d["t"]) > 0:
                        times.append(d["t"][0]); times.append(d["t"][-1])
                if not times:
                    start = 0.0; end = max(1.0, win)
                else:
                    start = max(0.0, min(times) - 0.5); end = max(times) + 0.5
                    if end - start < 1e-6: end = start + max(0.5, win)
            return start, end

    def _compute_and_set_slider_from_play_pos(self):
        if self.max_time > 0:
            spos = int((self.play_pos / self.max_time) * 10000)
            self.slider.blockSignals(True)
            self.slider.setValue(min(10000, spos))
            self.slider.blockSignals(False)
        else:
            self.slider.blockSignals(True)
            self.slider.setValue(0)
            self.slider.blockSignals(False)

    def update_plots_and_table(self):
        if not self.plots:
            return
            
        # ロード中にmax_timeが増加しても、play_posが未ロード部分（未来）に飛び出さないよう制限
        if self.play_pos > self.max_time:
            self.play_pos = self.max_time
            
        # ロードによるmax_timeの増加に伴い、現在のplay_pos（閲覧位置）に合わせてつまみ位置を逆算・更新
        self._compute_and_set_slider_from_play_pos()
            
        self.time_label.setText(f"Time: {self.play_pos:.3f} s")
        auto_y = self.autoy_checkbox.isChecked()
        
        if self.transition_active:
            self.transition_step += 1
            if self.transition_step >= self.transition_steps:
                self.transition_active = False

        base_start, base_end = self._get_current_display_range_estimate(p_data=None)
        if base_end <= base_start: base_end = base_start + 1.0
        
        self._auto_ranging = True
        try:
            for row in range(self.active_table.rowCount()):
                item = self.active_table.item(row, 0)
                if not item: continue
                key = item.data(QtCore.Qt.UserRole)
                if not key or key not in self.plots: continue
                
                p_data = self.plots[key]
                d = self.data.get(key)
                
                start, end = base_start, base_end
                if self.user_interacting and getattr(self, '_mouse_interacting', False):
                    start, end = self._get_current_display_range_estimate(p_data=p_data)
                    
                span = end - start
                if span <= 0: span = 1.0
                
                if d and len(d["t"]) > 0:
                    t_list = np.array(d["t"])
                    v_num_list = np.array(d["v_num"])
                    v_raw_list = d["v_raw"]
                    
                    idx_start = bisect.bisect_left(t_list, start - span * 0.1)
                    idx_end = bisect.bisect_right(t_list, end + span * 0.1)
                    
                    if idx_end > idx_start:
                        t_render = t_list[idx_start:idx_end]
                        v_render = v_num_list[idx_start:idx_end]
                        
                        choices = p_data["meta"].get("choices")
                        if choices:
                            t_render = np.repeat(t_render, 2)[1:]
                            v_render = np.repeat(v_render, 2)[:-1]

                        if (auto_y or not p_data.get("axis_applied", False)):
                            vmin, vmax = np.min(v_render), np.max(v_render)
                            if vmax > vmin:
                                pad = (vmax - vmin) * 0.1
                                try: p_data["widget"].getPlotItem().setYRange(vmin - pad, vmax + pad, padding=0)
                                except Exception: pass
                            else:
                                try: p_data["widget"].getPlotItem().setYRange(vmin - 0.5, vmax + 0.5, padding=0)
                                except Exception: pass

                        try: p_data["curve"].setData(t_render, v_render)
                        except Exception: pass
                    else:
                        p_data["curve"].setData([], [])
                        
                    if not getattr(self, '_mouse_interacting', False):
                        try: 
                            vb = p_data["widget"].getPlotItem().getViewBox()
                            vb.blockSignals(True)
                            vb.setXRange(start, end, padding=0)
                            vb.blockSignals(False)
                        except Exception: pass
                        
                    idx_cur = bisect.bisect_right(t_list, self.play_pos) - 1
                    val_str = "-"
                    
                    if len(t_list) > 0 and idx_cur >= 0:
                        choices = p_data["meta"].get("choices")
                        
                        if choices:
                            raw_val = v_raw_list[idx_cur]
                            val_str = str(raw_val)
                        else:
                            if idx_cur == len(t_list) - 1 or t_list[idx_cur] == self.play_pos:
                                val_num = v_num_list[idx_cur]
                            else:
                                t0, t1 = t_list[idx_cur], t_list[idx_cur+1]
                                v0, v1 = v_num_list[idx_cur], v_num_list[idx_cur+1]
                                val_num = v0 + (v1 - v0) * (self.play_pos - t0) / (t1 - t0) if t1 > t0 else v0
                                
                            val_str = f"{val_num:.3f}"
                            
                        unit = p_data['meta'].get('unit', "")
                        if unit: val_str += f" [{unit}]"
                        
                        p_data["curve"].setPen(pg.mkPen(color=p_data["color"], width=2))
                            
                        val_item = self.active_table.item(row, 3)
                        if val_item: val_item.setText(val_str)
                else:
                    p_data["curve"].setData([], [])
                    val_item = self.active_table.item(row, 3)
                    if val_item: val_item.setText("-")
        finally:
            self._auto_ranging = False

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()