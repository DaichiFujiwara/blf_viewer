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

# ---------------- Config File ----------------
CONFIG_FILE = "blf_viewer_workspace.json"

# ---------------- Helpers ----------------
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

# ---------------- TimeAxisItem ----------------
class TimeAxisItem(pg.AxisItem):
    def __init__(self, orientation='bottom', time_start_cb=None, *args, **kwargs):
        super().__init__(orientation=orientation, *args, **kwargs)
        self.time_start_cb = time_start_cb

    def tickStrings(self, values, scale, spacing):
        start = 0.0
        if callable(self.time_start_cb):
            start = float(self.time_start_cb())
            
        out = []
        for v in values:
            t = start + v
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

# ---------------- BLF Reader Thread (Multi-DBC Optimized) ----------------
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
            
            arb = msg.arbitration_id
            msgs_to_try = self.target_messages.get(arb)
            
            if not msgs_to_try:
                continue

            count += 1
            raw_ts = float(msg.timestamp)
            if base_ts is None:
                base_ts = raw_ts
                
            ts = raw_ts - base_ts
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
            
            if count % 10000 == 0:
                self.data_batch_ready.emit(dict(batch), max_ts)
                batch.clear()
                self.progress.emit(count)

        if batch:
            self.data_batch_ready.emit(dict(batch), max_ts)

        try:
            reader.close()
        except:
            pass
            
        self.finished.emit()

# ---------------- UI: Custom Widgets ----------------
class CheckableListWidget(QtWidgets.QListWidget):
    """
    Shift + Click でチェックボックスの一括選択/解除ができる拡張ListWidget
    """
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

# ---------------- UI: Dialogs ----------------
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
            if item.checkState() == QtCore.Qt.Checked and not item.isHidden():
                self.result_keys.append(item.data(QtCore.Qt.UserRole))
            elif item.checkState() == QtCore.Qt.Checked and item.isHidden():
                self.result_keys.append(item.data(QtCore.Qt.UserRole))
        super().accept()

# ---------------- Main Window ----------------
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
        tb.addWidget(QtWidgets.QLabel(" X-Mode: "))
        tb.addWidget(self.x_mode_combo)
        
        self.window_spin = QtWidgets.QDoubleSpinBox()
        self.window_spin.setRange(0.1, 3600.0)
        self.window_spin.valueChanged.connect(self.mark_workspace_modified)
        tb.addWidget(QtWidgets.QLabel(" Win(s): "))
        tb.addWidget(self.window_spin)
        
        self.autoy_checkbox = QtWidgets.QCheckBox(" Auto Y-Fit ")
        self.autoy_checkbox.toggled.connect(self.mark_workspace_modified)
        tb.addWidget(self.autoy_checkbox)

        tb.addSeparator()
        self.stale_spin = QtWidgets.QDoubleSpinBox()
        self.stale_spin.setRange(0.1, 3600.0); self.stale_spin.setSingleStep(0.5)
        self.stale_spin.valueChanged.connect(self.mark_workspace_modified)
        tb.addWidget(QtWidgets.QLabel(" Stale(s): "))
        tb.addWidget(self.stale_spin)

        tb.addSeparator()
        self.play_btn = QtWidgets.QPushButton("▶ Play / Load")
        self.play_btn.clicked.connect(self.toggle_play)
        tb.addWidget(self.play_btn)
        
        self.speed_spin = QtWidgets.QDoubleSpinBox()
        self.speed_spin.setRange(0.1, 50.0); self.speed_spin.setSuffix("x")
        self.speed_spin.valueChanged.connect(self.mark_workspace_modified)
        tb.addWidget(QtWidgets.QLabel(" Speed: "))
        tb.addWidget(self.speed_spin)

        tb.addSeparator()
        act_export = QtGui.QAction("💾 Export CSV", self); act_export.triggered.connect(self.export_csv)
        tb.addAction(act_export)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)

        h_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        v_splitter_left = QtWidgets.QSplitter(QtCore.Qt.Vertical)

        # === Frames List ===
        frame_group = QtWidgets.QGroupBox("1. Frames (Double-click)")
        frame_layout = QtWidgets.QVBoxLayout(frame_group)
        self.frame_tree = QtWidgets.QTreeWidget()
        self.frame_tree.setHeaderLabels(["DBC", "ID", "Message"])
        
        # UI改善: 階層ツリーの余白を消去し、完全なフラットリストとして扱う
        self.frame_tree.setRootIsDecorated(False)
        self.frame_tree.setUniformRowHeights(True)
        
        header_tree = self.frame_tree.header()
        header_tree.setSectionResizeMode(0, QtWidgets.QHeaderView.Interactive)
        header_tree.setSectionResizeMode(1, QtWidgets.QHeaderView.Interactive)
        header_tree.setSectionResizeMode(2, QtWidgets.QHeaderView.Interactive)
        header_tree.setStretchLastSection(True)
        self.frame_tree.setColumnWidth(0, 100)
        self.frame_tree.setColumnWidth(1, 60)
        
        self.frame_tree.itemDoubleClicked.connect(self.open_signal_popup)
        frame_layout.addWidget(self.frame_tree)
        v_splitter_left.addWidget(frame_group)

        # === Active Signals Table ===
        active_group = QtWidgets.QGroupBox("2. Active Signals & Values")
        active_layout = QtWidgets.QVBoxLayout(active_group)
        
        self.active_table = QtWidgets.QTableWidget(0, 5)
        self.active_table.setHorizontalHeaderLabels(["👁", "Frame", "Signal", "Value", ""])
        
        header_tbl = self.active_table.horizontalHeader()
        header_tbl.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header_tbl.setSectionResizeMode(1, QtWidgets.QHeaderView.Interactive)
        header_tbl.setSectionResizeMode(2, QtWidgets.QHeaderView.Interactive)
        header_tbl.setSectionResizeMode(3, QtWidgets.QHeaderView.Interactive)
        header_tbl.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeToContents)
        
        self.active_table.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.active_table.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.active_table.setColumnWidth(1, 100)
        self.active_table.setColumnWidth(2, 120)
        self.active_table.setColumnWidth(3, 80)
        self.active_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.active_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.active_table.itemChanged.connect(self.on_active_table_item_changed)
        
        active_layout.addWidget(self.active_table)
        v_splitter_left.addWidget(active_group)

        # === Plots ===
        plot_group = QtWidgets.QGroupBox("3. Signal Plots")
        plot_layout = QtWidgets.QVBoxLayout(plot_group)
        self.plot_scroll = QtWidgets.QScrollArea()
        self.plot_scroll.setWidgetResizable(True)
        self.plot_container = QtWidgets.QWidget()
        self.plot_vbox = QtWidgets.QVBoxLayout(self.plot_container)
        self.plot_vbox.setContentsMargins(0, 0, 0, 0)
        self.plot_vbox.addStretch()
        self.plot_scroll.setWidget(self.plot_container)
        plot_layout.addWidget(self.plot_scroll)
        
        slider_layout = QtWidgets.QHBoxLayout()
        self.time_label = QtWidgets.QLabel("Time: 0.000 s")
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, 10000)
        self.slider.sliderMoved.connect(self.on_slider_moved)
        slider_layout.addWidget(self.time_label)
        slider_layout.addWidget(self.slider)
        plot_layout.addLayout(slider_layout)

        h_splitter.addWidget(v_splitter_left)
        h_splitter.addWidget(plot_group)
        h_splitter.setSizes([450, 950])
        main_layout.addWidget(h_splitter)

        self.status = self.statusBar()

    def mark_workspace_modified(self, *args):
        self.is_workspace_modified = True

    def manual_save(self):
        self.save_config()
        self.status.showMessage("Workspace saved successfully.", 3000)

    def load_config(self):
        default_config = {
            "x_mode": "Fixed Window",
            "window_span": 10.0,
            "auto_y": False,
            "stale_time": 2.0,
            "speed": 1.0,
            "dbc_paths": [],
            "blf_path": None,
            "selected_signals": []
        }
        
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                print(f"Failed to load workspace: {e}")

        idx = self.x_mode_combo.findText(default_config["x_mode"])
        if idx >= 0:
            self.x_mode_combo.setCurrentIndex(idx)
        self.window_spin.setValue(default_config["window_span"])
        self.autoy_checkbox.setChecked(default_config["auto_y"])
        self.stale_spin.setValue(default_config["stale_time"])
        self.speed_spin.setValue(default_config["speed"])

        valid_dbcs = [p for p in default_config["dbc_paths"] if os.path.exists(p)]
        if valid_dbcs:
            self._load_dbc_files(valid_dbcs)
            
        blf_path = default_config["blf_path"]
        if blf_path and os.path.exists(blf_path):
            self.blf_path = blf_path
            self.status.showMessage(f"Selected BLF: {self.blf_path}")
            
        for key in default_config["selected_signals"]:
            if key in self.meta and key not in self.plots:
                self.add_signal_plot(key)
                
        self.is_workspace_modified = False

    def save_config(self):
        config = {
            "x_mode": self.x_mode_combo.currentText(),
            "window_span": self.window_spin.value(),
            "auto_y": self.autoy_checkbox.isChecked(),
            "stale_time": self.stale_spin.value(),
            "speed": self.speed_spin.value(),
            "dbc_paths": self.dbc_paths,
            "blf_path": getattr(self, 'blf_path', None),
            "selected_signals": list(self.plots.keys())
        }
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4)
            self.is_workspace_modified = False
        except Exception as e:
            print(f"Failed to save workspace: {e}")

    def closeEvent(self, event):
        if not self.dbc_paths and not getattr(self, 'blf_path', None) and not self.plots:
            event.accept()
            return

        if not self.is_workspace_modified:
            event.accept()
            return

        reply = QtWidgets.QMessageBox.question(
            self, 'Save Workspace',
            'You have unsaved changes. Do you want to save the current workspace?\n(Settings, Loaded DBCs, BLF, and Selected Signals)',
            QtWidgets.QMessageBox.StandardButton.Yes | 
            QtWidgets.QMessageBox.StandardButton.No | 
            QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Yes
        )

        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            self.save_config()
            event.accept()
        elif reply == QtWidgets.QMessageBox.StandardButton.No:
            event.accept()
        else:
            event.ignore()

    def setup_timers(self):
        self.is_playing = False
        self.play_timer = QtCore.QTimer()
        self.play_timer.timeout.connect(self.advance_playback)
        self.ui_timer = QtCore.QTimer()
        self.ui_timer.setInterval(100)
        self.ui_timer.timeout.connect(self.update_plots_and_table)
        self.ui_timer.start()

    def add_dbc_dialog(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Add DBC(s)", "", "DBC Files (*.dbc)")
        if paths:
            self._load_dbc_files(paths)

    def _load_dbc_files(self, paths):
        loaded_any = False
        for path in paths:
            if path in self.dbc_paths:
                continue 
                
            try:
                db = cantools.database.load_file(path)
                dbc_name = os.path.basename(path)
                self.dbs_info.append((db, dbc_name))
                self.dbc_paths.append(path)
                
                # --- UI改善: 階層ノードを廃止し、完全なフラットアイテムとして直接追加 ---
                is_first_msg = True
                for msg in db.messages:
                    display_dbc = dbc_name if is_first_msg else ""
                    is_first_msg = False
                    
                    item = QtWidgets.QTreeWidgetItem(self.frame_tree, [display_dbc, hex(msg.frame_id), msg.name])
                    
                    item.setToolTip(0, dbc_name)
                    item.setToolTip(2, msg.name)
                    
                    item.setData(0, QtCore.Qt.UserRole, msg.frame_id)
                    item.setData(1, QtCore.Qt.UserRole, dbc_name)
                    
                    for s in msg.signals:
                        key = f"{dbc_name}:{msg.frame_id}:{msg.name}:{s.name}"
                        sm = get_signal_meta_from_cantools_signal(s)
                        self.meta[key] = {
                            "dbc_name": dbc_name,
                            "frame_id": msg.frame_id, 
                            "msg": msg.name, 
                            "sig": s.name, 
                            "min": sm.get("min"),
                            "max": sm.get("max"),
                            "unit": sm.get("unit", ""), 
                            "choices": sm.get("choices")
                        }
                loaded_any = True
                self.status.showMessage(f"Loaded total {len(self.dbs_info)} DBC(s)")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "DBC Error", f"Failed to load {path}\n{str(e)}")
                
        if loaded_any:
            self.mark_workspace_modified()

    def open_blf(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open BLF", "", "BLF Files (*.blf)")
        if path:
            self.blf_path = path
            self.status.showMessage(f"Selected BLF: {path}")
            self.mark_workspace_modified()

    def open_signal_popup(self, item, column):
        frame_id = item.data(0, QtCore.Qt.UserRole)
        dbc_name = item.data(1, QtCore.Qt.UserRole)
        msg_name = item.text(2)
        
        if frame_id is None or dbc_name is None:
            return
            
        signals = [{"name": v["sig"], "unit": v["unit"]} for k, v in self.meta.items() if v["frame_id"] == frame_id and v["dbc_name"] == dbc_name and v["msg"] == msg_name]
        if not signals:
            return

        dialog = SignalSelectionDialog(dbc_name, frame_id, msg_name, signals, list(self.plots.keys()), self)
        if dialog.exec():
            for key in dialog.result_keys:
                if key not in self.plots:
                    self.add_signal_plot(key)

    def open_global_search(self):
        if not self.meta:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please load a DBC file first.")
            return

        dialog = GlobalSearchDialog(self.meta, list(self.plots.keys()), self)
        if dialog.exec():
            for key in dialog.result_keys:
                if key not in self.plots:
                    self.add_signal_plot(key)

    def export_csv(self):
        if not self.plots:
            QtWidgets.QMessageBox.warning(self, "No Signals", "Please select and plot signals to export.")
            return
            
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save CSV", "can_data_export.csv", "CSV Files (*.csv)")
        if not path:
            return

        keys = list(self.plots.keys())
        all_times = set()
        for k in keys:
            all_times.update(self.data[k]["t"])
        sorted_times = sorted(list(all_times))

        try:
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                headers = ["Time"] + [f"[{self.meta[k]['dbc_name']}] {hex(self.meta[k]['frame_id'])} {self.meta[k]['msg']}.{self.meta[k]['sig']}" for k in keys]
                writer.writerow(headers)

                for t in sorted_times:
                    row = [t]
                    for k in keys:
                        t_list = self.data[k]["t"]
                        v_list = self.data[k]["v_raw"]
                        idx = bisect.bisect_right(t_list, t) - 1
                        if idx >= 0:
                            row.append(v_list[idx])
                        else:
                            row.append("")
                    writer.writerow(row)
            self.status.showMessage(f"Exported to {path}")
            QtWidgets.QMessageBox.information(self, "Success", "CSV Export completed successfully.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Error", str(e))

    def add_signal_plot(self, key):
        meta = self.meta[key]
        color = pg.intColor(len(self.plots), hues=15)
        
        axis = TimeAxisItem(orientation='bottom', time_start_cb=lambda: self.current_display_start)
        pw = pg.PlotWidget(axisItems={'bottom': axis})
        pw.setFixedHeight(180)
        pi = pw.getPlotItem()
        pi.showGrid(x=True, y=True, alpha=0.5)
        
        title = f"[{meta['dbc_name']}] {meta['msg']} . {meta['sig']}"
        if meta['unit']:
            title += f" [{meta['unit']}]"
        pi.setTitle(title, size="10pt")
        
        choices = meta.get("choices")
        min_v = meta.get("min")
        max_v = meta.get("max")
        
        if choices:
            numeric_keys = []
            for k in choices.keys():
                try: numeric_keys.append(float(k))
                except: pass
            if numeric_keys:
                pi.setYRange(min(numeric_keys) - 0.5, max(numeric_keys) + 0.5)
        elif min_v is not None and max_v is not None and max_v > min_v:
            span = float(max_v) - float(min_v)
            pi.setYRange(float(min_v) - span * 0.05, float(max_v) + span * 0.05)

        curve = pi.plot([], [], pen=pg.mkPen(color=color, width=2))
        curve.setClipToView(True)
        curve.setDownsampling(ds=True, auto=True, method='peak')
        
        line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('r', width=1.5, style=QtCore.Qt.DashLine))
        pi.addItem(line)

        for other in self.plots.values():
            pi.setXLink(other['widget'].getPlotItem())
            break 
            
        self.plot_vbox.insertWidget(self.plot_vbox.count() - 1, pw)
        self.plots[key] = {"widget": pw, "curve": curve, "line": line, "meta": meta, "color": color}
        self.add_to_active_table(key)
        self.mark_workspace_modified()

    def add_to_active_table(self, key):
        row = self.active_table.rowCount()
        self.active_table.insertRow(row)
        
        chk_item = QtWidgets.QTableWidgetItem()
        chk_item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
        chk_item.setCheckState(QtCore.Qt.Checked)
        chk_item.setData(QtCore.Qt.UserRole, key)
        self.active_table.setItem(row, 0, chk_item)
        
        frame_name = self.meta[key]['msg']
        frame_item = QtWidgets.QTableWidgetItem(frame_name)
        frame_item.setToolTip(frame_name)
        self.active_table.setItem(row, 1, frame_item)
        
        sig_name = self.meta[key]['sig']
        sig_item = QtWidgets.QTableWidgetItem(sig_name)
        sig_item.setForeground(QtGui.QBrush(self.plots[key]['color']))
        font = sig_item.font()
        font.setBold(True)
        sig_item.setFont(font)
        sig_item.setToolTip(sig_name)
        self.active_table.setItem(row, 2, sig_item)
        
        val_item = QtWidgets.QTableWidgetItem("-")
        val_item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.active_table.setItem(row, 3, val_item)
        
        del_btn = QtWidgets.QPushButton("✖")
        del_btn.setFixedSize(24, 24)
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
                self.active_table.removeRow(r)
                break
                
        if key in self.plots:
            pw = self.plots[key]['widget']
            self.plot_vbox.removeWidget(pw)
            pw.deleteLater()
            del self.plots[key]
            self.mark_workspace_modified()

    def toggle_play(self):
        if not hasattr(self, 'blf_path') or not self.dbs_info:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please load at least one DBC and a BLF file first.")
            return

        if not self.is_playing:
            self.data.clear()
            self.max_time = 0.0
            self.play_pos = 0.0
            
            target_keys = list(self.plots.keys())
            if not target_keys:
                QtWidgets.QMessageBox.information(self, "Info", "Please select at least one signal to plot first.")
                return

            self.reader = BLFReaderThread(self.blf_path, self.dbs_info, target_keys)
            self.reader.data_batch_ready.connect(self.on_data_batch)
            self.reader.progress.connect(lambda n: self.status.showMessage(f"Reading... {n} target frames parsed"))
            self.reader.finished.connect(lambda: self.status.showMessage("Reading Finished / Paused"))
            self.reader.start()
            
            self.is_playing = True
            self.play_btn.setText("⏸ Pause")
            self.play_timer.start(50)
        else:
            self.reader.stop()
            self.is_playing = False
            self.play_btn.setText("▶ Play / Load Data")
            self.play_timer.stop()

    @QtCore.Slot(dict, float)
    def on_data_batch(self, batch, new_max_time):
        if new_max_time > self.max_time:
            self.max_time = new_max_time

        for key, new_data in batch.items():
            if key not in self.data:
                self.data[key] = {"t": [], "v_raw": [], "v_num": []}
            
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
                            try:
                                mapped = float(k_c)
                            except:
                                pass
                            break
                    nums.append(mapped)
                else:
                    try:
                        nums.append(float(val))
                    except:
                        nums.append(0.0)
            self.data[key]["v_num"].extend(nums)

    def advance_playback(self):
        dt = 0.05 * self.speed_spin.value()
        self.play_pos += dt
        if self.max_time > 0:
            spos = int((self.play_pos / self.max_time) * 10000)
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
        win = self.window_spin.value()
        stale_threshold = self.stale_spin.value()
        auto_y = self.autoy_checkbox.isChecked()
        
        if mode == "Fixed Window":
            start = max(0.0, self.play_pos - win/2)
            end = start + win
        elif mode == "Follow (Trailing)":
            end = max(self.play_pos, win)
            start = end - win
        else:
            start = 0.0
            end = max(self.max_time, 1.0)
            
        self.current_display_start = start
        
        for row in range(self.active_table.rowCount()):
            item = self.active_table.item(row, 0)
            if not item:
                continue
                
            key = item.data(QtCore.Qt.UserRole)
            if not key or key not in self.plots:
                continue
                
            p_data = self.plots[key]
            d = self.data.get(key)
            
            if d and len(d["t"]) > 0:
                t_list = d["t"]
                
                idx_start = bisect.bisect_left(t_list, start - win)
                idx_end = bisect.bisect_right(t_list, end + win)
                
                if idx_end > idx_start:
                    t_slice = t_list[idx_start:idx_end]
                    v_slice = d["v_num"][idx_start:idx_end]

                    if len(t_slice) > 30000:
                        step = len(t_slice) // 30000
                        t_slice = t_slice[::step]
                        v_slice = v_slice[::step]
                    
                    t_plot = np.array(t_slice) - start
                    v_plot = np.array(v_slice)

                    if auto_y and len(v_plot) > 0:
                        vmin = np.min(v_plot)
                        vmax = np.max(v_plot)
                        if vmax > vmin:
                            pad = (vmax - vmin) * 0.1
                            p_data["widget"].getPlotItem().setYRange(vmin - pad, vmax + pad, padding=0)
                        else:
                            p_data["widget"].getPlotItem().setYRange(vmin - 0.5, vmax + 0.5, padding=0)
                    
                    t_step = np.repeat(t_plot, 2)[1:]
                    v_step = np.repeat(v_plot, 2)[:-1]
                    p_data["curve"].setData(t_step, v_step)
                
                cursor_pos = self.play_pos - start
                p_data["line"].setPos(cursor_pos)
                p_data["widget"].getPlotItem().setXRange(0, end - start, padding=0)

                idx_cur = bisect.bisect_right(t_list, self.play_pos) - 1
                if idx_cur >= 0:
                    age = self.play_pos - t_list[idx_cur]
                    is_stale = (age > stale_threshold)
                    
                    raw_val = d["v_raw"][idx_cur]
                    val_str = f"{raw_val:.3f}" if isinstance(raw_val, float) else str(raw_val)
                    if p_data['meta']['unit']:
                        val_str += f" {p_data['meta']['unit']}"
                        
                    if is_stale:
                        val_str += " (stale)"
                        p_data["curve"].setPen(pg.mkPen(color=(160, 160, 160), width=1.5, style=QtCore.Qt.DashLine))
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