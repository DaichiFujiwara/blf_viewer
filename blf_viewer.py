import sys
import time
import bisect
import csv
from collections import defaultdict

from PySide6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import numpy as np
import cantools
import can

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

# ---------------- BLF Reader Thread (Optimized) ----------------
class BLFReaderThread(QtCore.QThread):
    data_batch_ready = QtCore.Signal(dict, float) 
    progress = QtCore.Signal(int)
    finished = QtCore.Signal()
    error = QtCore.Signal(str)

    def __init__(self, blf_path, db, target_frame_ids, parent=None):
        super().__init__(parent)
        self.blf_path = blf_path
        self.db = db
        self.target_frame_ids = set(target_frame_ids)
        self._running = True

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
            if arb not in self.target_frame_ids:
                continue

            count += 1
            raw_ts = float(msg.timestamp)
            if base_ts is None:
                base_ts = raw_ts
                
            ts = raw_ts - base_ts
            if ts > max_ts:
                max_ts = ts
            
            try:
                message = self.db.get_message_by_frame_id(arb)
                decoded = message.decode(msg.data)
                for sname, sval in decoded.items():
                    key = f"{arb}:{message.name}:{sname}"
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

# ---------------- UI: Dialogs ----------------
class SignalSelectionDialog(QtWidgets.QDialog):
    def __init__(self, frame_id, frame_name, signals, selected_keys, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Select Signals - {frame_name} ({hex(frame_id)})")
        self.resize(350, 400)
        self.result_keys = []

        layout = QtWidgets.QVBoxLayout(self)
        self.list_widget = QtWidgets.QListWidget()
        layout.addWidget(self.list_widget)

        for sig in signals:
            key = f"{frame_id}:{frame_name}:{sig['name']}"
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
        self.resize(450, 500)
        self.meta_dict = meta_dict
        self.result_keys = []

        layout = QtWidgets.QVBoxLayout(self)
        self.search_input = QtWidgets.QLineEdit()
        self.search_input.setPlaceholderText("Search by frame name, ID, or signal name...")
        self.search_input.textChanged.connect(self.filter_list)
        layout.addWidget(self.search_input)

        self.list_widget = QtWidgets.QListWidget()
        layout.addWidget(self.list_widget)

        # Populate all
        for key, m in self.meta_dict.items():
            fid_hex = hex(m['frame_id'])
            display_text = f"{fid_hex} : {m['msg']} . {m['sig']}"
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

    def accept(self):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == QtCore.Qt.Checked and not item.isHidden():
                self.result_keys.append(item.data(QtCore.Qt.UserRole))
            elif item.checkState() == QtCore.Qt.Checked and item.isHidden():
                # Keep checked items even if hidden during search
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

        self.db = None
        self.meta = {}
        self.data = defaultdict(lambda: {"t": [], "v_raw": [], "v_num": []})
        self.plots = {} 
        self.max_time = 0.0
        self.play_pos = 0.0
        self.current_display_start = 0.0

        self.setup_ui()
        self.setup_timers()

    def setup_ui(self):
        tb = self.addToolBar("Main")
        tb.setMovable(False)
        
        act_blf = QtGui.QAction("📂 Open BLF", self); act_blf.triggered.connect(self.open_blf)
        tb.addAction(act_blf)
        act_dbc = QtGui.QAction("📄 Load DBC", self); act_dbc.triggered.connect(self.open_dbc)
        tb.addAction(act_dbc)
        tb.addSeparator()

        act_search = QtGui.QAction("🔍 Search", self); act_search.triggered.connect(self.open_global_search)
        tb.addAction(act_search)
        tb.addSeparator()
        
        self.x_mode_combo = QtWidgets.QComboBox()
        self.x_mode_combo.addItems(["Fixed Window", "Follow (Trailing)", "Auto Fit"])
        tb.addWidget(QtWidgets.QLabel(" X-Mode: "))
        tb.addWidget(self.x_mode_combo)
        
        self.window_spin = QtWidgets.QDoubleSpinBox(); self.window_spin.setRange(0.1, 3600.0); self.window_spin.setValue(10.0)
        tb.addWidget(QtWidgets.QLabel(" Win(s): "))
        tb.addWidget(self.window_spin)
        
        tb.addSeparator()
        self.stale_spin = QtWidgets.QDoubleSpinBox(); self.stale_spin.setRange(0.1, 3600.0); self.stale_spin.setValue(2.0); self.stale_spin.setSingleStep(0.5)
        tb.addWidget(QtWidgets.QLabel(" Stale(s): "))
        tb.addWidget(self.stale_spin)

        tb.addSeparator()
        self.play_btn = QtWidgets.QPushButton("▶ Play / Load")
        self.play_btn.clicked.connect(self.toggle_play)
        tb.addWidget(self.play_btn)
        
        self.speed_spin = QtWidgets.QDoubleSpinBox(); self.speed_spin.setRange(0.1, 50.0); self.speed_spin.setValue(1.0); self.speed_spin.setSuffix("x")
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

        frame_group = QtWidgets.QGroupBox("1. Frames (Double-click)")
        frame_layout = QtWidgets.QVBoxLayout(frame_group)
        self.frame_tree = QtWidgets.QTreeWidget()
        self.frame_tree.setHeaderLabels(["ID", "Message"])
        self.frame_tree.itemDoubleClicked.connect(self.open_signal_popup)
        frame_layout.addWidget(self.frame_tree)
        v_splitter_left.addWidget(frame_group)

        active_group = QtWidgets.QGroupBox("2. Active Signals & Values")
        active_layout = QtWidgets.QVBoxLayout(active_group)
        
        self.active_table = QtWidgets.QTableWidget(0, 4)
        self.active_table.setHorizontalHeaderLabels(["Vis", "Signal", "Value", ""])
        self.active_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.active_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.active_table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self.active_table.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
        self.active_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.active_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        active_layout.addWidget(self.active_table)
        v_splitter_left.addWidget(active_group)

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
        h_splitter.setSizes([350, 1000])
        main_layout.addWidget(h_splitter)

        self.status = self.statusBar()

    def setup_timers(self):
        self.is_playing = False
        self.play_timer = QtCore.QTimer()
        self.play_timer.timeout.connect(self.advance_playback)
        self.ui_timer = QtCore.QTimer()
        self.ui_timer.setInterval(100)
        self.ui_timer.timeout.connect(self.update_plots_and_table)
        self.ui_timer.start()

    def open_dbc(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open DBC", "", "DBC Files (*.dbc)")
        if not path:
            return
            
        try:
            self.db = cantools.database.load_file(path)
            self.frame_tree.clear()
            self.meta.clear()
            for msg in self.db.messages:
                item = QtWidgets.QTreeWidgetItem(self.frame_tree, [hex(msg.frame_id), msg.name])
                item.setData(0, QtCore.Qt.UserRole, msg.frame_id)
                for s in msg.signals:
                    key = f"{msg.frame_id}:{msg.name}:{s.name}"
                    sm = get_signal_meta_from_cantools_signal(s)
                    self.meta[key] = {
                        "frame_id": msg.frame_id, 
                        "msg": msg.name, 
                        "sig": s.name, 
                        "unit": sm.get("unit", ""), 
                        "choices": sm.get("choices")
                    }
            self.status.showMessage(f"Loaded DBC: {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "DBC Error", str(e))

    def open_blf(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open BLF", "", "BLF Files (*.blf)")
        if path:
            self.blf_path = path
            self.status.showMessage(f"Selected BLF: {path}")

    def open_signal_popup(self, item, column):
        frame_id = item.data(0, QtCore.Qt.UserRole)
        if frame_id is None:
            return
            
        msg_name = item.text(1)
        signals = [{"name": v["sig"], "unit": v["unit"]} for k, v in self.meta.items() if v["frame_id"] == frame_id]
        if not signals:
            return

        dialog = SignalSelectionDialog(frame_id, msg_name, signals, list(self.plots.keys()), self)
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

        # すべての時間をマージしてユニークなタイムスタンプのリストを作成
        keys = list(self.plots.keys())
        all_times = set()
        for k in keys:
            all_times.update(self.data[k]["t"])
        sorted_times = sorted(list(all_times))

        try:
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                headers = ["Time"] + [f"{hex(self.meta[k]['frame_id'])} {self.meta[k]['msg']}.{self.meta[k]['sig']}" for k in keys]
                writer.writerow(headers)

                # Step-Hold補間で全時間軸のデータを出力
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
            QtWidgets.QMessageBox.information(self, "Success", f"CSV Export completed successfully.")
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
        
        title = f"{meta['msg']} . {meta['sig']}"
        if meta['unit']:
            title += f" [{meta['unit']}]"
        pi.setTitle(title, size="10pt")
        
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

    def add_to_active_table(self, key):
        row = self.active_table.rowCount()
        self.active_table.insertRow(row)
        
        chk = QtWidgets.QCheckBox()
        chk.setChecked(True)
        chk.setStyleSheet("QCheckBox::indicator { width: 16px; height: 16px; }")
        chk.toggled.connect(lambda checked, k=key: self.plots[k]['widget'].setVisible(checked))
        
        name_lbl = QtWidgets.QLabel(f" {self.meta[key]['sig']}")
        name_lbl.setStyleSheet(f"color: {self.plots[key]['color'].name()}; font-weight: bold;")
        
        val_lbl = QtWidgets.QLabel("-")
        val_lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        
        del_btn = QtWidgets.QPushButton("✖")
        del_btn.setFixedSize(24, 24)
        del_btn.clicked.connect(lambda _, r=row, k=key: self.remove_signal(k))

        self.active_table.setCellWidget(row, 0, chk)
        self.active_table.setCellWidget(row, 1, name_lbl)
        self.active_table.setCellWidget(row, 2, val_lbl)
        self.active_table.setCellWidget(row, 3, del_btn)
        self.active_table.setItem(row, 0, QtWidgets.QTableWidgetItem(key))

    def remove_signal(self, key):
        for r in range(self.active_table.rowCount()):
            item = self.active_table.item(r, 0)
            if item and item.text() == key:
                self.active_table.removeRow(r)
                break
                
        pw = self.plots[key]['widget']
        self.plot_vbox.removeWidget(pw)
        pw.deleteLater()
        del self.plots[key]

    def toggle_play(self):
        if not hasattr(self, 'blf_path') or not self.db:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please load DBC and BLF first.")
            return

        if not self.is_playing:
            self.data.clear()
            self.max_time = 0.0
            self.play_pos = 0.0
            
            target_ids = set([self.meta[k]["frame_id"] for k in self.plots.keys()])
            if not target_ids:
                QtWidgets.QMessageBox.information(self, "Info", "Please select at least one signal to plot first.")
                return

            self.reader = BLFReaderThread(self.blf_path, self.db, target_ids)
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
                
            key = item.text()
            p_data = self.plots[key]
            d = self.data.get(key)
            
            # --- プロットの更新 ---
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
                    
                    t_step = np.repeat(t_plot, 2)[1:]
                    v_step = np.repeat(v_plot, 2)[:-1]
                    p_data["curve"].setData(t_step, v_step)
                
                cursor_pos = self.play_pos - start
                p_data["line"].setPos(cursor_pos)
                p_data["widget"].getPlotItem().setXRange(0, end - start, padding=0)

                # --- 現在値とStaleの更新 ---
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
                        
                    lbl = self.active_table.cellWidget(row, 2)
                    if lbl:
                        lbl.setText(val_str)

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()