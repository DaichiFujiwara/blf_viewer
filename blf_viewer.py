# blf_viewer.py
# Modified: add thorough English comments that explain operation and rationale.
# Base functionality:
#  - Loads DBC files (via cantools) and displays CAN signals decoded from BLF files.
#  - Uses a background QThread (BLFReaderThread) to stream data from BLF and emit batches to the main thread.
#  - Play / Pause controls: on Pause the reader thread is stopped; on Resume it restarts and skips data up to the last play position.
#
# Requirements (pip):
#   PySide6, pyqtgraph, python-can, cantools, numpy
#
# NOTE:
#   This file preserves the original runtime behavior but replaces/expands comments with explanatory English text.
#   Keep backups before replacing existing file.

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


# ----------------------------
# Helper utilities
# ----------------------------

def _try_get_attr(obj, names, default=None):
    """
    Try to retrieve the first available attribute from obj among names.
    This is a small utility to extract metadata from cantools signal objects,
    which may have different attribute names across versions.
    """
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default


def get_signal_meta_from_cantools_signal(s):
    """
    Convert a cantools Signal object into a small metadata dict used by the UI:
      - min / max values
      - unit string
      - choices (enumeration values) if present
    This helps set axis ranges and label ticks when plotting.
    """
    meta = {}
    meta['min'] = _try_get_attr(s, ['minimum', 'min', 'physical_minimum'], None)
    meta['max'] = _try_get_attr(s, ['maximum', 'max', 'physical_maximum'], None)
    meta['unit'] = _try_get_attr(s, ['unit'], None)
    meta['choices'] = _try_get_attr(s, ['choices', 'enum', 'values'], None)
    return meta


# ----------------------------
# Custom Axis for time formatting
# ----------------------------

class TimeAxisItem(pg.AxisItem):
    """
    Custom axis item that formats seconds into human-friendly strings:
      - for >= 1 hour: H:MM:SS.sss
      - for >= 1 minute: M:SS.sss
      - else: seconds with 3 decimals
    This keeps time ticks readable even for long logs.
    """
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


# ----------------------------
# BLF Reader Thread
# ----------------------------

class BLFReaderThread(QtCore.QThread):
    """
    Background thread that reads CAN frames from a BLF file using python-can's BLFReader.

    Key behaviors:
      - It decodes messages according to the provided DBC(s) via cantools objects.
      - It emits data in batches (data_batch_ready) to the main thread to avoid too-frequent cross-thread signals.
      - On pause the main thread calls stop(), which sets _running=False; the for-loop inside run() checks this flag and exits gracefully.
      - resume_time: when provided (float seconds), the reader will skip (discard) messages whose relative timestamp is less than resume_time.
        This is how "resume from last play_pos" is implemented without keeping the reader active during pause.
      - The thread emits progress signals periodically so the UI can show reading progress if desired.

    Signals:
      - data_batch_ready(dict, float): dict maps key -> {"t": [...], "v": [...]}, second arg is latest timestamp seen (relative seconds)
      - progress(int): number of frames processed (or coarse progress indicator)
      - finished(): emitted when the reader exits normally (either EOF or stop requested)
      - error(str): error messages for UI reporting
    """
    data_batch_ready = QtCore.Signal(dict, float)
    progress = QtCore.Signal(int)
    finished = QtCore.Signal()
    error = QtCore.Signal(str)

    def __init__(self, blf_path, dbs_info, target_keys, resume_time=None, parent=None):
        """
        Parameters:
          - blf_path: path to the BLF file to open
          - dbs_info: list of tuples (cantools.Database, dbc_name)
          - target_keys: list or set of keys corresponding to signals to decode and emit
          - resume_time: if provided, skip messages whose relative timestamp < resume_time
        """
        super().__init__(parent)
        self.blf_path = blf_path
        self.dbs_info = dbs_info
        self.target_keys = set(target_keys)
        self._running = True
        # resume_time is number of seconds from the BLF file's first timestamp (relative)
        # If None, reading starts from the file beginning (no skip).
        self.resume_time = resume_time
        # Build a quick lookup from arbitration id -> list of (cantools message, dbc_name)
        # This accelerates per-frame decoding lookups.
        self.target_messages = defaultdict(list)
        for db, dbc_name in self.dbs_info:
            for msg in db.messages:
                self.target_messages[msg.frame_id].append((msg, dbc_name))

    def stop(self):
        """
        Request the thread to stop. The run loop must check self._running and exit.
        This is used when the UI user presses Pause, or the app shuts down.
        """
        self._running = False

    def run(self):
        """
        Main loop that reads from BLFReader and emits batches of decoded signals.

        Implementation notes:
          - BLFReader yields message-like objects with attributes: timestamp, arbitration_id, data
          - We treat the first timestamp as the base and compute relative seconds (ts = raw_ts - base_ts)
          - If resume_time is provided, messages with ts < resume_time are skipped (not decoded nor emitted).
          - Decoding uses cantools' Message.decode; any decode errors are ignored for robustness.
          - To limit signal emission overhead, every N frames a batch is emitted and cleared.
        """
        try:
            reader = can.BLFReader(self.blf_path)
        except Exception as e:
            # if opening fails, notify main thread and finish
            self.error.emit(f"BLF Open Error: {e}")
            self.finished.emit()
            return

        count = 0
        base_ts = None
        max_ts = 0.0
        # batch: key -> {"t": [...], "v": [...]}
        batch = defaultdict(lambda: {"t": [], "v": []})

        for msg in reader:
            # allow quick exit if requested
            if not self._running:
                break

            # timestamp retrieval can occasionally fail; guard it
            try:
                raw_ts = float(msg.timestamp)
            except Exception:
                # skip messages without timestamp
                continue

            if base_ts is None:
                # establish the zero point of the relative timeline
                base_ts = raw_ts
            ts = raw_ts - base_ts

            # If resume_time is set (we are resuming), drop earlier messages until we reach resume_time
            if self.resume_time is not None and ts < self.resume_time:
                # reading and discarding until we reach resume_time
                continue

            arb = msg.arbitration_id
            msgs_to_try = self.target_messages.get(arb)
            if not msgs_to_try:
                # we don't have DBC info for this arbitration id, skip
                continue

            count += 1
            if ts > max_ts:
                max_ts = ts

            # try decoding with each message definition matching the arbitration id
            for message, dbc_name in msgs_to_try:
                try:
                    decoded = message.decode(msg.data)
                    # decoded is a dict: signal_name -> value
                    for sname, sval in decoded.items():
                        key = f"{dbc_name}:{arb}:{message.name}:{sname}"
                        if key in self.target_keys:
                            batch[key]["t"].append(ts)
                            batch[key]["v"].append(sval)
                except Exception:
                    # if decode fails for one message, skip it but continue other possibilities
                    pass

            # periodically emit a batch to reduce signal frequency overhead.
            # The modulus threshold (10000) is tuned for large files; you may lower it for more responsive UI.
            if count % 10000 == 0:
                # emit a shallow copy (dict) to transfer ownership safely between threads
                self.data_batch_ready.emit(dict(batch), max_ts)
                batch.clear()
                self.progress.emit(count)

        # emit any remaining collected data
        if batch:
            self.data_batch_ready.emit(dict(batch), max_ts)

        try:
            reader.close()
        except Exception:
            # closing failure is non-fatal; we just ignore it
            pass

        # final signal to indicate the reader has stopped
        self.finished.emit()


# ----------------------------
# Small UI helper widgets
# ----------------------------

class CheckableListWidget(QtWidgets.QListWidget):
    """
    A QListWidget that supports shift-click range checking.
    This is used for multi-selecting signal checkboxes in the selection dialogs.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.last_clicked_row = -1
        self.itemClicked.connect(self.on_item_clicked)

    def on_item_clicked(self, item):
        """
        When a user clicks an item with Shift pressed, copy the checkbox state across
        the range between the last-clicked and current clicked rows. This mimics
        typical file-selection behavior and speeds up selection when many signals exist.
        """
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
    """
    Dialog to display all signals for a given message/frame so the user can pick which signals to activate.
    The list shows the signal name (and unit if present).
    """
    def __init__(self, dbc_name, frame_id, frame_name, signals, selected_keys, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Select Signals - {frame_name} ({hex(frame_id)})")
        self.resize(350, 400)
        self.result_keys = []
        layout = QtWidgets.QVBoxLayout(self)
        self.list_widget = CheckableListWidget()
        layout.addWidget(self.list_widget)

        # Populate the list widget with signals as checkable items.
        for sig in signals:
            key = f"{dbc_name}:{frame_id}:{frame_name}:{sig['name']}"
            unit_str = f" [{sig['unit']}]" if sig.get('unit') else ""
            item = QtWidgets.QListWidgetItem(f"{sig['name']}{unit_str}")
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            # check initial state based on existing selection
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
        """
        When the user accepts the dialog, collect the checked keys and close.
        """
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == QtCore.Qt.Checked:
                self.result_keys.append(item.data(QtCore.Qt.UserRole))
        super().accept()


class GlobalSearchDialog(QtWidgets.QDialog):
    """
    Dialog that allows text search across all frames/signals loaded from DBCs.
    This is useful when many DBCs/frames exist and the user wants to quickly find signals.
    """
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

        # Fill list with human-friendly strings for each signal metadata entry
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
        """
        Filter visible list items by the search query (case-insensitive).
        """
        query = text.lower()
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setHidden(query not in item.text().lower())
        # reset shift-select state to avoid accidental multi-range selections after filtering
        self.list_widget.last_clicked_row = -1

    def accept(self):
        """
        Collect checked keys and close the dialog.
        """
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == QtCore.Qt.Checked:
                self.result_keys.append(item.data(QtCore.Qt.UserRole))
        super().accept()


# ----------------------------
# Main Window
# ----------------------------

class MainWindow(QtWidgets.QMainWindow):
    """
    The main application window that manages:
      - loading DBC files
      - opening BLF files
      - selecting which signals to plot
      - launching and stopping the BLFReaderThread
      - plotting data with pyqtgraph and showing current values in a table
    Behavior highlights:
      - On Play: if resume_ts is None, start fresh (data cleared). If resume_ts is set (after a Pause), keep existing data and start the reader with resume_time=resume_ts so the reader skips up to that timestamp.
      - On Pause: the reader thread's stop() is called and resume_ts is set to the current play_pos; the UI timer stops updating playback position.
      - update_plots_and_table handles slicing of per-signal arrays for display using bisect for efficient index lookup.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BLF Viewer (modified, commented)")
        self.resize(1400, 850)
        # set pyqtgraph colors: white background, black foreground
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        pg.setConfigOptions(antialias=False)

        # application state
        self.dbs_info = []     # list of tuples (cantools.Database, dbc_basename)
        self.dbc_paths = []    # list of loaded DBC file paths
        self.meta = {}         # meta info for each signal key used by UI / export
        # self.data maps key -> {"t": [times], "v_raw": [original values], "v_num": [numeric mapped values]}
        self.data = defaultdict(lambda: {"t": [], "v_raw": [], "v_num": []})
        self.plots = {}        # active plot entries keyed by signal key
        self.max_time = 0.0    # maximum observed time across all parsed data
        self.play_pos = 0.0    # current playback position in seconds
        self.current_display_start = 0.0
        self.is_workspace_modified = False

        # resume timestamp in seconds. Set when Paused; cleared/used when Play/Resume.
        self.resume_ts = None

        # additional state for UI (stale detection etc.)
        self.active_state = {}

        # build UI and timers
        self.setup_ui()
        self.load_config()
        self.setup_timers()
        # create Save shortcut
        save_shortcut = QtGui.QShortcut(QtGui.QKeySequence.Save, self)
        save_shortcut.activated.connect(self.manual_save)

    def setup_ui(self):
        """
        Build the main toolbar and central UI layout:
          - Left: DBC frames list + active signal table
          - Right: Scrollable plot area for selected signals + timeline slider
          - Toolbar: Open BLF, Add DBC(s), Search, Play/Pause, Speed, Export
        """
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
        # X-axis mode selection: Fixed Window, Follow trailing window, Auto Fit to data
        self.x_mode_combo = QtWidgets.QComboBox()
        self.x_mode_combo.addItems(["Fixed Window", "Follow (Trailing)", "Auto Fit"])
        self.x_mode_combo.currentIndexChanged.connect(self.mark_workspace_modified)
        self.x_mode_combo.currentIndexChanged.connect(lambda: self.update_plots_and_table())
        tb.addWidget(QtWidgets.QLabel(" X-Mode: ")); tb.addWidget(self.x_mode_combo)
        # window span for Fixed/Follow modes (seconds)
        self.window_spin = QtWidgets.QDoubleSpinBox(); self.window_spin.setRange(0.1, 3600.0);
        self.window_spin.setValue(5.0); self.window_spin.setSingleStep(0.5)
        self.window_spin.valueChanged.connect(self.mark_workspace_modified)
        tb.addWidget(QtWidgets.QLabel(" Window(s): ")); tb.addWidget(self.window_spin)
        # auto Y-fit option
        self.autoy_checkbox = QtWidgets.QCheckBox(" Auto Y-Fit "); self.autoy_checkbox.toggled.connect(self.mark_workspace_modified)
        tb.addWidget(self.autoy_checkbox)
        tb.addSeparator()
        # stale detection threshold
        self.stale_spin = QtWidgets.QDoubleSpinBox(); self.stale_spin.setRange(0.1, 3600.0); self.stale_spin.setSingleStep(0.5); self.stale_spin.setValue(2.0)
        self.stale_spin.valueChanged.connect(self.mark_workspace_modified)
        tb.addWidget(QtWidgets.QLabel(" Stale(s): ")); tb.addWidget(self.stale_spin)
        tb.addSeparator()
        # play / pause button
        self.play_btn = QtWidgets.QPushButton("▶ Play / Load"); self.play_btn.clicked.connect(self.toggle_play)
        tb.addWidget(self.play_btn)
        # playback speed multiplier
        self.speed_spin = QtWidgets.QDoubleSpinBox(); self.speed_spin.setRange(0.1, 50.0); self.speed_spin.setValue(1.0); self.speed_spin.setSuffix("x")
        self.speed_spin.valueChanged.connect(self.mark_workspace_modified)
        tb.addWidget(QtWidgets.QLabel(" Speed: ")); tb.addWidget(self.speed_spin)
        tb.addSeparator()
        # export action
        act_export = QtGui.QAction("💾 Export CSV", self); act_export.triggered.connect(self.export_csv); tb.addAction(act_export)

        # central layout: left (frames + active table) and right (plots)
        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central); main_layout.setContentsMargins(4,4,4,4)
        h_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        v_splitter_left = QtWidgets.QSplitter(QtCore.Qt.Vertical)

        # Frames tree (double-click to select signals for a message)
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
        # double click signal -> open signal selection dialog (SignalSelectionDialog)
        self.frame_tree.itemDoubleClicked.connect(self.open_signal_popup)
        frame_layout.addWidget(self.frame_tree)
        v_splitter_left.addWidget(frame_group)

        # Active signals table: shows visibility checkbox, frame, signal, current value, and remove button
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
        # when the visibility checkbox changes, update plot visibility
        self.active_table.itemChanged.connect(self.on_active_table_item_changed)
        active_layout.addWidget(self.active_table)
        v_splitter_left.addWidget(active_group)

        # Plot area: scrollable container where each active signal gets a small plot widget
        plot_group = QtWidgets.QGroupBox("3. Signal Plots")
        plot_layout = QtWidgets.QVBoxLayout(plot_group)
        self.plot_scroll = QtWidgets.QScrollArea(); self.plot_scroll.setWidgetResizable(True)
        self.plot_container = QtWidgets.QWidget(); self.plot_vbox = QtWidgets.QVBoxLayout(self.plot_container)
        self.plot_vbox.setContentsMargins(0,0,0,0); self.plot_vbox.addStretch()
        self.plot_scroll.setWidget(self.plot_container)
        plot_layout.addWidget(self.plot_scroll)
        # timeline slider and label
        slider_layout = QtWidgets.QHBoxLayout()
        self.time_label = QtWidgets.QLabel("Time: 0.000 s")
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.slider.setRange(0,10000); self.slider.sliderMoved.connect(self.on_slider_moved)
        slider_layout.addWidget(self.time_label); slider_layout.addWidget(self.slider)
        plot_layout.addLayout(slider_layout)

        h_splitter.addWidget(v_splitter_left); h_splitter.addWidget(plot_group)
        h_splitter.setSizes([450,950]); main_layout.addWidget(h_splitter)
        self.status = self.statusBar()

    def mark_workspace_modified(self, *args):
        """
        Mark that workspace settings have changed (so we can offer Save on exit).
        """
        self.is_workspace_modified = True

    def manual_save(self):
        """
        Manual save action triggered by toolbar or shortcut.
        """
        self.save_config()
        self.status.showMessage("Workspace saved successfully.", 3000)

    def load_config(self):
        """
        Load workspace configuration from CONFIG_FILE:
          - ui mode, window span, auto-y, stale time, speed
          - last used DBC list and BLF path
          - selected signals saved across runs
        If values are missing or file does not exist, default_config is used.
        """
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
            self.blf_path = blf_path; self.status.showMessage(f"Selected BLF: {blf_path}")
        for key in default_config["selected_signals"]:
            if key in self.meta and key not in self.plots:
                self.add_signal_plot(key)
        self.is_workspace_modified = False

    def save_config(self):
        """
        Save workspace config to CONFIG_FILE for next session.
        """
        config = {"x_mode":self.x_mode_combo.currentText(),"window_span":self.window_spin.value(),"auto_y":self.autoy_checkbox.isChecked(),"stale_time":self.stale_spin.value(),"speed":self.speed_spin.value(),"dbc_paths":self.dbc_paths,"blf_path":getattr(self,'blf_path',None),"selected_signals":list(self.plots.keys())}
        try:
            with open(CONFIG_FILE,"w",encoding="utf-8") as f: json.dump(config,f,indent=4)
            self.is_workspace_modified = False
        except Exception as e:
            print(f"Failed to save workspace: {e}")

    def closeEvent(self, event):
        """
        On window close, attempt to stop any running reader thread cleanly,
        then ask the user to save workspace if there are unsaved changes.
        """
        try:
            if hasattr(self, 'reader') and getattr(self, 'reader') is not None and getattr(self, 'reader').isRunning():
                # request stop and wait briefly for the thread to exit
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
        """
        Initialize timers used by the app:
          - play_timer: advances playback position when playing
          - ui_timer: periodic UI updates (plot redraws / table updates)
        """
        self.is_playing = False
        self.play_timer = QtCore.QTimer(); self.play_timer.timeout.connect(self.advance_playback)
        self.play_timer.setInterval(50)  # control playback granularity / responsiveness
        self.ui_timer = QtCore.QTimer(); self.ui_timer.setInterval(100); self.ui_timer.timeout.connect(self.update_plots_and_table); self.ui_timer.start()

    def add_dbc_dialog(self):
        """
        Show file dialog to add one or more DBC files. After selecting, call _load_dbc_files.
        """
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self,"Add DBC(s)","","DBC Files (*.dbc)")
        if paths: self._load_dbc_files(paths)

    def _load_dbc_files(self, paths):
        """
        Load the selected DBC files using cantools and populate the frames tree and internal metadata.
        """
        loaded_any = False
        for path in paths:
            if path in self.dbc_paths: continue
            try:
                db = cantools.database.load_file(path); dbc_name = os.path.basename(path)
                self.dbs_info.append((db, dbc_name)); self.dbc_paths.append(path)
                for msg in db.messages:
                    # create a tree item for this frame/message
                    is_first_msg = True
                    display_dbc = dbc_name if is_first_msg else ""
                    is_first_msg = False
                    item = QtWidgets.QTreeWidgetItem(self.frame_tree, [display_dbc, hex(msg.frame_id), msg.name])
                    item.setToolTip(0, dbc_name); item.setToolTip(2, msg.name)
                    item.setData(0, QtCore.Qt.UserRole, msg.frame_id)
                    item.setData(1, QtCore.Qt.UserRole, dbc_name)
                    # register signals metadata for global search / export
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
        """
        Prompt for a BLF file path and store it for subsequent reading.
        """
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self,"Open BLF","","BLF Files (*.blf)")
        if path:
            self.blf_path = path; self.status.showMessage(f"Selected BLF: {path}"); self.mark_workspace_modified()

    def open_signal_popup(self, item, column):
        """
        On double-clicking a frame tree row, open a dialog listing all signals in that message,
        so the user can choose which signals to add to the active plot list.
        """
        frame_id = item.data(0, QtCore.Qt.UserRole); dbc_name = item.data(1, QtCore.Qt.UserRole); msg_name = item.text(2)
        if frame_id is None or dbc_name is None: return
        signals = [{"name": v["sig"], "unit": v["unit"]} for k,v in self.meta.items() if v["frame_id"]==frame_id and v["dbc_name"]==dbc_name and v["msg"]==msg_name]
        if not signals: return
        dialog = SignalSelectionDialog(dbc_name, frame_id, msg_name, signals, list(self.plots.keys()), self)
        if dialog.exec():
            for key in dialog.result_keys:
                if key not in self.plots: self.add_signal_plot(key)

    def open_global_search(self):
        """
        Open the global search dialog that lists all known signals across loaded DBC(s).
        """
        if not self.meta:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please load a DBC file first."); return
        dialog = GlobalSearchDialog(self.meta, list(self.plots.keys()), self)
        if dialog.exec():
            for key in dialog.result_keys:
                if key not in self.plots: self.add_signal_plot(key)

    def export_csv(self):
        """
        Export currently selected signals to CSV with a common time axis.
        For each distinct time across all plotted signals, write a row with the nearest previous value for each signal.
        This is a simple export suitable for offline analysis.
        """
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
        """
        Add a new plot widget for the given signal key and register it in:
          - self.plots: holds widget, curve, meta, color, axis_applied flag
          - the active table (with visibility checkbox and remove button)
        The plotting uses pyqtgraph.PlotWidget with downsampling and clipping enabled to handle large datasets.
        """
        meta = self.meta[key]
        color = pg.intColor(len(self.plots), hues=15)
        axis = TimeAxisItem(orientation='bottom')
        pw = pg.PlotWidget(axisItems={'bottom': axis}); pw.setFixedHeight(180); pi = pw.getPlotItem()
        pi.showGrid(x=True, y=True, alpha=0.5)
        title = f"[{meta['dbc_name']}] {meta['msg']} . {meta['sig']}"
        if meta['unit']: title += f" [{meta['unit']}]"
        pi.setTitle(title, size="10pt")
        curve = pi.plot([], [], pen=pg.mkPen(color=color, width=2))
        # performance-related settings
        curve.setClipToView(True)
        curve.setDownsampling(ds=True, auto=True, method='peak')
        # try to link X axis across multiple plots for synchronized zoom/pan
        for other in self.plots.values():
            try:
                pi.setXLink(other['widget'].getPlotItem())
            except Exception:
                pass
        self.plot_vbox.insertWidget(self.plot_vbox.count() - 1, pw)
        # store plot entry
        self.plots[key] = {"widget": pw, "curve": curve, "meta": meta, "color": color, "axis_applied": False}
        # apply DBC-provided axis settings (choices / min / max) if available
        self.apply_meta_to_axis(key)
        self.add_to_active_table(key)
        self.mark_workspace_modified()

    def apply_meta_to_axis(self, key):
        """
        Use DBC metadata to apply Y-axis ticks or range:
          - If 'choices' (enum) exist, create textual ticks on the left axis and set range.
          - Else if numeric min/max exist, set the Y range with a small padding.
          - Otherwise leave axis_autoscaling to the periodic update function.
        We mark axis_applied True when we have applied DBC-driven axis settings.
        """
        if key not in self.plots:
            return
        entry = self.plots[key]
        pi = entry["widget"].getPlotItem()
        meta = entry.get("meta", {}) or {}
        choices = meta.get("choices"); minv = meta.get("min"); maxv = meta.get("max")
        # If choices dictionary is present, convert it to ticks [(value,label), ...]
        if choices and isinstance(choices, dict) and len(choices) > 0:
            tick_items = []
            numeric_keys = []
            # Cantools choices keys may be ints or strings; normalize to numeric where possible
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
        # numeric min/max from DBC: apply a range with small padding
        if (isinstance(minv, (int, float)) and isinstance(maxv, (int, float)) and maxv > minv):
            try:
                span = float(maxv) - float(minv)
                pad = span * 0.05 if span > 0 else 0.5
                pi.setYRange(float(minv) - pad, float(maxv) + pad)
            except Exception:
                pass
            entry["axis_applied"] = True
            return
        # else: axis_applied stays False and auto-Y logic will apply during plotting

    def add_to_active_table(self, key):
        """
        Add a row to the active signals table with:
          - a checkbox that controls visibility
          - frame/message name
          - signal name (colored)
          - current value (updated periodically)
          - remove button
        """
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
        """
        Handle visibility checkbox toggles by hiding/showing the corresponding plot widget.
        """
        if item.column() == 0:
            key = item.data(QtCore.Qt.UserRole)
            if key and key in self.plots:
                is_visible = (item.checkState() == QtCore.Qt.Checked)
                self.plots[key]['widget'].setVisible(is_visible)

    def remove_signal(self, key):
        """
        Remove a signal from the active table and delete its associated plot widget.
        """
        for r in range(self.active_table.rowCount()):
            item = self.active_table.item(r, 0)
            if item and item.data(QtCore.Qt.UserRole) == key:
                self.active_table.removeRow(r); break
        if key in self.plots:
            pw = self.plots[key]['widget']; self.plot_vbox.removeWidget(pw); pw.deleteLater()
            del self.plots[key]; self.mark_workspace_modified()

    def toggle_play(self):
        """
        Toggle Play / Pause behavior.

        Play behavior:
          - If resume_ts is None: treat as a fresh playback; clear internal buffers and start reader from beginning.
          - If resume_ts is set: this is a resume after Pause; keep existing data buffers and start reader with resume_time=resume_ts to skip already-played data.

        Pause behavior:
          - Save the current play_pos into self.resume_ts and call reader.stop() to halt the background reading thread.
          - The reader will exit quickly (upon next loop iteration) and emit finished; UI indicates paused state.
        """
        if not hasattr(self, 'blf_path') or not self.dbs_info:
            QtWidgets.QMessageBox.warning(self,"Warning","Please load at least one DBC and a BLF file first."); return

        if not self.is_playing:
            # Play or Resume
            target_keys = list(self.plots.keys())
            if not target_keys:
                QtWidgets.QMessageBox.information(self,"Info","Please select at least one signal to plot first."); return

            resume_ts = getattr(self, 'resume_ts', None)
            if resume_ts is None:
                # Fresh start: clear existing data and reset play position
                self.data.clear(); self.max_time = 0.0; self.play_pos = 0.0
                self.status.showMessage("Starting playback from beginning...")
            else:
                # Resuming after Pause: keep existing data; set play_pos to the resume timestamp
                self.play_pos = resume_ts
                self.status.showMessage(f"Resuming playback from {resume_ts:.3f} s...")

            # Create and start the reader thread, passing resume_time if present.
            self.reader = BLFReaderThread(self.blf_path, self.dbs_info, target_keys, resume_time=resume_ts)
            self.reader.data_batch_ready.connect(self.on_data_batch)
            self.reader.progress.connect(lambda n: self.status.showMessage(f"Reading... {n} target frames parsed"))
            self.reader.finished.connect(self._on_reader_finished)
            self.reader.error.connect(lambda s: self.status.showMessage(s))
            self.reader.start()
            self.is_playing = True; self.play_btn.setText("⏸ Pause"); self.play_timer.start()
        else:
            # Pause: store current playback position and request the reader to stop
            try:
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
        """
        Called when the BLFReaderThread emits finished (either EOF or stop requested).
        We clear the reference so a new thread can be created on next Play.
        """
        try:
            self.reader = None
        except Exception:
            pass
        if getattr(self, 'resume_ts', None) is not None and not self.is_playing:
            self.status.showMessage("Paused")
        else:
            self.status.showMessage("Reading Finished")

    @QtCore.Slot(dict, float)
    def on_data_batch(self, batch, new_max_time):
        """
        Slot to receive batches of decoded signal values from the reader thread.

        The batch format:
          { key: {"t": [t1,t2,...], "v": [v1,v2,...]}, ... }
        We append these to internal buffers (self.data). v_raw keeps original values (strings/enums),
        and v_num stores numeric-mapped values for plotting (attempted conversion, enums mapped to numeric keys).
        """
        if new_max_time > self.max_time: self.max_time = new_max_time
        for key, new_data in batch.items():
            if key not in self.data: self.data[key] = {"t": [], "v_raw": [], "v_num": []}
            # Append timestamps and raw values
            self.data[key]["t"].extend(new_data["t"]); self.data[key]["v_raw"].extend(new_data["v"])
            # Map values to numeric form for plotting
            choices = self.meta.get(key, {}).get("choices")
            nums = []
            for val in new_data["v"]:
                if isinstance(val, (int, float)):
                    nums.append(float(val))
                elif choices:
                    # if DBC specified enumerations, try to map textual labels to numeric keys
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
            # If the plot for this key exists and no axis has been applied (by DBC), attempt to apply now.
            if key in self.plots:
                if not self.plots[key].get("axis_applied", False):
                    self.apply_meta_to_axis(key)

    def advance_playback(self):
        """
        Called by play_timer to advance the logical playback position (self.play_pos).
        The increment dt is scaled by the speed multiplier.
        For Follow (Trailing) mode, if the reader is active, play_pos follows the latest parsed timestamp (low-latency).
        Otherwise, it advances linearly and is clamped to max_time.
        """
        dt = 0.05 * max(0.001, float(self.speed_spin.value()))
        try:
            mode = self.x_mode_combo.currentText()
        except Exception:
            mode = 'Fixed Window'
        # If following trailing and the reader is running, keep play_pos at the newest available time
        if hasattr(self, 'reader') and getattr(self, 'reader') is not None and getattr(self, 'reader').isRunning() and mode == 'Follow (Trailing)':
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
        """
        When the user moves the timeline slider, update play_pos and refresh plots.
        """
        if self.max_time > 0:
            self.play_pos = (val / 10000.0) * self.max_time
            self.update_plots_and_table()

    def update_plots_and_table(self):
        """
        Periodic UI update that:
          - updates the time label
          - computes the current display window [start, end] based on X-mode
          - for each active plotted signal, uses bisect to find indices for the time slice and sets plot data
          - performs 'stepify' to draw each sample as a hold/step plot (repeat points)
          - updates current value cell with stale detection based on stale_threshold
        Optimization notes:
          - large slices are down-sampled with a simple step to limit plot points (cap ~30000 samples)
          - numpy arrays are used for min/max computations for auto-Y scaling
        """
        if not self.plots:
            return
        self.time_label.setText(f"Time: {self.play_pos:.3f} s")
        mode = self.x_mode_combo.currentText()
        win = float(self.window_spin.value())
        stale_threshold = float(self.stale_spin.value())
        auto_y = self.autoy_checkbox.isChecked()

        # determine the time window to display based on X-mode
        if mode == "Fixed Window":
            start = max(0.0, self.play_pos - win/2)
            end = start + win
        elif mode == "Follow (Trailing)":
            end = max(0.0, self.play_pos)
            start = max(0.0, end - win)
        else:  # Auto Fit: compute min/max across available data
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

        self.current_display_start = start
        window_span = max(1e-6, end - start)

        # iterate active table rows to refresh each plot and value cell
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
                # find indices within current window using bisect (O(log n))
                idx_start = bisect.bisect_left(t_list, start)
                idx_end = bisect.bisect_right(t_list, end)
                if idx_end > idx_start:
                    t_slice = t_list[idx_start:idx_end]
                    v_slice = d["v_num"][idx_start:idx_end]
                    # guard against extremely large arrays: crude downsample by stepping
                    if len(t_slice) > 30000:
                        step = max(1, len(t_slice) // 30000)
                        t_slice = t_slice[::step]; v_slice = v_slice[::step]
                    t_plot = np.array(t_slice)
                    v_plot = np.array(v_slice)
                    # Auto Y-scaling if enabled or no DBC-provided axis applied
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
                    # create a step-like curve by repeating points:
                    # e.g., t: [t0,t1] -> [t0, t0, t1], v: [v0,v1] -> [v0, v0, v1]
                    if len(t_plot) > 0:
                        t_step = np.repeat(t_plot, 2)[1:]
                        v_step = np.repeat(v_plot, 2)[:-1]
                        p_data["curve"].setData(t_step, v_step)
                else:
                    # no data in this window: clear the curve
                    p_data["curve"].setData([], [])
                try:
                    p_data["widget"].getPlotItem().setXRange(start, end, padding=0)
                except Exception:
                    pass
                # update the current value shown in the active table (nearest previous sample to play_pos)
                idx_cur = bisect.bisect_right(t_list, self.play_pos) - 1
                if idx_cur >= 0:
                    age = self.play_pos - t_list[idx_cur]
                    is_stale = (age > stale_threshold)
                    raw_val = d["v_raw"][idx_cur]
                    val_str = f"{raw_val:.3f}" if isinstance(raw_val, float) else str(raw_val)
                    unit = p_data['meta'].get('unit', "")
                    if unit:
                        val_str += f" {unit}"
                    # visually indicate stale signals by changing curve pen to grey dashed
                    if is_stale:
                        val_str += " (stale)"
                        p_data["curve"].setPen(pg.mkPen(color=(160,160,160), width=1.5, style=QtCore.Qt.DashLine))
                    else:
                        p_data["curve"].setPen(pg.mkPen(color=p_data["color"], width=2))
                    val_item = self.active_table.item(row, 3)
                    if val_item:
                        val_item.setText(val_str)


# ----------------------------
# Application entrypoint
# ----------------------------

def main():
    """
    Create and run the Qt application. This is the standard entrypoint when running
    the script directly (python blf_viewer.py).
    """
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()