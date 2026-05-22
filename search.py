# search.py
# Requirements: PySide6, cantools

import sys
import os
from PySide6 import QtWidgets, QtCore, QtGui
import cantools

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

class CheckableListWidget(QtWidgets.QListWidget):
    """Shiftキーでの複数項目の一括チェックに対応したリストウィジェット"""
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

class DBCSearchWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-DBC Signal Searcher & Browser")
        self.resize(1000, 650)
        
        self.dbs_info = []
        self.dbc_paths = []
        self.meta = {}  # シグナル一意キー -> メタデータ辞書
        
        self.setup_ui()
        
    def setup_ui(self):
        # ツールバー（DBC追加用）
        tb = self.addToolBar("Main")
        tb.setMovable(False)
        act_dbc = QtGui.QAction("➕ Add DBC(s)", self)
        act_dbc.setToolTip("複数選択して一括で読み込めます")
        act_dbc.triggered.connect(self.add_dbc_dialog)
        tb.addAction(act_dbc)
        
        # メインレイアウト（左右スプリッター）
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        
        # 左側: 読み込んだDBCファイルたちのツリー構造
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        lbl_tree = QtWidgets.QLabel("<b>1. Loaded DBC Structure</b>")
        self.frame_tree = QtWidgets.QTreeWidget()
        self.frame_tree.setHeaderLabels(["DBC / Message", "Frame ID"])
        self.frame_tree.setUniformRowHeights(True)
        self.frame_tree.header().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.frame_tree.itemDoubleClicked.connect(self.on_tree_item_double_clicked)
        
        left_layout.addWidget(lbl_tree)
        left_layout.addWidget(self.frame_tree)
        
        # 右側: 全DBC横断の検索・選択エリア
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        lbl_search = QtWidgets.QLabel("<b>2. Global Signal Search (Across all DBCs)</b>")
        self.search_input = QtWidgets.QLineEdit()
        self.search_input.setPlaceholderText("Search by DBC, Message, Hex/Dec ID, or Signal name...")
        self.search_input.textChanged.connect(self.filter_list)
        
        self.list_widget = CheckableListWidget()
        # エラー箇所を修正: itemCurrentChanged -> currentItemChanged
        self.list_widget.currentItemChanged.connect(self.on_list_item_selection_changed)
        
        # 下部: 選択されたシグナルの詳細情報表示
        lbl_detail = QtWidgets.QLabel("<b>3. Signal Meta Details</b>")
        self.detail_viewer = QtWidgets.QTextBrowser()
        self.detail_viewer.setFixedHeight(180)
        
        right_layout.addWidget(lbl_search)
        right_layout.addWidget(self.search_input)
        right_layout.addWidget(self.list_widget)
        right_layout.addWidget(lbl_detail)
        right_layout.addWidget(self.detail_viewer)
        
        # スプリッターに配置
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([380, 620])
        
        main_layout.addWidget(splitter)
        self.status = self.statusBar()
        self.status.showMessage("Click 'Add DBC(s)' to select and load multiple DBC files.")

    def add_dbc_dialog(self):
        # getOpenFileNames は標準でファイルの複数選択（CtrlやShiftを押しながら選択）に対応しています
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select One or Multiple DBC Files", "", "DBC Files (*.dbc)"
        )
        if paths:
            self._load_dbc_files(paths)

    def _load_dbc_files(self, paths):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            for path in paths:
                if path in self.dbc_paths: 
                    continue
                try:
                    db = cantools.database.load_file(path)
                    dbc_name = os.path.basename(path)
                    self.dbs_info.append((db, dbc_name))
                    self.dbc_paths.append(path)
                    
                    # 左側のツリーにDBCごとのルートノードを追加
                    dbc_root_item = QtWidgets.QTreeWidgetItem(self.frame_tree, [dbc_name, f"({len(db.messages)} msgs)"])
                    font = dbc_root_item.font(0)
                    font.setBold(True)
                    dbc_root_item.setFont(0, font)
                    
                    for msg in db.messages:
                        fid_hex = hex(msg.frame_id)
                        msg_item = QtWidgets.QTreeWidgetItem(dbc_root_item, [msg.name, fid_hex])
                        msg_item.setData(0, QtCore.Qt.UserRole, (dbc_name, msg.frame_id, msg.name))
                        
                        # メタデータの格納と右側検索リストへの追加
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
                                "choices": sm.get("choices"),
                                "comment": getattr(s, "comment", "")
                            }
                            
                            # 検索用テキスト（DBC名、16進数ID、10進数ID、メッセージ名、シグナル名をすべて含む）
                            display_text = f"[{dbc_name}] {fid_hex} ({msg.frame_id}) : {msg.name} ➔ {s.name}"
                            list_item = QtWidgets.QListWidgetItem(display_text)
                            list_item.setFlags(list_item.flags() | QtCore.Qt.ItemIsUserCheckable)
                            list_item.setData(QtCore.Qt.UserRole, key)
                            list_item.setCheckState(QtCore.Qt.Unchecked)
                            self.list_widget.addItem(list_item)
                            
                    dbc_root_item.setExpanded(False)  # 最初はすっきりさせるため閉じておく
                except Exception as e:
                    QtWidgets.QMessageBox.critical(self, "DBC Load Error", f"Failed to load {os.path.basename(path)}\n{str(e)}")
            
            self.status.showMessage(f"Total Loaded DBCs: {len(self.dbc_paths)} | Total Signals: {len(self.meta)}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def filter_list(self, text):
        query = text.lower().strip()
        # 描画更新の高速化のために一時的にシグナルを止める
        self.list_widget.setUpdatesEnabled(False)
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setHidden(query not in item.text().lower())
        self.list_widget.setUpdatesEnabled(True)
        self.list_widget.last_clicked_row = -1

    def on_tree_item_double_clicked(self, item, column):
        """左側ツリーのメッセージをダブルクリックした際、右側リストをそのメッセージ名で自動フィルタする"""
        data = item.data(0, QtCore.Qt.UserRole)
        if data:
            _, _, msg_name = data
            self.search_input.setText(msg_name)

    def on_list_item_selection_changed(self, current, previous):
        """シグナルを選択した時に詳細情報を下部パネルにHTML形式で綺麗に表示する"""
        if not current:
            self.detail_viewer.clear()
            return
            
        key = current.data(QtCore.Qt.UserRole)
        m = self.meta.get(key)
        if not m:
            return
            
        # 見栄えを整えたリッチテキスト
        html = f"""
        <table width="100%" cellpadding="2" cellspacing="0">
            <tr><td><b>Signal Name:</b></td><td><span style='color: #0055ff; font-size: 11pt;'><b>{m['sig']}</b></span></td></tr>
            <tr><td><b>Message Name:</b></td><td><b>{m['msg']}</b> (ID: {hex(m['frame_id'])} / {m['frame_id']})</td></tr>
            <tr><td><b>Source DBC:</b></td><td><span style='color: #227722;'>{m['dbc_name']}</span></td></tr>
        </table>
        <hr size="1" color="#ccc">
        <b>Physical Range:</b> Min: {m['min']} ~ Max: {m['max']} &nbsp;&nbsp;|&nbsp;&nbsp; <b>Unit:</b> '{m['unit']}'<br>
        """
        
        if m['comment']:
            html += f"<br><b>Comment:</b> {m['comment']}<br>"
            
        if m['choices']:
            html += "<br><b>Value Table (Choices / Enums):</b><ul>"
            for val, label in sorted(m['choices'].items(), key=lambda x: x[0]):
                html += f"<li><b>{val}</b>: {label}</li>"
            html += "</ul>"
            
        self.detail_viewer.setHtml(html)

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = DBCSearchWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()