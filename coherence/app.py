"""
coherence.app  —  v0.1
──────────────────────
Ventana principal estilo SMAART:
  Toolbar superior  |  Área de gráficas (tabs)  |  Panel de settings (derecha)

Tabs:
  MEASUREMENT  →  IR  /  Transfer Function + Coherencia  /  Phase
  SPECTRUM     →  RTA Spectrum

GPL v3
"""

import sys
import os
import datetime
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QSlider, QLabel, QSpinBox,
    QComboBox, QFrame, QGroupBox, QSizePolicy,
    QTabWidget, QTabBar, QStatusBar, QDoubleSpinBox,
    QProgressBar, QFileDialog, QDialog,
    QDialogButtonBox, QFormLayout, QScrollArea,
    QMessageBox, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QTreeWidget, QTreeWidgetItem,
    QLineEdit, QSplitter, QStackedWidget, QMenu,
)
from PyQt6.QtCore  import Qt, QTimer, QDateTime
from PyQt6.QtGui   import QFont, QPalette, QColor, QKeySequence, QShortcut
from PyQt6.QtWidgets import QMenuBar

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

from .dsp   import compute_analysis, setup_smaart_axis, find_delay, welch_spectra, RunningTF
from .dsp   import BLUE, GREEN, ORANGE, PURPLE, GRAY, CYAN, RED
from .audio import AudioEngine


# ── Paleta ────────────────────────────────────────────────────────────
BG_APP      = '#111311'   # SMAART: negro con tinte verde muy sutil
BG_PANEL    = '#161916'
BG_SETTINGS = '#1a1f1a'
BG_PLOT     = '#0c0f0c'   # fondo del gráfico oscuro-verde
BG_TOOLBAR  = '#141714'
ACCENT      = '#00bcd4'   # cyan accent
TEXT_HI     = '#e8ece8'
TEXT_MID    = '#8a9e8a'   # texto medio con tinte verde
TEXT_DIM    = '#3a4a3a'
BORDER      = '#252925'   # borde con tinte verde
COH_COLOR   = '#ef5350'   # coherencia: rojo
IR_COLOR    = '#80cbc4'   # verde agua para IR

# ── Paleta de engines (vivos) — SMAART style ─────────────────────────
ENGINE_PALETTE = [
    '#3cf53c',   # 0 SMAART verde brillante — engine 1 (live)
    '#4fc3f7',   # 1 cyan — engine 2
    '#ffb74d',   # 2 naranja — engine 3
    '#ce93d8',   # 3 violeta — engine 4
]

# ── Paleta de trazas almacenadas (distinta de engines) ────────────────
TRACE_PALETTE = [
    '#fff176',   # 0 amarillo
    '#ef9a9a',   # 1 rojo suave
    '#80deea',   # 2 turquesa
    '#ffcc80',   # 3 naranja suave
    '#ce93d8',   # 4 violeta
    '#a5d6a7',   # 5 verde suave
    '#4fc3f7',   # 6 cyan
]
MAX_TRACES = len(TRACE_PALETTE)


# ── Stylesheet ────────────────────────────────────────────────────────
QSS = f"""
* {{
    background-color: {BG_APP};
    color: {TEXT_HI};
    font-family: 'Menlo', 'Monaco', 'Courier New';
    font-size: 11px;
}}
QMainWindow, QWidget {{ background-color: {BG_APP}; }}

/* ── Toolbar ── */
QToolBar {{
    background-color: {BG_TOOLBAR};
    border-bottom: 1px solid {BORDER};
    spacing: 4px;
    padding: 3px 6px;
}}
QToolBar::separator {{
    background: {BORDER};
    width: 1px;
    margin: 4px 6px;
}}
QToolBar QLabel {{
    color: {TEXT_DIM};
    font-size: 9px;
    letter-spacing: 1px;
    padding: 0 4px;
}}

/* ── Tabs ── */
QTabWidget::pane {{
    border: none;
    background: {BG_PANEL};
}}
QTabBar::tab {{
    background: {BG_APP};
    color: {TEXT_DIM};
    padding: 5px 18px;
    border: none;
    border-right: 1px solid {BORDER};
    font-size: 10px;
    letter-spacing: 2px;
}}
QTabBar::tab:selected {{
    background: {BG_PANEL};
    color: {ACCENT};
    border-bottom: 2px solid {ACCENT};
}}
QTabBar::tab:hover {{ color: {TEXT_HI}; }}

/* ── Settings panel ── */
#settings_panel {{
    background-color: #1e2020;
    border-left: 1px solid {BORDER};
}}

/* ── GroupBox (global) ── */
QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 3px;
    margin-top: 8px;
    padding-top: 6px;
    color: {TEXT_DIM};
    font-size: 9px;
    letter-spacing: 1px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    top: 0px;
}}

/* ── Settings panel compact overrides ── */
#settings_panel QGroupBox {{
    margin-top: 4px;
    padding-top: 1px;
    padding-bottom: 0px;
    font-size: 8px;
}}
#settings_panel QSpinBox,
#settings_panel QDoubleSpinBox,
#settings_panel QComboBox {{
    max-height: 18px;
    min-height: 18px;
    padding: 0px 3px;
    font-size: 10px;
}}
#settings_panel QPushButton {{
    padding: 3px 6px;
    font-size: 10px;
    min-height: 22px;
    max-height: 22px;
}}
#settings_panel QSlider::groove:horizontal {{
    height: 3px;
}}
#settings_panel QSlider::handle:horizontal {{
    width: 9px;
    height: 9px;
    margin: -3px 0;
}}

/* ── Botones generales ── */
QPushButton {{
    border: 1px solid {BORDER};
    border-radius: 3px;
    padding: 5px 10px;
    color: {TEXT_HI};
    background-color: #1e1e1e;
    font-size: 11px;
}}
QPushButton:hover   {{ background-color: #252525; border-color: {ACCENT}; }}
QPushButton:disabled {{ color: {TEXT_DIM}; background-color: #141414; }}
QPushButton:checked  {{ background-color: #00363a; border-color: {ACCENT}; color: {ACCENT}; }}

/* Botones de toolbar: sin borde, más pequeños */
QToolBar QPushButton {{
    padding: 4px 12px;
    font-size: 11px;
    min-width: 60px;
}}

#btn_start {{ background-color: #1b5e20; border-color: #388e3c; font-weight: bold; }}
#btn_start:hover {{ background-color: #2e7d32; }}
#btn_start:disabled {{ background-color: #141414; color: {TEXT_DIM}; border-color:{BORDER}; }}

#btn_stop  {{ background-color: #7f0000; border-color: #c62828; font-weight: bold; }}
#btn_stop:hover  {{ background-color: #b71c1c; }}
#btn_stop:disabled {{ background-color: #141414; color: {TEXT_DIM}; border-color:{BORDER}; }}

#btn_noise {{ min-width: 70px; }}
#btn_noise:checked {{ background-color: #1a237e; border-color: #3949ab; color: #90caf9; }}

#btn_delay {{
    background-color: #1a1a1a;
    border-color: {ACCENT};
    color: {ACCENT};
    font-size: 10px;
    min-width: 90px;
}}
#btn_delay:hover {{ background-color: #003038; }}

/* ── Sliders ── */
QSlider::groove:horizontal {{
    background: #1e1e1e;
    height: 3px;
    border-radius: 1px;
}}
QSlider::handle:horizontal {{
    background: {ACCENT};
    width: 11px; height: 11px;
    margin: -4px 0;
    border-radius: 5px;
}}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 1px; }}

/* ── SpinBox / ComboBox ── */
QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: #1a1a1a;
    border: 1px solid {BORDER};
    border-radius: 3px;
    padding: 3px 5px;
    color: {TEXT_HI};
}}
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 14px; }}
QComboBox QAbstractItemView {{
    background-color: #1e1e1e;
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT};
    color: {TEXT_HI};
}}

/* ── Status bar ── */
QStatusBar {{
    background-color: #0d0d0d;
    border-top: 1px solid {BORDER};
    color: {TEXT_DIM};
    font-size: 10px;
}}

/* ── Info bar (arriba de tabs) ── */
#info_bar {{
    background-color: #0d0d0d;
    border-bottom: 1px solid {BORDER};
}}
#lbl_app_title {{
    color: {ACCENT};
    font-size: 14px;
    font-weight: bold;
    letter-spacing: 4px;
    padding: 0 10px;
}}
#lbl_cursor_info {{
    color: {TEXT_MID};
    font-size: 11px;
    padding: 0 8px;
}}
#lbl_clock {{
    color: {TEXT_DIM};
    font-size: 13px;
    padding: 0 10px;
    letter-spacing: 2px;
}}

/* ── Panel guardar (izquierda) ── */
#save_panel {{
    background-color: #0d0d0d;
    border-right: 1px solid {BORDER};
}}
#save_panel QPushButton {{
    background-color: transparent;
    border: 1px solid {BORDER};
    border-radius: 3px;
    color: {TEXT_MID};
    font-size: 9px;
    padding: 5px 4px;
    min-width: 0px;
    letter-spacing: 1px;
}}
#save_panel QPushButton:hover {{ color: {ACCENT}; background-color: #141414; border-color: {ACCENT}; }}

/* ── Toggle panel ── */
#btn_toggle_panel {{
    background-color: #0d0d0d;
    border: none;
    border-left: 1px solid {BORDER};
    color: {TEXT_DIM};
    font-size: 10px;
    padding: 0;
    min-width: 14px;
    max-width: 14px;
}}
#btn_toggle_panel:hover {{ color: {ACCENT}; background-color: #141414; }}

/* ── Frame separador ── */
QFrame#sep {{ background-color: {BORDER}; max-height: 1px; margin: 3px 0; }}

/* ── SPL bars ── */
QProgressBar {{
    background-color: #1a1a1a;
    border: 1px solid {BORDER};
    border-radius: 2px;
    height: 7px;
    text-align: center;
}}
QProgressBar::chunk {{ background-color: #66bb6a; border-radius: 2px; }}
QProgressBar#bar_warn::chunk  {{ background-color: #ffb74d; }}
QProgressBar#bar_clip::chunk  {{ background-color: #ef5350; }}

/* ── Label destacado ── */
#lbl_delay_val {{
    color: {ACCENT};
    font-size: 13px;
    font-weight: bold;
    padding: 2px 0;
}}
#lbl_coh_val {{
    color: {GREEN};
    font-size: 12px;
    font-weight: bold;
    padding: 2px 0;
}}
"""


# ── Helpers ───────────────────────────────────────────────────────────

def sep():
    f = QFrame()
    f.setObjectName('sep')
    f.setFrameShape(QFrame.Shape.HLine)
    return f

def lbl(text, color=TEXT_MID, size=9, parent=None):
    w = QLabel(text, parent)
    w.setStyleSheet(f'color:{color};font-size:{size}px;letter-spacing:1px;')
    return w

def fmt_freq(f_hz):
    if f_hz >= 1000:
        return f'{f_hz/1000:.2f} kHz'
    return f'{f_hz:.1f} Hz'


# ── Datos de una traza almacenada ─────────────────────────────────────

class TraceData:
    """Snapshot de una medición para mostrar como referencia estática."""
    def __init__(self, name, color, freqs, mag_db, phase_deg, gamma2, ir,
                 delay_ref_ms, coh_thresh, lev_x, lev_y):
        self.name          = name
        self.color         = color
        self.visible       = True
        self.freqs         = freqs
        self.mag_db        = mag_db
        self.phase_deg     = phase_deg
        self.gamma2        = gamma2
        self.ir            = ir
        self.delay_ref_ms  = delay_ref_ms
        self.coh_thresh    = coh_thresh
        self.lev_x         = lev_x   # CPB REF 31 bandas
        self.lev_y         = lev_y   # CPB MED 31 bandas


# ── Fila de una traza en el panel ─────────────────────────────────────

class TraceRow(QWidget):
    """Fila compacta: ■ nombre  [●]  [✕]"""

    def __init__(self, idx, name, color, on_vis, on_del, visible=True, parent=None):
        super().__init__(parent)
        self.setFixedHeight(20)
        h = QHBoxLayout(self)
        h.setContentsMargins(2, 0, 2, 0)
        h.setSpacing(3)

        dot = QLabel('■')
        dot.setStyleSheet(f'color:{color};font-size:10px;max-width:12px;background:transparent;')
        h.addWidget(dot)

        lbl_name = QLabel(name[:12])
        lbl_name.setStyleSheet(f'color:{TEXT_MID};font-size:9px;background:transparent;')
        h.addWidget(lbl_name, stretch=1)

        self.btn_v = QPushButton('●')
        self.btn_v.setFixedSize(16, 16)
        self.btn_v.setCheckable(True)
        self.btn_v.setChecked(visible)
        self.btn_v.setStyleSheet(
            f'font-size:8px;padding:0;border:none;background:transparent;color:{color};'
        )
        self.btn_v.clicked.connect(lambda checked, i=idx: on_vis(i, checked))
        h.addWidget(self.btn_v)

        btn_d = QPushButton('✕')
        btn_d.setFixedSize(14, 14)
        btn_d.setStyleSheet(
            'font-size:9px;padding:0;border:none;background:transparent;color:#555555;'
        )
        btn_d.clicked.connect(lambda _, i=idx: on_del(i))
        h.addWidget(btn_d)


# ── Dialog de configuración de Spectrum Engine ────────────────────────

class SpectrumEngineDialog(QDialog):
    """
    Dialog estilo SMAART para crear / editar un Spectrum Engine.
    Muestra: Name, Device (solo inputs), Ch (canales del device).
    Botones: Create + (crea y abre otro) | Create & Close / Save & Close.
    """

    def __init__(self, input_devices, title='New Spectrum Measurement',
                 name='', device_id=None, ch=1,
                 on_create=None, parent=None):
        """
        input_devices : list de dict {'id', 'name', 'max_in'}
        on_create     : callable(name, device_id, ch) — se llama al crear
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setFixedWidth(440)
        self._devices  = input_devices
        self._on_create = on_create
        self._close_after = False

        # ── Estilo oscuro ─────────────────────────────────────────────
        self.setStyleSheet(f"""
            QDialog   {{background:#1a1a1a; color:{TEXT_HI}; font-size:11px;}}
            QLabel    {{color:{TEXT_MID}; background:transparent; font-size:11px;}}
            QLineEdit {{background:#252525; color:{TEXT_HI}; border:1px solid #3a3a3a;
                        border-radius:3px; padding:4px 6px; font-size:11px;}}
            QLineEdit:focus {{border:1px solid {GREEN};}}
            QComboBox {{background:#252525; color:{TEXT_HI}; border:1px solid #3a3a3a;
                        border-radius:3px; padding:3px 6px; font-size:11px;}}
            QComboBox::drop-down {{border:none; width:20px;}}
            QComboBox QAbstractItemView {{background:#252525; color:{TEXT_HI};
                        selection-background-color:#303830;}}
            QPushButton {{background:#2e2e2e; color:{TEXT_HI}; border:1px solid #444;
                          border-radius:3px; padding:5px 16px; font-size:11px;}}
            QPushButton:hover  {{background:#3a3a3a; border-color:{GREEN};}}
            QPushButton:pressed{{background:#1e1e1e;}}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 16)
        layout.setSpacing(14)

        grid = QGridLayout()
        grid.setSpacing(10)
        grid.setColumnMinimumWidth(0, 70)
        grid.setColumnStretch(1, 1)

        # Name
        grid.addWidget(QLabel('Name:'), 0, 0, Qt.AlignmentFlag.AlignRight)
        self._name_edit = QLineEdit(name)
        self._name_edit.setPlaceholderText('e.g. FOH L, Subs, …')
        grid.addWidget(self._name_edit, 0, 1)

        # Device
        grid.addWidget(QLabel('Device:'), 1, 0, Qt.AlignmentFlag.AlignRight)
        self._dev_combo = QComboBox()
        for d in input_devices:
            self._dev_combo.addItem(d['name'], d['id'])
        if device_id is not None:
            for i, d in enumerate(input_devices):
                if d['id'] == device_id:
                    self._dev_combo.setCurrentIndex(i); break
        self._dev_combo.currentIndexChanged.connect(self._on_device_changed)
        grid.addWidget(self._dev_combo, 1, 1)

        # Ch
        grid.addWidget(QLabel('Ch:'), 2, 0, Qt.AlignmentFlag.AlignRight)
        self._ch_combo = QComboBox()
        self._populate_channels()
        if ch - 1 < self._ch_combo.count():
            self._ch_combo.setCurrentIndex(ch - 1)
        grid.addWidget(self._ch_combo, 2, 1)

        layout.addLayout(grid)

        # ── Buttons ───────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.addStretch()

        is_edit = (title != 'New Spectrum Measurement')
        if not is_edit:
            btn_more = QPushButton('Create +')
            btn_more.clicked.connect(self._on_create_more)
            btn_row.addWidget(btn_more)

        lbl_confirm = 'Save & Close' if is_edit else 'Create && Close'
        btn_close = QPushButton(lbl_confirm)
        btn_close.setDefault(True)
        btn_close.clicked.connect(self._on_create_close)
        btn_row.addWidget(btn_close)

        layout.addLayout(btn_row)

    # ── Internal ──────────────────────────────────────────────────────

    def _populate_channels(self):
        self._ch_combo.clear()
        idx = self._dev_combo.currentIndex()
        n = 8
        if 0 <= idx < len(self._devices):
            n = max(self._devices[idx].get('max_in', 8), 1)
        for i in range(1, n + 1):
            self._ch_combo.addItem(f'Input {i}', i)

    def _on_device_changed(self):
        self._populate_channels()

    def _get_values(self):
        name = self._name_edit.text().strip() or 'Analog'
        idx  = self._dev_combo.currentIndex()
        dev_id = self._devices[idx]['id'] if 0 <= idx < len(self._devices) else 0
        ch     = self._ch_combo.currentData() or 1
        return name, dev_id, ch

    def _on_create_more(self):
        """Crea el engine y limpia el form para agregar otro."""
        name, dev_id, ch = self._get_values()
        if self._on_create:
            self._on_create(name, dev_id, ch)
        self._name_edit.clear()
        self._name_edit.setFocus()

    def _on_create_close(self):
        """Crea / guarda y cierra."""
        name, dev_id, ch = self._get_values()
        if self._on_create:
            self._on_create(name, dev_id, ch)
        self.accept()


# ── Fila de un Spectrum Engine — estilo SMAART ───────────────────────

class SpectrumEngineRow(QWidget):
    """
    Fila de un Spectrum Engine — estilo SMAART.
    ● | Nombre  | ▶/■ | ⚙ | ✕
    ▶/■ activa / pausa la captura de ese engine.
    Borde de color cuando está activo.
    """

    def __init__(self, idx, name, color, on_config, on_del, on_play,
                 active=True, selected=False, parent=None):
        super().__init__(parent)
        self.setFixedHeight(34)
        self._color    = color
        self._idx      = idx
        self._active   = active
        self._selected = selected
        self._update_style()

        h = QHBoxLayout(self)
        h.setContentsMargins(5, 3, 5, 3)
        h.setSpacing(4)

        # Dot de color
        dot = QLabel('●')
        dot.setFixedWidth(14)
        dot.setStyleSheet(
            f'color:{color};font-size:12px;background:transparent;border:none;')
        h.addWidget(dot)

        # Nombre del canal
        self._lbl = QLabel(name[:20])
        self._lbl.setStyleSheet(
            f'color:{TEXT_HI};font-size:10px;background:transparent;border:none;')
        h.addWidget(self._lbl, stretch=1)

        # Botón ▶ / ■  — activa/pausa el engine
        self._btn_play = QPushButton('▶' if not active else '■')
        self._btn_play.setFixedSize(22, 22)
        _play_col = color if active else '#555'
        self._btn_play.setStyleSheet(
            f'QPushButton{{font-size:9px;padding:0;border:1px solid #333;'
            f'background:transparent;color:{_play_col};border-radius:2px;}}'
            f'QPushButton:hover{{color:{color};border-color:{color};}}')
        self._btn_play.clicked.connect(lambda: on_play(idx))
        h.addWidget(self._btn_play)

        # Botón ⚙ — abre configuración
        btn_cfg = QPushButton('⚙')
        btn_cfg.setFixedSize(20, 22)
        btn_cfg.setToolTip('Configure')
        btn_cfg.setStyleSheet(
            f'QPushButton{{font-size:10px;padding:0;border:1px solid #2a2a2a;'
            f'background:transparent;color:#555;border-radius:2px;}}'
            f'QPushButton:hover{{color:{color};border-color:{color};}}')
        btn_cfg.clicked.connect(lambda: on_config(idx))
        h.addWidget(btn_cfg)

        # Botón ✕ — elimina
        btn_d = QPushButton('✕')
        btn_d.setFixedSize(14, 22)
        btn_d.setStyleSheet(
            'QPushButton{font-size:8px;padding:0;border:none;'
            'background:transparent;color:#3a3a3a;}'
            'QPushButton:hover{color:#cc4444;}')
        btn_d.clicked.connect(lambda: on_del(idx))
        h.addWidget(btn_d)

    def _update_style(self):
        border_col = self._color if self._active else '#252525'
        self.setStyleSheet(
            f'SpectrumEngineRow{{background:#161916;border:1px solid {border_col};'
            f'border-radius:3px;}}'
            f'QLabel{{background:transparent;border:none;}}'
            f'QPushButton{{background:transparent;}}')

    def set_active(self, active: bool):
        self._active = active
        self._btn_play.setText('■' if active else '▶')
        col = self._color if active else '#555'
        self._btn_play.setStyleSheet(
            f'QPushButton{{font-size:9px;padding:0;border:1px solid #333;'
            f'background:transparent;color:{col};border-radius:2px;}}'
            f'QPushButton:hover{{color:{self._color};border-color:{self._color};}}')
        self._update_style()

    def set_selected(self, sel: bool):
        self._selected = sel
        self._update_style()


# ── Canvas de Medición (IR / TF+Coh / Phase) ─────────────────────────

class CanvasOverlay(QWidget):
    """
    Wrapper que pone un canvas matplotlib dentro de un QWidget
    y añade un botón dropdown flotante en la esquina superior izquierda
    (estilo SMAART: RTA▼ / Magnitude▼ etc.) más un engrane ⚙.

    El botón está posicionado absolutamente sobre el canvas con resizeEvent.
    """

    def __init__(self, canvas, options, current, on_select, on_gear,
                 label='', on_add=None, parent=None):
        super().__init__(parent)
        self._canvas    = canvas
        self._options   = options
        self._current   = current
        self._on_select = on_select
        self._on_gear   = on_gear
        self._on_add    = on_add

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        if canvas is not None:
            lay.addWidget(canvas)
        elif label:
            ph = QLabel(label)
            ph.setAlignment(Qt.AlignmentFlag.AlignCenter)
            ph.setStyleSheet(f'background:#080808;color:#333;font-size:10px;')
            lay.addWidget(ph)

        _overlay_ss = (
            'QPushButton{font-size:10px;font-weight:bold;color:#bbbbbb;'
            'background:rgba(12,12,12,200);border:1px solid rgba(60,60,60,120);'
            'padding:2px 6px;border-radius:2px;}'
            'QPushButton:hover{color:#ffffff;background:rgba(30,40,30,230);'
            'border-color:rgba(80,160,80,180);}'
        )

        # ── Dropdown vista (top-left) ─────────────────────────────────
        self._btn = QPushButton(f'{current} ▾', self)
        self._btn.setStyleSheet(_overlay_ss)
        self._btn.setFixedHeight(20)
        self._btn.adjustSize()
        self._btn.move(6, 6)
        self._btn.clicked.connect(self._show_dropdown)
        self._btn.raise_()

        # ── Engrane (top, a la derecha del dropdown) ─────────────────
        self._gear = QPushButton('⚙', self)
        self._gear.setStyleSheet(
            'QPushButton{font-size:11px;color:#444;'
            'background:rgba(12,12,12,200);border:1px solid rgba(60,60,60,80);'
            'padding:2px 4px;border-radius:2px;}'
            'QPushButton:hover{color:#888;background:rgba(30,30,30,220);}')
        self._gear.setFixedSize(20, 20)
        self._gear.clicked.connect(on_gear)
        self._gear.raise_()

        # ── "+" agregar panel (bottom-left) ──────────────────────────
        self._btn_add = QPushButton('＋', self)
        self._btn_add.setStyleSheet(
            'QPushButton{font-size:11px;font-weight:bold;color:#3a5a3a;'
            'background:rgba(12,12,12,200);border:1px solid rgba(50,80,50,100);'
            'padding:1px 5px;border-radius:2px;}'
            'QPushButton:hover{color:#88cc88;border-color:rgba(80,160,80,200);'
            'background:rgba(20,36,20,230);}')
        self._btn_add.setFixedSize(22, 20)
        self._btn_add.raise_()
        if on_add:
            self._btn_add.clicked.connect(on_add)
        else:
            self._btn_add.setVisible(False)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._btn.adjustSize()
        self._btn.move(6, 6)
        # Engrane a la derecha del dropdown
        self._gear.move(self._btn.x() + self._btn.width() + 4, 6)
        # "+" en la esquina inferior-izquierda
        self._btn_add.move(6, self.height() - 26)

    def _show_dropdown(self):
        from PyQt6.QtWidgets import QMenu
        from PyQt6.QtGui import QAction
        menu = QMenu(self)
        menu.setStyleSheet(
            f'QMenu{{background:#1e1e1e;color:#cccccc;'
            f'border:1px solid #3a3a3a;font-size:11px;}}'
            f'QMenu::item{{padding:5px 24px 5px 12px;}}'
            f'QMenu::item:selected{{background:#1e3a4a;color:#ffffff;}}'
            f'QMenu::item:checked{{color:{ACCENT};padding-left:4px;}}')
        for opt in self._options:
            act = QAction(opt, menu)
            act.setCheckable(True)
            act.setChecked(opt == self._current)
            def _on(checked, o=opt):
                self._current = o
                self._btn.setText(f'{o} ▾')
                self._btn.adjustSize()
                self._gear.move(self._btn.x() + self._btn.width() + 4, 4)
                if self._on_select:
                    self._on_select(o)
            act.triggered.connect(_on)
            menu.addAction(act)
        menu.exec(self._btn.mapToGlobal(self._btn.rect().bottomLeft()))


class MeasurementCanvas(FigureCanvas):
    """
    3 paneles apilados verticalmente:
      [0] Impulse Response
      [1] Transfer Function  +  Coherencia (eje derecho)
      [2] Phase
    """

    def __init__(self):
        self.fig = Figure(facecolor=BG_PANEL)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # Callback que MainWindow conecta para recibir info del cursor
        self.on_cursor_update = None
        # Últimos datos para interpolación del cursor
        self._last_freqs     = None
        self._last_gamma2    = None
        self._last_mag_db    = None
        self._last_phase_deg = None
        self._last_ir        = None
        self._fs             = 48000   # actualizado por MainWindow en cada refresh
        # Listas paralelas de líneas matplotlib para trazas almacenadas
        self._trace_tf_lines: List = []
        self._trace_ph_lines: List = []
        self._trace_ir_lines: List = []
        self._eng_colors = [ENGINE_PALETTE[0], ENGINE_PALETTE[1]]
        self._build()
        self._build_tf_overlay()

    def _build(self):
        from matplotlib.transforms import blended_transform_factory

        self._delay_ref_ms = 0.0
        self._nperseg      = 4096   # actualizado por MainWindow antes de update_plots
        self._ir_visible   = True   # controlado por Cmd+I desde MainWindow

        gs = gridspec.GridSpec(
            3, 1, figure=self.fig,
            hspace=0.16,
            left=0.055, right=0.935,
            top=0.99, bottom=0.07,
            height_ratios=[0.64, 2.2, 0.85]   # IR doble altura
        )

        # ── IR ──
        self.ax_ir = self.fig.add_subplot(gs[0])
        self.ax_ir.set_facecolor(BG_PLOT)
        self.ax_ir.set_xlim(-100, 100)   # siempre fijo, simétrico, NUNCA se mueve
        self.ax_ir.set_ylim(-1.1, 1.1)
        self.ax_ir.axhline(0, color=BORDER, lw=0.6)
        self.ax_ir.axvline(0, color='#2a2a2a', lw=0.7, ls='--')   # referencia fija 0 ms
        self.ax_ir.set_ylabel('amp', fontsize=6, color=TEXT_DIM, labelpad=1)
        self._style_ax(self.ax_ir, log=False)
        # Eje X en ms — ticks cada 50 ms (mayor) y 25 ms (menor)
        self.ax_ir.xaxis.set_major_locator(ticker.MultipleLocator(50))
        self.ax_ir.xaxis.set_minor_locator(ticker.MultipleLocator(25))
        self.ax_ir.tick_params(axis='x', labelsize=7, colors=TEXT_MID)
        self.ax_ir.set_xlabel('ms', fontsize=7, color=TEXT_MID, labelpad=1)

        self.line_ir, = self.ax_ir.plot([], [], color=self._eng_colors[0], lw=1.2, alpha=0.90)
        self.line_ir_peak = self.ax_ir.axvline(0, color=ORANGE, lw=1.2, ls='--', alpha=0.8)

        # Etiqueta del delay real en el marcador naranja
        _blend_ir = blended_transform_factory(self.ax_ir.transData, self.ax_ir.transAxes)
        self.txt_ir_delay = self.ax_ir.text(
            0, 0.97, '',
            transform=_blend_ir,
            color=ORANGE, fontsize=7, ha='center', va='top',
            fontfamily='monospace'
        )

        # ── TF (izquierda) + Coherencia (derecha twinx) ──
        self.ax_tf  = self.fig.add_subplot(gs[1])
        self.ax_coh = self.ax_tf.twinx()

        self.ax_tf.set_facecolor(BG_PLOT)
        self.ax_tf.set_ylim(-30, 30)
        setup_smaart_axis(self.ax_tf, bg=BG_PLOT)
        self.ax_tf.set_ylabel('dB', fontsize=7, color=TEXT_MID, labelpad=1)
        # Grilla horizontal cada 6 dB — estilo SMAART
        for _db in range(-24, 25, 6):
            _lw, _ls = (0.8, '--') if _db == 0 else (0.4, '-')
            _col = '#253225' if _db == 0 else '#181e18'
            self.ax_tf.axhline(_db, color=_col, lw=_lw, ls=_ls, zorder=0)

        # Alinear 0 de coherencia con 0 dB de TF
        self._tf_ymin, self._tf_ymax = -30, 30
        self._coh_ymax = 1.05
        self.ax_coh.set_ylim(*self._coh_aligned_lim())
        self.ax_coh.set_ylabel('γ²', fontsize=7, color=COH_COLOR, labelpad=1)
        self.ax_coh.tick_params(axis='y', colors=COH_COLOR, labelsize=6)
        self.ax_coh.set_yticks([0.0, 0.5, 1.0])
        self.ax_coh.spines['right'].set_color(COH_COLOR)
        self.ax_coh.spines['right'].set_alpha(0.4)
        for sp in ('top', 'left', 'bottom'):
            self.ax_coh.spines[sp].set_visible(False)

        # Línea referencia coherencia 0.9 — discreta
        self.ax_coh.axhline(0.9, color=COH_COLOR, lw=0.5, ls=':', alpha=0.35)

        # Colores de engines — SMAART verde + cyan
        self._eng_colors = [ENGINE_PALETTE[0], ENGINE_PALETTE[1]]

        f0 = np.array([20.0, 20000.0])
        self.line_tf,     = self.ax_tf.semilogx(f0, [0, 0], color=self._eng_colors[0], lw=2.2, label='TF1')
        self.line_tf2,    = self.ax_tf.semilogx([], [],     color=self._eng_colors[1], lw=1.2, alpha=0.55, ls='-', label='TF2')
        self.line_tf_avg, = self.ax_tf.semilogx([], [],     color='#ffffff', lw=1.8, alpha=0.85, ls='--', label='AVG')
        # Coherencia: fill blanco semitransparente (estilo SMAART)
        self.line_coh,    = self.ax_coh.semilogx(f0, [0, 0], color=COH_COLOR, lw=0.8, alpha=0.5)
        self._coh_fill = self.ax_coh.fill_between(f0, [0, 0],
                                                   color='#d0d8d0', alpha=0.09)

        # ── Phase ──  (fase envuelta −180…+180, estilo SMAART)
        self.ax_ph = self.fig.add_subplot(gs[2])
        self.ax_ph.set_facecolor(BG_PLOT)
        self.ax_ph.set_ylim(-185, 185)
        setup_smaart_axis(self.ax_ph, bg=BG_PLOT)
        self.ax_ph.set_ylabel('°', fontsize=7, color=TEXT_MID, labelpad=1)
        self.ax_ph.axhline(  0,   color='#253225', lw=0.9, ls='--')
        self.ax_ph.axhline( 90,   color='#1d261d', lw=0.5, ls=':')
        self.ax_ph.axhline(-90,   color='#1d261d', lw=0.5, ls=':')
        self.ax_ph.axhline( 180,  color='#1a221a', lw=0.4, ls=':')
        self.ax_ph.axhline(-180,  color='#1a221a', lw=0.4, ls=':')
        self.ax_ph.set_yticks([-180, -90, 0, 90, 180])
        self.ax_ph.tick_params(axis='y', labelsize=7, colors=TEXT_MID)

        self.line_ph,     = self.ax_ph.semilogx(f0, [0, 0], color=self._eng_colors[0], lw=1.5, alpha=0.90)
        self.line_ph2,    = self.ax_ph.semilogx([], [],     color=self._eng_colors[1], lw=0.8, alpha=0.45, ls='-')
        self.line_ph_avg, = self.ax_ph.semilogx([], [],     color=ORANGE, lw=1.2, alpha=0.9,  ls='-')

        # ── Crosshairs (cursor) — vertical + horizontal por panel ──
        _ck = dict(color=TEXT_MID, lw=0.7, ls=':', alpha=0.0, zorder=20)
        self._cxh_ir,  = self.ax_ir.plot([], [], **_ck)
        self._cyh_ir,  = self.ax_ir.plot([], [], **_ck)
        self._cxh_tf,  = self.ax_tf.plot([], [], **_ck)
        self._cyh_tf,  = self.ax_tf.plot([], [], **_ck)
        self._cxh_ph,  = self.ax_ph.plot([], [], **_ck)
        self._cyh_ph,  = self.ax_ph.plot([], [], **_ck)

        self.fig.patch.set_facecolor(BG_PANEL)
        self.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.draw()

    # ── Modos de vista (Magnitude completo / Phase-only) ──────────────

    def set_view_mode(self, mode: str):
        """
        'full'            → IR + TF + Coherencia + Phase (layout original)
        'magnitude_only'  → IR + TF + Coherencia sin Phase (Phase panel oculto)
        'phase_only'      → solo Phase expandido, oculta IR/TF/Coh
        """
        # Guardar posiciones originales la primera vez (antes de cualquier cambio)
        if not hasattr(self, '_saved_ax_pos'):
            self._saved_ax_pos = {
                'ir':  self.ax_ir.get_position().bounds,
                'tf':  self.ax_tf.get_position().bounds,
                'coh': self.ax_coh.get_position().bounds,
                'ph':  self.ax_ph.get_position().bounds,
            }

        if mode == 'magnitude_only':
            # Ocultar Phase, expandir TF para absorber ese espacio
            self.ax_ph.set_visible(False)
            self.ax_tf.set_visible(True)
            self.ax_coh.set_visible(True)
            # ax_ir: respetar Cmd+I — no forzar visible si el usuario lo ocultó
            self.ax_ir.set_visible(self._ir_visible)
            p = self._saved_ax_pos
            # Nueva altura de TF: desde el fondo del Phase hasta el tope actual del TF
            new_tf_y = p['ph'][1]                          # bottom del Phase original
            new_tf_h = p['tf'][3] + (p['tf'][1] - p['ph'][1])  # +gap+phase_h
            self.ax_tf.set_position( [p['tf'][0], new_tf_y, p['tf'][2], new_tf_h])
            self.ax_coh.set_position([p['tf'][0], new_tf_y, p['tf'][2], new_tf_h])

        elif mode == 'phase_only':
            for ax in (self.ax_ir, self.ax_tf, self.ax_coh):
                ax.set_visible(False)
            self.ax_ph.set_visible(True)
            # Expandir Phase para llenar el canvas
            self.ax_ph.set_position([0.055, 0.07, 0.880, 0.91])

        else:  # 'full'
            for ax in (self.ax_ir, self.ax_tf, self.ax_coh, self.ax_ph):
                ax.set_visible(True)
            p = self._saved_ax_pos
            self.ax_ir.set_position(p['ir'])
            self.ax_tf.set_position(p['tf'])
            self.ax_coh.set_position(p['coh'])
            self.ax_ph.set_position(p['ph'])

        self.draw_idle()

    # ── Overlay de suavizado — esquina inferior derecha del TF panel ──

    def _build_tf_overlay(self):
        _btn_ss = (
            'QPushButton{'
            f'  background:#1e221e; color:{TEXT_HI};'
            '   border:1px solid #3a3a3a; border-radius:2px;'
            '   font-size:10px; padding:2px 6px;'
            '}'
            'QPushButton:hover{'
            f'  background:#2a322a; border-color:{GREEN};'
            '}'
        )
        self._smooth_btn = QPushButton('1/12 Oct ▼', self)
        self._smooth_btn.setStyleSheet(_btn_ss)
        self._smooth_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._smooth_btn.setToolTip('TF Smoothing')
        self._smooth_btn.clicked.connect(self._show_smooth_menu)
        self._smooth_btn.raise_()
        self.on_smooth_changed = None   # callable(fraction) — MainWindow lo enlaza
        self._position_tf_overlay()

    def _position_tf_overlay(self):
        """
        Posiciona el botón en la esquina inferior-derecha del panel TF (ax_tf).
        gridspec: top=0.99, bottom=0.07, height_ratios=[0.32, 2.2, 0.85]
        """
        w, h = self.width(), self.height()
        if w < 30 or h < 30:
            return
        # El panel TF ocupa el tramo medio de la figura.
        # Total ratios = 3.37 → TF fraction = 2.2/3.37 = 0.653
        # Con top=0.99, bottom=0.07 el span total = 0.92
        # Bottom of ax_tf (in fig coords from bottom) = 0.07 + 0.92*(0.85/3.37) = 0.07+0.232 = 0.302
        # En Qt (y desde arriba): h * (1 - 0.302) = h * 0.698
        rw = max(self._smooth_btn.sizeHint().width(), 84)
        rh = 22
        self._smooth_btn.resize(rw, rh)
        x_btn = int(w * 0.935) - rw - 4   # right=0.935 del subplots_adjust
        y_btn = int(h * 0.695) - rh - 2   # justo encima del borde inferior del ax_tf
        self._smooth_btn.move(x_btn, y_btn)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, '_smooth_btn'):
            self._position_tf_overlay()

    def _show_smooth_menu(self):
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu{{background:#1e221e; color:{TEXT_HI}; border:1px solid #3a3a3a;
                   font-size:10px; padding:2px 0;}}
            QMenu::item{{padding:4px 24px 4px 10px;}}
            QMenu::item:selected{{background:#2a382a; color:{GREEN};}}
            QMenu::item:checked{{font-weight:bold;}}
        """)
        cur_frac = getattr(self, '_current_smooth', 12)
        for label, frac in _TF_SMOOTH_OPTIONS:
            act = menu.addAction(label)
            act.setCheckable(True)
            act.setChecked(frac == cur_frac)
            act.triggered.connect(
                lambda checked, f=frac, l=label: self._set_smooth(f, l))
        menu.exec(self._smooth_btn.mapToGlobal(
            self._smooth_btn.rect().bottomLeft()))

    def _set_smooth(self, frac: int, label: str):
        self._current_smooth = frac
        self._smooth_btn.setText(f'{label} ▼')
        if self.on_smooth_changed:
            self.on_smooth_changed(frac)

    def _coh_aligned_lim(self):
        """Calcula ylim del eje coherencia para alinear coh=0 con TF=0 dB."""
        tf_min, tf_max = self._tf_ymin, self._tf_ymax
        coh_max = self._coh_ymax
        pos_zero = -tf_min / (tf_max - tf_min)   # fracción desde abajo donde está 0 dB
        if pos_zero <= 0 or pos_zero >= 1:
            return (-0.05, coh_max)
        coh_min = -(pos_zero / (1.0 - pos_zero)) * coh_max
        return (coh_min, coh_max)

    def align_coh_axis(self, tf_min, tf_max):
        """Llama esto cuando cambia el rango de la TF."""
        self._tf_ymin, self._tf_ymax = tf_min, tf_max
        self.ax_coh.set_ylim(*self._coh_aligned_lim())
        self.draw_idle()

    def _style_ax(self, ax, log=True):
        ax.tick_params(axis='both', labelsize=7, colors=TEXT_MID)
        ax.grid(True, which='major', lw=0.4, color='#1e261e')
        ax.grid(True, which='minor', lw=0.25, color='#171d17', ls=':')
        for sp in ax.spines.values():
            sp.set_color(BORDER)
            sp.set_linewidth(0.6)

    # ── Engine color / highlight API ─────────────────────────────────

    def set_engine_colors(self, colors: list):
        """Actualiza los colores de las líneas según los engines registrados."""
        self._eng_colors = list(colors)
        c0 = colors[0] if len(colors) > 0 else ENGINE_PALETTE[0]
        c1 = colors[1] if len(colors) > 1 else ENGINE_PALETTE[1]
        self.line_tf.set_color(c0)
        self.line_tf2.set_color(c1)
        self.line_ph.set_color(c0)
        self.line_ph2.set_color(c1)
        self.draw_idle()

    def highlight_engine(self, idx: int):
        """
        Destaca el engine seleccionado:
        - Líneas del engine activo: lw grueso, alpha=1.0
        - Líneas del resto: lw fino, alpha=0.35
        - IR cambia al color del engine seleccionado
        """
        # TF lines
        pairs_tf = [(self.line_tf,  self.line_ph,  0),
                    (self.line_tf2, self.line_ph2, 1)]
        for ltf, lph, i in pairs_tf:
            if i == idx:
                ltf.set(linewidth=2.2, alpha=1.0)
                lph.set(linewidth=1.6, alpha=1.0)
            else:
                ltf.set(linewidth=0.9, alpha=0.35)
                lph.set(linewidth=0.8, alpha=0.35)

        # IR sigue al engine seleccionado
        color = self._eng_colors[idx] if idx < len(self._eng_colors) else self._eng_colors[0]
        self.line_ir.set_color(color)
        self.line_ir_peak.set_color(color)
        self.draw_idle()

    # ── Update ────────────────────────────────────────────────────────

    def update_plots(self, freqs, gamma2, mag_db, phase_deg, gxx_db, ir, coh_thresh=0.5):
        # Guardar para interpolación del cursor
        self._last_freqs     = freqs
        self._last_gamma2    = gamma2
        self._last_mag_db    = mag_db
        self._last_phase_deg = phase_deg
        self._last_ir        = ir

        mask = (freqs >= 20) & (freqs <= 20000)
        f    = freqs[mask]

        # TF
        self.line_tf.set_data(f, mag_db[mask])

        # Coherencia
        self._coh_fill.remove()
        self._coh_fill = self.ax_coh.fill_between(
            f, gamma2[mask], color=COH_COLOR, alpha=0.10
        )
        self.ax_coh.axhline(0.9, color=COH_COLOR, lw=0.7, ls=':', alpha=0.4)
        self.line_coh.set_data(f, gamma2[mask])

        # Phase — envuelta −180…+180, solo donde γ² >= thresh
        ok = gamma2[mask] >= coh_thresh
        if ok.sum() > 2:
            ph_wrap = ((phase_deg[mask][ok] + 180.0) % 360.0) - 180.0
            self.line_ph.set_data(f[ok], ph_wrap)
        else:
            self.line_ph.set_data([], [])

        # IR — eje X simétrico, siempre auto-centrado en el pico.
        # La IR ya viene fftshifted (energía distribuida en ambos lados).
        #
        # NOTA IMPORTANTE — eje de tiempo correcto para irfft zero-padded:
        #   irfft(H, n=N_ir) con H de nperseg//2+1 bins NO tiene dt=1/fs.
        #   Los bins de H están espaciados en fs/nperseg Hz, así que el
        #   periodo efectivo del irfft es  dt = nperseg / (N_ir * fs).
        #   Usar 1/fs daría un error de factor N_ir/nperseg (=4 típicamente).
        if ir is not None and len(ir) > 0:
            N_ir     = len(ir)
            nperseg  = getattr(self, '_nperseg', N_ir // 4)
            # dt correcto en ms: nperseg / (N_ir * fs) * 1000
            dt_ms    = nperseg * 1000.0 / (N_ir * float(self._fs))
            t_abs    = (np.arange(N_ir) - N_ir // 2) * dt_ms
            peak_idx = int(np.argmax(np.abs(ir)))
            peak_t   = t_abs[peak_idx]   # retardo real en ms ✓
            t_rel    = t_abs - peak_t    # pico SIEMPRE en t_rel = 0 ms
            # Ajustar xlim al rango real de la IR (±nperseg/(2fs) ms)
            half_ms  = (N_ir // 2) * dt_ms
            cur_xlim = self.ax_ir.get_xlim()
            if abs(cur_xlim[1] - half_ms) > 0.5:
                self.ax_ir.set_xlim(-half_ms, half_ms)
            xmin, xmax = self.ax_ir.get_xlim()
            mask_t   = (t_rel >= xmin) & (t_rel <= xmax)
            if mask_t.sum() > 2:
                self.line_ir.set_data(t_rel[mask_t], ir[mask_t])
            self.line_ir_peak.set_xdata([0, 0])  # marcador siempre en centro
            # Mostrar el retardo real medido
            if abs(peak_t) > 0.05:
                self.txt_ir_delay.set_text(f'({peak_t:.2f} ms)')
            else:
                self.txt_ir_delay.set_text('')

        self.draw_idle()

    def update_ch2(self, freqs, mag_db, phase_deg, gamma2, coh_thresh=0.5):
        """Actualiza las líneas del 2do canal de medición."""
        if freqs is None:
            self.line_tf2.set_data([], [])
            self.line_ph2.set_data([], [])
            return
        mask = (freqs >= 20) & (freqs <= 20000)
        f    = freqs[mask]
        self.line_tf2.set_data(f, mag_db[mask])
        ok = gamma2[mask] >= coh_thresh
        if ok.sum() > 2:
            ph_wrap = ((phase_deg[mask][ok] + 180.0) % 360.0) - 180.0
            self.line_ph2.set_data(f[ok], ph_wrap)
        else:
            self.line_ph2.set_data([], [])

    def update_avg(self, freqs, mag_db_avg, phase_deg_avg, gamma2, coh_thresh=0.5):
        """Actualiza la línea de promedio CH1+CH2."""
        if freqs is None:
            self.line_tf_avg.set_data([], [])
            self.line_ph_avg.set_data([], [])
            return
        mask = (freqs >= 20) & (freqs <= 20000)
        f    = freqs[mask]
        self.line_tf_avg.set_data(f, mag_db_avg[mask])
        ok = gamma2[mask] >= coh_thresh
        if ok.sum() > 2:
            ph_wrap = ((phase_deg_avg[mask][ok] + 180.0) % 360.0) - 180.0
            self.line_ph_avg.set_data(f[ok], ph_wrap)
        else:
            self.line_ph_avg.set_data([], [])

    def update_ir_range(self, ms):
        """Cambia el rango visible ±ms/2. El eje siempre es simétrico alrededor de 0."""
        half = ms / 2.0
        self.ax_ir.set_xlim(-half, half)
        self.draw_idle()

    def set_delay_ref(self, delay_ms):
        """
        Registra el retardo de referencia (delay finder).
        La IR SIEMPRE se auto-centra en su pico — este valor se usa
        solo para compatibilidad y tracking externo.
        """
        self._delay_ref_ms = delay_ms
        self.draw_idle()

    # ── Cursor / crosshair ────────────────────────────────────────────

    def _on_mouse_move(self, event):
        """Actualiza crosshair y emite info del cursor al MainWindow."""
        # Apagar todos los crosshairs
        for ln in (self._cxh_ir, self._cyh_ir,
                   self._cxh_tf, self._cyh_tf,
                   self._cxh_ph, self._cyh_ph):
            ln.set_alpha(0.0)

        ax = event.inaxes
        if ax is None or event.xdata is None:
            if self.on_cursor_update:
                self.on_cursor_update('—')
            self.draw_idle()
            return

        x, y = event.xdata, event.ydata

        def _show_v(ln, xv, ax_ref):
            """Línea vertical en datos del eje."""
            xlo, xhi = ax_ref.get_xlim()
            ylo, yhi = ax_ref.get_ylim()
            ln.set_data([xv, xv], [ylo, yhi]); ln.set_alpha(0.55)

        def _show_h(ln, yv, ax_ref):
            xlo, xhi = ax_ref.get_xlim()
            ln.set_data([xlo, xhi], [yv, yv]); ln.set_alpha(0.35)

        info = '—'

        if ax is self.ax_ir:
            _show_v(self._cxh_ir, x, self.ax_ir)
            _show_h(self._cyh_ir, y, self.ax_ir)
            info = self._fmt_ir(x, y)

        elif ax in (self.ax_tf, self.ax_coh):
            _show_v(self._cxh_tf, x, self.ax_tf)
            _show_h(self._cyh_tf, y, self.ax_tf)
            info = self._fmt_tf(x)

        elif ax is self.ax_ph:
            _show_v(self._cxh_ph, x, self.ax_ph)
            _show_h(self._cyh_ph, y, self.ax_ph)
            info = self._fmt_ph(x)

        if self.on_cursor_update:
            self.on_cursor_update(info)
        self.draw_idle()

    @staticmethod
    def _fmt_freq(f_hz):
        return fmt_freq(f_hz)

    def _interp(self, x_query, x_arr, y_arr):
        """Interpolación lineal segura sobre arrays de datos."""
        if x_arr is None or len(x_arr) < 2:
            return None
        mask = x_arr > 0
        xf, yf = x_arr[mask], y_arr[mask]
        if len(xf) < 2:
            return None
        return float(np.interp(x_query, xf, yf))

    def _fmt_ir(self, t_ms, amp):
        return (f'IR  │  t: {t_ms:+.2f} ms  │  amp: {amp:.4f}')

    def _fmt_tf(self, f_hz):
        db  = self._interp(f_hz, self._last_freqs, self._last_mag_db)
        g2  = self._interp(f_hz, self._last_freqs, self._last_gamma2)
        fs  = self._fmt_freq(f_hz)
        s_db = f'{db:+.1f} dB' if db is not None else '—'
        s_g2 = f'{g2*100:.0f}%' if g2 is not None else '—'
        return f'TF  │  {fs}  │  {s_db}  │  γ²: {s_g2}'

    def _fmt_ph(self, f_hz):
        ph = self._interp(f_hz, self._last_freqs, self._last_phase_deg)
        g2 = self._interp(f_hz, self._last_freqs, self._last_gamma2)
        fs = self._fmt_freq(f_hz)
        # Wrap for display
        s_ph = f'{((ph+180)%360)-180:.1f}°' if ph is not None else '—'
        s_g2 = f'{g2*100:.0f}%' if g2 is not None else '—'
        return f'PH  │  {fs}  │  {s_ph}  │  γ²: {s_g2}'

    # ── Trazas almacenadas ────────────────────────────────────────────

    def store_trace(self, freqs, mag_db, phase_deg, gamma2, ir,
                    delay_ref_ms, color, coh_thresh=0.5):
        """Dibuja una traza estática en los tres paneles y la registra."""
        mask = (freqs >= 20) & (freqs <= 20000)
        f    = freqs[mask]

        # TF mag — línea punteada fina
        line_tf, = self.ax_tf.semilogx(
            f, mag_db[mask], color=color, lw=1.0, alpha=0.70, ls='--', zorder=2)
        self._trace_tf_lines.append(line_tf)

        # Phase envuelta (solo donde γ² >= thresh)
        ok = gamma2[mask] >= coh_thresh
        if ok.sum() > 2:
            ph_wrap = ((phase_deg[mask][ok] + 180.0) % 360.0) - 180.0
            line_ph, = self.ax_ph.semilogx(
                f[ok], ph_wrap, color=color, lw=1.0, alpha=0.70, ls='--', zorder=2)
        else:
            line_ph, = self.ax_ph.semilogx([], [], color=color, lw=1.0, ls='--')
        self._trace_ph_lines.append(line_ph)

        # IR
        if ir is not None and len(ir) > 0:
            t_abs  = np.arange(len(ir)) / 48000.0 * 1000.0
            t_rel  = t_abs - delay_ref_ms
            xmin, xmax = self.ax_ir.get_xlim()
            mask_t = (t_rel >= xmin) & (t_rel <= xmax)
            if mask_t.sum() > 2:
                line_ir, = self.ax_ir.plot(
                    t_rel[mask_t], ir[mask_t],
                    color=color, lw=0.9, alpha=0.60, ls='--', zorder=2)
            else:
                line_ir, = self.ax_ir.plot([], [], color=color, lw=0.9)
        else:
            line_ir, = self.ax_ir.plot([], [], color=color, lw=0.9)
        self._trace_ir_lines.append(line_ir)

        self.draw_idle()

    def remove_trace(self, idx):
        """Elimina la traza en posición idx de todos los paneles."""
        if 0 <= idx < len(self._trace_tf_lines):
            for lst in (self._trace_tf_lines, self._trace_ph_lines, self._trace_ir_lines):
                lst[idx].remove()
                lst.pop(idx)
            self.draw_idle()

    def set_trace_visible(self, idx, visible):
        """Muestra u oculta la traza idx en todos los paneles."""
        if 0 <= idx < len(self._trace_tf_lines):
            for lst in (self._trace_tf_lines, self._trace_ph_lines, self._trace_ir_lines):
                lst[idx].set_visible(visible)
            self.draw_idle()

    def clear(self):
        for line in (self.line_tf, self.line_coh, self.line_ph, self.line_ir):
            line.set_data([], [])
        self.draw_idle()


# ── Opciones de resolución RTA — estilo SMAART ───────────────────────
_RES_OPTIONS = [
    ('No Banding',  0),
    ('1/48 Oct',   48),
    ('1/24 Oct',   24),
    ('1/12 Oct',   12),
    ('1/6 Oct',     6),
    ('1/3 Oct',     3),
    ('Octave',      1),
]

# ── Opciones de suavizado TF — estilo SMAART ─────────────────────────
_TF_SMOOTH_OPTIONS = [
    ('Off',        0),
    ('1/48 Oct',  48),
    ('1/24 Oct',  24),
    ('1/12 Oct',  12),
    ('1/6 Oct',    6),
    ('1/3 Oct',    3),
    ('Octave',     1),
]

# ── Frecuencias centrales ISO 1/3-octava (20 Hz … 20 kHz) ────────────
_ISO_CENTERS = np.array([
    20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
    200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
    2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000
], dtype=float)

# Bordes de banda contiguos: media geométrica entre centros adyacentes.
# Así no hay ni gaps ni solapamientos → línea CPB sin saltos extraños.
_ISO_EDGES = np.concatenate([
    [_ISO_CENTERS[0] / 2.0 ** (1/6)],
    np.sqrt(_ISO_CENTERS[:-1] * _ISO_CENTERS[1:]),   # media geométrica
    [_ISO_CENTERS[-1] * 2.0 ** (1/6)],
])


# ── Canvas de Spectrum ────────────────────────────────────────────────

class SpectrumCanvas(FigureCanvas):
    """
    RTA 1/3-octava CPB.

    Muestra los niveles como una línea suave conectando los centros ISO.
    El ruido rosa sale plano porque el ancho de banda proporcional
    compensa la pendiente 1/f del PSD.

    Cursor: muestra frecuencia + niveles REF y MED.
    """

    @staticmethod
    def _compute_bands(bpo: int, f_low=20.0, f_high=20000.0):
        """Genera centros y bordes de bandas para cualquier resolución (1/N oct)."""
        ref  = 1000.0
        k_lo = int(np.ceil( bpo * np.log2(f_low  / ref)))
        k_hi = int(np.floor(bpo * np.log2(f_high / ref)))
        centers = ref * 2.0 ** (np.arange(k_lo, k_hi + 1) / bpo)
        edges   = np.concatenate([
            [centers[0]  / 2.0 ** (1.0 / (2 * bpo))],
            np.sqrt(centers[:-1] * centers[1:]),
            [centers[-1] * 2.0 ** (1.0 / (2 * bpo))],
        ])
        return centers, edges

    def __init__(self):
        self.fig = Figure(facecolor=BG_PANEL)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.on_cursor_update = None   # conectar desde MainWindow
        self._last_lx  = None
        self._last_ly  = None
        self._last_Gxx = None
        self._last_Gyy = None
        self._trace_sp_lines: List = []   # lista de tuplas (line_x, line_y)
        self._sp_eng_lines: List  = []   # lista de dict {line, fill, color} por engine
        self._bpo = 12  # default 1/12 Oct (igual que SMAART)
        self._centers, self._edges = self._compute_bands(self._bpo)
        self._last_freqs = None
        self._build()
        self._build_overlay()

    def _build(self):
        self.ax = self.fig.add_subplot(111, facecolor=BG_PLOT)
        self.fig.subplots_adjust(left=0.07, right=0.97, top=0.95, bottom=0.09)

        setup_smaart_axis(self.ax, bg=BG_PLOT)
        self.ax.set_ylim(-80, 6)
        self.ax.set_autoscale_on(False)        # eje Y fijo siempre
        self.ax.set_ylabel('dBFS', fontsize=7, color=TEXT_MID, labelpad=1)
        # Grilla horizontal cada 6 dB — estilo SMAART
        for _db in [-60, -54, -48, -42, -36, -30, -24, -18, -12, -6, 0]:
            _col = '#232e23' if _db == 0 else '#181e18'
            _lw  = 0.6 if _db == 0 else 0.4
            self.ax.axhline(_db, color=_col, lw=_lw, ls='-', zorder=0)
        self.ax.tick_params(axis='y', labelsize=7, colors=TEXT_MID)
        for sp in self.ax.spines.values():
            sp.set_color(BORDER)

        # Línea AVG (opcional, no ligada a engines)
        self.line_y_avg, = self.ax.semilogx([], [], color=CYAN, lw=1.5, alpha=0.90, label='AVG', ls='--')
        # Las líneas por engine se crean dinámicamente con add_sp_engine_line()

        # Crosshair cursor
        _ck = dict(color=TEXT_MID, lw=0.7, ls=':', alpha=0.0, zorder=20)
        self._cxh, = self.ax.plot([], [], **_ck)
        self._cyh, = self.ax.plot([], [], **_ck)

        self.ax.legend(fontsize=7, facecolor='#1a1a1a', edgecolor=BORDER,
                       labelcolor=TEXT_MID, loc='upper right', ncol=2)
        self.fig.patch.set_facecolor(BG_PANEL)
        self.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.draw()

    # ── Overlay buttons (resolución + config) ────────────────────────

    def _build_overlay(self):
        """Crea los botones flotantes sobre el canvas."""
        _btn_ss = (
            'QPushButton{'
            f'  background:#1e221e; color:{TEXT_HI};'
            '   border:1px solid #3a3a3a; border-radius:2px;'
            '   font-size:10px; padding:2px 6px;'
            '}'
            'QPushButton:hover{'
            f'  background:#2a322a; border-color:{GREEN};'
            '}'
        )
        # Botón resolución — bottom-right del plot
        lbl0 = next(l for l, b in _RES_OPTIONS if b == self._bpo)
        self._res_btn = QPushButton(f'{lbl0} ▼', self)
        self._res_btn.setStyleSheet(_btn_ss)
        self._res_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._res_btn.clicked.connect(self._show_res_menu)
        self._res_btn.raise_()

        # Botón engrane — fuera del área de ejes, esquina inferior derecha
        self._cfg_btn = QPushButton('⚙', self)
        self._cfg_btn.setFixedSize(24, 24)
        self._cfg_btn.setStyleSheet(_btn_ss)
        self._cfg_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._cfg_btn.setToolTip('Settings (O)')
        self._cfg_btn.clicked.connect(
            lambda: self.on_config_clicked() if self.on_config_clicked else None)
        self._cfg_btn.raise_()
        self.on_config_clicked = None   # MainWindow lo enlaza

        self._position_overlay()

    def _position_overlay(self):
        """Reposiciona los botones al redimensionar el canvas."""
        w, h = self.width(), self.height()
        if w < 30 or h < 30:
            return

        # subplots_adjust: right=0.97, bottom=0.09
        # En Qt: x_right = w*0.97,  y_ax_bottom = h*(1-0.09) = h*0.91
        rw = max(self._res_btn.sizeHint().width(), 84)
        rh = 22
        self._res_btn.resize(rw, rh)

        x_res = int(w * 0.97) - rw - 4
        y_res = int(h * 0.91) - rh - 2   # justo encima del borde inferior del ax
        self._res_btn.move(x_res, y_res)

        # ⚙ fuera del eje (debajo del eje X, extremo derecho)
        self._cfg_btn.move(w - 26, h - 26)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, '_res_btn'):
            self._position_overlay()

    def _show_res_menu(self):
        """Muestra el popup de resoluciones al hacer click."""
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu{{background:#1e221e; color:{TEXT_HI}; border:1px solid #3a3a3a;
                   font-size:10px; padding:2px 0;}}
            QMenu::item{{padding:4px 24px 4px 10px;}}
            QMenu::item:selected{{background:#2a382a; color:{GREEN};}}
            QMenu::item:checked{{font-weight:bold;}}
            QMenu::indicator{{width:14px; height:14px;}}
        """)
        for label, bpo in _RES_OPTIONS:
            act = menu.addAction(label)
            act.setCheckable(True)
            act.setChecked(bpo == self._bpo)
            act.triggered.connect(
                lambda checked, b=bpo, l=label: self._set_resolution_bpo(b, l))
        menu.exec(self._res_btn.mapToGlobal(
            self._res_btn.rect().bottomLeft()))

    def _set_resolution_bpo(self, bpo: int, label: str):
        self.set_resolution(bpo)
        self._res_btn.setText(f'{label} ▼')

    def set_resolution(self, bpo: int):
        """Cambia la resolución. bpo=0 → No Banding (FFT raw)."""
        if bpo == self._bpo:
            return
        self._bpo = bpo
        if bpo > 0:
            self._centers, self._edges = self._compute_bands(bpo)
        else:
            self._centers = np.array([])
            self._edges   = np.array([])
        # Limpiar datos de engines y trazas
        for eng in self._sp_eng_lines:
            eng['line'].set_data([], [])
        for lx, ly in self._trace_sp_lines:
            lx.remove(); ly.remove()
        self._trace_sp_lines.clear()
        # Actualizar label si el botón ya existe
        if hasattr(self, '_res_btn'):
            lbl = next((l for l, b in _RES_OPTIONS if b == bpo), f'1/{bpo} Oct')
            self._res_btn.setText(f'{lbl} ▼')
        self.draw_idle()

    # ── Gestión de líneas por Spectrum Engine ─────────────────────────

    def add_sp_engine_line(self, color: str) -> int:
        """Agrega una línea para un nuevo engine. Retorna el índice."""
        line, = self.ax.semilogx([], [], color=color, lw=1.8, alpha=0.90)
        fill  = self.ax.fill_between([], [], -80, color=color, alpha=0.06)
        self._sp_eng_lines.append({'line': line, 'fill': fill, 'color': color})
        return len(self._sp_eng_lines) - 1

    def update_sp_engine(self, idx: int, freqs, Gxx):
        """Actualiza la línea del engine idx con los datos Gxx (PSD)."""
        if idx >= len(self._sp_eng_lines):
            return None
        eng = self._sp_eng_lines[idx]
        eps = 1e-12

        if self._bpo == 0:
            # No Banding — FFT raw, crop 20…20 kHz
            mask = (freqs >= 20) & (freqs <= 20000)
            fc   = freqs[mask]
            lev  = 10.0 * np.log10(Gxx[mask] + eps)
            self._last_freqs = fc
        else:
            fc  = self._centers
            lev = self._cpb(freqs, Gxx, fc, self._edges)
            self._last_freqs = fc

        eng['line'].set_data(fc, lev)
        try:
            eng['fill'].remove()
        except Exception:
            pass
        eng['fill'] = self.ax.fill_between(fc, lev, -80,
                                            color=eng['color'], alpha=0.06)
        self._sp_eng_lines[idx] = eng
        return lev

    def remove_sp_engine_line(self, idx: int):
        """Elimina la línea del engine idx del canvas."""
        if 0 <= idx < len(self._sp_eng_lines):
            eng = self._sp_eng_lines.pop(idx)
            try:
                eng['line'].remove()
                eng['fill'].remove()
            except ValueError:
                pass
            self.draw_idle()

    def set_sp_engine_visible(self, idx: int, visible: bool):
        if 0 <= idx < len(self._sp_eng_lines):
            eng = self._sp_eng_lines[idx]
            eng['line'].set_visible(visible)
            eng['fill'].set_visible(visible)
            self.draw_idle()

    # ── Análisis CPB ─────────────────────────────────────────────────

    @staticmethod
    def _cpb(freqs, G, centers=None, edges=None):
        """
        CPB de N bandas por octava.

        Para bandas con ≥2 bins FFT: suma directa.
        Para bandas con 0-1 bins:    interpola PSD × (bw/df).
        """
        if centers is None:
            centers = _ISO_CENTERS
        if edges is None:
            edges = _ISO_EDGES
        eps    = 1e-12
        df     = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
        levels = np.empty(len(centers))

        for i, fc in enumerate(centers):
            fl, fh = float(edges[i]), float(edges[i + 1])
            bw     = fh - fl
            mask   = (freqs >= fl) & (freqs < fh)
            n_bins = int(mask.sum())
            if n_bins >= 2:
                power = float(np.sum(G[mask]))
            else:
                g_fc  = float(np.interp(fc, freqs, G))
                power = g_fc * (bw / df)
            levels[i] = 10.0 * np.log10(power + eps)

        return levels

    # ── Update ────────────────────────────────────────────────────────

    def update_plots(self, freqs, Gxx, Gyy=None):
        """Backward-compat: delega a update_sp_engine(0)."""
        if self._sp_eng_lines:
            lev = self.update_sp_engine(0, freqs, Gxx)
            self._last_lx  = lev
            self._last_Gxx = Gxx
        self.draw_idle()

    def update_ch2_avg(self, lev_y2, lev_avg):
        """Actualiza la línea AVG del spectrum."""
        fc = self._centers
        if lev_avg is not None:
            self.line_y_avg.set_data(fc, lev_avg)
        else:
            self.line_y_avg.set_data([], [])

    # ── Trazas almacenadas ────────────────────────────────────────────

    def store_trace(self, lev_x, lev_y, color):
        """Dibuja trazas REF (--) y MED (:) como referencia estática."""
        fc = self._centers
        line_x, = self.ax.semilogx(
            fc, lev_x, color=color, lw=1.0, alpha=0.70, ls='--', zorder=2)
        line_y, = self.ax.semilogx(
            fc, lev_y, color=color, lw=0.8, alpha=0.55, ls=':',  zorder=2)
        self._trace_sp_lines.append((line_x, line_y))
        self.draw_idle()

    def remove_trace(self, idx):
        if 0 <= idx < len(self._trace_sp_lines):
            lx, ly = self._trace_sp_lines[idx]
            lx.remove(); ly.remove()
            self._trace_sp_lines.pop(idx)
            self.draw_idle()

    def set_trace_visible(self, idx, visible):
        if 0 <= idx < len(self._trace_sp_lines):
            lx, ly = self._trace_sp_lines[idx]
            lx.set_visible(visible)
            ly.set_visible(visible)
            self.draw_idle()

    def clear(self):
        for eng in self._sp_eng_lines:
            eng['line'].set_data([], [])
        self._last_lx = None; self._last_freqs = None
        self.draw_idle()

    # ── Cursor ────────────────────────────────────────────────────────

    def _on_mouse_move(self, event):
        self._cxh.set_alpha(0.0)
        self._cyh.set_alpha(0.0)

        if event.inaxes is not self.ax or event.xdata is None:
            if self.on_cursor_update:
                self.on_cursor_update('—')
            self.draw_idle()
            return

        x = event.xdata
        xlo, xhi = self.ax.get_xlim()
        ylo, yhi = self.ax.get_ylim()
        self._cxh.set_data([x, x], [ylo, yhi]);  self._cxh.set_alpha(0.55)
        self._cyh.set_data([xlo, xhi], [event.ydata, event.ydata])
        self._cyh.set_alpha(0.35)

        if (self._last_lx is not None and self._last_freqs is not None
                and len(self._last_lx) == len(self._last_freqs)
                and len(self._last_freqs) > 1):
            try:
                log_fc = np.log10(self._last_freqs)
                lx = float(np.interp(np.log10(x), log_fc, self._last_lx))
                info = f'SP  │  {fmt_freq(x)}  │  {lx:.1f} dBFS'
            except Exception:
                info = '—'
        else:
            info = '—'

        if self.on_cursor_update:
            self.on_cursor_update(info)
        self.draw_idle()


# ── Canvas de Espectrograma ───────────────────────────────────────────

class SpectrogramCanvas(FigureCanvas):
    """
    Espectrograma scrolling waterfall:
      X = frecuencia (Hz, escala log)
      Y = tiempo (nuevo arriba, viejo abajo)
      Color = nivel dBFS

    Usa pcolormesh — solo actualiza set_array() en cada frame
    para no redibujar ejes (performance).
    """

    N_TIME = 150   # filas de tiempo visibles

    def __init__(self):
        self.fig = Figure(facecolor=BG_PANEL)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._data  = None   # shape (N_TIME, N_FREQ)
        self._freqs = None
        self._mesh  = None
        self._cbar  = None
        self._build()

    def _build(self):
        self.ax = self.fig.add_subplot(111, facecolor=BG_PLOT)
        self.fig.subplots_adjust(left=0.07, right=0.90, top=0.95, bottom=0.09)
        self.ax.set_title('SPECTROGRAM', fontsize=9,
                          fontfamily='monospace', color=TEXT_MID, loc='left')
        self.ax.set_xlabel('Frecuencia (Hz)', fontsize=8, color=TEXT_MID)
        self.ax.set_ylabel('tiempo  ↑ nuevo', fontsize=8, color=TEXT_MID)
        self.ax.set_xscale('log')
        self.ax.set_xlim(20, 20000)
        self._apply_xticks()
        for sp in self.ax.spines.values():
            sp.set_color(BORDER)
        self.ax.tick_params(colors=TEXT_MID, labelsize=7)
        self.ax.set_yticks([])
        self.fig.patch.set_facecolor(BG_PANEL)
        self.draw()

    def _apply_xticks(self):
        _freqs_ticks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        self.ax.set_xticks(_freqs_ticks)
        self.ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: fmt_freq(x)))
        self.ax.xaxis.set_minor_locator(ticker.NullLocator())

    def update_spectrogram(self, freqs, Gxx):
        """Recibe PSD lineal → dBFS, scroll hacia arriba (nuevo arriba)."""
        mask = (freqs >= 20) & (freqs <= 20000)
        f    = freqs[mask]
        row  = 10.0 * np.log10(Gxx[mask] + 1e-12).astype(np.float32)

        rebuild = (self._data is None or self._data.shape[1] != len(f))
        if rebuild:
            self._data  = np.full((self.N_TIME, len(f)), -80.0, dtype=np.float32)
            self._freqs = f
            self._mesh  = None

        # Scroll: shift filas hacia abajo, nueva fila en index 0 (arriba)
        self._data = np.roll(self._data, 1, axis=0)
        self._data[0, :] = row

        if self._mesh is None:
            self._rebuild_mesh(f)
        else:
            # Solo actualizar datos — ejes no se redibujan
            self._mesh.set_array(self._data)
            self.draw_idle()

    def _rebuild_mesh(self, f):
        """Construye pcolormesh desde cero (solo en primer frame o cambio de fs)."""
        self.ax.cla()
        self.ax.set_facecolor(BG_PLOT)
        self.ax.set_title('SPECTROGRAM', fontsize=9,
                          fontfamily='monospace', color=TEXT_MID, loc='left')
        self.ax.set_xlabel('Frecuencia (Hz)', fontsize=8, color=TEXT_MID)
        self.ax.set_ylabel('tiempo  ↑ nuevo', fontsize=8, color=TEXT_MID)

        t = np.arange(self.N_TIME)   # eje Y: índice de tiempo (0 = más nuevo)
        self._mesh = self.ax.pcolormesh(
            f, t, self._data,
            cmap='inferno', vmin=-80, vmax=-10,
            shading='nearest',
        )
        self.ax.set_xscale('log')
        self.ax.set_xlim(20, 20000)
        self.ax.invert_yaxis()        # fila 0 (nuevo) arriba
        self.ax.set_yticks([])
        self._apply_xticks()

        for sp in self.ax.spines.values():
            sp.set_color(BORDER)
        self.ax.tick_params(colors=TEXT_MID, labelsize=7)

        # Colorbar (solo primera vez)
        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
        self._cbar = self.fig.colorbar(self._mesh, ax=self.ax, pad=0.01,
                                        fraction=0.03)
        self._cbar.ax.tick_params(colors=TEXT_MID, labelsize=6)
        self._cbar.set_label('dBFS', color=TEXT_MID, fontsize=7)
        self.fig.subplots_adjust(left=0.07, right=0.88, top=0.95, bottom=0.09)
        self.draw()

    def clear(self):
        self._data = None
        self._mesh = None
        self.ax.cla()
        self.draw_idle()


# ── Diálogo de configuración de medición ──────────────────────────────

class MeasurementConfigDialog(QDialog):
    """
    Parámetros de medición: FFT size, sample rate, overlap.
    Se abre desde Config → Measurement Config o Ctrl+M.
    """

    _FFT_OPTS = [
        ('1024  — Δf ≈ 47 Hz  (rápido)',   1024),
        ('2048  — Δf ≈ 23 Hz',             2048),
        ('4096  — Δf ≈ 12 Hz  (default)',  4096),
        ('8192  — Δf ≈ 6 Hz   (preciso)',  8192),
        ('16384 — Δf ≈ 3 Hz   (lento)',   16384),
    ]
    _FS_OPTS  = [44100, 48000, 96000]
    _OVL_OPTS = [0, 50, 75]

    def __init__(self, engine, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Measurement Config')
        self.setModal(True)
        self.setFixedSize(340, 200)
        if parent:
            self.setStyleSheet(parent.styleSheet())

        lay = QVBoxLayout(self)
        lay.setSpacing(8)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setSpacing(6)

        self.cmb_fft = QComboBox()
        self._fft_vals = []
        for label, val in self._FFT_OPTS:
            self.cmb_fft.addItem(label)
            self._fft_vals.append(val)
        try:
            self.cmb_fft.setCurrentIndex(self._fft_vals.index(engine.nperseg))
        except ValueError:
            self.cmb_fft.setCurrentIndex(2)
        form.addRow('FFT Size:', self.cmb_fft)

        self.cmb_fs = QComboBox()
        for fs in self._FS_OPTS:
            self.cmb_fs.addItem(f'{fs} Hz')
        self.cmb_fs.setCurrentIndex(self._FS_OPTS.index(engine.fs)
                                    if engine.fs in self._FS_OPTS else 1)
        form.addRow('Sample Rate:', self.cmb_fs)

        self.cmb_ovl = QComboBox()
        for o in self._OVL_OPTS:
            self.cmb_ovl.addItem(f'{o}%')
        form.addRow('Overlap:', self.cmb_ovl)

        lay.addLayout(form)
        lay.addStretch()

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def values(self):
        return {
            'nperseg': self._fft_vals[self.cmb_fft.currentIndex()],
            'fs':      self._FS_OPTS[self.cmb_fs.currentIndex()],
            'overlap': self._OVL_OPTS[self.cmb_ovl.currentIndex()],
        }


# ── Helper ───────────────────────────────────────────────────────────

def _lbl_tiny(text):
    """Label 7px para micro-etiquetas dentro de engine rows."""
    w = QLabel(text)
    w.setStyleSheet(f'color:{TEXT_DIM};font-size:7px;background:transparent;')
    return w


# ── TF Engine Widget ──────────────────────────────────────────────────

class TFEngine(QWidget):
    """
    Card estilo referencia:
      ┌─────────────────────────────┐  ← borde del color del engine
      │  ● N            [▶]    [●] │
      │  M ████████████       42.9 │
      │  R ████████████            │
      └─────────────────────────────┘

    Clic en la card → selecciona (borde más brillante).
    ▶ → activa/desactiva la medición de este engine.
    ● (pequeño) → remove.
    Canales M/R ocultos en spn_m / spn_r.
    """

    def __init__(self, number: int, color: str,
                 on_channels_changed, on_remove, on_find_delay,
                 on_select=None, on_normalize=None,
                 parent=None):
        super().__init__(parent)
        self._color          = color
        self._number         = number
        self._delay_comp_ms  = 0.0
        self._active         = True
        self._selected       = False
        self._gain_offset_db = 0.0
        self._on_select_cb   = on_select
        self._on_norm_cb     = on_normalize

        # Spinboxes ocultos — lógica de canales
        self.spn_m = QSpinBox(); self.spn_m.setRange(1, 32); self.spn_m.hide()
        self.spn_r = QSpinBox(); self.spn_r.setRange(1, 32); self.spn_r.hide()
        self.spn_m.valueChanged.connect(on_channels_changed)
        self.spn_r.valueChanged.connect(on_channels_changed)

        # ── Card con borde de color ───────────────────────────────────
        self._card = QFrame()
        self._card.setStyleSheet(
            f'QFrame#tfcard{{background:#181c1c;border:2px solid {color};'
            f'border-radius:6px;}}')
        self._card.setObjectName('tfcard')
        self._card.mousePressEvent = lambda e: self._on_body_click()

        card_lay = QVBoxLayout(self._card)
        card_lay.setContentsMargins(8, 6, 8, 6)
        card_lay.setSpacing(5)

        # ── Fila 1: ● color  N  |  ▶  ✕ ─────────────────────────────
        r1 = QHBoxLayout(); r1.setSpacing(6)

        # Círculo de color
        dot = QLabel('●')
        dot.setStyleSheet(
            f'color:{color};font-size:18px;background:transparent;')
        r1.addWidget(dot)

        # Número
        lbl_num = QLabel(str(number))
        lbl_num.setStyleSheet(
            f'color:#cccccc;font-size:16px;font-weight:bold;'
            f'background:transparent;')
        r1.addWidget(lbl_num, stretch=1)

        # Botón ▶ (play/pause por engine)
        self.btn_play = QPushButton('▶')
        self.btn_play.setFixedSize(26, 22)
        self.btn_play.setCheckable(True)
        self.btn_play.setChecked(True)
        self.btn_play.setStyleSheet(
            f'QPushButton{{font-size:11px;padding:0;border:none;'
            f'background:transparent;color:{color};}}'
            f'QPushButton:checked{{color:{color};}}'
            f'QPushButton:!checked{{color:#444444;}}')
        self.btn_play.clicked.connect(self._on_toggle_active)
        r1.addWidget(self.btn_play)

        # Botón remove (pequeño círculo gris)
        btn_rm = QPushButton('●')
        btn_rm.setFixedSize(16, 16)
        btn_rm.setStyleSheet(
            'QPushButton{font-size:9px;padding:0;border:none;'
            'background:transparent;color:#333333;}'
            'QPushButton:hover{color:#666666;}')
        btn_rm.clicked.connect(on_remove)
        r1.addWidget(btn_rm)

        card_lay.addLayout(r1)

        # ── Fila 2: M bars + valor ────────────────────────────────────
        r2 = QHBoxLayout(); r2.setSpacing(6)

        lbl_m = QLabel('M')
        lbl_m.setFixedWidth(10)
        lbl_m.setStyleSheet(f'color:{TEXT_DIM};font-size:8px;background:transparent;')
        r2.addWidget(lbl_m)

        bars_col = QVBoxLayout(); bars_col.setSpacing(3)
        self.bar_m = QProgressBar()
        self.bar_m.setRange(0, 100); self.bar_m.setValue(0)
        self.bar_m.setTextVisible(False); self.bar_m.setFixedHeight(4)
        self.bar_m.setStyleSheet(
            f'QProgressBar{{background:#222;border:none;border-radius:2px;}}'
            f'QProgressBar::chunk{{background:{color};border-radius:2px;}}')
        bars_col.addWidget(self.bar_m)

        self.bar_r = QProgressBar()
        self.bar_r.setRange(0, 100); self.bar_r.setValue(0)
        self.bar_r.setTextVisible(False); self.bar_r.setFixedHeight(4)
        self.bar_r.setStyleSheet(
            'QProgressBar{background:#222;border:none;border-radius:2px;}'
            'QProgressBar::chunk{background:#555;border-radius:2px;}')
        bars_col.addWidget(self.bar_r)
        r2.addLayout(bars_col, stretch=1)

        self.lbl_db = QLabel('—')
        self.lbl_db.setFixedWidth(46)
        self.lbl_db.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.lbl_db.setStyleSheet(
            f'color:#cccccc;font-size:11px;font-weight:bold;background:transparent;')
        r2.addWidget(self.lbl_db)

        card_lay.addLayout(r2)

        # ── Fila 3: R label + delay info ─────────────────────────────
        r3 = QHBoxLayout(); r3.setSpacing(6)

        lbl_r = QLabel('R')
        lbl_r.setFixedWidth(10)
        lbl_r.setStyleSheet(f'color:{TEXT_DIM};font-size:8px;background:transparent;')
        r3.addWidget(lbl_r)

        self.lbl_delay = QLabel('—')
        self.lbl_delay.setStyleSheet(
            f'color:{TEXT_DIM};font-size:8px;background:transparent;')
        self.lbl_delay.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        r3.addWidget(self.lbl_delay, stretch=1)

        # Botón NORM (normaliza la TF a 0 dB en banda de referencia)
        if on_normalize is not None:
            btn_norm = QPushButton('norm')
            btn_norm.setFixedHeight(14)
            btn_norm.setStyleSheet(
                'QPushButton{font-size:7px;padding:0 4px;border:1px solid #2a4a3a;'
                'color:#3a6a4a;background:transparent;border-radius:2px;}'
                'QPushButton:hover{border-color:#4a9a6a;color:#4a9a6a;}')
            btn_norm.clicked.connect(on_normalize)
            r3.addWidget(btn_norm)

        # Botón delay (pequeño, discreto)
        btn_find = QPushButton('delay')
        btn_find.setFixedHeight(14)
        btn_find.setStyleSheet(
            'QPushButton{font-size:7px;padding:0 5px;border:1px solid #2a3a3a;'
            'color:#3a7a7a;background:transparent;border-radius:2px;}'
            'QPushButton:hover{border-color:#4a9a9a;color:#4a9a9a;}')
        btn_find.clicked.connect(on_find_delay)
        r3.addWidget(btn_find)

        card_lay.addLayout(r3)

        # ── Layout raíz ──────────────────────────────────────────────
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 4)
        root.setSpacing(0)
        root.addWidget(self._card)

    # ── Slots ─────────────────────────────────────────────────────────

    def _on_toggle_active(self, checked: bool):
        self._active = checked
        self.btn_play.setText('▶' if checked else '⏸')

    def _on_body_click(self):
        if self._on_select_cb:
            self._on_select_cb()

    def set_selected(self, selected: bool):
        self._selected = selected
        if selected:
            self._card.setStyleSheet(
                f'QFrame#tfcard{{background:#1c2222;border:2px solid {self._color};'
                f'border-radius:6px;}}')
        else:
            # Borde atenuado cuando no está seleccionado
            self._card.setStyleSheet(
                f'QFrame#tfcard{{background:#161919;border:2px solid {self._color}55;'
                f'border-radius:6px;}}')

    # ── API pública ───────────────────────────────────────────────────

    def update_levels(self, m_db: float, r_db: float):
        def _pct(db): return max(0, min(100, int((db + 80) / 80 * 100)))
        self.bar_m.setValue(_pct(m_db))
        self.bar_r.setValue(_pct(r_db))
        self.lbl_db.setText(f'{m_db:.2f}')

    def set_delay(self, ms: float):
        self._delay_comp_ms = ms
        sign = '+' if ms >= 0 else ''
        self.lbl_delay.setText(f'{sign}{ms:.2f} ms')

    def reset_delay(self):
        self._delay_comp_ms = 0.0
        self.lbl_delay.setText('—')

    @property
    def active(self) -> bool:
        return self._active

    @property
    def ch_m(self) -> int:
        return self.spn_m.value()

    @property
    def ch_r(self) -> int:
        return self.spn_r.value()


# ── Panel secundario para split-view ─────────────────────────────────

class SecondaryPanel(QWidget):
    """
    Panel independiente creado al pulsar "+" en el panel principal.

    Tiene sus propios canvases (MeasurementCanvas, SpectrumCanvas,
    SpectrogramCanvas), dropdown de vista en top-left y botón × para cerrar.
    Los datos llegan desde MainWindow en el ciclo de refresh.
    """

    _VIEWS = ['Magnitude', 'Phase', 'Spectrum', 'Spectrograph']

    def __init__(self, on_close, parent=None):
        super().__init__(parent)
        self._current_view = 'Spectrum'
        self._on_close = on_close

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # ── Separador top (2 px verde muy sutil) ─────────────────────
        sep = QFrame(self)
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet('background:#1e3a1e;max-height:2px;min-height:2px;')
        lay.addWidget(sep)

        # ── Canvases propios ──────────────────────────────────────────
        self.canvas_meas  = MeasurementCanvas()
        self.canvas_spec  = SpectrumCanvas()
        self.canvas_sgram = SpectrogramCanvas()
        for c in (self.canvas_meas, self.canvas_spec, self.canvas_sgram):
            c.on_cursor_update = None  # sin cursor compartido por ahora

        self._stack = QStackedWidget()
        self._stack.addWidget(self.canvas_meas)   # 0 → Magnitude / Phase
        self._stack.addWidget(self.canvas_spec)   # 1 → Spectrum
        self._stack.addWidget(self.canvas_sgram)  # 2 → Spectrograph
        lay.addWidget(self._stack)
        self._stack.setCurrentIndex(1)

        # ── Overlays ──────────────────────────────────────────────────
        _oss = (
            'QPushButton{font-size:10px;font-weight:bold;color:#bbbbbb;'
            'background:rgba(12,12,12,200);border:1px solid rgba(60,60,60,120);'
            'padding:2px 6px;border-radius:2px;}'
            'QPushButton:hover{color:#ffffff;background:rgba(30,40,30,230);'
            'border-color:rgba(80,160,80,180);}'
        )
        self._btn_view = QPushButton('Spectrum ▾', self)
        self._btn_view.setStyleSheet(_oss)
        self._btn_view.setFixedHeight(20)
        self._btn_view.adjustSize()
        self._btn_view.move(6, 8)
        self._btn_view.clicked.connect(self._show_dropdown)
        self._btn_view.raise_()

        self._btn_close = QPushButton('×', self)
        self._btn_close.setStyleSheet(
            'QPushButton{font-size:13px;font-weight:bold;color:#553333;'
            'background:rgba(12,12,12,200);border:1px solid rgba(80,40,40,100);'
            'padding:1px 5px;border-radius:2px;}'
            'QPushButton:hover{color:#ff6666;border-color:rgba(180,60,60,200);'
            'background:rgba(36,12,12,230);}')
        self._btn_close.setFixedSize(22, 20)
        self._btn_close.clicked.connect(on_close)
        self._btn_close.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._btn_view.adjustSize()
        self._btn_view.move(6, 8)
        self._btn_close.move(self.width() - 28, 8)

    def _show_dropdown(self):
        from PyQt6.QtWidgets import QMenu
        from PyQt6.QtGui import QAction
        menu = QMenu(self)
        menu.setStyleSheet(
            'QMenu{background:#1a1a1a;color:#cccccc;'
            'border:1px solid #3a3a3a;font-size:11px;}'
            'QMenu::item{padding:5px 24px 5px 12px;}'
            'QMenu::item:selected{background:#1e3a1e;color:#ffffff;}'
            f'QMenu::item:checked{{color:{ACCENT};}}')
        for opt in self._VIEWS:
            act = QAction(opt, menu)
            act.setCheckable(True)
            act.setChecked(opt == self._current_view)
            def _cb(checked, o=opt): self.set_view(o)
            act.triggered.connect(_cb)
            menu.addAction(act)
        menu.exec(self._btn_view.mapToGlobal(self._btn_view.rect().bottomLeft()))

    def set_view(self, view: str):
        self._current_view = view
        self._btn_view.setText(f'{view} ▾')
        self._btn_view.adjustSize()
        if view in ('Magnitude', 'Phase'):
            self._stack.setCurrentIndex(0)
            self.canvas_meas.set_view_mode(
                'phase_only' if view == 'Phase' else 'full')
        elif view == 'Spectrum':
            self._stack.setCurrentIndex(1)
        elif view == 'Spectrograph':
            self._stack.setCurrentIndex(2)


# ── Ventana principal ─────────────────────────────────────────────────

class MainWindow(QMainWindow):

    UPDATE_MS = 120

    def __init__(self):
        super().__init__()
        self.engine           = AudioEngine()
        self._frozen          = False
        self._delay_comp_ms   = 0.0    # retardo compensado (delay finder → fase)
        self._traces: List[TraceData] = []
        self._sp_traces: list        = []   # trazas spectrum independientes
        self._sp_engines: list       = []   # Spectrum Engines activos
        self._workspaces: list       = []        # list of workspace dicts
        self._current_ws_idx: int    = 0          # active workspace index
        self._ws_loading: bool       = False      # True while loading a workspace (suppress _save_prefs)
        self._trace_color_idx = 0           # cicla por TRACE_PALETTE
        self._mic_cal         = None   # tuple (freqs, dB_correction) o None
        self._mic_cal_name    = ''     # nombre del archivo cargado
        self._show_ch2        = True   # mostrar 2do canal en gráficas
        self._show_avg        = True   # mostrar promedio

        # Listas de botones replicados (uno por sub-panel)
        self._all_start_btns  = []
        self._all_stop_btns   = []
        self._all_freeze_btns = []
        self._all_noise_btns  = []

        self.setWindowTitle('Coherence  v0.2')
        self.setMinimumSize(1100, 720)
        self.resize(1380, 820)
        self.setStyleSheet(QSS)

        self._build_central()
        self._build_menubar()
        self._build_statusbar()
        self._populate_devices()

        # Carga engines guardados; si no hay prefs, crea uno por defecto
        self._load_prefs()

        self.timer = QTimer(self)
        self.timer.setInterval(self.UPDATE_MS)
        self.timer.timeout.connect(self._refresh)

        self._set_stopped()
        # Vista inicial: Spectrum (RTA) al abrir
        self._on_view_mode_changed('Spectrum')

    # ── Central widget ────────────────────────────────────────────────

    def _build_central(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Panel izquierdo: guardar TXT (colapsable) ──
        root.addWidget(self._build_collapsible_save_panel())

        # ── Centro: info bar + splitter de paneles ──
        center = QWidget()
        cv = QVBoxLayout(center)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(0)
        cv.addWidget(self._build_info_bar())

        # ── Canvases ──────────────────────────────────────────────────
        self.canvas_meas  = MeasurementCanvas()   # IR + Phase + TF
        self.canvas_meas.on_cursor_update = self._update_cursor
        self.canvas_spec  = SpectrumCanvas()       # RTA / Spectrum
        self.canvas_spec.on_cursor_update  = self._update_cursor
        # ⚙ botón del canvas → abre el panel de settings
        self.canvas_spec.on_config_clicked = self._on_toggle_settings

        # Botón de suavizado del canvas TF → sync con cmb_smooth del panel
        self.canvas_meas.on_smooth_changed = self._set_smooth_fraction
        self.canvas_sgram = SpectrogramCanvas()    # Spectrogram

        self._current_view = 'magnitude'   # 'magnitude'|'phase'|'spectrum'|'spectrograph'
        self._secondary_panel: 'SecondaryPanel | None' = None

        # Dropdown — mismas opciones en los 3 wrappers
        _OPTS = ['Magnitude', 'Phase', 'Spectrum', 'Spectrograph']

        self._wrap_meas  = CanvasOverlay(self.canvas_meas,  _OPTS, 'Magnitude',
                                         self._on_view_mode_changed,
                                         self._show_measurement_config,
                                         on_add=self._on_add_panel)
        self._wrap_spec  = CanvasOverlay(self.canvas_spec,  _OPTS, 'Spectrum',
                                         self._on_view_mode_changed,
                                         self._show_measurement_config,
                                         on_add=self._on_add_panel)
        self._wrap_sgram = CanvasOverlay(self.canvas_sgram, _OPTS, 'Spectrograph',
                                         self._on_view_mode_changed,
                                         self._show_measurement_config,
                                         on_add=self._on_add_panel)

        # Solo Magnitude visible al inicio
        self._wrap_spec.setVisible(False)
        self._wrap_sgram.setVisible(False)

        # Slot principal (3 wraps apilados, uno visible a la vez)
        self._main_area = QWidget()
        ma_lay = QVBoxLayout(self._main_area)
        ma_lay.setContentsMargins(0, 0, 0, 0)
        ma_lay.setSpacing(0)
        ma_lay.addWidget(self._wrap_meas,  stretch=1)
        ma_lay.addWidget(self._wrap_spec,  stretch=1)
        ma_lay.addWidget(self._wrap_sgram, stretch=1)

        # Splitter vertical — soporta 1 o 2 paneles
        self._panel_splitter = QSplitter(Qt.Orientation.Vertical)
        self._panel_splitter.setHandleWidth(4)
        self._panel_splitter.setStyleSheet(
            'QSplitter::handle{background:#1e2e1e;}'
            'QSplitter::handle:hover{background:#3a5a3a;}')
        self._panel_splitter.addWidget(self._main_area)

        # ── Workspace tab bar ─────────────────────────────────────────────
        _ws_row = QWidget()
        _ws_row.setFixedHeight(28)
        _ws_row.setStyleSheet('background:#0d0d0d;')
        _ws_lay = QHBoxLayout(_ws_row)
        _ws_lay.setContentsMargins(4, 2, 4, 0)
        _ws_lay.setSpacing(0)

        self._ws_tab_bar = QTabBar()
        self._ws_tab_bar.setExpanding(False)
        self._ws_tab_bar.setTabsClosable(True)
        self._ws_tab_bar.setMovable(False)
        self._ws_tab_bar.setStyleSheet(
            'QTabBar::tab{'
            '  background:#161616; color:#5a7a5a;'
            '  border:1px solid #2a2a2a; border-bottom:none;'
            '  padding:3px 10px; margin-right:2px; min-width:70px;'
            '  font-size:11px;'
            '}'
            'QTabBar::tab:selected{'
            '  background:#1a2a1a; color:#3cf53c; border-color:#3a4a3a;'
            '}'
            'QTabBar::tab:hover:!selected{background:#1a1a1a; color:#8a9a8a;}'
            'QTabBar{background:#0d0d0d;}'
            'QTabBar::close-button{image:none;}'
        )
        self._ws_tab_bar.tabCloseRequested.connect(self._ws_close_tab)
        self._ws_tab_bar.currentChanged.connect(self._ws_on_tab_changed)
        self._ws_tab_bar.mouseDoubleClickEvent = self._ws_rename_tab_evt

        _ws_add_btn = QPushButton('＋')
        _ws_add_btn.setFixedSize(24, 22)
        _ws_add_btn.setStyleSheet(
            'QPushButton{background:#161616;color:#5a7a5a;border:1px solid #2a2a2a;'
            'border-radius:2px;font-size:14px;padding:0;}'
            'QPushButton:hover{background:#1e2e1e;color:#3cf53c;}'
        )
        _ws_add_btn.clicked.connect(self._ws_add_new)

        _ws_lay.addWidget(self._ws_tab_bar, stretch=1)
        _ws_lay.addWidget(_ws_add_btn)

        cv.addWidget(_ws_row)
        cv.addWidget(self._panel_splitter, stretch=1)
        root.addWidget(center, stretch=1)

        # ── Derecha: panel settings ──
        root.addWidget(self._build_collapsible_settings())

        # Compat tabs
        class _FakeTabs:
            def currentIndex(self): return 0
            def setCurrentIndex(self, i): pass
        self.tabs = _FakeTabs()

        # IR toggle — muestra/oculta la fila IR dentro de canvas_meas
        self._ir_visible = True   # IR siempre visible dentro del canvas_meas

    def _make_panel_wrapper(self, canvas, title, current_mode,
                            menu_options=None, on_option=None):
        """
        Panel estilo SMAART con header que tiene dropdown RTA▼ / Magnitude▼ etc.

        canvas        : QWidget a mostrar debajo del header (None = solo header)
        title         : label fijo a la derecha
        current_mode  : texto inicial del botón dropdown
        menu_options  : lista de opciones del dropdown
        on_option     : callback(option_str) cuando se elige una opción
        """
        from PyQt6.QtWidgets import QMenu
        from PyQt6.QtGui import QAction

        w = QWidget()
        w.setStyleSheet(f'background:{BG_PANEL};')
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        # ── Header bar ───────────────────────────────────────────────
        hdr = QWidget()
        hdr.setFixedHeight(24)
        hdr.setStyleSheet('background:#0a0a0a;border-bottom:1px solid #1e1e1e;')
        hh = QHBoxLayout(hdr)
        hh.setContentsMargins(6, 0, 6, 0); hh.setSpacing(4)

        # [X] close button (pequeño)
        btn_close = QPushButton('✕')
        btn_close.setFixedSize(14, 14)
        btn_close.setStyleSheet(
            'QPushButton{font-size:8px;color:#333;background:transparent;'
            'border:none;}QPushButton:hover{color:#888;}')
        btn_close.clicked.connect(w.hide)
        hh.addWidget(btn_close)

        # Dropdown botón estilo "RTA▼"
        if menu_options:
            btn_mode = QPushButton(f'{current_mode} ▾')
            btn_mode.setStyleSheet(
                f'QPushButton{{font-size:10px;font-weight:bold;color:{TEXT_HI};'
                f'background:transparent;border:none;padding:0 4px;}}'
                f'QPushButton:hover{{color:{ACCENT};}}')
            btn_mode.setFixedHeight(22)

            def _show_menu(_checked=False, options=menu_options, btn=btn_mode, cb=on_option):
                menu = QMenu(btn)
                menu.setStyleSheet(
                    f'QMenu{{background:#2a2a2a;color:{TEXT_HI};border:1px solid #3a3a3a;'
                    f'font-size:11px;}}'
                    f'QMenu::item{{padding:5px 20px;}}'
                    f'QMenu::item:selected{{background:#1e3a4a;}}'
                    f'QMenu::item:checked{{color:{ACCENT};}}')
                for opt in options:
                    act = QAction(opt, menu)
                    act.setCheckable(True)
                    act.setChecked(btn.text().startswith(opt))
                    def _triggered(checked, o=opt, b=btn, c=cb):
                        b.setText(f'{o} ▾')
                        if c: c(o)
                    act.triggered.connect(_triggered)
                    menu.addAction(act)
                menu.exec(btn.mapToGlobal(btn.rect().bottomLeft()))

            btn_mode.clicked.connect(_show_menu)
            hh.addWidget(btn_mode)

        hh.addStretch()

        # Label de measurement name (derecha, estilo SMAART)
        lbl_name = QLabel(title.upper())
        lbl_name.setStyleSheet(
            f'color:{TEXT_DIM};font-size:8px;letter-spacing:1px;'
            f'background:transparent;')
        hh.addWidget(lbl_name)

        # Engrane ⚙ → abre Measurement Config
        btn_gear = QPushButton('⚙')
        btn_gear.setFixedSize(18, 18)
        btn_gear.setStyleSheet(
            f'QPushButton{{font-size:11px;color:#444;background:transparent;border:none;}}'
            f'QPushButton:hover{{color:{TEXT_MID};}}')
        btn_gear.clicked.connect(lambda: self._show_measurement_config())
        hh.addWidget(btn_gear)

        v.addWidget(hdr)
        if canvas is not None:
            v.addWidget(canvas, stretch=1)
        return w

    def _build_info_bar(self):
        bar = QWidget()
        bar.setObjectName('info_bar')
        bar.setFixedHeight(38)
        h = QHBoxLayout(bar)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)

        self.lbl_app_title = QLabel('  COHERENCE  v0.2  ')
        self.lbl_app_title.setObjectName('lbl_app_title')
        h.addWidget(self.lbl_app_title)

        sep1 = QFrame(); sep1.setFrameShape(QFrame.Shape.VLine)
        sep1.setStyleSheet(f'color:{BORDER};')
        h.addWidget(sep1)

        self.lbl_cursor_info = QLabel('  —')
        self.lbl_cursor_info.setObjectName('lbl_cursor_info')
        self.lbl_cursor_info.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        h.addWidget(self.lbl_cursor_info, stretch=1)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.Shape.VLine)
        sep2.setStyleSheet(f'color:{BORDER};')
        h.addWidget(sep2)

        self.lbl_clock = QLabel('  00:00:00  ')
        self.lbl_clock.setObjectName('lbl_clock')
        h.addWidget(self.lbl_clock)
        return bar

    def _build_save_panel(self):
        """Panel SAVE izquierdo — vista TF / IR (índice 0 del _save_stack)."""
        panel = QWidget()
        panel.setObjectName('save_panel')
        v = QVBoxLayout(panel)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(3)

        # ── Header: título + botón lupa ───────────────────────────────
        hdr = QHBoxLayout(); hdr.setSpacing(4)
        title = QLabel('TF / IR')
        title.setStyleSheet(
            f'color:{ACCENT};font-size:9px;letter-spacing:2px;font-weight:bold;'
            f'padding:2px 0;border-bottom:1px solid {BORDER};background:transparent;')
        hdr.addWidget(title, stretch=1)
        btn_lupa_tf = QPushButton('🔍')
        btn_lupa_tf.setFixedSize(22, 20)
        btn_lupa_tf.setToolTip('Buscar traza…')
        btn_lupa_tf.setStyleSheet(
            'QPushButton{font-size:11px;padding:0;border:none;background:transparent;color:#555;}'
            'QPushButton:hover{color:#aaa;}')
        hdr.addWidget(btn_lupa_tf)
        v.addLayout(hdr)

        # Barra de búsqueda (oculta por defecto)
        self._tf_search = QLineEdit()
        self._tf_search.setPlaceholderText('Buscar traza...')
        self._tf_search.setFixedHeight(20)
        self._tf_search.setStyleSheet(
            f'QLineEdit{{background:#161616;border:1px solid {BORDER};border-radius:3px;'
            f'color:{TEXT_MID};font-size:9px;padding:0 4px;}}')
        self._tf_search.setVisible(False)
        self._tf_search.textChanged.connect(self._filter_tf_traces)
        v.addWidget(self._tf_search)
        btn_lupa_tf.clicked.connect(lambda: (
            self._tf_search.setVisible(not self._tf_search.isVisible()),
            self._tf_search.setFocus() if self._tf_search.isVisible() else None
        ))

        v.addWidget(sep())

        for label, tip, fn in [
            ('SAVE  IR',    'Guardar Impulse Response como .txt',  self._save_ir_txt),
            ('SAVE  TF',    'Guardar Transfer Function como .txt', self._save_tf_txt),
            ('SAVE  PHASE', 'Guardar Phase como .txt',             self._save_ph_txt),
        ]:
            btn = QPushButton(label)
            btn.setToolTip(tip)
            btn.setMinimumHeight(28)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn.clicked.connect(fn)
            v.addWidget(btn)

        # ── Sección TRAZAS TF ─────────────────────────────────────────
        v.addWidget(sep())

        tt = QLabel('TRAZAS  TF')
        tt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tt.setStyleSheet(
            f'color:{ACCENT};font-size:9px;letter-spacing:2px;font-weight:bold;'
            f'padding:2px 0;background:transparent;')
        v.addWidget(tt)

        btn_save_trace = QPushButton('📌  GUARDAR TRAZA')
        btn_save_trace.setToolTip(
            f'Guarda la medición actual como traza de referencia (máx {MAX_TRACES})')
        btn_save_trace.setMinimumHeight(28)
        btn_save_trace.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        btn_save_trace.clicked.connect(self._save_trace)
        v.addWidget(btn_save_trace)

        # Lista de filas de trazas
        self._trace_rows_widget = QWidget()
        self._trace_rows_widget.setStyleSheet('background:transparent;')
        self._trace_rows_layout = QVBoxLayout(self._trace_rows_widget)
        self._trace_rows_layout.setContentsMargins(0, 2, 0, 2)
        self._trace_rows_layout.setSpacing(1)
        v.addWidget(self._trace_rows_widget)

        v.addStretch()
        return panel

    def _build_collapsible_save_panel(self):
        """Panel izquierdo colapsable — stack 0=TF, 1=Spectrum."""
        from PyQt6.QtWidgets import QStackedWidget
        container = QWidget()
        h = QHBoxLayout(container)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)

        self._save_stack = QStackedWidget()
        self._save_stack.setFixedWidth(155)
        self._save_stack.addWidget(self._build_save_panel())       # 0 — TF / IR
        self._save_stack.addWidget(self._build_save_spec_panel())  # 1 — Spectrum
        h.addWidget(self._save_stack)

        self.btn_toggle_save = QPushButton('▶')
        self.btn_toggle_save.setObjectName('btn_toggle_panel')
        self.btn_toggle_save.setFixedWidth(14)
        self.btn_toggle_save.clicked.connect(self._on_toggle_save)
        h.addWidget(self.btn_toggle_save)
        return container

    def _on_toggle_save(self):
        visible = self._save_stack.isVisible()
        self._save_stack.setVisible(not visible)
        self.btn_toggle_save.setText('◀' if visible else '▶')

    def _build_save_spec_panel(self):
        """Panel SAVE izquierdo — vista Spectrum (índice 1 del _save_stack)."""
        panel = QWidget()
        panel.setObjectName('save_panel')
        v = QVBoxLayout(panel)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(3)

        # ── Header: título + botón lupa ───────────────────────────────
        hdr = QHBoxLayout(); hdr.setSpacing(4)
        title = QLabel('SPECTRUM')
        title.setStyleSheet(
            f'color:{GREEN};font-size:9px;letter-spacing:2px;font-weight:bold;'
            f'padding:2px 0;border-bottom:1px solid {BORDER};background:transparent;')
        hdr.addWidget(title, stretch=1)
        btn_lupa_sp = QPushButton('🔍')
        btn_lupa_sp.setFixedSize(22, 20)
        btn_lupa_sp.setToolTip('Buscar traza…')
        btn_lupa_sp.setStyleSheet(
            'QPushButton{font-size:11px;padding:0;border:none;background:transparent;color:#555;}'
            'QPushButton:hover{color:#aaa;}')
        hdr.addWidget(btn_lupa_sp)
        v.addLayout(hdr)

        # Barra de búsqueda (oculta por defecto)
        self._sp_search = QLineEdit()
        self._sp_search.setPlaceholderText('Buscar traza...')
        self._sp_search.setFixedHeight(20)
        self._sp_search.setStyleSheet(
            f'QLineEdit{{background:#161616;border:1px solid {BORDER};border-radius:3px;'
            f'color:{TEXT_MID};font-size:9px;padding:0 4px;}}')
        self._sp_search.setVisible(False)
        self._sp_search.textChanged.connect(self._filter_sp_traces)
        v.addWidget(self._sp_search)
        btn_lupa_sp.clicked.connect(lambda: (
            self._sp_search.setVisible(not self._sp_search.isVisible()),
            self._sp_search.setFocus() if self._sp_search.isVisible() else None
        ))

        v.addWidget(sep())

        for label, tip, fn in [
            ('SAVE SPECTRUM', 'Guardar Spectrum como .txt', self._save_sp_txt),
            ('SAVE  PNG',     'Guardar gráfica como PNG',  self._save_graph_png),
        ]:
            btn = QPushButton(label)
            btn.setToolTip(tip)
            btn.setMinimumHeight(28)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn.clicked.connect(fn)
            v.addWidget(btn)

        # ── Sección TRAZAS Spectrum ───────────────────────────────────
        v.addWidget(sep())

        tt = QLabel('TRAZAS  SP')
        tt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tt.setStyleSheet(
            f'color:{GREEN};font-size:9px;letter-spacing:2px;font-weight:bold;'
            f'padding:2px 0;background:transparent;')
        v.addWidget(tt)

        btn_save_sp_trace = QPushButton('📌  GUARDAR TRAZA')
        btn_save_sp_trace.setToolTip('Guarda el spectrum actual como referencia')
        btn_save_sp_trace.setMinimumHeight(28)
        btn_save_sp_trace.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        btn_save_sp_trace.clicked.connect(self._save_spectrum_trace)
        v.addWidget(btn_save_sp_trace)

        self._sp_trace_rows_widget = QWidget()
        self._sp_trace_rows_widget.setStyleSheet('background:transparent;')
        self._sp_trace_rows_layout = QVBoxLayout(self._sp_trace_rows_widget)
        self._sp_trace_rows_layout.setContentsMargins(0, 2, 0, 2)
        self._sp_trace_rows_layout.setSpacing(1)
        v.addWidget(self._sp_trace_rows_widget)

        v.addStretch()
        return panel

    def _build_collapsible_settings(self):
        """Envuelve el panel de settings con un botón de colapso."""
        container = QWidget()
        h = QHBoxLayout(container)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)

        self.btn_toggle_settings = QPushButton('◀')
        self.btn_toggle_settings.setObjectName('btn_toggle_panel')
        self.btn_toggle_settings.setFixedWidth(14)
        self.btn_toggle_settings.clicked.connect(self._on_toggle_settings)
        h.addWidget(self.btn_toggle_settings)

        self._settings_widget = self._build_settings()
        h.addWidget(self._settings_widget)
        return container

    def _on_toggle_settings(self):
        visible = self._settings_widget.isVisible()
        self._settings_widget.setVisible(not visible)
        self.btn_toggle_settings.setText('▶' if visible else '◀')

    def _update_cursor(self, info: str):
        self.lbl_cursor_info.setText(f'  {info}')

    def _build_settings(self):
        """
        Panel derecho — estructura fija:
          ┌─────────────────────────┐
          │  SPL METER  (fijo)      │  ← siempre visible, mismo tamaño
          ├─────────────────────────┤
          │  StackedWidget (scroll) │  ← cambia según vista (TF / Spectrum / Sgram)
          ├─────────────────────────┤
          │  SIGNAL GENERATOR (fijo)│  ← siempre visible, mismo tamaño
          └─────────────────────────┘
        """
        from PyQt6.QtWidgets import QStackedWidget

        outer = QWidget()
        outer.setFixedWidth(240)
        outer.setStyleSheet(
            f'background:{BG_SETTINGS};border-left:1px solid {BORDER};')
        ov = QVBoxLayout(outer)
        ov.setContentsMargins(0, 0, 0, 0)
        ov.setSpacing(0)

        # ── SPL fijo arriba ───────────────────────────────────────────
        spl_container = QWidget()
        spl_container.setStyleSheet(f'background:{BG_SETTINGS};')
        spl_lay = QVBoxLayout(spl_container)
        spl_lay.setContentsMargins(6, 6, 6, 4)
        spl_lay.setSpacing(4)

        # Inicializar listas antes del primer _make_spl_block
        if not hasattr(self, '_spl_labels'): self._spl_labels = []
        if not hasattr(self, '_spl_bars'):   self._spl_bars   = []
        if not hasattr(self, '_spl_max_db'): self._spl_max_db = -200.0
        if not hasattr(self, '_spl_warn_db'):   self._spl_warn_db   = 100.0
        if not hasattr(self, '_spl_clip_db'):   self._spl_clip_db   = 118.0
        if not hasattr(self, '_spl_offset_db'): self._spl_offset_db =   0.0

        self._make_spl_block(spl_lay)
        ov.addWidget(spl_container)

        # Separador
        _s = QFrame(); _s.setFrameShape(QFrame.Shape.HLine)
        _s.setStyleSheet(f'color:{BORDER};')
        ov.addWidget(_s)

        # ── StackedWidget con scroll (contenido variable) ─────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            f'QScrollArea{{background:{BG_SETTINGS};border:none;}}'
            'QScrollBar:vertical{width:5px;background:transparent;}'
            f'QScrollBar::handle:vertical{{background:{BORDER};border-radius:2px;}}')

        self._settings_stack = QStackedWidget()
        self._settings_stack.setStyleSheet(f'background:{BG_SETTINGS};')
        scroll.setWidget(self._settings_stack)

        self._settings_stack.addWidget(self._build_settings_tf())        # 0
        self._settings_stack.addWidget(self._build_settings_spectrum())  # 1
        self._settings_stack.addWidget(self._build_settings_sgram())     # 2

        ov.addWidget(scroll, stretch=1)

        # Separador
        _s2 = QFrame(); _s2.setFrameShape(QFrame.Shape.HLine)
        _s2.setStyleSheet(f'color:{BORDER};')
        ov.addWidget(_s2)

        # ── Signal Generator fijo abajo ───────────────────────────────
        sg_container = QWidget()
        sg_container.setStyleSheet(f'background:{BG_SETTINGS};')
        sg_lay = QVBoxLayout(sg_container)
        sg_lay.setContentsMargins(6, 4, 6, 6)
        sg_lay.setSpacing(4)
        self._build_signal_generator_block(sg_lay)
        ov.addWidget(sg_container)

        return outer

    # ── Signal Generator block (compartido, aparece siempre abajo) ───

    def _build_signal_generator_block(self, layout):
        """Construye el bloque Signal Generator y lo agrega a layout."""
        sg_frame = QFrame()
        sg_frame.setStyleSheet(
            f'QFrame{{background:#161616;border:1px solid #2a2a2a;border-radius:5px;}}')
        sg_v = QVBoxLayout(sg_frame)
        sg_v.setContentsMargins(8, 7, 8, 7); sg_v.setSpacing(6)

        # Header
        sg_hdr = QLabel('Signal Generator')
        sg_hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sg_hdr.setStyleSheet(
            f'color:{TEXT_HI};font-size:12px;font-weight:bold;background:transparent;'
            f'border-bottom:1px solid #252525;padding-bottom:5px;')
        sg_v.addWidget(sg_hdr)

        # Fila 1: Tipo | dB | ON
        r1 = QHBoxLayout(); r1.setSpacing(4)

        if not hasattr(self, 'cmb_signal'):
            self.cmb_signal = QComboBox()
            for s in ['Noise', 'Pink', 'White', 'Tone', 'Sweep']:
                self.cmb_signal.addItem(s)
            self.cmb_signal.currentIndexChanged.connect(self._on_signal_type)
        self.cmb_signal.setFixedHeight(26)
        self.cmb_signal.setStyleSheet(
            f'font-size:11px;color:{TEXT_HI};background:#222;'
            f'border:1px solid #333;border-radius:4px;padding:0 4px;')
        r1.addWidget(self.cmb_signal, stretch=2)

        if not hasattr(self, 'lbl_gain_p'):
            self.lbl_gain_p = QLabel('-19 dB')
        self.lbl_gain_p.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_gain_p.setStyleSheet(
            f'color:{TEXT_HI};font-size:11px;background:#222;'
            f'border:1px solid #333;border-radius:4px;padding:0 6px;')
        self.lbl_gain_p.setFixedHeight(26)
        r1.addWidget(self.lbl_gain_p, stretch=1)

        if not hasattr(self, 'btn_noise_p'):
            self.btn_noise_p = QPushButton('On')
            self.btn_noise_p.setCheckable(True)
            self.btn_noise_p.setChecked(False)
            self.btn_noise_p.setObjectName('btn_noise')
            self.btn_noise_p.clicked.connect(self._on_noise_panel)
            self._all_noise_btns.append(self.btn_noise_p)
        self.btn_noise_p.setFixedHeight(26)
        self.btn_noise_p.setStyleSheet(
            f'QPushButton{{font-size:11px;font-weight:bold;border-radius:4px;'
            f'border:1px solid #333;background:#1e2e1e;color:#66aa66;}}'
            f'QPushButton:checked{{background:#2a3a2a;color:#88cc88;}}'
            f'QPushButton:!checked{{background:#222;color:{TEXT_DIM};}}')
        r1.addWidget(self.btn_noise_p, stretch=1)
        sg_v.addLayout(r1)

        # Fila 2: Output | - | +
        r2 = QHBoxLayout(); r2.setSpacing(4)

        if not hasattr(self, 'cmb_noise_dev'):
            self.cmb_noise_dev = QComboBox()
        self.cmb_noise_dev.setFixedHeight(26)
        self.cmb_noise_dev.setStyleSheet(
            f'font-size:11px;color:{TEXT_HI};background:#222;'
            f'border:1px solid #333;border-radius:4px;padding:0 4px;')
        r2.addWidget(self.cmb_noise_dev, stretch=2)

        _pm = (f'QPushButton{{font-size:16px;font-weight:bold;padding:0;'
               f'border:1px solid #333;background:#222;color:{TEXT_MID};border-radius:4px;}}'
               f'QPushButton:hover{{background:#2a2a2a;color:{TEXT_HI};}}')
        btn_dn = QPushButton('−'); btn_dn.setFixedHeight(26); btn_dn.setStyleSheet(_pm)
        btn_up = QPushButton('+'); btn_up.setFixedHeight(26); btn_up.setStyleSheet(_pm)

        if not hasattr(self, 'sld_gain_p'):
            self.sld_gain_p = QSlider(Qt.Orientation.Horizontal)
            self.sld_gain_p.setRange(0, 100); self.sld_gain_p.setValue(30)
            self.sld_gain_p.hide()
            self.sld_gain_p.valueChanged.connect(self._on_gain_panel)

        def _step(d):
            self.sld_gain_p.setValue(max(0, min(100, self.sld_gain_p.value() + d)))
        btn_dn.clicked.connect(lambda: _step(-5))
        btn_up.clicked.connect(lambda: _step(+5))
        r2.addWidget(btn_dn, stretch=1)
        r2.addWidget(btn_up, stretch=1)
        sg_v.addLayout(r2)

        layout.addWidget(sg_frame)

    # ── Helper compartido ─────────────────────────────────────────────

    def _make_transport(self, parent_layout, freeze=True, gen=True):
        """
        Crea un bloque TRANSPORT (START/STOP + opcionales FREEZE/GEN).
        Devuelve (btn_start, btn_stop, btn_freeze, btn_noise) — algunos pueden ser None.
        Cada llamada crea botones nuevos; todos conectan a los mismos slots.
        """
        grp = QGroupBox('TRANSPORT')
        gt = QGridLayout(grp)
        gt.setContentsMargins(5, 6, 5, 5)
        gt.setSpacing(3)

        btn_start = QPushButton('▶')
        btn_start.setObjectName('btn_start')
        btn_start.setToolTip('Start  [Space]')
        btn_start.clicked.connect(self._on_start)
        gt.addWidget(btn_start, 0, 0)

        btn_stop = QPushButton('⏹')
        btn_stop.setObjectName('btn_stop')
        btn_stop.setToolTip('Stop')
        btn_stop.clicked.connect(self._on_stop)
        gt.addWidget(btn_stop, 0, 1)

        btn_freeze = None
        if freeze:
            btn_freeze = QPushButton('❄')
            btn_freeze.setCheckable(True)
            btn_freeze.setToolTip('Freeze  [F]')
            btn_freeze.clicked.connect(self._on_freeze_panel)
            gt.addWidget(btn_freeze, 1, 0)

        btn_noise = None
        if gen:
            btn_noise = QPushButton('◉ GEN')
            btn_noise.setObjectName('btn_noise')
            btn_noise.setCheckable(True)
            btn_noise.setChecked(True)
            btn_noise.clicked.connect(self._on_noise_panel)
            gt.addWidget(btn_noise, 1, 1)

        parent_layout.addWidget(grp)

        # Registrar en listas para sincronizar estado
        self._all_start_btns.append(btn_start)
        self._all_stop_btns.append(btn_stop)
        if btn_freeze:
            self._all_freeze_btns.append(btn_freeze)
        if btn_noise:
            self._all_noise_btns.append(btn_noise)

        return btn_start, btn_stop, btn_freeze, btn_noise

    # ── Sub-panel TF ──────────────────────────────────────────────────

    def _make_spl_block(self, layout):
        """
        Construye el bloque SPL meter completo (una sola instancia, fija arriba).
        """
        meter_frame = QFrame()
        meter_frame.setStyleSheet(
            f'QFrame{{background:#111111;border:1px solid #2a2a2a;border-radius:4px;}}')
        mf = QVBoxLayout(meter_frame)
        mf.setContentsMargins(8, 6, 8, 6); mf.setSpacing(3)

        # Fila superior: canal + weighting
        top_row = QHBoxLayout(); top_row.setSpacing(4)
        top_row.addStretch()

        if not hasattr(self, 'spn_spl_ch'):
            self.spn_spl_ch = QSpinBox()
            self.spn_spl_ch.setRange(1, 32); self.spn_spl_ch.setValue(1)
            self.spn_spl_ch.setFixedHeight(18); self.spn_spl_ch.setFixedWidth(42)
            self.spn_spl_ch.setStyleSheet(
                f'font-size:10px;color:{TEXT_MID};background:#1e1e1e;'
                f'border:1px solid #2a2a2a;border-radius:3px;')
            self.spn_spl_ch.valueChanged.connect(
                lambda v: setattr(self.engine, 'ch_spl', v))
        top_row.addWidget(QLabel('CH'))
        top_row.addWidget(self.spn_spl_ch)

        if not hasattr(self, 'cmb_spl_weighting'):
            self.cmb_spl_weighting = QComboBox()
            self.cmb_spl_weighting.addItems(['dB SPL Slow', 'dB SPL Fast', 'dB SPL Imp', 'dBFS'])
            self.cmb_spl_weighting.setFixedHeight(18)
            self.cmb_spl_weighting.setStyleSheet(
                f'font-size:9px;color:{TEXT_MID};background:#1e1e1e;border:none;border-radius:3px;')
        top_row.addWidget(self.cmb_spl_weighting)
        mf.addLayout(top_row)

        # Número grande + Max
        mid_row = QHBoxLayout(); mid_row.setSpacing(6)
        if not hasattr(self, 'lbl_spl_val'):
            self.lbl_spl_val = QLabel('0.0')
        self.lbl_spl_val.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.lbl_spl_val.setStyleSheet(
            'color:#66bb6a;font-size:42px;font-weight:bold;'
            'background:transparent;letter-spacing:-1px;')
        mid_row.addWidget(self.lbl_spl_val, stretch=1)

        right_col = QVBoxLayout(); right_col.setSpacing(2); right_col.addStretch()
        if not hasattr(self, 'lbl_spl_max'):
            self.lbl_spl_max = QLabel('Max  0.0')
        self.lbl_spl_max.setStyleSheet(
            f'color:{TEXT_MID};font-size:9px;background:transparent;')

        max_row = QHBoxLayout(); max_row.setSpacing(4)
        max_row.addWidget(self.lbl_spl_max, stretch=1)
        btn_rst = QPushButton('●')
        btn_rst.setFixedSize(14, 14)
        btn_rst.setStyleSheet(
            'QPushButton{font-size:8px;padding:0;border:none;'
            'background:#2a2a2a;color:#555;border-radius:7px;}'
            'QPushButton:hover{color:#888;}')
        btn_rst.clicked.connect(self._on_spl_max_reset)
        max_row.addWidget(btn_rst)
        right_col.addLayout(max_row)
        mid_row.addLayout(right_col)
        mf.addLayout(mid_row)

        # Barra
        if not hasattr(self, 'bar_spl'):
            self.bar_spl = QProgressBar()
            self.bar_spl.setRange(0, 100); self.bar_spl.setValue(0)
            self.bar_spl.setTextVisible(False)
        self.bar_spl.setFixedHeight(4)
        self.bar_spl.setStyleSheet(
            'QProgressBar{background:#222;border:none;border-radius:2px;}'
            'QProgressBar::chunk{background:#66bb6a;border-radius:2px;}')
        mf.addWidget(self.bar_spl)
        layout.addWidget(meter_frame)

        # Registrar para _refresh()
        if not hasattr(self, '_spl_labels'): self._spl_labels = []
        if not hasattr(self, '_spl_bars'):   self._spl_bars   = []
        if self.lbl_spl_val not in self._spl_labels:
            self._spl_labels.append(self.lbl_spl_val)
        if self.bar_spl not in self._spl_bars:
            self._spl_bars.append(self.bar_spl)
        if not hasattr(self, '_spl_max_db'): self._spl_max_db = -200.0

    def _build_settings_tf(self):
        # SPL meter thresholds (dB SPL after calibration offset)
        self._spl_warn_db   = 100.0
        self._spl_clip_db   = 118.0
        self._spl_offset_db =   0.0   # dBFS → dB SPL calibration offset

        w = QWidget()
        w.setStyleSheet(f'background:{BG_SETTINGS};')
        layout = QVBoxLayout(w)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        def rl(t): return lbl(t, color=TEXT_MID, size=9)

        # ── TRANSFER FUNCTION header ──────────────────────────────────
        tf_hdr = QLabel('Transfer Function')
        tf_hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tf_hdr.setStyleSheet(
            f'color:{TEXT_MID};font-size:11px;font-weight:bold;'
            f'background:#252525;border-radius:3px;padding:4px 0;'
            f'margin:2px 0;')
        layout.addWidget(tf_hdr)

        # Averaging Depth
        avg_row = QHBoxLayout(); avg_row.setSpacing(6)
        avg_row.addWidget(lbl('Averaging Depth:', color=TEXT_MID, size=10))
        self.cmb_avg = QComboBox()
        self._avg_values = [1, 2, 4, 6, 8, 12, 16, 32]
        for v in self._avg_values: self.cmb_avg.addItem(str(v))
        self.cmb_avg.setCurrentIndex(3)   # 6 por defecto
        self.cmb_avg.setFixedWidth(52)
        self.cmb_avg.currentIndexChanged.connect(
            lambda i: setattr(self.engine, 'n_averages', self._avg_values[i]))
        avg_row.addWidget(self.cmb_avg)
        layout.addLayout(avg_row)

        layout.addWidget(sep())

        # ── TRANSPORT row ─────────────────────────────────────────────
        tr_row = QHBoxLayout(); tr_row.setSpacing(3)
        _tr_style = (f'QPushButton{{font-size:14px;padding:4px 0;border-radius:4px;'
                     f'border:1px solid #333;background:#252525;color:{TEXT_MID};}}'
                     f'QPushButton:hover{{background:#303030;color:{TEXT_HI};}}')
        btn_tr_play  = QPushButton('▶')
        btn_tr_stop  = QPushButton('■')
        btn_tr_pause = QPushButton('⏸')
        for b in (btn_tr_play, btn_tr_stop, btn_tr_pause):
            b.setStyleSheet(_tr_style)
            tr_row.addWidget(b, stretch=1)
        btn_tr_play.clicked.connect(self._on_start)
        btn_tr_stop.clicked.connect(self._on_stop)
        btn_tr_pause.setCheckable(True)
        btn_tr_pause.clicked.connect(self._on_freeze_panel)
        self._all_start_btns.append(btn_tr_play)
        self._all_stop_btns.append(btn_tr_stop)
        self._all_freeze_btns.append(btn_tr_pause)
        self.btn_freeze_p = btn_tr_pause   # alias para shortcut F
        layout.addLayout(tr_row)

        # Segunda fila: Track / timer / config
        tr_row2 = QHBoxLayout(); tr_row2.setSpacing(3)
        _tr2_style = (f'QPushButton{{font-size:10px;padding:3px 0;border-radius:4px;'
                      f'border:1px solid #2a2a2a;background:#1e1e1e;color:{TEXT_DIM};}}'
                      f'QPushButton:hover{{color:{TEXT_MID};}}')
        btn_track = QPushButton('Track')
        btn_timer = QPushButton('⏱')
        btn_cfg   = QPushButton('⚙')
        for b in (btn_track, btn_timer, btn_cfg):
            b.setStyleSheet(_tr2_style)
            tr_row2.addWidget(b, stretch=1)
        btn_cfg.clicked.connect(self._show_measurement_config)
        layout.addLayout(tr_row2)

        layout.addWidget(sep())

        # ── TF ENGINES ───────────────────────────────────────────────

        engines_container = QWidget()
        engines_container.setStyleSheet(f'background:{BG_SETTINGS};')
        self._engines_layout = QVBoxLayout(engines_container)
        self._engines_layout.setContentsMargins(0, 0, 0, 0)
        self._engines_layout.setSpacing(3)
        layout.addWidget(engines_container)

        # Inicializar lista de engines y RunningTF por engine (vacía al inicio)
        self._tf_engines: List[TFEngine] = []
        self._running_tfs: List[RunningTF] = []
        self._selected_engine_idx: int = 0

        # Aliases de compatibilidad — se actualizan en _add_tf_engine()
        self.spn_ch_meas  = QSpinBox(); self.spn_ch_meas.hide()
        self.spn_ch_ref   = QSpinBox(); self.spn_ch_ref.hide()
        self.spn_ch_meas2 = QSpinBox(); self.spn_ch_meas2.hide()
        self.bar_spl_meas = QProgressBar(); self.bar_spl_meas.hide()
        self.bar_spl_ref  = QProgressBar(); self.bar_spl_ref.hide()
        self.lbl_spl_meas = QLabel(); self.lbl_spl_meas.hide()
        self.lbl_spl_ref  = QLabel(); self.lbl_spl_ref.hide()

        # Botón + TF Engine
        btn_add = QPushButton('+ TF Engine')
        btn_add.setStyleSheet(
            f'QPushButton{{font-size:11px;color:{TEXT_MID};'
            f'background:#1e1e1e;border:1px solid {BORDER};'
            f'border-radius:5px;padding:6px 0;}}'
            f'QPushButton:hover{{background:#252525;color:{TEXT_HI};'
            f'border-color:#3a3a3a;}}')
        btn_add.clicked.connect(self._show_new_tf_dialog)
        layout.addWidget(btn_add)

        # Botón AVG — promedio de todos los engines activos
        _avg_on_style  = (f'QPushButton{{font-size:10px;color:#e0e0e0;font-weight:bold;'
                          f'background:#1e2a1e;border:1px solid #3cf53c;'
                          f'border-radius:4px;padding:4px 0;letter-spacing:1px;}}'
                          f'QPushButton:hover{{background:#253325;}}')
        _avg_off_style = (f'QPushButton{{font-size:10px;color:{TEXT_DIM};'
                          f'background:#1a1a1a;border:1px solid #333;'
                          f'border-radius:4px;padding:4px 0;letter-spacing:1px;}}'
                          f'QPushButton:hover{{background:#222;color:{TEXT_MID};}}')
        self.btn_show_avg = QPushButton('⊘  AVG')
        self.btn_show_avg.setCheckable(True)
        self.btn_show_avg.setChecked(False)
        self._show_avg = False
        self.btn_show_avg.setStyleSheet(_avg_off_style)
        def _toggle_avg(checked):
            self._show_avg = checked
            self.btn_show_avg.setText('◎  AVG' if checked else '⊘  AVG')
            self.btn_show_avg.setStyleSheet(_avg_on_style if checked else _avg_off_style)
            if not checked:
                self.canvas_meas.update_avg(None, None, None, None)
        self.btn_show_avg.clicked.connect(_toggle_avg)
        layout.addWidget(self.btn_show_avg)

        # Config hint — configuración de canales en el dialog
        btn_tf_cfg = QPushButton('⊞  TF ENGINE CONFIG…')
        btn_tf_cfg.setStyleSheet(
            f'font-size:8px;color:{TEXT_DIM};background:transparent;'
            f'border:none;text-align:left;padding:1px 2px;')
        btn_tf_cfg.clicked.connect(self._show_tf_engine_config)
        layout.addWidget(btn_tf_cfg)

        layout.addWidget(sep())

        # ── SMOOTH ────────────────────────────────────────────────────────
        sm_row = QHBoxLayout(); sm_row.setSpacing(6)
        sm_row.addWidget(lbl('Smooth:', color=TEXT_MID, size=10))
        self.cmb_smooth = QComboBox()
        for _s in ['OFF', '1/3', '1/6', '1/12', '1/24']:
            self.cmb_smooth.addItem(_s)
        self.cmb_smooth.setCurrentIndex(3)   # 1/12 oct por defecto
        self._smooth_values = [0, 3, 6, 12, 24]
        sm_row.addWidget(self.cmb_smooth, stretch=1)
        layout.addLayout(sm_row)

        # ── γ² DISPLAY + THRESHOLD ────────────────────────────────────────
        g2_row = QHBoxLayout(); g2_row.setSpacing(4)
        g2_row.addWidget(lbl('γ²:', color=TEXT_MID, size=10))
        self.lbl_gamma2 = QLabel('—')
        self.lbl_gamma2.setStyleSheet(
            'color:#66bb6a;font-size:15px;font-weight:bold;background:transparent;')
        g2_row.addWidget(self.lbl_gamma2, stretch=1)
        g2_row.addWidget(lbl('thr:', color=TEXT_DIM, size=9))
        self.spn_thresh = QDoubleSpinBox()
        self.spn_thresh.setRange(0.0, 1.0); self.spn_thresh.setValue(0.10)
        self.spn_thresh.setDecimals(2); self.spn_thresh.setSingleStep(0.05)
        self.spn_thresh.setFixedWidth(54)
        g2_row.addWidget(self.spn_thresh)
        layout.addLayout(g2_row)
        self.lbl_coh_full    = self.lbl_gamma2
        self.lbl_gamma2_mark = QLabel()
        self.bar_coh         = QProgressBar()

        # ── VIEW / TIMER ──────────────────────────────────────────────────
        vt_row = QHBoxLayout(); vt_row.setSpacing(3)
        _vt_style = (f'QPushButton{{font-size:10px;padding:5px 0;border-radius:4px;'
                     f'border:1px solid #333;background:#252525;color:{TEXT_DIM};}}'
                     f'QPushButton:hover{{background:#303030;color:{TEXT_HI};}}')
        btn_view_d   = QPushButton('View…')
        btn_timer_sg = QPushButton('Timer')
        for b in (btn_view_d, btn_timer_sg):
            b.setStyleSheet(_vt_style)
            vt_row.addWidget(b, stretch=1)
        btn_view_d.clicked.connect(self._show_display_settings)
        layout.addLayout(vt_row)

        layout.addStretch()

        # ── Atributos de compatibilidad — NO en layout ─────────────────────
        self.cmb_dev_in  = QComboBox()
        self.cmb_dev_out = QComboBox()

        self.spn_ir_ms = QSpinBox()
        self.spn_ir_ms.setRange(10, 1000); self.spn_ir_ms.setSingleStep(10)
        self.spn_ir_ms.setValue(200)
        self.spn_ir_ms.valueChanged.connect(lambda v: self.canvas_meas.update_ir_range(v))

        self.spn_tf_min = QSpinBox()
        self.spn_tf_min.setRange(-60, 0); self.spn_tf_min.setValue(-30)
        self.spn_tf_min.valueChanged.connect(
            lambda v: (self.canvas_meas.ax_tf.set_ylim(v, self.spn_tf_max.value()),
                       self.canvas_meas.align_coh_axis(v, self.spn_tf_max.value()),
                       self.canvas_meas.draw_idle()))

        self.spn_tf_max = QSpinBox()
        self.spn_tf_max.setRange(0, 60); self.spn_tf_max.setValue(30)
        self.spn_tf_max.valueChanged.connect(
            lambda v: (self.canvas_meas.ax_tf.set_ylim(self.spn_tf_min.value(), v),
                       self.canvas_meas.align_coh_axis(self.spn_tf_min.value(), v),
                       self.canvas_meas.draw_idle()))

        self.lbl_tone_freq = QLabel()
        self.spn_tone_freq = QSpinBox()
        self.spn_tone_freq.setRange(20, 20000); self.spn_tone_freq.setValue(1000)
        self.spn_tone_freq.valueChanged.connect(lambda v: setattr(self.engine, 'tone_freq', v))

        self.spn_noise_ch = QSpinBox()
        self.spn_noise_ch.setRange(1, 8); self.spn_noise_ch.setValue(1)
        self.spn_noise_ch.valueChanged.connect(self._on_noise_channel)

        self.lbl_delay_full = QLabel('—')

        return w

    # ── Measurement Configuration dialog ─────────────────────────────

    def _show_measurement_config(self):
        """
        Panel principal de configuración de mediciones estilo SMAART.
        Árbol con secciones Spectrum y Transfer Function.
        Columnas: Measurement Configuration | Device | Meas. Ch. | Ref. Ch.
        Botones: New Spectrum Measurement | New TF Measurement | Delete | etc.
        """
        import sounddevice as sd

        dlg = QDialog(self)
        dlg.setWindowTitle('Measurement Configuration')
        dlg.resize(860, 700)
        dlg.setStyleSheet(
            f'QDialog{{background:#111111;color:{TEXT_HI};font-size:11px;}}'
            f'QTreeWidget{{background:#111111;color:{TEXT_HI};border:none;'
            f'alternate-background-color:#141414;font-size:11px;}}'
            f'QTreeWidget::item{{padding:3px 4px;border-bottom:1px solid #1a1a1a;}}'
            f'QTreeWidget::item:selected{{background:#1e3a4a;color:{TEXT_HI};}}'
            f'QHeaderView::section{{background:#1e1e1e;color:{TEXT_MID};'
            f'padding:5px 6px;border:none;border-right:1px solid #2a2a2a;'
            f'font-size:10px;font-weight:bold;}}'
        )

        root_lay = QVBoxLayout(dlg)
        root_lay.setContentsMargins(0, 0, 0, 0)
        root_lay.setSpacing(0)

        # ── Árbol principal ───────────────────────────────────────────
        tree = QTreeWidget()
        tree.setColumnCount(4)
        tree.setHeaderLabels(['Measurement Configuration', 'Device', 'Meas. Ch.', 'Ref. Ch.'])
        tree.setAlternatingRowColors(True)
        tree.setRootIsDecorated(True)
        tree.setIndentation(20)
        tree.header().setStretchLastSection(False)
        tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        tree.header().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        tree.header().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        tree.setColumnWidth(1, 180)
        tree.setColumnWidth(2, 130)
        tree.setColumnWidth(3, 130)

        # Helper: añadir ítem con swatch de color
        def _add_row(parent, color, name, device, meas_ch, ref_ch=''):
            item = QTreeWidgetItem(parent)
            item.setText(1, device)
            item.setText(2, meas_ch)
            item.setText(3, ref_ch)
            # Widget con swatch + nombre
            w = QWidget()
            w.setStyleSheet('background:transparent;')
            h = QHBoxLayout(w)
            h.setContentsMargins(4, 2, 4, 2)
            h.setSpacing(6)
            swatch = QLabel('■')
            swatch.setStyleSheet(f'color:{color};font-size:16px;background:transparent;')
            h.addWidget(swatch)
            lbl_n = QLabel(name)
            lbl_n.setStyleSheet(f'color:{TEXT_HI};font-size:11px;background:transparent;')
            h.addWidget(lbl_n, stretch=1)
            tree.setItemWidget(item, 0, w)
            return item

        # ── Sección Spectrum ──────────────────────────────────────────
        sec_sp = QTreeWidgetItem(tree)
        sec_sp.setText(0, 'Spectrum')
        sec_sp.setExpanded(True)
        sec_sp.setFlags(sec_sp.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        sec_sp.setForeground(0, QColor(TEXT_DIM))

        # Mostrar el canal de spectrum actual
        try:
            dev_name = sd.query_devices(self.engine.dev_in)['name']
        except Exception:
            dev_name = '—'
        sp_colors = ['#ffff00', '#00ff00', '#ff00ff', '#ff8800']
        for i in range(1):   # por ahora solo 1 spectrum
            _add_row(sec_sp,
                     sp_colors[i % len(sp_colors)],
                     f'Spectrum {i+1}',
                     dev_name,
                     f'Ch {self.engine.ch_spec}')

        # ── Sección Transfer Function ─────────────────────────────────
        sec_tf = QTreeWidgetItem(tree)
        sec_tf.setText(0, 'Transfer Function')
        sec_tf.setExpanded(True)
        sec_tf.setFlags(sec_tf.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        sec_tf.setForeground(0, QColor(TEXT_DIM))

        try:
            in_name  = sd.query_devices(self.engine.dev_in)['name']
        except Exception:
            in_name = '—'

        for i, eng in enumerate(self._tf_engines):
            try:
                dname = sd.query_devices(self.engine.dev_in)['name']
            except Exception:
                dname = in_name
            _add_row(sec_tf,
                     eng._color,
                     str(i + 1),
                     dname,
                     f'Ch {eng.ch_m}',
                     f'Ch {eng.ch_r}')

        root_lay.addWidget(tree, stretch=1)

        # ── Botones inferiores ────────────────────────────────────────
        _btn_style = (
            f'QPushButton{{background:#252525;color:{TEXT_MID};border:1px solid #333;'
            f'border-radius:3px;padding:5px 12px;font-size:11px;}}'
            f'QPushButton:hover{{background:#303030;color:{TEXT_HI};}}')

        bot1 = QHBoxLayout()
        bot1.setContentsMargins(6, 4, 6, 2)
        bot1.setSpacing(4)
        btn_del  = QPushButton('🗑')
        btn_del.setFixedWidth(40)
        btn_del.setStyleSheet(_btn_style)
        btn_new_sp  = QPushButton('New Spectrum Measurement')
        btn_new_tf  = QPushButton('New TF Measurement')
        for b in (btn_del, btn_new_sp, btn_new_tf):
            b.setStyleSheet(_btn_style)
        bot1.addWidget(btn_del)
        bot1.addWidget(btn_new_sp, stretch=1)
        bot1.addWidget(btn_new_tf, stretch=1)

        bot2 = QHBoxLayout()
        bot2.setContentsMargins(6, 2, 6, 6)
        bot2.setSpacing(4)
        btn_info = QPushButton('ℹ')
        btn_info.setFixedWidth(40)
        btn_avg_sp = QPushButton('New Spectrum Average')
        btn_avg_tf = QPushButton('New TF Average')
        for b in (btn_info, btn_avg_sp, btn_avg_tf):
            b.setStyleSheet(_btn_style)
        bot2.addWidget(btn_info)
        bot2.addWidget(btn_avg_sp, stretch=1)
        bot2.addWidget(btn_avg_tf, stretch=1)

        root_lay.addLayout(bot1)
        root_lay.addLayout(bot2)

        # Conectar botones
        def _new_tf():
            dlg.accept()
            self._show_new_tf_dialog()

        def _del_selected():
            item = tree.currentItem()
            if item and item.parent() == sec_tf:
                idx = sec_tf.indexOfChild(item)
                if 0 <= idx < len(self._tf_engines):
                    self._remove_tf_engine(idx)
                sec_tf.removeChild(item)

        btn_new_tf.clicked.connect(_new_tf)
        btn_del.clicked.connect(_del_selected)

        dlg.exec()

    # ── New TF Measurement dialog ─────────────────────────────────────

    def _show_new_tf_dialog(self):
        """
        Dialog 'New TF Measurement' estilo SMAART.
        Permite elegir nombre, device, Mea Ch y Ref Ch antes de crear el engine.
        Create +        → crea y reabre el dialog
        Create & Close  → crea y cierra
        """
        import sounddevice as sd

        # Obtener dispositivos de entrada disponibles
        try:
            raw = sd.query_devices()
            in_devs = [(i, d) for i, d in enumerate(raw) if d['max_input_channels'] > 0]
        except Exception:
            in_devs = []

        while True:
            dlg = QDialog(self)
            dlg.setWindowTitle('New TF Measurement')
            dlg.setFixedSize(420, 230)
            dlg.setStyleSheet(
                f'QDialog{{background:#2a2a2a;}}'
                f'QLabel{{color:{TEXT_HI};font-size:12px;background:transparent;}}'
                f'QLineEdit{{background:#3a3a3a;color:{TEXT_HI};border:1px solid #555;'
                f'border-radius:3px;padding:4px 6px;font-size:12px;}}'
                f'QComboBox{{background:#3a3a3a;color:{TEXT_HI};border:1px solid #444;'
                f'border-radius:3px;padding:3px 6px;font-size:12px;}}'
                f'QComboBox::drop-down{{border:none;width:20px;}}'
                f'QComboBox QAbstractItemView{{background:#3a3a3a;color:{TEXT_HI};'
                f'selection-background-color:#1e3a4a;}}'
            )

            from PyQt6.QtWidgets import QLineEdit
            lay = QVBoxLayout(dlg)
            lay.setContentsMargins(20, 16, 20, 16)
            lay.setSpacing(10)

            form = QFormLayout()
            form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
            form.setSpacing(8)

            # Name
            txt_name = QLineEdit()
            txt_name.setPlaceholderText(f'TF {len(self._tf_engines)+1}')
            form.addRow('Name:', txt_name)

            # Device
            cmb_dev = QComboBox()
            for _, d in in_devs:
                cmb_dev.addItem(d['name'])
            form.addRow('Device:', cmb_dev)

            # Mea Ch / Ref Ch — se actualizan según device seleccionado
            cmb_mea = QComboBox()
            cmb_ref = QComboBox()

            def _populate_channels(dev_idx):
                cmb_mea.clear(); cmb_ref.clear()
                if dev_idx < len(in_devs):
                    n = in_devs[dev_idx][1]['max_input_channels']
                    for c in range(1, n + 1):
                        cmb_mea.addItem(f'Input {c}')
                        cmb_ref.addItem(f'Input {c}')
                    if n >= 2:
                        cmb_ref.setCurrentIndex(1)

            cmb_dev.currentIndexChanged.connect(_populate_channels)
            _populate_channels(0)

            form.addRow('Mea Ch:', cmb_mea)
            form.addRow('Ref Ch:', cmb_ref)
            lay.addLayout(form)
            lay.addStretch()

            # Botones
            _btn_style = (
                f'QPushButton{{background:#3a3a3a;color:{TEXT_HI};border:1px solid #555;'
                f'border-radius:3px;padding:5px 16px;font-size:11px;text-align:right;}}'
                f'QPushButton:hover{{background:#454545;}}')
            btn_create_more  = QPushButton('Create +')
            btn_create_close = QPushButton('Create && Close')
            for b in (btn_create_more, btn_create_close):
                b.setStyleSheet(_btn_style)

            btn_row = QVBoxLayout()
            btn_row.setSpacing(4)
            btn_row.addWidget(btn_create_more)
            btn_row.addWidget(btn_create_close)
            lay.addLayout(btn_row)

            _action = [None]   # 'more' | 'close' | None (cancel)
            btn_create_more.clicked.connect( lambda: (_action.__setitem__(0,'more'),  dlg.accept()))
            btn_create_close.clicked.connect(lambda: (_action.__setitem__(0,'close'), dlg.accept()))

            dlg.exec()

            if _action[0] is None:
                break   # cerrado con X

            # Crear engine
            name   = txt_name.text().strip() or f'TF {len(self._tf_engines)+1}'
            ch_m   = cmb_mea.currentIndex() + 1
            ch_r   = cmb_ref.currentIndex() + 1
            dev_idx = cmb_dev.currentIndex()
            if dev_idx < len(in_devs):
                dev_id = in_devs[dev_idx][0]
                self.engine.dev_in = dev_id
                self.cmb_dev_in.setCurrentIndex(
                    next((i for i in range(self.cmb_dev_in.count())
                          if str(dev_id) in self.cmb_dev_in.itemText(i)), 0))

            color = ENGINE_PALETTE[len(self._tf_engines) % len(ENGINE_PALETTE)]
            eng = self._add_tf_engine(ch_m=ch_m, ch_r=ch_r, color=color)
            self._select_engine(len(self._tf_engines) - 1)
            self._on_channels()

            if _action[0] == 'close':
                break

    # ── Gestión dinámica de TF engines ───────────────────────────────

    def _add_tf_engine(self, ch_m: int = None, ch_r: int = 4,
                       color: str = None) -> 'TFEngine':
        idx   = len(self._tf_engines)
        color = color or ENGINE_PALETTE[idx % len(ENGINE_PALETTE)]
        ch_m  = ch_m if ch_m is not None else (idx + 1)

        # Callbacks (capturan referencia a la lista en ese momento)
        def _on_ch(): self._on_channels()
        def _on_rm(): self._remove_tf_engine(self._tf_engines.index(eng))
        def _on_fd(): self._on_find_delay_for_engine(self._tf_engines.index(eng))
        def _on_sel(): self._select_engine(self._tf_engines.index(eng))
        def _on_nrm(): self._normalize_engine(self._tf_engines.index(eng))

        eng = TFEngine(idx + 1, color, _on_ch, _on_rm, _on_fd,
                       on_select=_on_sel, on_normalize=_on_nrm, parent=None)
        eng.spn_m.setValue(ch_m)
        eng.spn_r.setValue(ch_r)

        # Asegurar que el AudioEngine tiene buffer para este canal
        while len(self.engine.buf_meas_list) <= idx:
            self.engine.add_meas_channel(ch_m)

        self._tf_engines.append(eng)
        self._engines_layout.addWidget(eng)

        # Actualizar aliases de compatibilidad
        if len(self._tf_engines) == 1:
            self.spn_ch_meas  = eng.spn_m
            self.spn_ch_ref   = eng.spn_r
            self.bar_spl_meas = eng.bar_m
            self.bar_spl_ref  = eng.bar_r
            self.lbl_spl_meas = eng.lbl_db
        elif len(self._tf_engines) == 2:
            self.spn_ch_meas2 = eng.spn_m

        # Crear RunningTF para este engine
        rtf = RunningTF(
            nperseg    = self.engine.nperseg,
            fs         = self.engine.fs,
            n_averages = self.engine.n_averages,
        )
        if not hasattr(self, '_running_tfs'):
            self._running_tfs = []
        self._running_tfs.append(rtf)

        # Sincronizar colores del canvas con los engines registrados
        colors = [e._color for e in self._tf_engines]
        self.canvas_meas.set_engine_colors(colors)
        self.canvas_meas.highlight_engine(
            min(getattr(self, '_selected_engine_idx', 0), len(self._tf_engines) - 1))

        return eng

    def _remove_tf_engine(self, idx: int):
        if len(self._tf_engines) <= 1:
            return   # siempre queda al menos 1
        eng = self._tf_engines.pop(idx)
        self._engines_layout.removeWidget(eng)
        eng.setParent(None)
        eng.deleteLater()
        # Quitar buffer de medición y RunningTF correspondiente
        if idx < len(self.engine.buf_meas_list):
            self.engine.remove_meas_channel(idx)
        if hasattr(self, '_running_tfs') and idx < len(self._running_tfs):
            self._running_tfs.pop(idx)

    # ── Spectrum Engine management ────────────────────────────────────

    def _rebuild_sp_engine_rows(self):
        """Reconstruye el panel de filas de engines spectrum."""
        if not hasattr(self, '_sp_eng_layout'):
            return
        lay = self._sp_eng_layout
        # Borrar filas previas (no borrar el stretch al final)
        while lay.count() > 1:
            item = lay.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
                w.deleteLater()
        for i, eng in enumerate(self._sp_engines):
            row = SpectrumEngineRow(
                idx=i,
                name=eng.get('name', f'Analog {eng.get("ch", i+1)}'),
                color=eng.get('color', ENGINE_PALETTE[i % len(ENGINE_PALETTE)]),
                on_config=self._open_sp_engine_config,
                on_del=self._remove_spectrum_engine,
                on_play=self._toggle_sp_engine_active,
                active=eng.get('active', True),
                selected=(i == getattr(self, '_sp_eng_selected', 0)),
            )
            lay.insertWidget(lay.count() - 1, row)

    def _add_spectrum_engine(self, ch: int = None, color: str = None,
                              name: str = None) -> int:
        """Agrega un Spectrum Engine. Retorna el índice."""
        idx   = len(self._sp_engines)
        color = color or ENGINE_PALETTE[idx % len(ENGINE_PALETTE)]
        ch    = ch or (idx + 1)
        name  = name or f'Analog {ch}'
        # audio backend
        if idx == 0:
            self.engine._ch_spec_list[0] = ch
        else:
            self.engine.add_spec_channel(ch)
        # canvas line
        line_idx = self.canvas_spec.add_sp_engine_line(color)
        # running avg
        rtf = RunningTF(nperseg=self.engine.nperseg,
                        fs=self.engine.fs,
                        n_averages=self.engine.n_averages)
        self._sp_engines.append({
            'ch': ch, 'color': color, 'name': name,
            'rtf': rtf, 'canvas_idx': line_idx, 'active': True
        })
        self._rebuild_sp_engine_rows()
        return idx

    def _remove_spectrum_engine(self, idx: int):
        """Elimina el Spectrum Engine en posición idx."""
        if len(self._sp_engines) <= 1:
            return
        self._sp_engines.pop(idx)
        self.canvas_spec.remove_sp_engine_line(idx)
        if idx > 0:
            self.engine.remove_spec_channel(idx)
        if getattr(self, '_sp_eng_selected', 0) >= len(self._sp_engines):
            self._sp_eng_selected = len(self._sp_engines) - 1
        self._rebuild_sp_engine_rows()
        self._save_prefs()

    def _select_sp_engine(self, idx: int):
        self._sp_eng_selected = idx
        self._rebuild_sp_engine_rows()

    def _toggle_sp_engine_active(self, idx: int):
        """Activa / pausa la captura del engine idx."""
        if idx >= len(self._sp_engines):
            return
        self._sp_engines[idx]['active'] = not self._sp_engines[idx].get('active', True)
        self._rebuild_sp_engine_rows()
        # Si se activa, resetear su RunningTF para que arranque limpio
        if self._sp_engines[idx]['active']:
            self._sp_engines[idx]['rtf'].reset()

    # ── Config dialog helpers ─────────────────────────────────────────

    def _get_input_devices_for_dialog(self) -> list:
        """Retorna lista de dicts {'id', 'name', 'max_in'} para el dialog."""
        devs = AudioEngine.list_devices()
        _FORCE = ('apogee', 'symphony', 'duet', 'quartet', 'ensemble',
                  'ua ', 'universal audio', 'apollo', 'uad',
                  'rme', 'motu', 'focusrite', 'scarlett', 'clarett',
                  'presonus', 'avid', 'pro tools', 'dante')
        result = []
        for d in devs:
            nl = d['name'].lower()
            force = any(k in nl for k in _FORCE)
            if d.get('in', 0) >= 1 or force:
                result.append({
                    'id'    : d['id'],
                    'name'  : d['name'],
                    'max_in': max(d.get('in', 8), 1),
                })
        # Fallback: si no hay nada, poner un entry genérico
        if not result:
            result.append({'id': 0, 'name': 'Default Input', 'max_in': 8})
        return result

    def _on_add_spectrum_engine(self):
        """Abre el dialog 'New Spectrum Measurement'."""
        devs = self._get_input_devices_for_dialog()

        def _on_create(name, device_id, ch):
            # Elegir el primer color que no esté ya en uso
            used = {e.get('color') for e in self._sp_engines}
            color = next(
                (c for c in ENGINE_PALETTE if c not in used),
                ENGINE_PALETTE[len(self._sp_engines) % len(ENGINE_PALETTE)]
            )
            self._add_spectrum_engine(ch=ch, color=color, name=name)
            self._save_prefs()

        dlg = SpectrumEngineDialog(
            input_devices=devs,
            title='New Spectrum Measurement',
            on_create=_on_create,
            parent=self,
        )
        dlg.exec()

    def _open_sp_engine_config(self, idx: int):
        """Abre el dialog de edición para el engine idx."""
        if idx >= len(self._sp_engines):
            return
        eng  = self._sp_engines[idx]
        devs = self._get_input_devices_for_dialog()

        def _on_save(name, device_id, ch):
            old_ch = eng.get('ch', 1)
            eng['name'] = name
            eng['ch']   = ch
            # Actualizar canal en audio backend
            if idx < len(self.engine._ch_spec_list):
                self.engine._ch_spec_list[idx] = ch
            elif old_ch != ch:
                # si el índice ya no existe (no debería pasar), ignorar
                pass
            self._rebuild_sp_engine_rows()
            self._save_prefs()

        dlg = SpectrumEngineDialog(
            input_devices=devs,
            title='Edit Spectrum Engine',
            name=eng.get('name', f'Analog {eng.get("ch", 1)}'),
            device_id=eng.get('device_id'),
            ch=eng.get('ch', 1),
            on_create=_on_save,
            parent=self,
        )
        dlg.exec()

    # ── Persistencia de preferencias ─────────────────────────────────

    def _prefs_path(self) -> str:
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(script_dir, 'coherence_prefs.json')

    # ── Workspace management ──────────────────────────────────────────

    def _ws_current_state(self) -> dict:
        """Captura el estado actual como dict de workspace."""
        name = ''
        if hasattr(self, '_ws_tab_bar') and self._ws_tab_bar.count() > self._current_ws_idx:
            name = self._ws_tab_bar.tabText(self._current_ws_idx)
        return {
            'name': name or f'Workspace {self._current_ws_idx + 1}',
            'dev_in':     self.engine.dev_in,
            'dev_out':    self.engine.dev_out,
            'ch_ref':     self.engine.ch_ref,
            'ch_spl':     self.engine.ch_spl,
            'n_averages': self.engine.n_averages,
            'nperseg':    self.engine.nperseg,
            'smooth':     getattr(self, '_smooth_fraction', 0.0),
            'tf_engines': [
                {
                    'ch_m':        eng.spn_m.value(),
                    'ch_r':        eng.spn_r.value(),
                    'delay_ms':    eng._delay_comp_ms,
                    'gain_offset': eng._gain_offset_db,
                    'color':       eng._color,
                }
                for eng in self._tf_engines
            ],
            'sp_engines': [
                {
                    'ch':    e.get('ch', 1),
                    'color': e.get('color', ENGINE_PALETTE[0]),
                    'name':  e.get('name', 'Analog'),
                }
                for e in self._sp_engines
            ],
        }

    def _ws_save_current(self):
        """Guarda el estado actual en _workspaces[_current_ws_idx]."""
        if not self._workspaces:
            return
        if self._current_ws_idx < len(self._workspaces):
            self._workspaces[self._current_ws_idx] = self._ws_current_state()

    def _ws_load(self, idx: int):
        """Carga el workspace en idx, limpiando el estado actual."""
        if idx < 0 or idx >= len(self._workspaces):
            return
        ws = self._workspaces[idx]
        self._current_ws_idx = idx
        self._ws_loading = True
        try:
            # ── Limpiar TF engines ────────────────────────────────────
            while self._tf_engines:
                eng = self._tf_engines.pop(0)
                self._engines_layout.removeWidget(eng)
                eng.setParent(None)
                eng.deleteLater()
                if self.engine.buf_meas_list:
                    try:
                        self.engine.remove_meas_channel(0)
                    except Exception:
                        pass
                if hasattr(self, '_running_tfs') and self._running_tfs:
                    self._running_tfs.pop(0)

            # ── Limpiar Spectrum engines ──────────────────────────────
            while self._sp_engines:
                self._sp_engines.pop()
            # Limpiar líneas del canvas espectro
            try:
                while self.canvas_spec._sp_eng_lines:
                    ln = self.canvas_spec._sp_eng_lines.pop()
                    try:
                        ln.remove()
                    except Exception:
                        pass
                while self.canvas_spec._sp_eng_fills:
                    fl = self.canvas_spec._sp_eng_fills.pop()
                    try:
                        fl.remove()
                    except Exception:
                        pass
            except Exception:
                pass
            self._rebuild_sp_engine_rows()

            # ── Restaurar TF engines ──────────────────────────────────
            for e in ws.get('tf_engines', []):
                eng = self._add_tf_engine(
                    ch_m=e.get('ch_m', 1),
                    ch_r=e.get('ch_r', 2),
                    color=e.get('color'),
                )
                eng._delay_comp_ms  = e.get('delay_ms', 0.0)
                eng._gain_offset_db = e.get('gain_offset', 0.0)
                eng.set_delay(e.get('delay_ms', 0.0))

            # ── Restaurar Spectrum engines ────────────────────────────
            for e in ws.get('sp_engines', []):
                self._add_spectrum_engine(
                    ch=e.get('ch', 1),
                    color=e.get('color'),
                    name=e.get('name', 'Analog'),
                )

        finally:
            self._ws_loading = False

        # Después de cargar: forzar sync de colores canvas ↔ widgets
        if self._tf_engines:
            colors = [e._color for e in self._tf_engines]
            self.canvas_meas.set_engine_colors(colors)
            self.canvas_meas.highlight_engine(0)
            self._selected_engine_idx = 0

    def _ws_add_new(self, name: str = ''):
        """Agrega un workspace vacío y cambia a él."""
        self._ws_save_current()
        idx = len(self._workspaces)
        ws_name = name or f'Workspace {idx + 1}'
        self._workspaces.append({
            'name': ws_name,
            'tf_engines': [],
            'sp_engines': [],
        })
        self._ws_tab_bar.blockSignals(True)
        self._ws_tab_bar.addTab(ws_name)
        self._ws_tab_bar.blockSignals(False)
        self._ws_tab_bar.setCurrentIndex(idx)   # triggers _ws_on_tab_changed
        self._save_prefs()

    def _ws_close_tab(self, idx: int):
        """Cierra el workspace en idx (mínimo 1 tab)."""
        if self._ws_tab_bar.count() <= 1:
            return
        self._workspaces.pop(idx)
        self._ws_tab_bar.blockSignals(True)
        self._ws_tab_bar.removeTab(idx)
        self._ws_tab_bar.blockSignals(False)
        # Ajustar índice activo
        new_idx = min(self._current_ws_idx, self._ws_tab_bar.count() - 1)
        self._current_ws_idx = new_idx
        self._ws_tab_bar.setCurrentIndex(new_idx)
        self._ws_load(new_idx)
        self._save_prefs()

    def _ws_on_tab_changed(self, idx: int):
        """Callback cuando el usuario cambia de tab."""
        if self._ws_loading:
            return
        if idx == self._current_ws_idx:
            return
        self._ws_save_current()
        self._ws_load(idx)
        self._save_prefs()

    def _ws_rename_tab_evt(self, event):
        """Double-click en tab bar → dialog de renombrar."""
        tab_idx = self._ws_tab_bar.tabAt(event.pos())
        if tab_idx < 0:
            QTabBar.mouseDoubleClickEvent(self._ws_tab_bar, event)
            return
        from PyQt6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(
            self, 'Rename Workspace',
            'Nombre del workspace:',
            text=self._ws_tab_bar.tabText(tab_idx)
        )
        if ok and name.strip():
            self._ws_tab_bar.setTabText(tab_idx, name.strip())
            if tab_idx < len(self._workspaces):
                self._workspaces[tab_idx]['name'] = name.strip()
            self._save_prefs()

    def _save_prefs(self):
        if self._ws_loading:
            return
        import json, os
        self._ws_save_current()
        prefs = {
            'dev_in':  self.engine.dev_in,
            'dev_out': self.engine.dev_out,
            'ch_ref':  self.engine.ch_ref,
            'ch_spl':  self.engine.ch_spl,
            'current_ws': self._current_ws_idx,
            'workspaces': self._workspaces,
        }
        try:
            with open(self._prefs_path(), 'w') as f:
                json.dump(prefs, f, indent=2)
        except Exception:
            pass

    def _load_prefs(self):
        """
        Carga workspaces guardados.
        - Sin archivo → 1 workspace vacío, sin engines.
        - Formato viejo (spectrum_engines al top) → migrar a workspace.
        - Formato nuevo (workspaces list) → restaurar tabs.
        """
        import json, os
        path = self._prefs_path()

        workspaces = []
        current_ws = 0

        if os.path.exists(path):
            try:
                with open(path) as f:
                    prefs = json.load(f)

                # ── Restaurar dispositivos globales ────────────────────
                saved_in  = prefs.get('dev_in')
                saved_out = prefs.get('dev_out')
                if saved_in is not None:
                    self.engine.dev_in = int(saved_in)
                    for i, did in enumerate(self._dev_in_ids):
                        if did == self.engine.dev_in:
                            self.cmb_dev_in.blockSignals(True)
                            self.cmb_dev_in.setCurrentIndex(i)
                            self.cmb_dev_in.blockSignals(False)
                            break
                if saved_out is not None:
                    self.engine.dev_out = int(saved_out)
                    for i, did in enumerate(self._dev_out_ids):
                        if did == self.engine.dev_out:
                            self.cmb_dev_out.blockSignals(True)
                            self.cmb_dev_out.setCurrentIndex(i)
                            self.cmb_dev_out.blockSignals(False)
                            break
                if prefs.get('ch_ref') is not None:
                    self.engine.ch_ref = int(prefs['ch_ref'])
                if prefs.get('ch_spl') is not None:
                    self.engine.ch_spl = int(prefs['ch_spl'])

                if 'workspaces' in prefs:
                    # ── Formato nuevo ──────────────────────────────────
                    workspaces  = prefs['workspaces']
                    current_ws  = int(prefs.get('current_ws', 0))
                elif 'spectrum_engines' in prefs:
                    # ── Migrar formato viejo ───────────────────────────
                    sp_engs = prefs.get('spectrum_engines', [])
                    workspaces = [{
                        'name': 'Workspace 1',
                        'tf_engines': [],
                        'sp_engines': [
                            {'ch': e.get('ch', 1),
                             'color': e.get('color', ENGINE_PALETTE[0]),
                             'name': e.get('name', 'Analog')}
                            for e in sp_engs
                        ],
                    }]
            except Exception:
                pass

        if not workspaces:
            # Primera ejecución o archivo corrupto → workspace vacío
            workspaces = [{'name': 'Workspace 1', 'tf_engines': [], 'sp_engines': []}]
            current_ws = 0

        self._workspaces = workspaces

        # ── Poblar tab bar ────────────────────────────────────────────
        self._ws_tab_bar.blockSignals(True)
        for ws in self._workspaces:
            self._ws_tab_bar.addTab(ws.get('name', 'Workspace'))
        self._ws_tab_bar.blockSignals(False)

        current_ws = min(current_ws, len(self._workspaces) - 1)
        self._current_ws_idx = current_ws
        self._ws_tab_bar.setCurrentIndex(current_ws)

        # ── Cargar workspace activo ───────────────────────────────────
        self._ws_load(current_ws)

    # ── Sub-panel SPECTRUM ────────────────────────────────────────────

    def _build_settings_spectrum(self):
        w = QWidget()
        w.setStyleSheet(f'background:{BG_SETTINGS};')
        layout = QVBoxLayout(w)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(5)

        def rl(t): return lbl(t, color=TEXT_MID, size=9)

        # ── Header discreto ──────────────────────────────────────────
        h_lbl = QLabel('Spectrum')
        h_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        h_lbl.setStyleSheet(
            f'color:{TEXT_MID};font-size:9px;letter-spacing:2px;'
            f'padding:2px 0 5px 0;border-bottom:1px solid #222;background:transparent;')
        layout.addWidget(h_lbl)

        # ── Averaging Depth ───────────────────────────────────────────
        avg_row = QHBoxLayout()
        avg_row.setContentsMargins(0, 4, 0, 0)
        avg_row.setSpacing(6)
        avg_row.addWidget(lbl('Averaging Depth:', color=TEXT_MID, size=9))
        self.cmb_avg_spec = QComboBox()
        self._avg_spec_values = [1, 2, 4, 6, 8, 12, 16, 32]
        for _av in self._avg_spec_values:
            self.cmb_avg_spec.addItem(str(_av))
        self.cmb_avg_spec.setCurrentIndex(2)   # 4 por defecto (igual que SMAART)
        avg_row.addWidget(self.cmb_avg_spec, stretch=1)
        layout.addLayout(avg_row)

        # ── Transport (▶ ■ ‖ / ⏱ GEN) ───────────────────────────────
        self._make_transport(layout, freeze=True, gen=False)

        # ── Spectrum Engines ─────────────────────────────────────────
        sep_lbl = QLabel('Spectrum Engines')
        sep_lbl.setStyleSheet(
            f'color:{TEXT_DIM};font-size:8px;letter-spacing:1px;'
            f'padding:6px 0 2px 0;background:transparent;')
        layout.addWidget(sep_lbl)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet('background:transparent;')
        scroll.setFixedHeight(140)

        self._sp_eng_container = QWidget()
        self._sp_eng_container.setStyleSheet('background:transparent;')
        self._sp_eng_layout = QVBoxLayout(self._sp_eng_container)
        self._sp_eng_layout.setContentsMargins(0, 0, 0, 0)
        self._sp_eng_layout.setSpacing(3)
        self._sp_eng_layout.addStretch()
        scroll.setWidget(self._sp_eng_container)
        layout.addWidget(scroll)

        # Botón + Spectrum Engine (compacto)
        btn_add_sp_eng = QPushButton('＋  Spectrum Engine')
        btn_add_sp_eng.setFixedHeight(22)
        btn_add_sp_eng.setStyleSheet(
            f'QPushButton{{background:transparent;color:{TEXT_DIM};'
            f'border:1px solid #2a2a2a;border-radius:3px;font-size:9px;padding:0 4px;}}'
            f'QPushButton:hover{{color:{GREEN};border-color:{GREEN};}}')
        btn_add_sp_eng.clicked.connect(self._on_add_spectrum_engine)
        layout.addWidget(btn_add_sp_eng)

        layout.addStretch()

        # compat attrs
        self.cmb_sp_res = None
        self._sp_bpo    = [3, 12, 24]
        self._lbl_sp_dev = QLabel()
        self._lbl_sp_dev.hide()

        self.spn_ch_spec = QSpinBox()
        self.spn_ch_spec.setRange(1, 20)
        self.spn_ch_spec.setValue(1)
        self.spn_ch_spec.hide()

        return w

    # ── Sub-panel SPECTROGRAM ─────────────────────────────────────────

    def _build_settings_sgram(self):
        w = QWidget()
        w.setStyleSheet(f'background:{BG_SETTINGS};')
        layout = QVBoxLayout(w)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        def rl(t): return lbl(t, color=TEXT_MID, size=9)

        h_lbl = QLabel('SPECTROGRAM')
        h_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        h_lbl.setStyleSheet(
            f'color:{CYAN};font-size:8px;letter-spacing:3px;font-weight:bold;'
            f'padding:3px 0 4px 0;border-bottom:1px solid {BORDER};background:transparent;')
        layout.addWidget(h_lbl)

        # TRANSPORT propio
        self._make_transport(layout, freeze=False, gen=False)

        # SPECTROGRAM settings
        grp_sg = QGroupBox('ANALYSIS')
        gsg = QGridLayout(grp_sg)
        gsg.setContentsMargins(5, 6, 5, 4); gsg.setSpacing(3); gsg.setColumnStretch(1, 1)

        gsg.addWidget(rl('CH IN'), 0, 0)
        self.spn_ch_spec_sg = QSpinBox()
        self.spn_ch_spec_sg.setRange(1, 20); self.spn_ch_spec_sg.setValue(1)
        # Spectrogram usa el mismo ch_spec que Spectrum
        self.spn_ch_spec_sg.valueChanged.connect(
            lambda v: (setattr(self.engine, 'ch_spec', v),
                       self.spn_ch_spec.setValue(v)))
        gsg.addWidget(self.spn_ch_spec_sg, 0, 1)

        gsg.addWidget(rl('TIME WIN'), 1, 0)
        lbl_tw = QLabel('auto')
        lbl_tw.setStyleSheet(f'color:{TEXT_MID};font-size:9px;background:transparent;')
        gsg.addWidget(lbl_tw, 1, 1)
        layout.addWidget(grp_sg)

        # ROUTING (solo DEV IN)
        grp_rt = QGroupBox('ROUTING')
        gr = QGridLayout(grp_rt)
        gr.setContentsMargins(5, 6, 5, 4); gr.setSpacing(3); gr.setColumnStretch(1, 1)
        gr.addWidget(rl('DEV IN'), 0, 0)
        lbl_dev_sg = QLabel('—')
        lbl_dev_sg.setStyleSheet(f'color:{TEXT_MID};font-size:9px;background:transparent;')
        gr.addWidget(lbl_dev_sg, 0, 1)
        self._lbl_sg_dev = lbl_dev_sg
        layout.addWidget(grp_rt)

        layout.addStretch()
        return w

    def _on_tab_changed(self, idx: int):
        """Compat — ya no hay tabs reales."""
        pass

    def _toggle_ir_panel(self, *args):
        """
        Cmd+I — muestra/oculta el panel IR.
        Cuando se oculta: ax_tf y ax_coh se expanden hacia arriba para llenar
        el espacio liberado. El IR siempre tiene el mismo tamaño fijo cuando
        está visible — nunca se reescala.
        """
        self._ir_visible = not self._ir_visible
        canvas = self.canvas_meas
        canvas._ir_visible = self._ir_visible

        if self._ir_visible:
            # Restaurar posiciones previas de TF/Coh (guardadas al ocultarlo)
            if hasattr(self, '_ir_hide_saved'):
                s = self._ir_hide_saved
                canvas.ax_tf.set_position(s['tf'])
                canvas.ax_coh.set_position(s['coh'])
                canvas.ax_ph.set_position(s['ph'])
            canvas.ax_ir.set_visible(True)

        else:
            # Guardar posiciones actuales de TF/Coh/Ph antes de expandir
            self._ir_hide_saved = {
                'tf':  list(canvas.ax_tf.get_position().bounds),
                'coh': list(canvas.ax_coh.get_position().bounds),
                'ph':  list(canvas.ax_ph.get_position().bounds),
            }
            canvas.ax_ir.set_visible(False)

            # IR bounds (fijo en figura): bottom, height
            ir_l, ir_b, ir_w, ir_h = canvas.ax_ir.get_position().bounds
            tf_l, tf_b, tf_w, tf_h = canvas.ax_tf.get_position().bounds
            ph_l, ph_b, ph_w, ph_h = canvas.ax_ph.get_position().bounds

            # TF sube hasta donde llegaba el tope del IR (ir_b + ir_h)
            # bottom no cambia, height crece
            new_tf_h = (ir_b + ir_h) - tf_b
            canvas.ax_tf.set_position([tf_l, tf_b, tf_w, new_tf_h])
            canvas.ax_coh.set_position([tf_l, tf_b, tf_w, new_tf_h])

            # Phase sube proporcionalmente (si está visible, ej. modo Phase)
            if canvas.ax_ph.get_visible():
                new_ph_h = (ir_b + ir_h) - ph_b
                canvas.ax_ph.set_position([ph_l, ph_b, ph_w, new_ph_h])

        canvas.draw_idle()

        if hasattr(self, '_act_live_ir'):
            self._act_live_ir.setChecked(self._ir_visible)

    def _switch_to_tf(self):
        self._on_view_mode_changed('Magnitude')

    def _switch_to_spectrum(self):
        self._on_view_mode_changed('Spectrum')

    def _toggle_spec_panel(self):
        if self._current_view == 'magnitude':
            self._switch_to_spectrum()
        else:
            self._switch_to_tf()

    # ── Callbacks de los dropdowns de panel ───────────────────────────

    def _on_view_mode_changed(self, mode: str):
        """
        Cambia la vista principal y el panel de settings de la derecha.

        Magnitude    → canvas_meas (IR + TF + Phase completo), settings TF (idx 0)
        Phase        → canvas_meas solo Phase expandido,        settings TF (idx 0)
        Spectrum     → canvas_spec (RTA),                       settings Spec (idx 1)
        Spectrograph → canvas_sgram,                            settings Spec (idx 1)
        """
        # Sincronizar el label del botón en todos los wraps al modo elegido
        for w in (self._wrap_meas, self._wrap_spec, self._wrap_sgram):
            w._current = mode
            w._btn.setText(f'{mode} ▾')
            w._btn.adjustSize()
            w.setVisible(False)

        if mode == 'Magnitude':
            self._wrap_meas.setVisible(True)
            self.canvas_meas.set_view_mode('magnitude_only')   # IR + TF, sin Phase
            self._current_view = 'magnitude'
            settings_idx = 0

        elif mode == 'Phase':
            self._wrap_meas.setVisible(True)
            self.canvas_meas.set_view_mode('phase_only')
            self._current_view = 'phase'
            settings_idx = 0

        elif mode == 'Spectrum':
            self._wrap_spec.setVisible(True)
            self.canvas_meas.set_view_mode('magnitude_only')   # restaurar si venía de Phase
            self._current_view = 'spectrum'
            settings_idx = 1

        elif mode == 'Spectrograph':
            self._wrap_sgram.setVisible(True)
            self.canvas_meas.set_view_mode('magnitude_only')
            self._current_view = 'spectrograph'
            settings_idx = 1

        else:
            self._wrap_meas.setVisible(True)
            self.canvas_meas.set_view_mode('magnitude_only')
            self._current_view = 'magnitude'
            settings_idx = 0

        if hasattr(self, '_settings_stack'):
            self._settings_stack.setCurrentIndex(settings_idx)

        # Sincronizar panel SAVE izquierdo: 0=TF/Phase, 1=Spectrum/Sgram
        if hasattr(self, '_save_stack'):
            self._save_stack.setCurrentIndex(
                0 if self._current_view in ('magnitude', 'phase') else 1)

    def _on_add_panel(self):
        """Agrega o elimina el panel secundario (split view)."""
        if self._secondary_panel is not None:
            # Quitar panel secundario
            self._secondary_panel.setParent(None)
            self._secondary_panel.deleteLater()
            self._secondary_panel = None
            # Actualizar icono "+" → "＋" (sin cambios, solo toggle visual)
            return

        # Crear panel secundario
        self._secondary_panel = SecondaryPanel(on_close=self._on_add_panel)
        # Compartir cursor update
        self._secondary_panel.canvas_spec.on_cursor_update  = self._update_cursor
        self._secondary_panel.canvas_meas.on_cursor_update  = self._update_cursor
        self._secondary_panel.canvas_sgram.on_cursor_update = self._update_cursor
        # Agregar al splitter
        self._panel_splitter.addWidget(self._secondary_panel)
        # Dividir espacio equitativamente
        h = self._panel_splitter.height()
        self._panel_splitter.setSizes([max(h // 2, 200), max(h // 2, 200)])

    def _on_ir_display_mode(self, mode: str):
        """Dropdown del panel IR: Lin / Log / Normalize."""
        self.sb.showMessage(f'IR mode: {mode}', 2000)

    def _on_tf_display_mode(self, mode: str):
        """Dropdown del panel TF: Magnitude / Phase / Coherence."""
        # El panel IR (ax_ir arriba) es controlado EXCLUSIVAMENTE por Cmd+I (_ir_visible).
        # Este dropdown solo afecta ax_tf y ax_ph.
        has_mag = mode in ('Magnitude', 'Coherence')
        has_ph  = mode == 'Phase'
        if hasattr(self, 'canvas_meas'):
            self.canvas_meas.ax_tf.set_visible(has_mag)
            self.canvas_meas.ax_ph.set_visible(has_ph)
            # ax_ir: respetar siempre el flag _ir_visible (Cmd+I)
            self.canvas_meas.ax_ir.set_visible(getattr(self, '_ir_visible', True))
            self.canvas_meas.draw_idle()
        self.sb.showMessage(f'TF panel: {mode}', 2000)

    def _on_spec_display_mode(self, mode: str):
        """Dropdown del panel Spectrum: RTA / Spectrograph / Phase / Magnitude."""
        self.sb.showMessage(f'Spectrum mode: {mode}', 2000)
        if mode == 'Spectrograph':
            # Mostrar spectrogram canvas en lugar de spectrum
            pass  # TODO: swap canvas

    # ── Dialogs de configuración ──────────────────────────────────────

    def _show_tf_engine_config(self):
        """
        Dialog central de configuración de TF engines.
        Contenido:
          • DEVICE IN  (compartido)
          • Por cada engine: ● TF N  /  CH MEAS  /  CH REF
          • ＋ ADD TF ENGINE  (agrega engine y recarga el dialog)
          • CANCEL / APPLY
        """
        while True:   # loop para recargar al agregar engine
            dlg = QDialog(self)
            dlg.setWindowTitle('TF Engine Config')
            dlg.setModal(True)
            dlg.setMinimumWidth(300)
            dlg.setStyleSheet(self.styleSheet())

            lay = QVBoxLayout(dlg)
            lay.setSpacing(8)

            title = QLabel('TF  ENGINE  CONFIG')
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            title.setStyleSheet(
                f'color:{ACCENT};font-size:9px;letter-spacing:3px;font-weight:bold;'
                f'padding:4px 0 6px 0;border-bottom:1px solid {BORDER};'
                f'background:transparent;')
            lay.addWidget(title)

            # ── Device IN ────────────────────────────────────────────
            dev_row = QHBoxLayout()
            dev_row.addWidget(lbl('DEVICE  IN', color=TEXT_DIM, size=8))
            cmb_dev = QComboBox()
            for i in range(self.cmb_dev_in.count()):
                cmb_dev.addItem(self.cmb_dev_in.itemText(i))
            cmb_dev.setCurrentIndex(self.cmb_dev_in.currentIndex())
            dev_row.addWidget(cmb_dev, stretch=1)
            lay.addLayout(dev_row)

            sep0 = QFrame(); sep0.setFrameShape(QFrame.Shape.HLine)
            sep0.setStyleSheet(f'color:{BORDER};'); lay.addWidget(sep0)

            # ── Engines ──────────────────────────────────────────────
            spn_m_list: list[QSpinBox] = []
            spn_r_list: list[QSpinBox] = []

            for i, eng in enumerate(self._tf_engines):
                hdr = QLabel(f'● TF {i + 1}')
                hdr.setStyleSheet(
                    f'color:{eng._color};font-size:10px;font-weight:bold;'
                    f'letter-spacing:1px;background:transparent;')
                lay.addWidget(hdr)

                grid = QGridLayout()
                grid.setContentsMargins(14, 0, 0, 0)
                grid.setSpacing(5); grid.setColumnStretch(1, 1)

                grid.addWidget(lbl('CH  MEAS', color=TEXT_MID, size=9), 0, 0)
                spn_m = QSpinBox()
                spn_m.setRange(1, 32); spn_m.setValue(eng.spn_m.value())
                grid.addWidget(spn_m, 0, 1)
                spn_m_list.append(spn_m)

                grid.addWidget(lbl('CH  REF', color=TEXT_MID, size=9), 1, 0)
                spn_r = QSpinBox()
                spn_r.setRange(1, 32); spn_r.setValue(eng.spn_r.value())
                grid.addWidget(spn_r, 1, 1)
                spn_r_list.append(spn_r)

                lay.addLayout(grid)

                if i < len(self._tf_engines) - 1:
                    s2 = QFrame(); s2.setFrameShape(QFrame.Shape.HLine)
                    s2.setStyleSheet(f'color:{BORDER};'); lay.addWidget(s2)

            lay.addStretch()

            # ── ADD TF ENGINE ────────────────────────────────────────
            sep_add = QFrame(); sep_add.setFrameShape(QFrame.Shape.HLine)
            sep_add.setStyleSheet(f'color:{BORDER};'); lay.addWidget(sep_add)

            btn_add = QPushButton('＋  ADD TF ENGINE')
            btn_add.setStyleSheet(
                f'font-size:9px;color:{ACCENT};background:transparent;'
                f'border:1px dashed {BORDER};border-radius:3px;padding:4px;')
            lay.addWidget(btn_add)

            # ── CANCEL / APPLY ───────────────────────────────────────
            btns = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Apply |
                QDialogButtonBox.StandardButton.Cancel)
            btns.button(QDialogButtonBox.StandardButton.Apply).setStyleSheet(
                f'border:1px solid {ACCENT};color:{ACCENT};background:transparent;')
            btns.button(QDialogButtonBox.StandardButton.Cancel).setStyleSheet(
                f'border:1px solid {BORDER};color:{TEXT_MID};background:transparent;')
            btns.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(dlg.accept)
            btns.rejected.connect(dlg.reject)

            _add_requested = [False]
            def _on_add():
                _add_requested[0] = True
                dlg.accept()
            btn_add.clicked.connect(_on_add)
            lay.addWidget(btns)

            result = dlg.exec()

            if _add_requested[0]:
                # Aplicar cambios actuales antes de agregar
                self.cmb_dev_in.setCurrentIndex(cmb_dev.currentIndex())
                for i, eng in enumerate(self._tf_engines):
                    eng.spn_m.setValue(spn_m_list[i].value())
                    eng.spn_r.setValue(spn_r_list[i].value())
                self._add_tf_engine()   # agrega nuevo engine al panel
                continue                # reabrir dialog con el nuevo engine

            if result == QDialog.DialogCode.Accepted:
                self.cmb_dev_in.setCurrentIndex(cmb_dev.currentIndex())
                for i, eng in enumerate(self._tf_engines):
                    eng.spn_m.setValue(spn_m_list[i].value())
                    eng.spn_r.setValue(spn_r_list[i].value())
                self._on_channels()
            break

    def _on_meas_tab(self, idx: int):
        """Cambia el tab de medición (Real Time / Impulse / SPL)."""
        tabs = [self.btn_tab_rt, self.btn_tab_imp, self.btn_tab_spl]
        for i, b in enumerate(tabs):
            b.setChecked(i == idx)
        # Por ahora solo cambia el tab visual — funcionalidad futura
        self._meas_tab = idx

    def _on_spl_max_reset(self):
        """Resetea el valor máximo del SPL."""
        self._spl_max_db = -200.0
        if hasattr(self, 'lbl_spl_max'):
            self.lbl_spl_max.setText('Max  0.0')

    def _show_spl_settings(self):
        dlg = QDialog(self)
        dlg.setWindowTitle('SPL Meter Settings')
        dlg.setFixedSize(280, 220)
        dlg.setStyleSheet(self.styleSheet())
        lay = QVBoxLayout(dlg)
        form = QFormLayout(); form.setSpacing(8)

        spn_offset = QDoubleSpinBox()
        spn_offset.setRange(-60.0, 60.0); spn_offset.setSingleStep(0.5)
        spn_offset.setDecimals(1); spn_offset.setSuffix(' dB')
        spn_offset.setValue(self._spl_offset_db)
        form.addRow('Cal. Offset (dBFS→SPL):', spn_offset)

        spn_warn = QDoubleSpinBox()
        spn_warn.setRange(0.0, 160.0); spn_warn.setSingleStep(1.0)
        spn_warn.setDecimals(1); spn_warn.setSuffix(' dB')
        spn_warn.setValue(self._spl_warn_db)
        form.addRow('Warn Threshold:', spn_warn)

        spn_clip = QDoubleSpinBox()
        spn_clip.setRange(0.0, 160.0); spn_clip.setSingleStep(1.0)
        spn_clip.setDecimals(1); spn_clip.setSuffix(' dB')
        spn_clip.setValue(self._spl_clip_db)
        form.addRow('Clip Threshold:', spn_clip)

        lay.addLayout(form)
        lay.addStretch()
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._spl_offset_db = spn_offset.value()
            self._spl_warn_db   = spn_warn.value()
            self._spl_clip_db   = spn_clip.value()

    def _show_display_settings(self):
        dlg = QDialog(self)
        dlg.setWindowTitle('Display Settings')
        dlg.setFixedSize(260, 180)
        dlg.setStyleSheet(self.styleSheet())
        lay = QVBoxLayout(dlg)
        form = QFormLayout(); form.setSpacing(8)

        spn_ir = QSpinBox()
        spn_ir.setRange(10, 1000); spn_ir.setSingleStep(10)
        spn_ir.setSuffix(' ms'); spn_ir.setValue(self.spn_ir_ms.value())
        form.addRow('IR Window:', spn_ir)

        spn_min = QSpinBox()
        spn_min.setRange(-60, 0); spn_min.setValue(self.spn_tf_min.value())
        spn_min.setSuffix(' dB')
        form.addRow('TF Min:', spn_min)

        spn_max = QSpinBox()
        spn_max.setRange(0, 30); spn_max.setValue(self.spn_tf_max.value())
        spn_max.setSuffix(' dB')
        form.addRow('TF Max:', spn_max)

        lay.addLayout(form)
        lay.addStretch()
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.spn_ir_ms.setValue(spn_ir.value())
            self.spn_tf_min.setValue(spn_min.value())
            self.spn_tf_max.setValue(spn_max.value())

    def _show_routing_settings(self):
        dlg = QDialog(self)
        dlg.setWindowTitle('Routing / Devices')
        dlg.setFixedSize(320, 160)
        dlg.setStyleSheet(self.styleSheet())
        lay = QVBoxLayout(dlg)
        form = QFormLayout(); form.setSpacing(8)

        cmb_in = QComboBox()
        for i in range(self.cmb_dev_in.count()):
            cmb_in.addItem(self.cmb_dev_in.itemText(i))
        cmb_in.setCurrentIndex(self.cmb_dev_in.currentIndex())
        form.addRow('Device IN:', cmb_in)

        cmb_out = QComboBox()
        for i in range(self.cmb_dev_out.count()):
            cmb_out.addItem(self.cmb_dev_out.itemText(i))
        cmb_out.setCurrentIndex(self.cmb_dev_out.currentIndex())
        form.addRow('Device OUT:', cmb_out)

        lay.addLayout(form)
        lay.addStretch()
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.cmb_dev_in.setCurrentIndex(cmb_in.currentIndex())
            self.cmb_dev_out.setCurrentIndex(cmb_out.currentIndex())

    # ── Engine selection & normalization ─────────────────────────────

    def _select_engine(self, idx: int):
        """Selecciona el engine activo — IR, TF y Phase siguen su color."""
        self._selected_engine_idx = idx
        for i, eng in enumerate(self._tf_engines):
            eng.set_selected(i == idx)
        # Sincronizar colores y destacar en el canvas
        colors = [eng._color for eng in self._tf_engines]
        self.canvas_meas.set_engine_colors(colors)
        self.canvas_meas.highlight_engine(idx)

    def _normalize_engine(self, idx: int):
        """
        Captura la ganancia promedio del engine en 500 Hz–4 kHz y la
        guarda como offset negativo para centrar la TF en 0 dB.
        (Como 'normalize' en SMAART — corrige diferencia de ganancia entre canales.)
        """
        if idx >= len(self._running_tfs):
            return
        rtf = self._running_tfs[idx]
        if not rtf.ready:
            self.sb.showMessage('⚠  Sin datos aún para normalizar', 4000)
            return
        mask = (rtf.freqs >= 500) & (rtf.freqs <= 4000)
        if mask.sum() == 0:
            return
        eps    = 1e-12
        H      = rtf.Gxy[mask] / (rtf.Gxx[mask] + eps)
        avg_db = float(20.0 * np.log10(np.mean(np.abs(H)) + eps))
        self._tf_engines[idx]._gain_offset_db = -avg_db
        self.sb.showMessage(
            f'TF {idx+1}  normalizado  ({avg_db:+.1f} dB → 0 dB)', 4000)

    # ── Delay finder por engine ───────────────────────────────────────

    def _on_find_delay_for_engine(self, idx: int):
        if idx >= len(self._tf_engines):
            return
        eng = self._tf_engines[idx]
        x   = self.engine.buf_ref.read(self.engine.nperseg * self.engine.n_averages)
        y   = self.engine.get_buffer_meas(idx)
        delay_ms, smp = find_delay(x, y, fs=self.engine.fs)
        eng.set_delay(delay_ms)

        # Engine 0 actualiza también el canvas y el label legacy
        if idx == 0:
            self._delay_comp_ms = delay_ms
            dist_m = abs(delay_ms) / 1000.0 * 343.0
            self.lbl_delay_full.setText(
                f'{"+" if delay_ms >= 0 else ""}{delay_ms:.2f} ms\n'
                f'{smp:+d} smp  {dist_m:.2f} m')
            self.canvas_meas.set_delay_ref(delay_ms)


    # ── I-O Config dialog ─────────────────────────────────────────────

    def _show_io_config(self):
        """
        Dialog de configuración de I/O estilo SMAART.
        Muestra todas las interfaces detectadas por PortAudio con:
          - Tabla de dispositivos (Use / API:Driver / Friendly Name / Status)
          - Tabs por dispositivo con sus canales (Use / Ch / Name / Cal.Offset / Level)
        """
        import sounddevice as sd

        dlg = QDialog(self)
        dlg.setWindowTitle('I-O Config')
        dlg.resize(900, 680)
        dlg.setStyleSheet(f'background:#1e1e1e;color:{TEXT_HI};font-size:11px;')

        root = QVBoxLayout(dlg)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Tabs Input / Output ───────────────────────────────────────
        io_tabs = QTabWidget()
        io_tabs.setStyleSheet(
            f'QTabWidget::pane{{border:1px solid #333;background:#1a1a1a;}}'
            f'QTabBar::tab{{background:#252525;color:{TEXT_MID};padding:6px 20px;'
            f'border:1px solid #333;border-bottom:none;border-radius:3px 3px 0 0;}}'
            f'QTabBar::tab:selected{{background:#1a1a1a;color:{TEXT_HI};}}')
        root.addWidget(io_tabs)

        # Obtener dispositivos
        try:
            raw_devs = sd.query_devices()
        except Exception:
            raw_devs = []

        all_devs = []
        for i, d in enumerate(raw_devs):
            all_devs.append({
                'id':    i,
                'api':   f"CoreAudio: {d['name']}",
                'name':  d['name'],
                'in':    int(d['max_input_channels']),
                'out':   int(d['max_output_channels']),
                'fs':    int(d['default_samplerate']),
            })

        _tbl_style = (
            f'QTableWidget{{background:#111;color:{TEXT_HI};gridline-color:#2a2a2a;'
            f'border:1px solid #2a2a2a;font-size:11px;}}'
            f'QTableWidget::item{{padding:3px 6px;}}'
            f'QTableWidget::item:selected{{background:#1e3a4a;color:{TEXT_HI};}}'
            f'QHeaderView::section{{background:#2a2a2a;color:{TEXT_MID};'
            f'padding:4px 6px;border:none;border-right:1px solid #1a1a1a;font-size:10px;}}')

        def _make_io_panel(mode):
            """mode = 'in' o 'out'"""
            devs = [d for d in all_devs if d[mode] > 0]

            panel = QWidget()
            pv = QVBoxLayout(panel)
            pv.setContentsMargins(6, 6, 6, 6)
            pv.setSpacing(6)

            # ── Tabla de dispositivos ─────────────────────────────────
            dev_tbl = QTableWidget(len(devs), 4)
            dev_tbl.setStyleSheet(_tbl_style)
            dev_tbl.setHorizontalHeaderLabels(['Use', 'API : Driver Name', 'Friendly Name', 'Status'])
            dev_tbl.horizontalHeader().setStretchLastSection(False)
            dev_tbl.horizontalHeader().setSectionResizeMode(0, dev_tbl.horizontalHeader().ResizeMode.Fixed)
            dev_tbl.horizontalHeader().setSectionResizeMode(1, dev_tbl.horizontalHeader().ResizeMode.Stretch)
            dev_tbl.horizontalHeader().setSectionResizeMode(2, dev_tbl.horizontalHeader().ResizeMode.Stretch)
            dev_tbl.horizontalHeader().setSectionResizeMode(3, dev_tbl.horizontalHeader().ResizeMode.Fixed)
            dev_tbl.setColumnWidth(0, 44)
            dev_tbl.setColumnWidth(3, 60)
            dev_tbl.setSelectionBehavior(dev_tbl.SelectionBehavior.SelectRows)
            dev_tbl.setEditTriggers(dev_tbl.EditTrigger.NoEditTriggers)
            dev_tbl.verticalHeader().setVisible(False)
            dev_tbl.setFixedHeight(160)

            _use_devs = set()   # ids de dispositivos marcados como "Use"
            _chk_map  = {}      # row → QCheckBox

            for row, d in enumerate(devs):
                # Use checkbox
                chk = QCheckBox()
                chk.setChecked(True)
                _use_devs.add(d['id'])
                cell_w = QWidget()
                cell_l = QHBoxLayout(cell_w)
                cell_l.setContentsMargins(8, 0, 0, 0)
                cell_l.addWidget(chk)
                dev_tbl.setCellWidget(row, 0, cell_w)
                _chk_map[row] = chk

                dev_tbl.setItem(row, 1, QTableWidgetItem(d['api']))
                dev_tbl.setItem(row, 2, QTableWidgetItem(d['name']))

                # Status: intentar abrir stream
                try:
                    if mode == 'in':
                        sd.check_input_settings(device=d['id'], channels=1, samplerate=d['fs'])
                    else:
                        sd.check_output_settings(device=d['id'], channels=1, samplerate=d['fs'])
                    status = 'OK'
                    s_color = '#66bb6a'
                except Exception:
                    status = 'N/C'
                    s_color = '#ef5350'

                s_item = QTableWidgetItem(status)
                s_item.setForeground(QColor(s_color))
                dev_tbl.setItem(row, 3, s_item)

            pv.addWidget(dev_tbl)

            # Remove button
            btn_remove = QPushButton('Remove')
            btn_remove.setFixedWidth(80)
            btn_remove.setStyleSheet(
                f'QPushButton{{background:#2a2a2a;color:{TEXT_MID};border:1px solid #3a3a3a;'
                f'border-radius:3px;padding:3px 10px;}}'
                f'QPushButton:hover{{background:#333;}}')
            rm_row = QHBoxLayout()
            rm_row.addStretch()
            rm_row.addWidget(btn_remove)
            pv.addLayout(rm_row)

            # ── Tabs por dispositivo con sus canales ──────────────────
            ch_tabs = QTabWidget()
            ch_tabs.setStyleSheet(io_tabs.styleSheet())
            pv.addWidget(ch_tabs, stretch=1)

            for d in devs:
                n_ch = d[mode]
                tab_w = QWidget()
                tv = QVBoxLayout(tab_w)
                tv.setContentsMargins(4, 4, 4, 4)

                ch_tbl = QTableWidget(n_ch, 6)
                ch_tbl.setStyleSheet(_tbl_style)
                ch_tbl.setHorizontalHeaderLabels(
                    ['Use', 'Ch', 'Channel Name', 'Friendly Name', 'Cal. Offset', 'Level'])
                ch_tbl.horizontalHeader().setSectionResizeMode(0, ch_tbl.horizontalHeader().ResizeMode.Fixed)
                ch_tbl.horizontalHeader().setSectionResizeMode(1, ch_tbl.horizontalHeader().ResizeMode.Fixed)
                ch_tbl.horizontalHeader().setSectionResizeMode(2, ch_tbl.horizontalHeader().ResizeMode.Stretch)
                ch_tbl.horizontalHeader().setSectionResizeMode(3, ch_tbl.horizontalHeader().ResizeMode.Stretch)
                ch_tbl.horizontalHeader().setSectionResizeMode(4, ch_tbl.horizontalHeader().ResizeMode.Fixed)
                ch_tbl.horizontalHeader().setSectionResizeMode(5, ch_tbl.horizontalHeader().ResizeMode.Fixed)
                ch_tbl.setColumnWidth(0, 44)
                ch_tbl.setColumnWidth(1, 36)
                ch_tbl.setColumnWidth(4, 80)
                ch_tbl.setColumnWidth(5, 160)
                ch_tbl.verticalHeader().setVisible(False)
                ch_tbl.setEditTriggers(ch_tbl.EditTrigger.NoEditTriggers)


                for ch in range(n_ch):
                    # Use
                    chk_ch = QCheckBox()
                    chk_ch.setChecked(ch < 2)
                    cw = QWidget(); cl = QHBoxLayout(cw)
                    cl.setContentsMargins(8,0,0,0); cl.addWidget(chk_ch)
                    ch_tbl.setCellWidget(ch, 0, cw)

                    ch_tbl.setItem(ch, 1, QTableWidgetItem(str(ch + 1)))
                    ch_tbl.setItem(ch, 2, QTableWidgetItem(f'Input {ch+1}' if mode=='in' else f'Output {ch+1}'))
                    ch_tbl.setItem(ch, 3, QTableWidgetItem(f'Input {ch+1}' if mode=='in' else f'Output {ch+1}'))
                    ch_tbl.setItem(ch, 4, QTableWidgetItem('0.0'))

                    # Level bar
                    bar = QProgressBar()
                    bar.setRange(0, 100)
                    bar.setValue(30 if ch < 2 else 0)
                    bar.setTextVisible(False)
                    bar.setFixedHeight(14)
                    bar.setStyleSheet(
                        'QProgressBar{background:#111;border:none;border-radius:2px;}'
                        'QProgressBar::chunk{background:#44aa44;border-radius:2px;}')
                    ch_tbl.setCellWidget(ch, 5, bar)

                tv.addWidget(ch_tbl)
                ch_tabs.addTab(tab_w, d['name'][:20])

            # ── Botones inferiores ────────────────────────────────────
            bot = QHBoxLayout()
            btn_clear = QPushButton('Clear Settings')
            btn_cal   = QPushButton('Calibrate')
            for b in (btn_clear, btn_cal):
                b.setStyleSheet(
                    f'QPushButton{{background:#2a2a2a;color:{TEXT_MID};border:1px solid #3a3a3a;'
                    f'border-radius:3px;padding:4px 14px;}}'
                    f'QPushButton:hover{{background:#333;color:{TEXT_HI};}}')
            bot.addWidget(btn_clear)
            bot.addStretch()
            bot.addWidget(btn_cal)
            pv.addLayout(bot)

            return panel

        io_tabs.addTab(_make_io_panel('in'),  'Input')
        io_tabs.addTab(_make_io_panel('out'), 'Output')

        # ── Close ─────────────────────────────────────────────────────
        btn_close = QPushButton('Close')
        btn_close.setFixedWidth(80)
        btn_close.setStyleSheet(
            f'QPushButton{{background:#2a2a2a;color:{TEXT_MID};border:1px solid #3a3a3a;'
            f'border-radius:3px;padding:4px 14px;}}'
            f'QPushButton:hover{{background:#333;color:{TEXT_HI};}}')
        btn_close.clicked.connect(dlg.accept)
        close_row = QHBoxLayout()
        close_row.addStretch()
        close_row.addWidget(btn_close)
        root.addLayout(close_row)

        dlg.exec()

    def _build_menubar(self):
        mb_style = f"""
            QMenuBar {{
                background:{BG_TOOLBAR}; color:{TEXT_MID};
                font-size:11px; padding:1px 4px;
                border-bottom:1px solid {BORDER};
            }}
            QMenuBar::item {{ padding:3px 10px; border-radius:2px; }}
            QMenuBar::item:selected {{ background:{ACCENT}; color:#000; }}
            QMenu {{
                background:{BG_PANEL}; color:{TEXT_HI};
                border:1px solid {BORDER}; font-size:11px;
            }}
            QMenu::item {{ padding:4px 22px 4px 14px; }}
            QMenu::item:selected {{ background:{ACCENT}; color:#000; }}
            QMenu::separator {{ background:{BORDER}; height:1px; margin:3px 0; }}
        """
        mb = self.menuBar()
        mb.setStyleSheet(mb_style)

        # ── FILE ─────────────────────────────────────────────────────
        fm = mb.addMenu('File')
        fm.addAction('Save IR…',       self._save_ir_txt).setShortcut('Ctrl+Shift+I')
        fm.addAction('Save TF…',       self._save_tf_txt).setShortcut('Ctrl+Shift+T')
        fm.addAction('Save Phase…',    self._save_ph_txt).setShortcut('Ctrl+Shift+P')
        fm.addAction('Save Spectrum…', self._save_sp_txt).setShortcut('Ctrl+Shift+S')
        fm.addSeparator()
        fm.addAction('Save Graph (PNG)…', self._save_graph_png).setShortcut('Ctrl+G')
        fm.addSeparator()
        fm.addAction('Load Mic Calibration…', self._load_mic_cal).setShortcut('Ctrl+K')
        fm.addAction('Clear Mic Calibration', self._clear_mic_cal)
        fm.addSeparator()
        fm.addAction('Quit', self.close).setShortcut('Ctrl+Q')

        # ── CONFIG ───────────────────────────────────────────────────
        cm = mb.addMenu('Config')

        cm.addAction('Manage Configurations…',  self._show_manage_configs)
        cm.addSeparator()

        cm.addAction('I-O Config…',             self._show_io_config).setShortcut('Alt+A')
        cm.addAction('Measurement Config…',     self._show_measurement_config).setShortcut('Alt+G')
        cm.addAction('SPL Config…',             self._show_spl_settings).setShortcut('Ctrl+Shift+E')
        cm.addAction('Command Bar Config…',     self._show_command_bar_config)
        cm.addSeparator()

        cm.addAction('New Spectrum…',           self._new_spectrum_measurement).setShortcut('Ctrl+S')
        cm.addAction('New Spectrum Avg…',       self._new_spectrum_avg).setShortcut('Ctrl+Shift+S')
        cm.addAction('New TF…',                 self._show_new_tf_dialog).setShortcut('Ctrl+T')
        cm.addAction('New TF Avg…',             self._new_tf_avg).setShortcut('Ctrl+Shift+T')
        cm.addSeparator()

        cm.addAction('New Tab',                 self._new_tab)
        cm.addAction('Duplicate Tab',           self._duplicate_tab)
        cm.addAction('Delete Tab',              self._delete_tab)
        cm.addAction('Move Tab',                self._move_tab)
        cm.addSeparator()

        cm.addAction('Amplitude Calibration…',  self._show_amplitude_cal)

        # ── VIEW ─────────────────────────────────────────────────────
        vm = mb.addMenu('View')

        # ── Modos de medición ─────────────────────────────────────────
        from PyQt6.QtGui import QAction as _QA
        self._act_rt  = _QA('Real Time Mode',       self, checkable=True, checked=True)
        self._act_ir  = _QA('Impulse Response Mode', self, checkable=True)
        self._act_spl = _QA('SPL Mode',              self, checkable=True)
        self._act_rt.setShortcut('R')
        self._act_ir.setShortcut('I')
        self._act_spl.setShortcut('Alt+H')

        _mode_grp = [self._act_rt, self._act_ir, self._act_spl]
        def _set_mode(act, others):
            def _f(checked):
                if not checked:
                    act.setChecked(True); return   # no deseleccionar
                for o in others:
                    o.setChecked(False)
            return _f
        self._act_rt.triggered.connect( _set_mode(self._act_rt,  [self._act_ir,  self._act_spl]))
        self._act_ir.triggered.connect( _set_mode(self._act_ir,  [self._act_rt,  self._act_spl]))
        self._act_spl.triggered.connect(_set_mode(self._act_spl, [self._act_rt,  self._act_ir]))

        vm.addAction(self._act_rt)
        vm.addAction(self._act_ir)
        vm.addAction(self._act_spl)
        vm.addSeparator()

        # ── View Presets (submenu) ────────────────────────────────────
        vp_menu = vm.addMenu('View Presets')
        vp_menu.addAction('Default',  lambda: None)
        vp_menu.addAction('TF Only',  self._switch_to_tf)
        vp_menu.addAction('Spectrum', self._switch_to_spectrum)
        vm.addSeparator()

        # ── Ventanas / paneles ────────────────────────────────────────
        vm.addAction('Client Window…').setShortcut('Alt+R')

        inp_menu = vm.addMenu('Input Meters')
        inp_menu.addAction('Show Input Meters', self._show_input_meters)
        inp_menu.addAction('Large Meters',      lambda: None)

        vm.addAction('SPL Meters',   self._toggle_spl_meters).setShortcut('E')
        vm.addSeparator()

        # ── Toggles de gráfica ────────────────────────────────────────
        self._act_live_ir = _QA('Toggle Live IR', self, checkable=True, checked=True)
        self._act_live_ir.setShortcut('Meta+I')
        self._act_live_ir.triggered.connect(self._toggle_ir_panel)
        vm.addAction(self._act_live_ir)

        self._act_peak = _QA('Toggle Peak Hold', self, checkable=True)
        self._act_peak.setShortcut('P')
        self._act_peak.triggered.connect(self._toggle_peak_hold)
        vm.addAction(self._act_peak)

        self._act_coh = _QA('Toggle Coherence', self, checkable=True, checked=True)
        self._act_coh.setShortcut('C')
        self._act_coh.triggered.connect(self._toggle_coherence)
        vm.addAction(self._act_coh)

        act_target = vm.addAction('Show Target Curves')
        act_target.setShortcut('X')
        act_target.setEnabled(False)
        vm.addSeparator()

        act_decay1 = vm.addAction('Reset Decay Markers')
        act_decay1.setEnabled(False)
        act_decay2 = vm.addAction('Reset All Decay Markers')
        act_decay2.setEnabled(False)
        vm.addSeparator()

        # ── Skins (submenu) ───────────────────────────────────────────
        skins = vm.addMenu('Skins')
        skins.addAction('Dark (default)',  lambda: None).setCheckable(True)
        skins.actions()[0].setChecked(True)
        vm.addSeparator()

        # ── Barras / paneles UI ───────────────────────────────────────
        self._act_spl_bar  = _QA('Data/SPL Meter Bar', self, checkable=True, checked=True)
        self._act_ctrl_bar = _QA('Control Bar',         self, checkable=True, checked=True)
        self._act_cmd_bar  = _QA('Command Bar',         self, checkable=True)
        self._act_tab_bar  = _QA('Tab Bar',             self, checkable=True, checked=True)
        self._act_spl_met  = _QA('SPL Meter',           self, checkable=True, checked=True)
        self._act_compact  = _QA('Compact Signal Generator', self, checkable=True)

        self._act_spl_bar.setShortcut('B')
        self._act_ctrl_bar.setShortcut('O')
        self._act_cmd_bar.setShortcut('U')
        self._act_tab_bar.setShortcut('A')
        self._act_spl_met.setShortcut('Alt+K')

        self._act_spl_bar.triggered.connect( lambda c: self._toggle_ui_bar('spl_bar', c))
        self._act_ctrl_bar.triggered.connect(lambda c: self._toggle_ui_bar('ctrl_bar', c))
        self._act_spl_met.triggered.connect( lambda c: self._toggle_ui_bar('spl_meter', c))

        for a in (self._act_spl_bar, self._act_ctrl_bar, self._act_cmd_bar,
                  self._act_tab_bar, self._act_spl_met, self._act_compact):
            vm.addAction(a)
        vm.addSeparator()

        vm.addAction('Toggle SPL/Clock',         self._toggle_spl_clock).setShortcut('K')
        vm.addAction('Toggle Data/SPL Meter Bar', lambda: self._act_spl_bar.trigger()).setShortcut('Alt+E')
        vm.addSeparator()

        vm.addAction('Toggle Settings Panel',    self._on_toggle_settings).setShortcut('Ctrl+Right')
        vm.addAction('Toggle Save Panel',        self._on_toggle_save).setShortcut('Ctrl+Left')

        # ── COMMAND ──────────────────────────────────────────────────
        xm = mb.addMenu('Command')
        xm.addAction('Start  [Space]', self._on_start).setShortcut('Space')
        xm.addAction('Stop',           self._on_stop)
        xm.addSeparator()
        xm.addAction('Find Delay  [D]',   self._on_find_delay).setShortcut('D')
        xm.addAction('Reset Delay  [R]',  self._on_delay_reset).setShortcut('R')
        xm.addAction('Freeze  [F]',       lambda: self.btn_freeze_p.click()).setShortcut('F')
        xm.addSeparator()
        xm.addAction('Save Trace  [P]',   self._save_trace).setShortcut('P')
        xm.addAction('Toggle Noise  [G]', lambda: self.btn_noise_p.click()).setShortcut('G')

        # ── HELP ─────────────────────────────────────────────────────
        hm = mb.addMenu('Help')
        hm.addAction('Keyboard Shortcuts', self._show_shortcuts)
        hm.addAction('About Coherence…',   self._show_about)

        # ── Shortcuts globales — ApplicationShortcut para que funcionen ──
        # aunque el canvas matplotlib tenga el foco de teclado.
        # En macOS: Qt Ctrl = Command(⌘), Qt Meta = Control(⌃)
        from PyQt6.QtCore import Qt as _Qt

        def _sc(key, slot):
            s = QShortcut(QKeySequence(key), self)
            s.setContext(_Qt.ShortcutContext.ApplicationShortcut)
            s.activated.connect(slot)
            return s

        # Navegación de vista
        _sc('S',      self._switch_to_spectrum)
        _sc('T',      self._switch_to_tf)
        # Transport
        _sc('Space',  self._on_start)
        _sc('D',      self._on_find_delay)
        _sc('F',      lambda: self.btn_freeze_p.click())
        _sc('G',      lambda: self.btn_noise_p.click())
        _sc('P',      self._save_trace)
        # IR toggle — Cmd+I en macOS = Ctrl+I en Qt
        _sc('Ctrl+I', self._toggle_ir_panel)
        # SPL / misc
        _sc('E',      self._toggle_spl_meters)
        _sc('K',      self._toggle_spl_clock)

    def _build_statusbar(self):
        self.sb = QStatusBar()
        self.setStatusBar(self.sb)
        self.sb.showMessage('Coherence v0.2  —  GPL v3')

    # ── Populate devices ──────────────────────────────────────────────

    def _populate_devices(self):
        devices = AudioEngine.list_devices()
        self._dev_in_ids   = []
        self._dev_out_ids  = []
        self._dev_noise_ids = []

        # Marcas de fabricantes que CoreAudio/PortAudio a veces reporta con 0 canales
        # aunque funcionan. Los incluimos siempre como candidatos de entrada Y salida.
        _FORCE_SHOW = ('apogee', 'symphony', 'duet', 'quartet', 'ensemble',
                       'ua ', 'universal audio', 'apollo', 'uad',
                       'rme', 'motu', 'focusrite', 'scarlett', 'clarett',
                       'presonus', 'avid', 'pro tools', 'dante')

        for d in devices:
            name_low  = d['name'].lower()
            force     = any(k in name_low for k in _FORCE_SHOW)
            label     = f"[{d['id']}] {d['name'][:22]}"
            # Marcar con ★ los que PortAudio declara 0 canales pero forzamos
            if force and d['in'] == 0:
                label = f"[{d['id']}] ★{d['name'][:21]}"

            if d['in'] >= 1 or force:
                self.cmb_dev_in.addItem(label)
                self._dev_in_ids.append(d['id'])
                if d['id'] == self.engine.dev_in:
                    self.cmb_dev_in.setCurrentIndex(self.cmb_dev_in.count() - 1)
            if d['out'] >= 1 or force:
                self.cmb_dev_out.addItem(label)
                self._dev_out_ids.append(d['id'])
                if d['id'] == self.engine.dev_out:
                    self.cmb_dev_out.setCurrentIndex(self.cmb_dev_out.count() - 1)
                self.cmb_noise_dev.addItem(label)
                self._dev_noise_ids.append(d['id'])
                if d['id'] == self.engine.dev_out:
                    self.cmb_noise_dev.setCurrentIndex(self.cmb_noise_dev.count() - 1)

        # "Configure…" como primera opción del dropdown Output
        self.cmb_noise_dev.insertItem(0, 'Configure…')
        self._dev_noise_ids.insert(0, -1)   # -1 = acción especial
        # Ajustar selección actual (se desplazó por el insert)
        cur = self.cmb_noise_dev.currentIndex()
        self.cmb_noise_dev.setCurrentIndex(max(1, cur + 1))

        self.cmb_dev_in.currentIndexChanged.connect(self._on_dev_in)
        self.cmb_dev_out.currentIndexChanged.connect(self._on_dev_out)
        self.cmb_noise_dev.currentIndexChanged.connect(self._on_noise_dev)

        # Actualizar labels de dispositivo en los sub-paneles Spectrum/Sgram
        if devices and self._dev_in_ids:
            cur_name = next(
                (d['name'][:20] for d in devices if d['id'] == self.engine.dev_in),
                '—')
            if hasattr(self, '_lbl_sp_dev'):
                self._lbl_sp_dev.setText(cur_name)
            if hasattr(self, '_lbl_sg_dev'):
                self._lbl_sg_dev.setText(cur_name)

    # ── Handlers ──────────────────────────────────────────────────────

    def _on_start(self):
        try:
            self.engine.start()
        except Exception as exc:
            self.sb.showMessage(f'⚠  Error al abrir stream: {exc}', 8000)
            return
        self.timer.start()
        self._set_running()
        self._silent_ticks = 0   # contador para detección de señal cero
        # Programar chequeo de señal 3 s después de arrancar
        QTimer.singleShot(3000, self._check_signal_present)

    def _check_signal_present(self):
        """
        Llamado 3 s después de arrancar el stream.
        Si todos los buffers siguen en silencio absoluto → warning de permisos macOS.
        """
        if not self.engine.running:
            return
        buf = self.engine.buf_spl.read(self.engine.nperseg)
        rms = float(np.sqrt(np.mean(buf ** 2)))
        if rms < 1e-7:   # < −140 dBFS → señal completamente nula (zeros)
            self._show_no_signal_warning()

    def _show_no_signal_warning(self):
        dlg = QMessageBox(self)
        dlg.setWindowTitle('Sin señal de entrada')
        dlg.setIcon(QMessageBox.Icon.Warning)
        dlg.setText(
            '<b>El stream de audio está abierto pero no llega señal.</b><br><br>'
            'En macOS esto ocurre cuando Python no tiene permiso de micrófono.<br><br>'
            '<b>Cómo arreglarlo:</b><br>'
            '1. Abre <b>System Settings → Privacy &amp; Security → Microphone</b><br>'
            '2. Activa el permiso para <b>Python</b> (o Terminal / tu app).<br>'
            '3. Reinicia Coherence.<br><br>'
            'Si el permiso ya está activo, verifica que el canal seleccionado '
            'en TF Engine Config coincida con el canal físico de la interfaz.'
        )
        dlg.setStandardButtons(QMessageBox.StandardButton.Ok)
        dlg.setStyleSheet(
            f'QMessageBox{{background:{BG_APP};color:{TEXT_HI};font-size:11px;}}'
            f'QPushButton{{background:#2a2a2a;color:{TEXT_HI};padding:4px 12px;'
            f'border:1px solid {BORDER};border-radius:3px;}}'
        )
        dlg.exec()

    def _on_stop(self):
        self.timer.stop()
        self.engine.stop()
        # NO se borran las gráficas — la última medición queda visible (freeze)
        self._set_stopped()



    def _on_channels(self):
        # Sincronizar canales de medición desde TF engines
        if hasattr(self, '_tf_engines'):
            for i, eng in enumerate(self._tf_engines):
                if i < len(self.engine._ch_meas_list):
                    self.engine._ch_meas_list[i] = eng.ch_m
                else:
                    self.engine.add_meas_channel(eng.ch_m)
            # Referencia: usar el R del primer engine
            if self._tf_engines:
                self.engine.ch_ref = self._tf_engines[0].ch_r
        # SPL channel
        if hasattr(self, 'spn_spl_ch'):
            self.engine.ch_spl = self.spn_spl_ch.value()
        # Spectrum channel
        if hasattr(self, 'spn_ch_spec'):
            self.engine.ch_spec = self.spn_ch_spec.value()
        if self.engine.running:
            self._safe_restart()

    def _set_smooth_fraction(self, frac: int):
        """Sincroniza el suavizado TF desde el botón overlay del canvas."""
        if not hasattr(self, '_smooth_values') or not hasattr(self, 'cmb_smooth'):
            return
        if frac in self._smooth_values:
            idx = self._smooth_values.index(frac)
            self.cmb_smooth.setCurrentIndex(idx)
        # Ampliar la lista si se pide un valor nuevo (e.g. 48)
        elif frac == 48:
            if 48 not in self._smooth_values:
                self._smooth_values.append(48)
                self.cmb_smooth.addItem('1/48')
            self.cmb_smooth.setCurrentIndex(self._smooth_values.index(48))
        # Actualizar el botón del canvas para que refleje el estado del combo
        if hasattr(self, 'canvas_meas') and hasattr(self.canvas_meas, '_smooth_btn'):
            self.canvas_meas._current_smooth = frac

    def _safe_restart(self):
        """
        Reinicia el stream con manejo de error visible.
        Si falla: detiene timer, muestra mensaje en status bar.
        """
        # Resetear acumuladores RunningTF — nueva interfaz, nuevo contexto
        for rtf in getattr(self, '_running_tfs', []):
            rtf.reset()
        try:
            self.engine.restart()
            if not self.engine.running:
                raise RuntimeError('El stream no arrancó.')
            self._set_running()
            self.sb.showMessage('▶  Stream reiniciado', 3000)
        except Exception as exc:
            self.timer.stop()
            self._set_stopped()
            self.sb.showMessage(f'⚠  Error al reiniciar stream: {exc}', 10000)

    def _on_dev_in(self, idx):
        if 0 <= idx < len(self._dev_in_ids):
            self.engine.dev_in = self._dev_in_ids[idx]
            name = self.cmb_dev_in.itemText(idx)
            if hasattr(self, '_lbl_sp_dev'): self._lbl_sp_dev.setText(name[:18])
            if hasattr(self, '_lbl_sg_dev'): self._lbl_sg_dev.setText(name[:18])
            self._save_prefs()          # recordar elección
            if self.engine.running:
                self._safe_restart()

    def _on_dev_out(self, idx):
        if 0 <= idx < len(self._dev_out_ids):
            self.engine.dev_out = self._dev_out_ids[idx]
            self._save_prefs()          # recordar elección
            if self.engine.running:
                self._safe_restart()

    def _on_noise_panel(self, checked):
        self.engine.noise_on = checked
        # Sincronizar estado en todos los botones de noise
        for b in self._all_noise_btns:
            b.setChecked(checked)
        # Si el stream no está corriendo, mostrarlo en status bar
        if checked and not self.engine.running:
            self.sb.showMessage('▶  Presiona Play para activar el generador', 4000)

    def _on_signal_type(self, idx):
        # Noise/Pink/White/Tone/Sweep — primer item es alias de pink
        types = ['pink', 'pink', 'white', 'tone', 'sweep']
        t = types[idx] if idx < len(types) else 'pink'
        self.engine.signal_type = t
        is_tone = (t == 'tone')
        if hasattr(self, 'lbl_tone_freq'): self.lbl_tone_freq.setVisible(is_tone)
        if hasattr(self, 'spn_tone_freq'): self.spn_tone_freq.setVisible(is_tone)

    def _on_gain_panel(self, val):
        gain = val / 100.0
        self.engine.gain = gain
        db = 20 * np.log10(gain + 1e-9)
        db_str = f'{db:.0f} dB'
        if hasattr(self, 'lbl_gain_p'):
            self.lbl_gain_p.setText(db_str)

    def _on_noise_channel(self, val):
        self.engine.noise_ch_start = val

    def _on_noise_dev(self, idx):
        if not (0 <= idx < len(self._dev_noise_ids)):
            return
        dev_id = self._dev_noise_ids[idx]
        if dev_id == -1:
            # "Configure…" → abrir I-O Config y volver al último output válido
            # Bloquear señal para no re-entrar
            self.cmb_noise_dev.blockSignals(True)
            self.cmb_noise_dev.setCurrentIndex(
                next((i for i in range(1, self.cmb_noise_dev.count())
                      if self._dev_noise_ids[i] == self.engine.dev_out), 1))
            self.cmb_noise_dev.blockSignals(False)
            self._show_io_config()
            return
        self.engine.dev_out = dev_id
        if self.engine.running:
            self._safe_restart()

    def _on_delay_reset(self):
        """Reset global (tecla R) → resetea todos los engines."""
        self._delay_comp_ms = 0.0
        self.lbl_delay_full.setText('—')
        self.canvas_meas.set_delay_ref(0.0)
        if hasattr(self, '_tf_engines'):
            for eng in self._tf_engines:
                eng.reset_delay()

    def _on_freeze_panel(self, checked):
        self._frozen = checked
        for b in self._all_freeze_btns:
            b.setChecked(checked)

    def _on_find_delay(self):
        """
        Shortcut global (tecla D) → opera sobre el primer TF engine.
        """
        self._on_find_delay_for_engine(0)

    # ── Refresh helpers — Spectrum y Spectrogram independientes ──────

    def _refresh_spectrum(self):
        """Spectrum: itera todos los Spectrum Engines con RunningTF."""
        if not self._sp_engines:
            return

        n_avg = 6
        if hasattr(self, 'cmb_avg_spec') and hasattr(self, '_avg_spec_values'):
            n_avg = self._avg_spec_values[self.cmb_avg_spec.currentIndex()]

        need_draw = False
        for i, eng in enumerate(self._sp_engines):
            if not eng.get('active', True):
                continue
            sig = self.engine.get_buffer_spec(i)
            if np.max(np.abs(sig)) < 1e-7:
                continue

            rtf = eng['rtf']
            if (rtf.nperseg != self.engine.nperseg or rtf.fs != self.engine.fs):
                eng['rtf'] = RunningTF(
                    nperseg=self.engine.nperseg,
                    fs=self.engine.fs,
                    n_averages=n_avg,
                )
                rtf = eng['rtf']

            rtf.set_n_averages(n_avg)
            rtf.push(sig, sig)
            if not rtf._ready:
                continue

            self.canvas_spec.update_sp_engine(eng['canvas_idx'], rtf.freqs, rtf.Gxx)
            # ── Panel secundario — Spectrum ───────────────────────────
            if (self._secondary_panel is not None and
                    self._secondary_panel._current_view == 'Spectrum'):
                sp2 = self._secondary_panel.canvas_spec
                # Agregar líneas de engine al canvas secundario si faltan
                while len(sp2._sp_eng_lines) <= eng['canvas_idx']:
                    sp2.add_sp_engine_line(eng.get('color', ENGINE_PALETTE[i % len(ENGINE_PALETTE)]))
                sp2.update_sp_engine(eng['canvas_idx'], rtf.freqs, rtf.Gxx)
            need_draw = True

        if need_draw:
            self.canvas_spec.draw_idle()
            if (self._secondary_panel is not None and
                    self._secondary_panel._current_view == 'Spectrum'):
                self._secondary_panel.canvas_spec.draw_idle()

    def _refresh_spectrogram(self):
        """Spectrogram: usa ch_spec, mide siempre sin importar si hay REF."""
        sig = self.engine.get_buffer_spec()
        if np.max(np.abs(sig)) < 1e-7:
            return
        freqs, Gspec, _, _ = welch_spectra(
            sig, sig, nperseg=self.engine.nperseg, fs=self.engine.fs)
        self.canvas_sgram.update_spectrogram(freqs, Gspec)
        # Panel secundario — Spectrograph
        if (self._secondary_panel is not None and
                self._secondary_panel._current_view == 'Spectrograph'):
            self._secondary_panel.canvas_sgram.update_spectrogram(freqs, Gspec)

    # ── Refresh (QTimer) ──────────────────────────────────────────────

    def _refresh(self):
        try:
            self._refresh_inner()
        except Exception as exc:
            # Evitar que un error en el timer trabe la UI
            self.sb.showMessage(f'⚠  refresh error: {exc}', 3000)

    def _refresh_inner(self):
        # Reloj
        self.lbl_clock.setText(
            '  ' + QDateTime.currentDateTime().toString('hh:mm:ss') + '  ')

        if self._frozen:
            return

        # Si el engine no está corriendo, no hay nada que medir
        if not self.engine.running:
            return

        # Refrescar según la vista activa
        view = getattr(self, '_current_view', 'magnitude')

        # Si el panel secundario está en Spectrum/Spectrograph, refrescarlo siempre
        if self._secondary_panel is not None:
            sv = self._secondary_panel._current_view
            if sv == 'Spectrum':
                self._refresh_spectrum()
            elif sv == 'Spectrograph':
                self._refresh_spectrogram()

        if view == 'spectrum':
            if self._secondary_panel is None or self._secondary_panel._current_view != 'Spectrum':
                self._refresh_spectrum()
            return
        if view == 'spectrograph':
            if self._secondary_panel is None or self._secondary_panel._current_view != 'Spectrograph':
                self._refresh_spectrogram()
            return

        # ── Transfer Function ─────────────────────────────────────────
        # Leer solo nperseg muestras (el bloque más reciente) para RunningTF
        x = self.engine.buf_ref.read(self.engine.nperseg)
        rms_ref_db = 20.0 * np.log10(float(np.sqrt(np.mean(x ** 2))) + 1e-9)

        # Level gate: si la referencia está por debajo del noise floor,
        # NO actualizamos RunningTF — mantenemos la última medición buena.
        # (Igual que SMAART: la curva se "congela" pero no desaparece.)
        _signal_ok = rms_ref_db > -70.0

        rms_ref = rms_ref_db
        eps     = 1e-12

        smooth_frac = self._smooth_values[self.cmb_smooth.currentIndex()]
        thresh      = self.spn_thresh.value()

        # ── SPL meter (canal dedicado) ────────────────────────────────
        if not self.engine.running:
            # Stream caído: resetear display
            _dim = f'color:{TEXT_DIM};font-size:56px;font-weight:bold;background:transparent;'
            _bst = 'QProgressBar{background:#111;border:none;border-radius:3px;}QProgressBar::chunk{background:#333;border-radius:3px;}'
            for _lv, _br in zip(
                    getattr(self, '_spl_labels', [self.lbl_spl_val]),
                    getattr(self, '_spl_bars',   [self.bar_spl])):
                _lv.setText('—')
                _lv.setStyleSheet(_dim)
                _br.setValue(0)
                _br.setStyleSheet(_bst)
        else:
            spl_data  = self.engine.get_buffer_spl()
            rms_spl_f = 20.0 * np.log10(float(np.sqrt(np.mean(spl_data ** 2))) + 1e-9)
            spl_db    = rms_spl_f + self._spl_offset_db

            if spl_db >= self._spl_clip_db:
                spl_color = '#ef5350'
            elif spl_db >= self._spl_warn_db:
                spl_color = '#ffb74d'
            else:
                spl_color = '#66bb6a'

            bar_pct   = max(0, min(100, int((spl_db - 50) / 80.0 * 100)))
            spl_style = (f'color:{spl_color};font-size:56px;font-weight:bold;'
                         f'background:transparent;letter-spacing:-1px;')
            bar_style = (f'QProgressBar{{background:#111;border:none;border-radius:3px;}}'
                         f'QProgressBar::chunk{{background:{spl_color};border-radius:3px;}}')
            # Actualizar max
            if spl_db > getattr(self, '_spl_max_db', -200.0):
                self._spl_max_db = spl_db
                if hasattr(self, 'lbl_spl_max'):
                    self.lbl_spl_max.setText(f'Max  {spl_db:.1f}')
            for _lv, _br in zip(
                    getattr(self, '_spl_labels', [self.lbl_spl_val]),
                    getattr(self, '_spl_bars',   [self.bar_spl])):
                _lv.setText(f'{spl_db:.1f}')
                _lv.setStyleSheet(spl_style)
                _br.setValue(bar_pct)
                _br.setStyleSheet(bar_style)

        # ── TF engines (barras de nivel) ──────────────────────────────
        for i, eng in enumerate(self._tf_engines):
            y_i     = self.engine.get_buffer_meas(i)
            rms_m_i = 20.0 * np.log10(float(np.sqrt(np.mean(y_i ** 2))) + 1e-9)
            eng.update_levels(rms_m_i, rms_ref)

        # ── Análisis TF — solo si hay al menos un engine creado ──────
        if not self._tf_engines:
            return

        # Sincronizar n_averages en todos los RunningTF
        n_avg = self.engine.n_averages
        for rtf in getattr(self, '_running_tfs', []):
            rtf.set_n_averages(n_avg)

        y0     = self.engine.get_buffer_meas(0)
        delay0 = self._tf_engines[0]._delay_comp_ms

        rtf0 = self._running_tfs[0] if self._running_tfs else None

        # Solo actualizamos si la señal supera el noise floor (level gate)
        if rtf0 is not None:
            if _signal_ok:
                rtf0.push(x, y0)
            # Si RunningTF no tiene datos aún, no hay nada que mostrar
            if not rtf0.ready:
                return
            freqs, gamma2, mag_db, phase_deg, gxx_db = rtf0.get_tf(
                delay_comp_s    = delay0 / 1000.0,
                smooth_fraction = smooth_frac,
            )
        else:
            if not _signal_ok:
                return
            freqs, gamma2, mag_db, phase_deg, gxx_db, _ = compute_analysis(
                x, y0, nperseg=self.engine.nperseg, fs=self.engine.fs,
                smooth_fraction=smooth_frac, delay_comp_s=delay0/1000.0)

        # Aplicar offset de normalización del engine 0
        mag_db = mag_db + self._tf_engines[0]._gain_offset_db

        if self._mic_cal is not None:
            cal_f, cal_db = self._mic_cal
            mag_db = mag_db - np.interp(freqs, cal_f, cal_db, left=0.0, right=0.0)

        # Engine 1 → CH2 line
        freqs2 = mag_db2 = phase_deg2 = gamma2_2 = None
        if len(self._tf_engines) >= 2 and self._tf_engines[1].active:
            y1     = self.engine.get_buffer_meas(1)
            delay1 = self._tf_engines[1]._delay_comp_ms
            rtf1   = self._running_tfs[1] if len(self._running_tfs) > 1 else None
            if rtf1 is not None:
                if _signal_ok:
                    rtf1.push(x, y1)
                if rtf1.ready:
                    freqs2, gamma2_2, mag_db2, phase_deg2, _ = rtf1.get_tf(
                        delay_comp_s    = delay1 / 1000.0,
                        smooth_fraction = smooth_frac,
                    )
            else:
                if _signal_ok:
                    freqs2, gamma2_2, mag_db2, phase_deg2, _, _ = compute_analysis(
                        x, y1, nperseg=self.engine.nperseg, fs=self.engine.fs,
                        smooth_fraction=smooth_frac, delay_comp_s=delay1/1000.0)
            if mag_db2 is not None:
                mag_db2 = mag_db2 + self._tf_engines[1]._gain_offset_db
            if mag_db2 is not None and self._mic_cal is not None:
                mag_db2 = mag_db2 - np.interp(freqs2, cal_f, cal_db, left=0.0, right=0.0)

        # ── Promedio de TODOS los engines activos (N engines) ────────────
        # Recolectar resultados de engines 0 y 1 ya calculados
        _all_lin  = [10.0 ** (mag_db / 20.0)]
        _all_ph   = [phase_deg]
        _all_g2   = [gamma2]
        if mag_db2 is not None:
            _all_lin.append(10.0 ** (mag_db2 / 20.0))
            _all_ph.append(phase_deg2)
            _all_g2.append(gamma2_2)

        # Engines 2+ (si existen y están activos)
        for _ei in range(2, len(self._tf_engines)):
            if not self._tf_engines[_ei].active:
                continue
            _y_ei    = self.engine.get_buffer_meas(_ei)
            _delay_ei = self._tf_engines[_ei]._delay_comp_ms
            _rtf_ei   = self._running_tfs[_ei] if _ei < len(self._running_tfs) else None
            if _rtf_ei is not None:
                if _signal_ok:
                    _rtf_ei.push(x, _y_ei)
                if _rtf_ei.ready:
                    _, _g2ei, _mei, _phei, _ = _rtf_ei.get_tf(
                        delay_comp_s=_delay_ei / 1000.0,
                        smooth_fraction=smooth_frac)
                    _gain_ei = self._tf_engines[_ei]._gain_offset_db
                    _mei = _mei + _gain_ei
                    if self._mic_cal is not None:
                        _mei = _mei - np.interp(freqs, cal_f, cal_db, left=0.0, right=0.0)
                    _all_lin.append(10.0 ** (_mei / 20.0))
                    _all_ph.append(_phei)
                    _all_g2.append(_g2ei)

        mag_avg = ph_avg = g2_avg = None
        if len(_all_lin) > 1:
            mag_avg = 20.0 * np.log10(np.mean(_all_lin, axis=0) + eps)
            ph_avg  = np.mean(_all_ph, axis=0)
            g2_avg  = np.mean(_all_g2, axis=0)

        # ── IR via IFFT coherence-weighted — usa el engine seleccionado ─
        sel_idx  = getattr(self, '_selected_engine_idx', 0)
        sel_idx  = min(sel_idx, len(self._running_tfs) - 1)
        rtf_sel  = self._running_tfs[sel_idx] if self._running_tfs else rtf0
        y_sel    = self.engine.get_buffer_meas(sel_idx)
        if _signal_ok and rtf_sel is not None and rtf_sel != rtf0:
            rtf_sel.push(x, y_sel)   # ya empujado si sel_idx==0

        if rtf_sel is not None and rtf_sel.ready:
            H_ir        = rtf_sel.Gxy / (rtf_sel.Gxx + eps)
            gamma2_full = np.abs(rtf_sel.Gxy)**2 / (rtf_sel.Gxx * rtf_sel.Gyy + eps)
        else:
            _, Gxx_ir, Gyy_ir, Gxy_ir = welch_spectra(
                x, y_sel, nperseg=self.engine.nperseg, fs=self.engine.fs)
            H_ir        = Gxy_ir / (Gxx_ir + eps)
            gamma2_full = np.abs(Gxy_ir)**2 / (Gxx_ir * Gyy_ir + eps)
        H_w  = H_ir * np.sqrt(np.clip(gamma2_full, 0.0, 1.0))
        N_ir = self.engine.nperseg * 4
        ir   = np.fft.irfft(H_w, n=N_ir)
        ir   = np.fft.fftshift(ir)          # centra t=0 → pico queda en t_rel=0 ms
        ir   = ir / (np.max(np.abs(ir)) + eps)

        self.canvas_meas._fs      = self.engine.fs
        self.canvas_meas._nperseg = self.engine.nperseg  # necesario para eje IR correcto

        mask    = (freqs >= 20) & (freqs <= 20000)
        avg_coh = float(np.mean(gamma2[mask]))

        # ── Update canvas ─────────────────────────────────────────────
        self.canvas_meas.update_plots(freqs, gamma2, mag_db, phase_deg,
                                      gxx_db, ir, coh_thresh=thresh)
        if freqs2 is not None and self._show_ch2:
            self.canvas_meas.update_ch2(freqs2, mag_db2, phase_deg2,
                                        gamma2_2, coh_thresh=thresh)
        else:
            self.canvas_meas.update_ch2(None, None, None, None)

        if mag_avg is not None and self._show_avg:
            self.canvas_meas.update_avg(freqs, mag_avg, ph_avg,
                                        g2_avg, coh_thresh=thresh)
        else:
            self.canvas_meas.update_avg(None, None, None, None)

        # ── Panel secundario — Magnitude / Phase ──────────────────────
        if (self._secondary_panel is not None and
                self._secondary_panel._current_view in ('Magnitude', 'Phase')):
            cm2 = self._secondary_panel.canvas_meas
            cm2._fs = self.engine.fs
            cm2.update_plots(freqs, gamma2, mag_db, phase_deg,
                             gxx_db, ir, coh_thresh=thresh)
            if freqs2 is not None:
                cm2.update_ch2(freqs2, mag_db2, phase_deg2,
                               gamma2_2, coh_thresh=thresh)
            else:
                cm2.update_ch2(None, None, None, None)
            if mag_avg is not None:
                cm2.update_avg(freqs, mag_avg, ph_avg, g2_avg, coh_thresh=thresh)
            else:
                cm2.update_avg(None, None, None, None)

        # ── γ² display ───────────────────────────────────────────────
        mark  = '✓  GOOD' if avg_coh > 0.9 else ('~  OK' if avg_coh > 0.7 else '⚠  LOW')
        ccolor = '#66bb6a' if avg_coh > 0.9 else ('#ffb74d' if avg_coh > 0.7 else '#ef5350')
        self.lbl_coh_full.setText(f'{avg_coh*100:.0f}%')
        self.lbl_gamma2.setText(f'{avg_coh*100:.0f}%')
        self.lbl_gamma2.setStyleSheet(
            f'color:{ccolor};font-size:20px;font-weight:bold;'
            f'padding:0;letter-spacing:1px;background:transparent;')
        self.lbl_gamma2_mark.setText(mark)
        self.lbl_gamma2_mark.setStyleSheet(
            f'font-size:7px;color:{ccolor};letter-spacing:1px;background:transparent;')
        self.bar_coh.setValue(int(avg_coh * 100))

        self.sb.showMessage(
            f'TF1 M={self.engine.ch_meas}  REF={self.engine.ch_ref}   |   '
            f'γ² avg={avg_coh*100:.0f}%   |   '
            f'n={self.engine.n_averages}   |   '
            f'Δf={self.engine.fs/self.engine.nperseg:.1f} Hz/bin'
        )

    # ── Estados ───────────────────────────────────────────────────────

    def _set_running(self):
        for b in self._all_start_btns:
            b.setDisabled(True)
        for b in self._all_stop_btns:
            b.setEnabled(True)

    def _set_stopped(self):
        for b in self._all_start_btns:
            b.setEnabled(True)
        for b in self._all_stop_btns:
            b.setDisabled(True)

    # ── Guardar TXT ───────────────────────────────────────────────────

    def _default_path(self, suffix):
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        folder = os.path.expanduser('~/Desktop')
        return os.path.join(folder, f'coherence_{suffix}_{ts}.txt')

    def _save_txt(self, default_name, header, data_cols):
        path, _ = QFileDialog.getSaveFileName(
            self, 'Guardar TXT', default_name, 'Text files (*.txt);;All files (*)')
        if not path:
            return
        comp = self._delay_comp_ms
        meta = (f'# Coherence v0.1\n'
                f'# {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n'
                f'# Avg: {self.engine.n_averages}  |  '
                f'Smooth: {self.cmb_smooth.currentText()}  |  '
                f'Delay comp: {comp:.3f} ms\n')
        with open(path, 'w') as f:
            f.write(meta)
            f.write('# ' + '\t'.join(header) + '\n')
            np.savetxt(f, np.column_stack(data_cols), delimiter='\t', fmt='%.6f')
        self.sb.showMessage(f'Guardado: {os.path.basename(path)}', 4000)

    def _save_ir_txt(self):
        ir = self.canvas_meas._last_ir
        if ir is None:
            return
        t = np.arange(len(ir)) / self.engine.fs * 1000.0 - self.canvas_meas._delay_ref_ms
        self._save_txt(self._default_path('IR'),
                       ['time_ms', 'amplitude'], [t, ir])

    def _save_tf_txt(self):
        f = self.canvas_meas._last_freqs
        if f is None:
            return
        mask = (f >= 20) & (f <= 20000)
        self._save_txt(self._default_path('TF'),
                       ['freq_hz', 'mag_db', 'coherence'],
                       [f[mask],
                        self.canvas_meas._last_mag_db[mask],
                        self.canvas_meas._last_gamma2[mask]])

    def _save_ph_txt(self):
        f = self.canvas_meas._last_freqs
        if f is None:
            return
        mask = (f >= 20) & (f <= 20000)
        ph_wrap = ((self.canvas_meas._last_phase_deg[mask] + 180) % 360) - 180
        self._save_txt(self._default_path('PH'),
                       ['freq_hz', 'phase_deg', 'coherence'],
                       [f[mask], ph_wrap, self.canvas_meas._last_gamma2[mask]])

    def _save_sp_txt(self):
        f = self.canvas_meas._last_freqs
        if f is None:
            return
        mask = (f >= 20) & (f <= 20000)
        eps = 1e-12
        x, y = self.engine.get_buffers()
        from .dsp import welch_spectra
        freqs2, Gxx, Gyy, _ = welch_spectra(x, y, self.engine.nperseg, fs=self.engine.fs)
        gxx_db = 10 * np.log10(Gxx + eps)
        gyy_db = 10 * np.log10(Gyy + eps)
        m2 = (freqs2 >= 20) & (freqs2 <= 20000)
        self._save_txt(self._default_path('SP'),
                       ['freq_hz', 'ref_dbfs', 'meas_dbfs'],
                       [freqs2[m2], gxx_db[m2], gyy_db[m2]])

    # ── Calibración de micrófono ──────────────────────────────────────

    def _load_mic_cal(self):
        """Carga archivo .txt con columnas freq_hz / correction_dB."""
        path, _ = QFileDialog.getOpenFileName(
            self, 'Cargar calibración de micrófono',
            os.path.expanduser('~'),
            'Text files (*.txt);;CSV (*.csv);;All files (*)'
        )
        if not path:
            return
        try:
            data = np.loadtxt(path, comments='#')
            if data.ndim != 2 or data.shape[1] < 2:
                raise ValueError('El archivo debe tener 2 columnas: freq_hz  correction_dB')
            self._mic_cal      = (data[:, 0], data[:, 1])
            self._mic_cal_name = os.path.basename(path)
            self.sb.showMessage(
                f'✓  Mic cal cargada: {self._mic_cal_name}  '
                f'({len(data)} puntos, '
                f'{data[0,0]:.0f}–{data[-1,0]:.0f} Hz)', 6000)
        except Exception as e:
            QMessageBox.critical(self, 'Error al cargar calibración', str(e))

    def _clear_mic_cal(self):
        self._mic_cal      = None
        self._mic_cal_name = ''
        self.sb.showMessage('Calibración de micrófono eliminada', 3000)

    # ── Measurement Config ────────────────────────────────────────────

    def _show_meas_config(self):
        dlg = MeasurementConfigDialog(self.engine, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            vals = dlg.values()
            was  = self.engine.running
            if was:
                self.engine.stop()
                self.timer.stop()
            self.engine.nperseg = vals['nperseg']
            self.engine.fs      = vals['fs']
            self.sb.showMessage(
                f"Config: FFT={vals['nperseg']}  fs={vals['fs']} Hz  "
                f"Δf={vals['fs']/vals['nperseg']:.1f} Hz/bin", 5000)
            if was:
                try:
                    self.engine.start()
                    self.timer.start()
                except Exception as exc:
                    self.sb.showMessage(f'⚠  {exc}', 6000)

    # ── Guardar gráfica PNG ───────────────────────────────────────────

    def _save_graph_png(self):
        tab = self.tabs.currentIndex()
        canvas_map = {
            0: (self.canvas_meas,  'TF_Phase_IR'),
            1: (self.canvas_spec,  'Spectrum'),
            2: (self.canvas_sgram, 'Spectrogram'),
        }
        canvas, name = canvas_map.get(tab, (self.canvas_meas, 'graph'))
        ts   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(os.path.expanduser('~/Desktop'),
                            f'coherence_{name}_{ts}.png')
        path, _ = QFileDialog.getSaveFileName(
            self, 'Guardar gráfica', path, 'PNG (*.png);;PDF (*.pdf)')
        if not path:
            return
        canvas.fig.savefig(path, dpi=150, facecolor=BG_PANEL,
                           bbox_inches='tight')
        self.sb.showMessage(f'Gráfica guardada: {os.path.basename(path)}', 4000)

    # ── About / Shortcuts ─────────────────────────────────────────────

    # ── Config menu stubs / implementaciones ──────────────────────────

    # ── View menu actions ─────────────────────────────────────────────

    def _toggle_peak_hold(self, checked: bool):
        """P — peak hold en el spectrum canvas."""
        if hasattr(self.canvas_spec, 'set_peak_hold'):
            self.canvas_spec.set_peak_hold(checked)
        self.sb.showMessage(f'Peak Hold {"ON" if checked else "OFF"}', 2000)

    def _toggle_coherence(self, checked: bool):
        """C — muestra/oculta la curva de coherencia en TF."""
        if hasattr(self, 'canvas_meas'):
            self.canvas_meas.line_coh.set_visible(checked)
            if hasattr(self.canvas_meas, '_coh_fill'):
                self.canvas_meas._coh_fill.set_visible(checked)
            self.canvas_meas.draw_idle()
        self.sb.showMessage(f'Coherence {"ON" if checked else "OFF"}', 2000)

    def _show_input_meters(self):
        """Muestra un diálogo con los niveles de entrada en tiempo real."""
        self.sb.showMessage('Input Meters — próximamente', 3000)

    def _toggle_spl_meters(self):
        """E — toggle del SPL meter en el panel."""
        if hasattr(self, '_spl_labels'):
            visible = not self._spl_labels[0].isVisible()
            for lbl in self._spl_labels:
                lbl.parentWidget().setVisible(visible) if lbl.parentWidget() else lbl.setVisible(visible)
        self.sb.showMessage('SPL Meters toggled', 2000)

    def _toggle_ui_bar(self, bar: str, checked: bool):
        """Toggle de barras de UI."""
        if bar == 'spl_meter' and hasattr(self, '_spl_labels'):
            for lbl in self._spl_labels:
                p = lbl.parent()
                if p: p.setVisible(checked)
        self.sb.showMessage(f'{bar} {"shown" if checked else "hidden"}', 2000)

    def _toggle_spl_clock(self):
        """K — alterna entre mostrar SPL y reloj en la toolbar."""
        if hasattr(self, 'lbl_clock'):
            vis = self.lbl_clock.isVisible()
            self.lbl_clock.setVisible(not vis)
        self.sb.showMessage('SPL/Clock toggled', 2000)

    def _show_manage_configs(self):
        QMessageBox.information(self, 'Manage Configurations',
            'Gestión de configuraciones guardadas.\n(Próximamente)')

    def _show_command_bar_config(self):
        QMessageBox.information(self, 'Command Bar Config',
            'Configuración de la barra de comandos.\n(Próximamente)')

    def _new_spectrum_measurement(self):
        """Nueva medición de espectro — cambia a vista Spectrum."""
        self._switch_to_spectrum()

    def _new_spectrum_avg(self):
        QMessageBox.information(self, 'New Spectrum Average',
            'Promedio de espectro.\n(Próximamente)')

    def _new_tf_avg(self):
        QMessageBox.information(self, 'New TF Average',
            'Promedio de Transfer Function.\n(Próximamente)')

    def _new_tab(self):
        QMessageBox.information(self, 'New Tab',
            'Nueva pestaña de medición.\n(Próximamente)')

    def _duplicate_tab(self):
        QMessageBox.information(self, 'Duplicate Tab',
            'Duplicar pestaña actual.\n(Próximamente)')

    def _delete_tab(self):
        QMessageBox.information(self, 'Delete Tab',
            'Eliminar pestaña actual.\n(Próximamente)')

    def _move_tab(self):
        QMessageBox.information(self, 'Move Tab',
            'Mover pestaña.\n(Próximamente)')

    def _show_amplitude_cal(self):
        """Calibración de amplitud — abre SPL Settings con sección de offset."""
        self._show_spl_settings()

    def _show_about(self):
        QMessageBox.about(self, 'Coherence v0.2',
            '<b>Coherence v0.2</b><br>'
            'Analizador de audio en tiempo real<br>'
            'Alternativa libre a SMAART<br><br>'
            'GPL v3 — github.com/JorgeChiles/coherence<br>'
            '<small>Jorge Peña (JorgeChiles) © 2024</small>'
        )

    def _show_shortcuts(self):
        msg = (
            '<b>Keyboard Shortcuts</b><br><br>'
            '<b>Navigation</b><br>'
            'T — Transfer Function tab<br>'
            'S — Spectrum tab<br>'
            'E — Spectrogram tab<br><br>'
            '<b>Transport</b><br>'
            'Space — Start<br>'
            'G — Toggle Noise Generator<br>'
            'F — Freeze<br><br>'
            '<b>Analysis</b><br>'
            'D — Find Delay<br>'
            'R — Reset Delay<br>'
            'P — Save Trace<br><br>'
            '<b>File</b><br>'
            'Ctrl+G — Save Graph PNG<br>'
            'Ctrl+K — Load Mic Calibration<br>'
            'Ctrl+M — Measurement Config<br>'
            'Ctrl+Q — Quit'
        )
        QMessageBox.information(self, 'Shortcuts', msg)

    # ── Overlay de trazas ─────────────────────────────────────────────

    def _save_trace(self):
        """Captura el estado actual y lo guarda como traza estática de referencia."""
        if len(self._traces) >= MAX_TRACES:
            self.sb.showMessage(
                f'⚠  Máximo {MAX_TRACES} trazas. Eliminá una antes de guardar.', 4000)
            return

        f = self.canvas_meas._last_freqs
        if f is None:
            self.sb.showMessage('⚠  Sin datos — iniciá la medición primero.', 3000)
            return

        color = TRACE_PALETTE[self._trace_color_idx % len(TRACE_PALETTE)]
        self._trace_color_idx += 1

        name      = f'T{len(self._traces) + 1}'
        thresh    = self.spn_thresh.value()

        # Niveles CPB del spectrum (pueden no existir si nunca se vio esa tab)
        lev_x = self.canvas_spec._last_lx
        lev_y = self.canvas_spec._last_ly
        if lev_x is None:
            lev_x = np.full(31, -80.0)
            lev_y = np.full(31, -80.0)

        tr = TraceData(
            name         = name,
            color        = color,
            freqs        = f.copy(),
            mag_db       = self.canvas_meas._last_mag_db.copy(),
            phase_deg    = self.canvas_meas._last_phase_deg.copy(),
            gamma2       = self.canvas_meas._last_gamma2.copy(),
            ir           = (self.canvas_meas._last_ir.copy()
                            if self.canvas_meas._last_ir is not None else None),
            delay_ref_ms = self.canvas_meas._delay_ref_ms,
            coh_thresh   = thresh,
            lev_x        = lev_x.copy(),
            lev_y        = lev_y.copy(),
        )
        self._traces.append(tr)

        # Dibujar en ambos canvas
        self.canvas_meas.store_trace(
            tr.freqs, tr.mag_db, tr.phase_deg, tr.gamma2, tr.ir,
            tr.delay_ref_ms, tr.color, tr.coh_thresh
        )
        self.canvas_spec.store_trace(tr.lev_x, tr.lev_y, tr.color)

        self._rebuild_trace_panel()
        self.sb.showMessage(f'Traza {name} guardada', 3000)

    def _delete_trace(self, idx):
        """Elimina la traza idx de canvas y lista."""
        if 0 <= idx < len(self._traces):
            self._traces.pop(idx)
            self.canvas_meas.remove_trace(idx)
            self.canvas_spec.remove_trace(idx)
            self._rebuild_trace_panel()

    def _toggle_trace_visible(self, idx, checked):
        """Muestra u oculta la traza idx."""
        if 0 <= idx < len(self._traces):
            self._traces[idx].visible = checked
            self.canvas_meas.set_trace_visible(idx, checked)
            self.canvas_spec.set_trace_visible(idx, checked)

    def _rebuild_trace_panel(self):
        """Reconstruye la lista de filas de trazas desde self._traces."""
        while self._trace_rows_layout.count():
            item = self._trace_rows_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for i, tr in enumerate(self._traces):
            row = TraceRow(
                idx=i, name=tr.name, color=tr.color,
                on_vis=self._toggle_trace_visible,
                on_del=self._delete_trace,
                visible=tr.visible,
            )
            self._trace_rows_layout.addWidget(row)

    # ── Búsqueda y trazas Spectrum ────────────────────────────────────

    def _filter_tf_traces(self, text: str):
        """Filtra las filas de trazas TF por nombre."""
        text = text.lower().strip()
        for i in range(self._trace_rows_layout.count()):
            item = self._trace_rows_layout.itemAt(i)
            w = item.widget() if item else None
            if w is None:
                continue
            name = self._traces[i].name.lower() if i < len(self._traces) else ''
            w.setVisible(not text or text in name)

    def _filter_sp_traces(self, text: str):
        """Filtra las filas de trazas Spectrum por nombre."""
        text = text.lower().strip()
        for i in range(self._sp_trace_rows_layout.count()):
            item = self._sp_trace_rows_layout.itemAt(i)
            w = item.widget() if item else None
            if w is None:
                continue
            name = f'sp {i + 1}'
            w.setVisible(not text or text in name)

    def _save_spectrum_trace(self):
        """Guarda el spectrum actual como traza de referencia visual."""
        lev_x = self.canvas_spec._last_lx
        if lev_x is None:
            self.sb.showMessage('⚠  Sin datos de spectrum — iniciá la medición primero.', 3000)
            return
        lev_y = self.canvas_spec._last_ly
        if lev_y is None:
            lev_y = lev_x
        color = TRACE_PALETTE[len(self._sp_traces) % len(TRACE_PALETTE)]
        self.canvas_spec.store_trace(lev_x.copy(), lev_y.copy(), color)
        idx = len(self._sp_traces)
        self._sp_traces.append({'lev_x': lev_x.copy(), 'lev_y': lev_y.copy(), 'color': color})
        row = TraceRow(
            idx=idx, name=f'Sp {idx + 1}', color=color,
            on_vis=lambda i, c: self.canvas_spec.set_trace_visible(i, c),
            on_del=self._delete_sp_trace,
        )
        self._sp_trace_rows_layout.addWidget(row)
        self.sb.showMessage(f'Traza spectrum Sp {idx + 1} guardada', 2000)

    def _delete_sp_trace(self, idx: int):
        """Elimina la traza spectrum en posición idx."""
        if not (0 <= idx < len(self._sp_traces)):
            return
        self._sp_traces.pop(idx)
        self.canvas_spec.remove_trace(idx)
        item = self._sp_trace_rows_layout.itemAt(idx)
        if item and item.widget():
            w = item.widget()
            self._sp_trace_rows_layout.removeWidget(w)
            w.deleteLater()

    def closeEvent(self, event):
        self.timer.stop()
        self.engine.stop()
        event.accept()


# ── Entry point ───────────────────────────────────────────────────────

def run():
    # macOS: cambiar "Python" por "Coherence" en el menú de la app
    if sys.platform == 'darwin':
        try:
            from Foundation import NSBundle  # type: ignore
            _info = NSBundle.mainBundle().localizedInfoDictionary() \
                    or NSBundle.mainBundle().infoDictionary()
            if _info is not None:
                _info['CFBundleName'] = 'Coherence'
        except ImportError:
            try:
                import ctypes as _ct
                _ct.cdll.LoadLibrary('libc.dylib').setprogname(b'Coherence')
            except Exception:
                pass

    app = QApplication(sys.argv)
    app.setApplicationName('Coherence')
    app.setApplicationDisplayName('Coherence')
    app.setApplicationVersion('0.1.0')

    pal = QPalette()
    for role, color in [
        (QPalette.ColorRole.Window,        BG_APP),
        (QPalette.ColorRole.WindowText,    TEXT_HI),
        (QPalette.ColorRole.Base,          BG_PANEL),
        (QPalette.ColorRole.AlternateBase, BG_SETTINGS),
        (QPalette.ColorRole.Text,          TEXT_HI),
        (QPalette.ColorRole.Button,        BG_SETTINGS),
        (QPalette.ColorRole.ButtonText,    TEXT_HI),
        (QPalette.ColorRole.Highlight,     ACCENT),
        (QPalette.ColorRole.HighlightedText, '#000000'),
    ]:
        pal.setColor(role, QColor(color))
    app.setPalette(pal)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())
