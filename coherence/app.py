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
    QMessageBox, QCheckBox, QRadioButton, QButtonGroup,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QTreeWidget, QTreeWidgetItem,
    QLineEdit, QSplitter, QStackedWidget, QMenu,
)
from PyQt6.QtCore  import Qt, QTimer, QDateTime
from PyQt6.QtGui   import QFont, QPalette, QColor, QKeySequence, QShortcut, QAction
from PyQt6.QtWidgets import QMenuBar

import matplotlib as _mpl_module
# Deshabilitar TODOS los atajos de teclado de matplotlib para que no bloqueen
# los QShortcut / event filter de la app (p=pan, s=save, g=grid, r=reset, c=back…)
for _k in ('keymap.fullscreen', 'keymap.home', 'keymap.back', 'keymap.forward',
           'keymap.pan', 'keymap.zoom', 'keymap.save', 'keymap.grid',
           'keymap.grid_minor', 'keymap.yscale', 'keymap.xscale',
           'keymap.quit', 'keymap.quit_all', 'keymap.all_axes'):
    try:
        _mpl_module.rcParams[_k] = []
    except KeyError:
        pass

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
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
    '#BF5AF2',   # 0 Violeta  — engine 1  (~380–450 nm)
    '#0A84FF',   # 1 Azul     — engine 2  (~450–495 nm)
    '#00E5FF',   # 2 Cian     — engine 3  (~495–530 nm)
    '#30D158',   # 3 Verde    — engine 4  (~530–560 nm)
    '#FFD60A',   # 4 Amarillo — engine 5  (~560–590 nm)
    '#FF9F0A',   # 5 Naranja  — engine 6  (~590–620 nm)
    '#FF453A',   # 6 Rojo     — engine 7  (~620–750 nm)
]

# ── Paleta de curvas objetivo ────────────────────────────────────────
TARGET_PALETTE = ['#FFD60A', '#FF9F0A', '#BF5AF2', '#FF375F', '#00C5E8', '#30D158', '#FF6B6B']

# ── Colormap para espectrograma — estilo thermal / analizador pro ─────
# negro (silencio) → azul → cyan → verde → amarillo → rojo → blanco (pico)
_SGRAM_CMAP = LinearSegmentedColormap.from_list(
    'audio_thermal',
    [
        # ── Zona de ruido/silencio: negro → azul nocturno (0–18%)
        (0.00, '#000000'),   # silencio total
        (0.03, '#010008'),
        (0.06, '#020012'),
        (0.09, '#030020'),
        (0.12, '#050038'),
        (0.15, '#07004e'),
        (0.18, '#0a0070'),
        # ── Zona baja: azul profundo → azul brillante (18–35%)
        (0.22, '#0a18a8'),
        (0.26, '#0835cc'),
        (0.30, '#0558e0'),
        (0.35, '#0585f0'),
        # ── Transición azul→cyan (35–46%)
        (0.38, '#00a0e8'),
        (0.42, '#00bcd4'),
        (0.46, '#00cfc5'),
        # ── Zona media: cyan → verde (46–58%)
        (0.50, '#00d4a0'),
        (0.54, '#00d870'),
        (0.58, '#00de45'),
        # ── Verde → amarillo-verde (58–68%)
        (0.62, '#40e000'),
        (0.66, '#85e800'),
        (0.68, '#aae800'),
        # ── Amarillo-verde → amarillo → naranja (68–86%)
        (0.71, '#cce000'),
        (0.74, '#f0d000'),
        (0.77, '#ffc000'),
        (0.80, '#ffaa00'),
        (0.83, '#ff8800'),
        (0.86, '#ff6000'),
        # ── Naranja → rojo (86–96%)
        (0.89, '#ff3800'),
        (0.92, '#ff1800'),
        (0.95, '#ff0a00'),
        # ── Rojo → blanco (clipping) (96–100%)
        (0.97, '#ff4040'),
        (0.99, '#ffb0b0'),
        (1.00, '#ffffff'),
    ],
    N=2048,   # más puntos de LUT → transiciones más suaves
)

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
    font-size: 11px;
    font-weight: bold;
    letter-spacing: 3px;
    padding: 0 8px;
}}
#lbl_cursor_info {{
    color: {TEXT_HI};
    font-size: 14px;
    font-family: 'Menlo', 'Monaco', 'Courier New';
    letter-spacing: 1px;
    padding: 0 10px;
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
    """Formatea una frecuencia en Hz como string compacto estilo audio.
    Ejemplos: 20 Hz, 50 Hz, 1 kHz, 2 kHz, 10 kHz, 20 kHz.
    """
    if f_hz >= 1000:
        v = f_hz / 1000.0
        return f'{v:.0f} kHz' if v == int(v) else f'{v:.1f} kHz'
    return f'{int(f_hz)} Hz' if f_hz == int(f_hz) else f'{f_hz:.0f} Hz'


def _load_curve_file(path: str):
    """
    Load a frequency / dB curve file.

    Supported formats
    -----------------
    • SMAART .crv   — tab-separated, lines starting with '*' are comments;
                      '* Curve Name:' / '* Name:' metadata extracted automatically.
    • REW  .txt     — header block then "Frequency Hz  Level dB  Phase degrees" data.
    • Generic .txt / .csv / .tsv  — space, tab, comma or semicolon separated.
    • 3-column files  (freq, dB, phase)  — only first two columns used.
    • kHz units     — values like "1.0k", "1kHz", "1.5K" auto-converted to Hz.
    • "dB" suffix   — stripped from amplitude column if present.

    Returns
    -------
    curve_name : str | None   — extracted from file metadata (or None)
    arr        : np.ndarray   — shape (N, 2), col 0 = Hz, col 1 = dB, sorted by Hz
    """
    import re as _re

    curve_name: str | None = None
    rows = []

    with open(path, 'r', encoding='utf-8', errors='ignore') as _f:
        for raw in _f:
            line = raw.strip()
            if not line:
                continue

            # ── Comment / metadata lines ──────────────────────────────
            if line.startswith('*') or line.startswith('#') or line.startswith('//'):
                # Try to extract curve name from SMAART .crv metadata
                # e.g.  "* Curve Name: My Target"  or  "* Name: ..."
                m = _re.search(r'(?i)(?:curve\s+)?name\s*[:\s]\s*(.+)', line[1:])
                if m and curve_name is None:
                    candidate = m.group(1).strip().strip('*').strip()
                    if candidate and len(candidate) < 80:
                        curve_name = candidate
                continue

            # ── Normalize delimiters ──────────────────────────────────
            # Replace comma / semicolon / tab with single space
            norm = line.replace(',', ' ').replace(';', ' ').replace('\t', ' ')
            parts = norm.split()

            if len(parts) < 2:
                continue

            # ── Parse frequency (col 0) ───────────────────────────────
            try:
                f_raw = parts[0].upper()
                if f_raw.endswith('KHZ'):
                    freq = float(f_raw[:-3]) * 1000.0
                elif f_raw.endswith('K'):
                    freq = float(f_raw[:-1]) * 1000.0
                elif f_raw.endswith('HZ'):
                    freq = float(f_raw[:-2])
                else:
                    freq = float(f_raw)
            except ValueError:
                continue   # header / label row — skip silently

            if freq <= 0:
                continue

            # ── Parse amplitude (col 1) ───────────────────────────────
            try:
                db_raw = parts[1].upper().replace('DB', '').strip()
                db = float(db_raw)
            except ValueError:
                continue

            rows.append((freq, db))

    if not rows:
        raise ValueError(
            'No valid frequency / dB data found.\n'
            'Expected two columns: Hz  dB  (space, tab or comma separated).')

    arr = np.array(rows, dtype=float)
    arr = arr[arr[:, 0].argsort()]
    return curve_name, arr


# ── Datos de una traza almacenada ─────────────────────────────────────

class TraceData:
    """Snapshot de una medición para mostrar como referencia estática."""
    def __init__(self, name, color, freqs, mag_db, phase_deg, gamma2, ir,
                 delay_ref_ms, coh_thresh, lev_x, lev_y, lev_freqs=None):
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
        self.lev_x         = lev_x     # niveles REF (CPB o FFT raw)
        self.lev_y         = lev_y     # niveles MED (CPB o FFT raw)
        self.lev_freqs     = lev_freqs  # eje de frecuencias de lev_x/lev_y


# ── Target curve overlay (SMAART-style) ───────────────────────────────

class TargetCurve:
    """Overlay curve drawn on magnitude / spectrum axes."""
    def __init__(self, name, freqs, mag_db, color='#FFD60A',
                 visible=False, line_width=2, offset_db=0.0,
                 tolerance=False, tolerance_db=0.0, builtin=False):
        self.name         = name
        self.freqs        = freqs        # Hz array
        self.mag_db       = mag_db       # dB array (base, without offset)
        self.color        = color
        self.visible      = visible
        self.line_width   = line_width   # 2 or 3
        self.offset_db    = offset_db    # vertical shift in dB
        self.tolerance    = tolerance    # show ±tolerance band
        self.tolerance_db = tolerance_db # half-width of band in dB
        self.builtin      = builtin      # factory curve (shown greyed in dialog)


# ── Built-in target curve generators ──────────────────────────────────

def _tc_base_freqs():
    import numpy as _np_tc
    return _np_tc.logspace(_np_tc.log10(20), _np_tc.log10(20000), 400)

def _builtin_tilt(tilt_db):
    """Linear dB slope, 0 dB pivot at 1 kHz, total swing = tilt_db."""
    import numpy as _np_tc
    f = _tc_base_freqs()
    log_f   = _np_tc.log10(f)
    log_min = _np_tc.log10(20)
    log_max = _np_tc.log10(20000)
    slope   = tilt_db / (log_max - log_min)
    return f, slope * (_np_tc.log10(1000) - log_f)

def _builtin_haystack(peak_db):
    """Gaussian hump peaking at ~250 Hz — classic live-sound target."""
    import numpy as _np_tc
    f        = _tc_base_freqs()
    log_f    = _np_tc.log10(f)
    peak_log = _np_tc.log10(250)
    sigma    = 0.78   # decades
    return f, peak_db * _np_tc.exp(-0.5 * ((log_f - peak_log) / sigma) ** 2)

def _make_builtin_target_curves():
    _COLS = ['#E040FB', '#AD1FD6', '#CC2222', '#FF5555']
    specs = [
        ('9dB Haystack',  _builtin_haystack(9),  _COLS[0], 3),
        ('15dB Haystack', _builtin_haystack(15), _COLS[1], 3),
        ('3dB Tilt',      _builtin_tilt(3),       _COLS[2], 2),
        ('6dB Tilt',      _builtin_tilt(6),       _COLS[3], 2),
    ]
    out = []
    for name, (f, db), col, lw in specs:
        out.append(TargetCurve(
            name=name, freqs=f, mag_db=db, color=col,
            visible=False, line_width=lw, builtin=True))
    return out


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


# ── TF Average config + dialog ────────────────────────────────────────

@dataclass
class TFAvgConfig:
    """Configuration for a named TF spatial average."""
    name:         str       = 'Average 1'
    avg_type:     str       = 'dB'       # 'dB' or 'Power'
    coh_weighted: bool      = True
    contributors: List[int] = field(default_factory=list)  # engine indices; [] = all


@dataclass
class SpectrumAvgConfig:
    """Configuration for a named Spectrum spatial average."""
    name:         str       = 'Spectrum Avg 1'
    avg_type:     str       = 'Power'    # 'dB' or 'Power'
    contributors: List[int] = field(default_factory=list)  # engine indices; [] = all


class TFAvgDialog(QDialog):
    """
    SMAART-style "New Measurement Average" dialog.

    engine_names : list of str — names of available TF engines (index = engine idx)
    config       : existing TFAvgConfig to edit, or None to create new
    """

    def __init__(self, engine_names: List[str],
                 parent=None, config=None, show_coh_weighted: bool = True):
        super().__init__(parent)
        self._engine_names      = engine_names
        self._editing           = config is not None
        self._result_cfg        = None
        self._show_coh_weighted = show_coh_weighted

        title = 'Edit Measurement Average' if self._editing else 'New Measurement Average'
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(380)

        # ── Dark palette ──────────────────────────────────────────────
        self.setStyleSheet(
            'QDialog{background:#2a2a2a;color:#cccccc;}'
            'QLabel{color:#cccccc;}'
            'QLineEdit{background:#1e1e1e;color:#eeeeee;border:1px solid #444;'
            '          border-radius:3px;padding:3px 6px;}'
            'QRadioButton{color:#cccccc;spacing:6px;}'
            'QCheckBox{color:#cccccc;spacing:6px;}'
            'QTableWidget{background:#1a1a1a;color:#cccccc;gridline-color:#333;'
            '             border:1px solid #444;}'
            'QHeaderView::section{background:#252525;color:#aaaaaa;'
            '                     border:none;padding:4px;font-size:11px;}'
            'QPushButton{background:#3a3a3a;color:#cccccc;border:1px solid #555;'
            '            border-radius:4px;padding:5px 14px;font-size:12px;}'
            'QPushButton:hover{background:#4a4a4a;}'
            'QPushButton:pressed{background:#252525;}'
        )

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # ── Name ──────────────────────────────────────────────────────
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel('Name:'))
        self._txt_name = QLineEdit(config.name if config else f'Average {1}')
        name_row.addWidget(self._txt_name)
        layout.addLayout(name_row)

        # ── Average as ────────────────────────────────────────────────
        avg_as_row = QHBoxLayout()
        avg_as_row.addWidget(QLabel('Average as:'))
        self._rb_db    = QRadioButton('dB')
        self._rb_power = QRadioButton('Power')
        self._rb_db.setChecked(True)
        if config and config.avg_type == 'Power':
            self._rb_power.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self._rb_db);  grp.addButton(self._rb_power)
        avg_as_row.addWidget(self._rb_db)
        avg_as_row.addWidget(self._rb_power)
        avg_as_row.addStretch()
        layout.addLayout(avg_as_row)

        # ── Coherence Weighted (TF only) ──────────────────────────────
        self._chk_coh = QCheckBox('Coherence Weighted')
        coh_val = getattr(config, 'coh_weighted', True) if config else True
        self._chk_coh.setChecked(coh_val)
        if self._show_coh_weighted:
            layout.addWidget(self._chk_coh)

        # ── Contributors table ────────────────────────────────────────
        layout.addWidget(QLabel('Contributors:'))

        self._tbl = QTableWidget(len(engine_names), 2)
        self._tbl.setHorizontalHeaderLabels(['Avg', 'Name'])
        self._tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self._tbl.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._tbl.verticalHeader().setVisible(False)
        self._tbl.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._tbl.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._tbl.setFixedHeight(min(len(engine_names) * 26 + 30, 180))

        initial_contribs = set(config.contributors) if (config and config.contributors) else set(range(len(engine_names)))
        for row, eng_name in enumerate(engine_names):
            chk = QCheckBox()
            chk.setChecked(row in initial_contribs)
            chk_cell = QWidget()
            chk_lay  = QHBoxLayout(chk_cell)
            chk_lay.addWidget(chk); chk_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
            chk_lay.setContentsMargins(0, 0, 0, 0)
            self._tbl.setCellWidget(row, 0, chk_cell)
            name_item = QTableWidgetItem(eng_name)
            name_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self._tbl.setItem(row, 1, name_item)
        layout.addWidget(self._tbl)

        # ── Buttons ───────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        if not self._editing:
            self._btn_create_plus = QPushButton('Create +')
            self._btn_create_plus.clicked.connect(self._on_create_plus)
            btn_row.addWidget(self._btn_create_plus)
        lbl_ok = 'Save & Close' if self._editing else 'Create & Close'
        self._btn_ok = QPushButton(lbl_ok)
        self._btn_ok.setDefault(True)
        self._btn_ok.clicked.connect(self._on_ok)
        btn_row.addWidget(self._btn_ok)
        layout.addLayout(btn_row)

    # ── Helpers ───────────────────────────────────────────────────────

    def _build_config(self) -> TFAvgConfig:
        contributors = [
            row for row in range(self._tbl.rowCount())
            if self._tbl.cellWidget(row, 0).layout().itemAt(0).widget().isChecked()
        ]
        return TFAvgConfig(
            name         = self._txt_name.text().strip() or 'Average',
            avg_type     = 'dB' if self._rb_db.isChecked() else 'Power',
            coh_weighted = self._chk_coh.isChecked(),
            contributors = contributors,
        )

    def _on_ok(self):
        self._result_cfg = self._build_config()
        self.accept()

    def _on_create_plus(self):
        """Emit config and reset form for another entry."""
        self._result_cfg = self._build_config()
        self.accept()   # caller checks _create_another flag
        # Mark as "create another" so caller can reopen
        self._create_another = True

    def get_config(self) -> TFAvgConfig:
        return self._result_cfg


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
    Card de Spectrum Engine — estilo idéntico a TFEngine.
      ┌─────────────────────────────┐
      │  ● Name          [⏸] [⚙] [✕] │
      │  ████████████ (SPL bar)     │
      └─────────────────────────────┘
    Dot click → trace ON/OFF.  Right-click → color picker.
    """

    def __init__(self, idx, name, color, on_config, on_del, on_play,
                 active=True, selected=False, on_trace_toggle=None, parent=None):
        super().__init__(parent)
        self._color          = color
        self._idx            = idx
        self._active         = active
        self._selected       = selected
        self._trace_visible  = True
        self._on_trace_cb    = on_trace_toggle  # callback(idx, visible)

        # ── Card con borde de color ───────────────────────────────────
        self._card = QFrame()
        self._card.setObjectName('spcard')
        self._apply_card_style()
        self._card.mousePressEvent = lambda e: None  # sin selección por ahora

        card_lay = QVBoxLayout(self._card)
        card_lay.setContentsMargins(8, 6, 8, 6)
        card_lay.setSpacing(5)

        # ── Fila 1: dot | nombre | ⏸ | ⚙ | ✕ ───────────────────────
        r1 = QHBoxLayout(); r1.setSpacing(6)

        # Dot — trace ON/OFF (left) / color picker (right-click)
        self._dot = QPushButton('●')
        self._dot.setFixedSize(26, 26)
        self._dot.setStyleSheet(
            f'QPushButton{{color:{color};font-size:22px;background:transparent;'
            f'border:none;padding:0;margin:0;}}'
            f'QPushButton:hover{{color:#ffffff;}}')
        self._dot.setCursor(Qt.CursorShape.PointingHandCursor)
        self._dot.setToolTip('Click: mostrar/ocultar trazo · Clic derecho: cambiar color')
        self._dot.clicked.connect(self._on_toggle_trace)
        self._dot.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._dot.customContextMenuRequested.connect(lambda _: self._on_pick_color())
        r1.addWidget(self._dot)

        # Nombre
        self._lbl = QLabel(name[:24])
        self._lbl.setStyleSheet(
            f'color:#cccccc;font-size:12px;font-weight:bold;background:transparent;')
        r1.addWidget(self._lbl, stretch=1)

        # ▶ / ⏸  — medir / detener (trazo queda)
        self._btn_play = QPushButton('⏸' if active else '▶')
        self._btn_play.setFixedSize(26, 22)
        self._btn_play.setCheckable(True)
        self._btn_play.setChecked(active)
        _pc = color if active else '#444444'
        self._btn_play.setStyleSheet(
            f'QPushButton{{font-size:12px;padding:0;border:none;'
            f'background:transparent;color:{_pc};}}'
            f'QPushButton:checked{{color:{color};}}'
            f'QPushButton:!checked{{color:#444444;}}')
        self._btn_play.clicked.connect(lambda: on_play(idx))
        r1.addWidget(self._btn_play)

        # ⚙ config
        btn_cfg = QPushButton('⚙')
        btn_cfg.setFixedSize(22, 22)
        btn_cfg.setToolTip('Configurar engine')
        btn_cfg.setStyleSheet(
            f'QPushButton{{font-size:12px;padding:0;border:1px solid #2a2a2a;'
            f'background:transparent;color:#555;border-radius:3px;}}'
            f'QPushButton:hover{{color:{color};border-color:{color};}}')
        btn_cfg.clicked.connect(lambda: on_config(idx))
        r1.addWidget(btn_cfg)

        # ✕ eliminar
        btn_d = QPushButton('✕')
        btn_d.setFixedSize(18, 22)
        btn_d.setToolTip('Eliminar engine')
        btn_d.setStyleSheet(
            'QPushButton{font-size:10px;padding:0;border:none;'
            'background:transparent;color:#3a3a3a;}'
            'QPushButton:hover{color:#cc4444;}')
        btn_d.clicked.connect(lambda: on_del(idx))
        r1.addWidget(btn_d)

        card_lay.addLayout(r1)

        # ── Fila 2: barra SPL ────────────────────────────────────────
        _BAR_CSS = ('QProgressBar{background:#222;border:none;border-radius:2px;}'
                    'QProgressBar::chunk{background:#4ab46a;border-radius:2px;}')
        self.bar_spl = QProgressBar()
        self.bar_spl.setRange(0, 100); self.bar_spl.setValue(0)
        self.bar_spl.setTextVisible(False); self.bar_spl.setFixedHeight(6)
        self.bar_spl.setStyleSheet(_BAR_CSS)
        card_lay.addWidget(self.bar_spl)

        # ── Layout raíz ──────────────────────────────────────────────
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 4)
        root.setSpacing(0)
        root.addWidget(self._card)

    # ── Helpers ───────────────────────────────────────────────────────

    def _apply_card_style(self):
        border = self._color if self._active else f'{self._color}44'
        self._card.setStyleSheet(
            f'QFrame#spcard{{background:#161916;border:2px solid {border};'
            f'border-radius:6px;}}'
            f'QLabel{{background:transparent;border:none;}}'
            f'QPushButton{{background:transparent;}}')

    def _on_toggle_trace(self):
        self._trace_visible = not self._trace_visible
        dim = self._color if self._trace_visible else '#333333'
        self._dot.setStyleSheet(
            f'QPushButton{{color:{dim};font-size:22px;background:transparent;'
            f'border:none;padding:0;margin:0;}}'
            f'QPushButton:hover{{color:{self._color};}}')
        if self._on_trace_cb is not None:
            self._on_trace_cb(self._idx, self._trace_visible)

    def _on_pick_color(self):
        from PyQt6.QtWidgets import QColorDialog
        from PyQt6.QtGui import QColor as _QColor
        c = QColorDialog.getColor(_QColor(self._color), self, 'Color del Engine')
        if c.isValid():
            self._color = c.name()
            self._dot.setStyleSheet(
                f'QPushButton{{color:{self._color};font-size:22px;background:transparent;'
                f'border:none;padding:0;margin:0;}}'
                f'QPushButton:hover{{color:#ffffff;}}')
            self._btn_play.setStyleSheet(
                f'QPushButton{{font-size:12px;padding:0;border:none;'
                f'background:transparent;color:{self._color};}}'
                f'QPushButton:checked{{color:{self._color};}}'
                f'QPushButton:!checked{{color:#444444;}}')
            self._apply_card_style()

    # ── API pública ───────────────────────────────────────────────────

    def update_level(self, db: float):
        pct = max(0, min(100, int((db + 80) / 80 * 100)))
        self.bar_spl.setValue(pct)
        # Color: green → yellow (≥−12 dB) → red (≥−3 dB)
        if db >= -3.0:
            chunk = '#ef5350'   # rojo — peak
        elif db >= -12.0:
            chunk = '#ffb74d'   # amarillo — warning
        else:
            chunk = '#4ab46a'   # verde — normal
        self.bar_spl.setStyleSheet(
            f'QProgressBar{{background:#222;border:none;border-radius:2px;}}'
            f'QProgressBar::chunk{{background:{chunk};border-radius:2px;}}')

    def set_active(self, active: bool):
        self._active = active
        self._btn_play.setChecked(active)
        self._btn_play.setText('⏸' if active else '▶')
        self._apply_card_style()

    def set_selected(self, sel: bool):
        self._selected = sel
        self._apply_card_style()


# ── Canvas de Medición (IR / TF+Coh / Phase) ─────────────────────────

class CanvasOverlay(QWidget):
    """
    Wrapper que pone un canvas matplotlib dentro de un QWidget
    y añade un botón dropdown flotante en la esquina superior izquierda
    (estilo SMAART: RTA▼ / Magnitude▼ etc.) más un engrane ⚙.

    El botón está posicionado absolutamente sobre el canvas con resizeEvent.
    """

    def __init__(self, canvas, options, current, on_select, on_gear,
                 label='', on_add=None, on_focus=None, parent=None):
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
            # Connect focus callback: clicking the canvas switches side panels
            if on_focus is not None:
                canvas.mpl_connect('button_press_event', lambda e: on_focus())
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
        w, h = self.width(), self.height()
        # Position button INSIDE the plot area (same style as _mag_lbl in MeasurementCanvas).
        # Read axes bounds from the canvas so the label never ends up in the top margin.
        lx, ty = 6, 6
        if self._canvas is not None and hasattr(self._canvas, 'ax'):
            try:
                p = self._canvas.ax.get_position()
                lx = max(6, int(w * p.x0) + 4)
                ty = max(4, int(h * (1.0 - p.y1)) + 4)
            except Exception:
                pass
        self._btn.adjustSize()
        self._btn.move(lx, ty)
        # Engrane a la derecha del dropdown
        self._gear.move(lx + self._btn.width() + 4, ty)
        # "+" en la esquina inferior-izquierda
        self._btn_add.move(6, h - 26)

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
        # Callbacks
        self.on_cursor_update    = None
        self.on_panel_menu       = None   # callable(anchor_widget) — abre menú de vistas
        self.on_gear_clicked       = None   # callable() — opens Graph Settings TF tab
        self.on_close_mag_panel    = None   # callable() — X/+ on Magnitude corner
        self.on_close_ph_panel     = None   # callable() — X/+ on Phase corner
        self.on_toggle_ir          = None   # callable() — toggle IR panel (Cmd+I)
        self.on_panel1_changed     = None   # callable(type_name) — user changed panel 1 (Magnitude slot)
        self.on_panel2_changed     = None   # callable(type_name) — user changed panel 2 (Phase slot)
        self.on_engine_select_menu = None   # callable(anchor_widget) — engine picker
        self.on_ir_peak_detected   = None   # callable(delay_ms: float) — IR argmax peak detected
        # Últimos datos para interpolación del cursor
        self._last_freqs     = None
        self._last_gamma2    = None
        self._last_mag_db    = None
        self._last_phase_deg = None
        self._last_ir        = None
        # Last AVG data (for saving AVG trace)
        self._last_avg_freqs     = None
        self._last_avg_mag_db    = None
        self._last_avg_phase_deg = None
        self._last_avg_gamma2    = None
        self._fs             = 48000   # actualizado por MainWindow en cada refresh
        # Listas paralelas de líneas matplotlib para trazas almacenadas
        self._trace_tf_lines:  List = []
        self._trace_ph_lines:  List = []
        self._trace_ir_lines:  List = []
        self._trace_coh_lines: List = []
        self._target_lines: List = []   # dashed overlay lines on ax_tf
        self._target_fills: List = []   # fill_between polycollections (tolerance bands)
        self._eng_colors = [ENGINE_PALETTE[0], ENGINE_PALETTE[1]]
        self._build()
        self._build_tf_overlay()

    def _build(self):
        from matplotlib.transforms import blended_transform_factory

        self._delay_ref_ms = 0.0
        self._nperseg      = 4096   # actualizado por MainWindow antes de update_plots
        self._ir_visible   = True   # controlado por Cmd+I desde MainWindow
        self._ir_centered  = False  # False=raw(eje real) / True=centrado en pico (Find Delay)

        # Axes created with placeholder positions — _relayout() sets the real ones
        # ── IR ──
        self.ax_ir = self.fig.add_axes([0.07, 0.80, 0.865, 0.18])
        self.ax_ir.set_facecolor(BG_PLOT)
        self.ax_ir.set_xlim(-100, 100)   # siempre fijo, simétrico, NUNCA se mueve
        self.ax_ir.set_ylim(-1.1, 1.1)
        self.ax_ir.axhline(0, color=BORDER, lw=0.6)
        self.ax_ir.axvline(0, color='#2a2a2a', lw=0.7, ls='--')   # referencia fija 0 ms
        self.ax_ir.set_ylabel('Level (%)', fontsize=6, color=TEXT_DIM, labelpad=1)
        self.ax_ir.tick_params(axis='y', length=0)
        self._style_ax(self.ax_ir, log=False)

        # Eje X tiempo en ms — labels arriba (panel superior, hay espacio libre)
        self.ax_ir.xaxis.set_major_locator(ticker.MultipleLocator(50))
        self.ax_ir.xaxis.set_minor_locator(ticker.MultipleLocator(25))
        self.ax_ir.tick_params(axis='x', which='both',
                               top=False, labeltop=False,
                               bottom=True, labelbottom=True,
                               labelsize=6, colors=TEXT_MID)
        self.ax_ir.xaxis.set_label_position('bottom')
        self.ax_ir.set_xlabel('ms', fontsize=6, color=TEXT_MID, labelpad=1)

        self.line_ir, = self.ax_ir.plot([], [], color=self._eng_colors[0], lw=0.9, alpha=0.90)
        self.line_ir_peak = self.ax_ir.axvline(0, color=ORANGE, lw=1.2, ls='--', alpha=0.8)

        # Etiqueta del delay real en el marcador naranja
        _blend_ir = blended_transform_factory(self.ax_ir.transData, self.ax_ir.transAxes)
        self.txt_ir_delay = self.ax_ir.text(
            0, 0.97, '',
            transform=_blend_ir,
            color=ORANGE, fontsize=7, ha='center', va='top',
            fontfamily='Menlo'
        )

        # ── TF (izquierda) + Coherencia (derecha twinx) ──
        self.ax_tf  = self.fig.add_axes([0.07, 0.26, 0.865, 0.52])
        self.ax_coh = self.ax_tf.twinx()

        self.ax_tf.set_facecolor(BG_PLOT)
        self.ax_tf.set_ylim(-18, 18)
        # Panel central: grilla de frecuencias CON labels (siempre visibles)
        setup_smaart_axis(self.ax_tf, bg=BG_PLOT,
                          show_xlabels=True, show_xlabel=False)
        self.ax_tf.set_ylabel('Level (dB)', fontsize=6, color=TEXT_MID, labelpad=0)
        self.ax_tf.tick_params(axis='y', length=0)
        # Grilla horizontal cada 6 dB — estilo SMAART
        for _db in range(-24, 25, 6):
            _lw, _ls = (0.8, '--') if _db == 0 else (0.4, '-')
            _col = '#253225' if _db == 0 else '#181e18'
            self.ax_tf.axhline(_db, color=_col, lw=_lw, ls=_ls, zorder=0)

        # Alinear 0 de coherencia con 0 dB de TF
        self._tf_ymin, self._tf_ymax = -18, 18
        self._coh_ymax = 1.05
        self.ax_coh.set_ylim(*self._coh_aligned_lim())
        self.ax_coh.set_ylabel('', fontsize=6, color=COH_COLOR, labelpad=2)
        self.ax_coh.tick_params(axis='y', colors=COH_COLOR, labelsize=6, length=0)
        self.ax_coh.set_yticks([0.2, 0.4, 0.6, 0.8])
        self.ax_coh.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: str(int(round(x * 100)))))
        # Disable offset/scientific notation on coherence axis — prevents "×" artifacts
        import matplotlib.ticker as _mticker
        self.ax_coh.yaxis.set_major_formatter(_mticker.ScalarFormatter(useOffset=False))
        self.ax_coh.yaxis.offsetText.set_visible(False)
        self.ax_coh.spines['right'].set_color(COH_COLOR)
        self.ax_coh.spines['right'].set_alpha(0.4)
        for sp in ('top', 'left', 'bottom'):
            self.ax_coh.spines[sp].set_visible(False)

        # Línea referencia coherencia 0.9 — discreta
        self.ax_coh.axhline(0.9, color=COH_COLOR, lw=0.5, ls=':', alpha=0.35)

        # Colores de engines — SMAART verde + cyan
        self._eng_colors = [ENGINE_PALETTE[0], ENGINE_PALETTE[1]]

        f0 = np.array([20.0, 20000.0])
        _l0, = self.ax_tf.semilogx(f0, [0, 0], color=self._eng_colors[0], lw=2.2, alpha=1.0, label='TF1')
        _l1, = self.ax_tf.semilogx([], [],     color=self._eng_colors[1], lw=1.2, alpha=0.55, ls='-', label='TF2')
        self._lines_tf    = [_l0, _l1]
        self.line_tf_avg, = self.ax_tf.semilogx([], [],     color='#ffffff', lw=1.8, alpha=0.85, ls='--', label='AVG')
        # Coherencia: solo el trazo rojo, sin área de relleno
        self.line_coh,    = self.ax_coh.semilogx(f0, [0, 0], color=COH_COLOR, lw=1.2, alpha=0.85)
        self._coh_fill    = None   # no fill

        # ── Phase ──  (fase envuelta −180…+180, estilo SMAART)
        self.ax_ph = self.fig.add_axes([0.07, 0.06, 0.865, 0.18])
        self.ax_ph.set_facecolor(BG_PLOT)
        self.ax_ph.set_ylim(-185, 185)
        # Panel inferior: muestra los labels de frecuencia en Hz/kHz
        setup_smaart_axis(self.ax_ph, bg=BG_PLOT,
                          show_xlabels=True, show_xlabel=True)
        self.ax_ph.set_ylabel('Phase (deg)', fontsize=6, color=TEXT_MID, labelpad=0)
        self.ax_ph.axhline(  0,   color='#253225', lw=0.9, ls='--')
        self.ax_ph.axhline( 90,   color='#1d261d', lw=0.5, ls=':')
        self.ax_ph.axhline(-90,   color='#1d261d', lw=0.5, ls=':')
        self.ax_ph.axhline( 180,  color='#1a221a', lw=0.4, ls=':')
        self.ax_ph.axhline(-180,  color='#1a221a', lw=0.4, ls=':')
        self.ax_ph.set_yticks([-180, -90, 0, 90, 180])
        self.ax_ph.tick_params(axis='y', labelsize=6, colors=TEXT_MID, length=0)

        _p0, = self.ax_ph.semilogx(f0, [0, 0], color=self._eng_colors[0], lw=2.2, alpha=1.0)
        _p1, = self.ax_ph.semilogx([], [],     color=self._eng_colors[1], lw=1.2, alpha=0.55, ls='-')
        self._lines_ph    = [_p0, _p1]
        self.line_ph_avg, = self.ax_ph.semilogx([], [],     color='#ffffff', lw=1.8, alpha=0.85, ls='--')
        # Aliases backward compat
        self.line_tf  = self._lines_tf[0]
        self.line_tf2 = self._lines_tf[1]
        self.line_ph  = self._lines_ph[0]
        self.line_ph2 = self._lines_ph[1]

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
        self._current_view_mode = 'tf_phase_ir'
        self._relayout()
        self.draw()

    # ── Layout engine — single source of truth for all axes positions ─

    def _relayout(self):
        """
        Compute and apply all axes positions from scratch.
        Called by set_view_mode() and resizeEvent().
        Positions are in figure-normalized coords (0.0–1.0); they scale
        automatically with the widget — no stale cached values.
        """
        if not hasattr(self, 'ax_ir'):
            return

        L, W  = 0.04, 0.92     # left=4%, right edge=96%  ← canonical for ALL canvases
        B, T  = 0.025, 0.995  # bottom=2.5%, top=99.5%
        GAP   = 0.025           # gap between stacked panels
        avail = T - B           # 0.92 total usable height

        # Panels ordered BOTTOM → TOP: [(name, proportion), ...]
        # Proportions must sum to 1.0 per row.
        _LAYOUTS = {
            'magnitude_only':         [('tf', 1.00)],
            'phase_only':             [('ph', 1.00)],
            'tf_phase':               [('ph', 0.50), ('tf', 0.50)],
            'magnitude_ir':           [('tf', 0.74), ('ir', 0.26)],
            'phase_ir':               [('ph', 0.74), ('ir', 0.26)],
            'tf_phase_ir':            [('ph', 0.425), ('tf', 0.425), ('ir', 0.15)],
            'magnitude_magnitude':    [('ph', 0.50), ('tf', 0.50)],
            'magnitude_magnitude_ir': [('ph', 0.425), ('tf', 0.425), ('ir', 0.15)],
        }
        mode   = getattr(self, '_current_view_mode', 'tf_phase_ir')
        panels = _LAYOUTS.get(mode, _LAYOUTS['tf_phase_ir'])

        # ── Visibility ────────────────────────────────────────────────
        show_ir = any(n == 'ir' for n, _ in panels)
        show_tf = any(n == 'tf' for n, _ in panels)
        show_ph = any(n == 'ph' for n, _ in panels)
        self.ax_ir.set_visible(show_ir)
        self.ax_tf.set_visible(show_tf)
        self.ax_coh.set_visible(show_tf)
        self.ax_ph.set_visible(show_ph)
        self._ir_visible = show_ir

        # ── Positions — bottom-up ─────────────────────────────────────
        n_gaps = len(panels) - 1
        usable = avail - n_gaps * GAP
        cur_b  = B
        for name, ratio in panels:
            h   = usable * ratio
            pos = [L, cur_b, W, h]
            if   name == 'ir': self.ax_ir.set_position(pos)
            elif name == 'tf': self.ax_tf.set_position(pos); self.ax_coh.set_position(pos)
            elif name == 'ph': self.ax_ph.set_position(pos)
            cur_b += h + GAP

        # ── X-axis tick labels — every visible frequency panel shows them ──
        self.ax_tf.tick_params(axis='x', which='major', labelsize=6,
                               colors='#9e9e9e', labelbottom=show_tf)
        self.ax_ph.tick_params(axis='x', which='major', labelsize=6,
                               colors='#9e9e9e', labelbottom=show_ph)

        # ── Frequency (Hz) xlabel on bottommost visible frequency panel ──
        if show_ph:
            self.ax_ph.set_xlabel('Frequency (Hz)', fontsize=6,
                                  color='#6a7a6a', labelpad=2)
            self.ax_tf.set_xlabel('')
        elif show_tf:
            self.ax_tf.set_xlabel('Frequency (Hz)', fontsize=6,
                                  color='#6a7a6a', labelpad=2)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'ax_ir'):
            self._relayout()

    # ── Modos de vista (Magnitude completo / Phase-only) ──────────────

    def set_view_mode(self, mode: str):
        """
        'tf_phase'       → Magnitude + Phase,          IR hidden
        'tf_phase_ir'    → IR + Magnitude + Phase       IR visible (20/40/40)
        'magnitude_only' → solo Magnitude,              IR oculto
        'magnitude_ir'   → IR + Magnitude solo          IR visible (30/70)
        'phase_only'     → solo Phase,                  IR oculto
        'phase_ir'       → IR + Phase solo              IR visible (30/70)
        IR always controlled by Cmd+I only — never by panel X/+ buttons.
        Positions are computed entirely by _relayout() — no math here.
        """
        # Restore ax_ph to phase mode when leaving magnitude_magnitude (any variant)
        if getattr(self, '_view_mag_mag', False) and mode not in ('magnitude_magnitude',
                                                                   'magnitude_magnitude_ir'):
            self._view_mag_mag = False
            self.ax_ph.set_ylim(-185, 185)
            self.ax_ph.set_ylabel('Phase (deg)', fontsize=6, color='#6a7a6a', labelpad=0)
            self.ax_ph.set_yticks([-180, -90, 0, 90, 180])
            self.ax_ph.tick_params(axis='y', labelsize=6, colors='#6a7a6a', length=0)

        self._current_view_mode = mode

        # ── magnitude_magnitude[_ir]: reconfigure ax_ph as a second magnitude axis ──
        if mode in ('magnitude_magnitude', 'magnitude_magnitude_ir'):
            self._view_mag_mag = True
            _ymin, _ymax = self.ax_tf.get_ylim()
            _step   = 6 if (_ymax - _ymin) <= 36 else 12
            _yticks = [v for v in range(int(_ymin) - _step, int(_ymax) + _step + 1, _step)
                       if _ymin <= v <= _ymax]
            self.ax_ph.set_yticks(_yticks)
            self.ax_ph.set_ylim(_ymin, _ymax)
            self.ax_ph.set_ylabel('Level (dB)', fontsize=6, color='#6a7a6a', labelpad=0)
            self.ax_ph.set_yscale('linear')
            self.ax_ph.tick_params(axis='y', labelsize=6, colors='#6a7a6a', length=0)

        # ── Phase trace lines visibility ──
        show_ph_lines = mode in ('tf_phase', 'tf_phase_ir', 'phase_only',
                                 'phase_ir', 'magnitude_magnitude',
                                 'magnitude_magnitude_ir')
        for ln in self._trace_ph_lines:
            ln.set_visible(show_ph_lines)

        # ── Smooth button visibility ──
        if hasattr(self, '_smooth_btn'):
            self._smooth_btn.setVisible(
                mode in ('tf_phase', 'tf_phase_ir', 'magnitude_only',
                         'magnitude_ir', 'magnitude_magnitude',
                         'magnitude_magnitude_ir'))

        # ── Corner +/✕ buttons ──
        if hasattr(self, '_mag_close_btn'):
            mag_dual = mode in ('tf_phase', 'tf_phase_ir')
            self._mag_close_btn.setText('✕' if mag_dual else '＋')
        if hasattr(self, '_ph_close_btn'):
            ph_dual = mode in ('tf_phase', 'tf_phase_ir')
            self._ph_close_btn.setText('✕' if ph_dual else '＋')

        # ── Layout: positions + visibility computed here ──
        self._relayout()

        self.draw_idle()
        if hasattr(self, '_smooth_btn') or hasattr(self, '_mag_lbl'):
            self._position_tf_overlay()

    # ── Panel overlays (labels + resolution buttons inside each panel) ──

    def _build_tf_overlay(self):
        _lbl_ss = (
            'QPushButton{'
            f'  background:rgba(20,26,20,180); color:{TEXT_HI};'
            '   border:none; border-radius:2px;'
            '   font-size:11px; font-weight:bold; padding:2px 6px;'
            '}'
            'QPushButton:hover{'
            f'  background:rgba(42,60,42,200); color:{GREEN};'
            '}'
        )
        _res_ss = (
            'QPushButton{'
            f'  background:#1e221e; color:{TEXT_HI};'
            '   border:1px solid #3a3a3a; border-radius:2px;'
            '   font-size:10px; padding:2px 6px;'
            '}'
            'QPushButton:hover{'
            f'  background:#2a322a; border-color:{GREEN};'
            '}'
        )
        _eng_ss = (
            'QPushButton{'
            f'  background:rgba(20,26,20,160); color:#6a8a6a;'
            '   border:none; border-radius:2px;'
            '   font-size:9px; padding:2px 5px;'
            '}'
        )
        _xplus_ss = (
            'QPushButton{'
            f'  background:rgba(20,26,20,180); color:{TEXT_MID};'
            '   border:1px solid #3a3a3a; border-radius:2px;'
            '   font-size:11px; font-weight:bold; padding:0px 5px;'
            '}'
            'QPushButton:hover{'
            f'  background:rgba(42,60,42,210); color:{TEXT_HI};'
            '}'
        )
        _gear_ss = (
            'QPushButton{'
            f'  background:#1e221e; color:{TEXT_MID};'
            '   border:1px solid #3a3a3a; border-radius:2px;'
            '   font-size:12px; padding:1px 4px;'
            '}'
            'QPushButton:hover{'
            f'  background:#2a322a; color:{TEXT_HI};'
            '}'
        )

        # ── Magnitude label — top-left of ax_tf ──────────────────────
        self._mag_lbl = QPushButton('Magnitude ▾', self)
        self._mag_lbl.setStyleSheet(_lbl_ss)
        self._mag_lbl.setCursor(Qt.CursorShape.PointingHandCursor)
        self._mag_lbl.clicked.connect(self._show_mag_panel_menu)
        self._mag_lbl.raise_()

        # ── Engine name — top-right of ax_tf (clickable: selects engine) ──
        self._mag_eng_lbl = QPushButton('—', self)
        self._mag_eng_lbl.setStyleSheet(_eng_ss)
        self._mag_eng_lbl.setCursor(Qt.CursorShape.PointingHandCursor)
        self._mag_eng_lbl.clicked.connect(
            lambda: self.on_engine_select_menu(self._mag_eng_lbl)
            if self.on_engine_select_menu else None)
        self._mag_eng_lbl.raise_()

        # ── X / + button — bottom-left of ax_tf ──────────────────────
        self._mag_close_btn = QPushButton('✕', self)
        self._mag_close_btn.setStyleSheet(_xplus_ss)
        self._mag_close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._mag_close_btn.clicked.connect(self._on_close_mag)
        self._mag_close_btn.raise_()

        # ── Phase label — top-left of ax_ph ──────────────────────────
        self._ph_lbl = QPushButton('Phase ▾', self)
        self._ph_lbl.setStyleSheet(_lbl_ss)
        self._ph_lbl.setCursor(Qt.CursorShape.PointingHandCursor)
        self._ph_lbl.clicked.connect(self._show_ph_panel_menu)
        self._ph_lbl.raise_()

        # ── Engine name — top-right of ax_ph (clickable: selects engine) ──
        self._ph_eng_lbl = QPushButton('—', self)
        self._ph_eng_lbl.setStyleSheet(_eng_ss)
        self._ph_eng_lbl.setCursor(Qt.CursorShape.PointingHandCursor)
        self._ph_eng_lbl.clicked.connect(
            lambda: self.on_engine_select_menu(self._ph_eng_lbl)
            if self.on_engine_select_menu else None)
        self._ph_eng_lbl.raise_()

        # ── X / + button — bottom-left of ax_ph ──────────────────────
        self._ph_close_btn = QPushButton('✕', self)
        self._ph_close_btn.setStyleSheet(_xplus_ss)
        self._ph_close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._ph_close_btn.clicked.connect(self._on_close_ph)
        self._ph_close_btn.raise_()

        # ── Resolution button — bottom-right of ax_tf ─────────────────
        self._smooth_btn = QPushButton('1/12 Oct ▼', self)
        self._smooth_btn.setStyleSheet(_res_ss)
        self._smooth_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._smooth_btn.setToolTip('Smoothing (linked Mag + Phase)')
        self._smooth_btn.clicked.connect(self._show_smooth_menu)
        self._smooth_btn.raise_()

        # ── Gear button — bottom-right of ax_tf (right of resolution) ─
        self._mag_gear_btn = QPushButton('⚙', self)
        self._mag_gear_btn.setStyleSheet(_gear_ss)
        self._mag_gear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._mag_gear_btn.setToolTip('Graph Settings')
        self._mag_gear_btn.clicked.connect(self._on_gear)
        self._mag_gear_btn.raise_()

        # ── Resolution button — bottom-right of ax_ph ─────────────────
        self._ph_smooth_btn = QPushButton('1/12 Oct ▼', self)
        self._ph_smooth_btn.setStyleSheet(_res_ss)
        self._ph_smooth_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._ph_smooth_btn.setToolTip('Smoothing (linked Mag + Phase)')
        self._ph_smooth_btn.clicked.connect(self._show_smooth_menu)
        self._ph_smooth_btn.raise_()

        # ── Gear button — bottom-right of ax_ph (right of resolution) ─
        self._ph_gear_btn = QPushButton('⚙', self)
        self._ph_gear_btn.setStyleSheet(_gear_ss)
        self._ph_gear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._ph_gear_btn.setToolTip('Graph Settings')
        self._ph_gear_btn.clicked.connect(self._on_gear)
        self._ph_gear_btn.raise_()

        self.on_smooth_changed = None   # callable(fraction) — MainWindow lo enlaza
        self._position_tf_overlay()

    def _show_mag_panel_menu(self):
        """Slot-1 panel menu — choose what this panel shows."""
        _ss = (
            'QMenu{background:#1a1a1a;color:#cccccc;border:1px solid #3a3a3a;'
            'font-size:11px;padding:2px 0;}'
            'QMenu::item{padding:5px 20px 5px 10px;}'
            'QMenu::item:selected{background:#2a382a;color:#88cc88;}'
            'QMenu::item:checked{font-weight:bold;}'
        )
        menu = QMenu(self)
        menu.setStyleSheet(_ss)
        _cur1 = getattr(self, '_slot1_type', 'magnitude')
        for name in ('Magnitude', 'Phase', 'RTA', 'Spectrograph'):
            act = menu.addAction(name)
            act.setCheckable(True)
            act.setChecked(name.lower() == _cur1)
            act.triggered.connect(
                lambda checked, n=name: self.on_panel1_changed(n) if self.on_panel1_changed else None)
        menu.exec(self._mag_lbl.mapToGlobal(self._mag_lbl.rect().bottomLeft()))

    def _show_ph_panel_menu(self):
        """Slot-2 panel menu — choose what this panel shows."""
        _ss = (
            'QMenu{background:#1a1a1a;color:#cccccc;border:1px solid #3a3a3a;'
            'font-size:11px;padding:2px 0;}'
            'QMenu::item{padding:5px 20px 5px 10px;}'
            'QMenu::item:selected{background:#2a382a;color:#88cc88;}'
            'QMenu::item:checked{font-weight:bold;}'
        )
        menu = QMenu(self)
        menu.setStyleSheet(_ss)
        _cur2 = getattr(self, '_slot2_type', 'phase')
        for name in ('Magnitude', 'Phase', 'RTA', 'Spectrograph'):
            act = menu.addAction(name)
            act.setCheckable(True)
            act.setChecked(name.lower() == _cur2)
            act.triggered.connect(
                lambda checked, n=name: self.on_panel2_changed(n) if self.on_panel2_changed else None)
        menu.exec(self._ph_lbl.mapToGlobal(self._ph_lbl.rect().bottomLeft()))

    def _emit_toggle_ir(self):
        if self.on_toggle_ir:
            self.on_toggle_ir()

    def add_target_curve(self, freqs, db, color='#FFD60A', name='Target'):
        """Add a dashed target curve overlay on the TF magnitude axis."""
        line, = self.ax_tf.plot(
            freqs, db, '--', color=color, linewidth=1.4,
            alpha=0.75, zorder=4, label=name)
        self._target_lines.append(line)
        self.draw_idle()
        return len(self._target_lines) - 1

    def remove_target_curve(self, idx):
        if 0 <= idx < len(self._target_lines):
            try:
                self._target_lines[idx].remove()
            except Exception:
                pass
            self._target_lines.pop(idx)
            self.draw_idle()

    def set_target_visible(self, idx, visible: bool):
        if 0 <= idx < len(self._target_lines):
            self._target_lines[idx].set_visible(visible)
            self.draw_idle()

    def update_target_curves(self, curves):
        """
        Full redraw of all target curve overlays on ax_tf.
        Replaces the old per-index add/remove API.
        curves: list of TargetCurve objects.
        """
        # Clear all existing target artists
        for ln in list(self._target_lines):
            try: ln.remove()
            except Exception: pass
        self._target_lines.clear()
        for poly in list(self._target_fills):
            try: poly.remove()
            except Exception: pass
        self._target_fills.clear()

        for tc in curves:
            if not tc.visible:
                continue
            mag = tc.mag_db + tc.offset_db
            lw  = float(tc.line_width)
            ln, = self.ax_tf.semilogx(
                tc.freqs, mag, '--',
                color=tc.color, linewidth=lw,
                alpha=0.80, zorder=4, label=tc.name)
            self._target_lines.append(ln)
            if tc.tolerance and tc.tolerance_db > 0:
                tdb = tc.tolerance_db
                # Shaded acceptance band
                poly = self.ax_tf.fill_between(
                    tc.freqs,
                    mag - tdb,
                    mag + tdb,
                    color=tc.color, alpha=0.16, zorder=3)
                self._target_fills.append(poly)
                # Upper and lower edge lines
                lo, = self.ax_tf.semilogx(
                    tc.freqs, mag + tdb, ':',
                    color=tc.color, linewidth=0.8, alpha=0.55, zorder=4)
                hi, = self.ax_tf.semilogx(
                    tc.freqs, mag - tdb, ':',
                    color=tc.color, linewidth=0.8, alpha=0.55, zorder=4)
                self._target_lines.extend([lo, hi])

        self.draw_idle()

    def _position_tf_overlay(self):
        """Posiciona los 10 overlays de esquinas en cada panel — estilo SMAART."""
        w, h = self.width(), self.height()
        if w < 30 or h < 30:
            return

        def _ax_rect(ax):
            """(left, top, right, bottom) en coordenadas Qt px (y=0 arriba)."""
            p = ax.get_position()
            return (int(w * p.x0), int(h * (1 - p.y1)),
                    int(w * p.x1), int(h * (1 - p.y0)))

        RES_W, RES_H = 84, 20
        GEAR_W, GEAR_H = 22, 20
        LBL_H  = 20
        ENG_H  = 18
        XP_W, XP_H = 22, 20   # X / + close-add button

        tf_vis = self.ax_tf.get_visible()
        ph_vis = self.ax_ph.get_visible()

        # ── MAGNITUDE panel ───────────────────────────────────────────
        if tf_vis:
            l, t, r, b = _ax_rect(self.ax_tf)

            # Top-left: Magnitude ▾
            if hasattr(self, '_mag_lbl'):
                lw = max(self._mag_lbl.sizeHint().width(), 90)
                self._mag_lbl.resize(lw, LBL_H)
                self._mag_lbl.move(l + 4, t + 4)
                self._mag_lbl.setVisible(True)

            # Top-right: engine name
            if hasattr(self, '_mag_eng_lbl'):
                ew = max(self._mag_eng_lbl.sizeHint().width(), 60)
                self._mag_eng_lbl.resize(ew, ENG_H)
                self._mag_eng_lbl.move(r - ew - 4, t + 4)
                self._mag_eng_lbl.setVisible(True)

            # Bottom-left: ✕ or ＋
            if hasattr(self, '_mag_close_btn'):
                self._mag_close_btn.resize(XP_W, XP_H)
                self._mag_close_btn.move(l + 4, b - XP_H - 4)
                self._mag_close_btn.setVisible(True)

            # Bottom-right: gear ⚙ then resolution
            if hasattr(self, '_mag_gear_btn') and hasattr(self, '_smooth_btn'):
                self._mag_gear_btn.resize(GEAR_W, GEAR_H)
                self._mag_gear_btn.move(r - GEAR_W - 2, b - GEAR_H - 4)
                self._mag_gear_btn.setVisible(True)
                self._smooth_btn.resize(RES_W, RES_H)
                self._smooth_btn.move(r - GEAR_W - RES_W - 6, b - RES_H - 4)
                self._smooth_btn.setVisible(True)
        else:
            for attr in ('_mag_lbl', '_mag_eng_lbl', '_mag_close_btn',
                         '_smooth_btn', '_mag_gear_btn'):
                if hasattr(self, attr):
                    getattr(self, attr).setVisible(False)

        # ── PHASE panel ───────────────────────────────────────────────
        if ph_vis:
            l, t, r, b = _ax_rect(self.ax_ph)

            # Top-left: Phase ▾ (or Magnitude ▾ in mag_mag mode — needs more width)
            if hasattr(self, '_ph_lbl'):
                lw = max(self._ph_lbl.sizeHint().width(), 90)
                self._ph_lbl.resize(lw, LBL_H)
                self._ph_lbl.move(l + 4, t + 4)
                self._ph_lbl.setVisible(True)

            # Top-right: engine name
            if hasattr(self, '_ph_eng_lbl'):
                ew = max(self._ph_eng_lbl.sizeHint().width(), 60)
                self._ph_eng_lbl.resize(ew, ENG_H)
                self._ph_eng_lbl.move(r - ew - 4, t + 4)
                self._ph_eng_lbl.setVisible(True)

            # Bottom-left: ✕ or ＋
            if hasattr(self, '_ph_close_btn'):
                self._ph_close_btn.resize(XP_W, XP_H)
                self._ph_close_btn.move(l + 4, b - XP_H - 4)
                self._ph_close_btn.setVisible(True)

            # Bottom-right: gear ⚙ then resolution
            if hasattr(self, '_ph_gear_btn') and hasattr(self, '_ph_smooth_btn'):
                self._ph_gear_btn.resize(GEAR_W, GEAR_H)
                self._ph_gear_btn.move(r - GEAR_W - 2, b - GEAR_H - 4)
                self._ph_gear_btn.setVisible(True)
                self._ph_smooth_btn.resize(RES_W, RES_H)
                self._ph_smooth_btn.move(r - GEAR_W - RES_W - 6, b - RES_H - 4)
                self._ph_smooth_btn.setVisible(True)
        else:
            for attr in ('_ph_lbl', '_ph_eng_lbl', '_ph_close_btn',
                         '_ph_smooth_btn', '_ph_gear_btn'):
                if hasattr(self, attr):
                    getattr(self, attr).setVisible(False)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, '_smooth_btn') or hasattr(self, '_mag_lbl'):
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
        txt = f'{label} ▼'
        # Sync both resolution buttons — resolution is always linked
        if hasattr(self, '_smooth_btn'):
            self._smooth_btn.setText(txt)
        if hasattr(self, '_ph_smooth_btn'):
            self._ph_smooth_btn.setText(txt)
        if self.on_smooth_changed:
            self.on_smooth_changed(frac)

    def _on_gear(self):
        """Gear button on any panel → open Graph Settings (TF tab)."""
        if self.on_gear_clicked:
            self.on_gear_clicked()

    def _on_close_mag(self):
        """X/+ on Magnitude panel corner."""
        if self.on_close_mag_panel:
            self.on_close_mag_panel()

    def _on_close_ph(self):
        """X/+ on Phase panel corner."""
        if self.on_close_ph_panel:
            self.on_close_ph_panel()

    def update_engine_label(self, name: str):
        """Update top-right engine name on both panels."""
        if hasattr(self, '_mag_eng_lbl'):
            self._mag_eng_lbl.setText(name)
            self._mag_eng_lbl.adjustSize()
        if hasattr(self, '_ph_eng_lbl'):
            self._ph_eng_lbl.setText(name)
            self._ph_eng_lbl.adjustSize()
        self._position_tf_overlay()

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
        ax.tick_params(axis='both', labelsize=6, colors=TEXT_MID, length=0)
        ax.grid(True, which='major', lw=0.4, color='#1e261e')
        ax.grid(True, which='minor', lw=0.25, color='#171d17', ls=':')
        for sp in ax.spines.values():
            sp.set_color(BORDER)
            sp.set_linewidth(0.6)

    # ── Engine color / highlight API ─────────────────────────────────

    def set_engine_colors(self, colors: list):
        """
        Actualiza los colores de TF, Phase e IR para todos los engines.
        TF y Phase del mismo engine siempre usan el mismo color.
        """
        self._eng_colors = list(colors)
        for i, color in enumerate(colors):
            if i < len(self._lines_tf):
                self._lines_tf[i].set_color(color)
                self._lines_ph[i].set_color(color)
        # IR sigue al engine primario
        if colors:
            self.line_ir.set_color(colors[0])
            self.line_ir_peak.set_color(colors[0])
        self.draw_idle()

    def highlight_engine(self, idx: int, show_avg: bool = False):
        """
        Destaca el engine seleccionado: TF, Phase e IR del engine activo
        se ven en primer plano; el resto se atenúa levemente.
        Cuando show_avg=True el promedio domina visualmente sobre todos los engines.
        """
        if not self._eng_colors:
            self.draw_idle()
            return

        if show_avg:
            # Con AVG activo: engines en segundo plano, AVG domina
            for i, (ltf, lph) in enumerate(zip(self._lines_tf, self._lines_ph)):
                if i == idx:
                    ltf.set(linewidth=1.8, alpha=0.80, zorder=4)
                    lph.set(linewidth=1.8, alpha=0.80, zorder=4)
                else:
                    ltf.set(linewidth=1.1, alpha=0.55, zorder=3)
                    lph.set(linewidth=1.1, alpha=0.55, zorder=3)
            # AVG encima de todo — línea gruesa, totalmente opaca
            self.line_tf_avg.set(linewidth=2.6, alpha=1.0, zorder=6)
            self.line_ph_avg.set(linewidth=2.6, alpha=1.0, zorder=6)
        else:
            # Sin AVG: engine seleccionado en primer plano, resto visible pero subordinado
            for i, (ltf, lph) in enumerate(zip(self._lines_tf, self._lines_ph)):
                if i == idx:
                    ltf.set(linewidth=2.2, alpha=1.0, zorder=5)
                    lph.set(linewidth=2.2, alpha=1.0, zorder=5)
                else:
                    ltf.set(linewidth=1.1, alpha=0.55, zorder=3)
                    lph.set(linewidth=1.1, alpha=0.55, zorder=3)
            # AVG oculto visualmente (los datos ya se limpian en _toggle_avg)
            self.line_tf_avg.set(linewidth=1.6, alpha=0.0, zorder=2)
            self.line_ph_avg.set(linewidth=1.6, alpha=0.0, zorder=2)

        # IR usa el color del engine seleccionado y queda en primer plano
        color = self._eng_colors[idx] if idx < len(self._eng_colors) else self._eng_colors[0]
        self.line_ir.set_color(color)
        self.line_ir_peak.set_color(color)
        self.line_ir.set(linewidth=1.4, alpha=1.0, zorder=5)
        self.line_ir_peak.set(alpha=0.7, zorder=4)
        self.draw_idle()

    def add_tf_engine_line(self, color: str) -> int:
        """Agrega una línea TF+Phase para un nuevo engine. Retorna el índice."""
        l, = self.ax_tf.semilogx([], [], color=color, lw=1.2, alpha=0.65)
        p, = self.ax_ph.semilogx([], [], color=color, lw=1.2, alpha=0.65)
        self._lines_tf.append(l)
        self._lines_ph.append(p)
        return len(self._lines_tf) - 1

    def remove_tf_engine_line(self, idx: int):
        """Elimina la línea del engine idx. Engines 0 y 1 son permanentes."""
        if idx < 2 or idx >= len(self._lines_tf):
            return
        self._lines_tf[idx].remove()
        self._lines_ph[idx].remove()
        del self._lines_tf[idx]
        del self._lines_ph[idx]
        self.draw_idle()

    def update_engine_n(self, idx: int, freqs, mag_db, phase_deg):
        """Actualiza TF + Phase para engines idx >= 2 (sin coherencia ni IR)."""
        if idx >= len(self._lines_tf):
            return
        if freqs is None or mag_db is None:
            self._lines_tf[idx].set_data([], [])
            self._lines_ph[idx].set_data([], [])
            return
        mask = (freqs >= 20) & (freqs <= 20000)
        f    = freqs[mask]
        self._lines_tf[idx].set_data(f, mag_db[mask])
        if getattr(self, '_view_mag_mag', False):
            self._lines_ph[idx].set_data(f, mag_db[mask])
        else:
            ph_wrap = ((phase_deg[mask] + 180.0) % 360.0) - 180.0
            self._lines_ph[idx].set_data(f, ph_wrap)

    # ── Update ────────────────────────────────────────────────────────

    def update_plots(self, freqs, gamma2, mag_db, phase_deg, gxx_db, ir,
                     coh_thresh=0.5, unwrap_phase=False,
                     coh_squared=True, phase_as_gd=False):
        # Sincronizar COLOR de Phase con TF cada frame (garantiza que siempre coincidan).
        # En magnitude_magnitude mode también sincronizar alpha/linewidth para que ambos
        # paneles muestren los engines con el mismo peso visual.
        for ltf, lph in zip(self._lines_tf, self._lines_ph):
            lph.set_color(ltf.get_color())
            if getattr(self, '_view_mag_mag', False):
                lph.set_alpha(ltf.get_alpha())
                lph.set_linewidth(ltf.get_linewidth())

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

        # Coherencia — γ² o γ según flag
        coh_display = gamma2[mask] if coh_squared else np.sqrt(np.clip(gamma2[mask], 0.0, 1.0))
        # no fill — solo actualizar la línea
        self.ax_coh.axhline(0.9 if coh_squared else 0.949, color=COH_COLOR,
                            lw=0.7, ls=':', alpha=0.4)
        self.line_coh.set_data(f, coh_display)

        # Máscara de blanqueo (siempre sobre γ²)
        ok = gamma2[mask] >= coh_thresh

        # Phase — sin gate de coherencia (se muestra siempre el rango completo)
        if getattr(self, '_view_mag_mag', False):
            # magnitude_magnitude mode: bottom panel mirrors top magnitude
            self.line_ph.set_data(f, mag_db[mask])
        elif phase_as_gd:
            ph_rad = np.unwrap(phase_deg[mask] * np.pi / 180.0)
            omega  = 2.0 * np.pi * f
            gd_ms  = -np.gradient(ph_rad, omega) * 1000.0
            gd_ms  = np.clip(gd_ms, -500.0, 500.0)
            self.line_ph.set_data(f, gd_ms)
        else:
            if unwrap_phase:
                ph_data = np.unwrap(phase_deg[mask] * np.pi / 180.0) * 180.0 / np.pi
            else:
                ph_data = ((phase_deg[mask] + 180.0) % 360.0) - 180.0
            self.line_ph.set_data(f, ph_data)

        # IR — dos modos controlados por _ir_centered:
        #   False (raw)      → eje de tiempo real; pico en su posición física
        #   True  (centrado) → pico en t=0 con zoom ±50 ms (activado por Find Delay)
        #
        # NOTA: irfft(H, n=N_ir) con H de nperseg//2+1 bins tiene
        #   dt = nperseg / (N_ir * fs), NO 1/fs.  Factor 4× si se usa 1/fs.
        if ir is not None and len(ir) > 0:
            N_ir    = len(ir)
            nperseg = getattr(self, '_nperseg', N_ir // 4)
            dt_ms   = nperseg * 1000.0 / (N_ir * float(self._fs))
            t_abs   = (np.arange(N_ir) - N_ir // 2) * dt_ms   # eje real simétrico en 0
            half_ms = (N_ir // 2) * dt_ms

            peak_idx = int(np.argmax(np.abs(ir)))
            peak_t   = t_abs[peak_idx]   # retardo real en ms

            if getattr(self, '_ir_centered', False):
                # ── Modo centrado: Find Delay ya fue presionado ──────────
                t_plot = t_abs - peak_t        # pico en 0 ms
                zoom   = min(50.0, half_ms)    # ventana ±50 ms alrededor del pico
                self.ax_ir.set_xlim(-zoom, zoom)
                self.line_ir_peak.set_xdata([0, 0])
                lbl = f'{peak_t:+.2f} ms'
            else:
                # ── Modo raw: eje de tiempo real, pico en su posición ────
                t_plot = t_abs
                self.ax_ir.set_xlim(-half_ms, half_ms)
                self.line_ir_peak.set_xdata([peak_t, peak_t])
                lbl = f'{peak_t:.2f} ms' if abs(peak_t) > 0.05 else ''

            xmin, xmax = self.ax_ir.get_xlim()
            mask_t = (t_plot >= xmin) & (t_plot <= xmax)
            if mask_t.sum() > 2:
                self.line_ir.set_data(t_plot[mask_t], ir[mask_t])
            self.txt_ir_delay.set_text(lbl)
            # Notify MainWindow so the active engine card can display the detected delay
            if abs(peak_t) > 0.01 and self.on_ir_peak_detected:
                self.on_ir_peak_detected(peak_t)

        self.draw_idle()

    def update_ir_only(self, ir):
        """Update ONLY the IR panel — for when the selected engine is not engine 0.
        Reuses identical logic to the IR section in update_plots()."""
        if ir is None or len(ir) == 0:
            self.draw_idle()
            return
        N_ir    = len(ir)
        nperseg = getattr(self, '_nperseg', N_ir // 4)
        dt_ms   = nperseg * 1000.0 / (N_ir * float(self._fs))
        t_abs   = (np.arange(N_ir) - N_ir // 2) * dt_ms
        half_ms = (N_ir // 2) * dt_ms

        peak_idx = int(np.argmax(np.abs(ir)))
        peak_t   = t_abs[peak_idx]

        if getattr(self, '_ir_centered', False):
            t_plot = t_abs - peak_t
            zoom   = min(50.0, half_ms)
            self.ax_ir.set_xlim(-zoom, zoom)
            self.line_ir_peak.set_xdata([0, 0])
            lbl = f'{peak_t:+.2f} ms'
        else:
            t_plot = t_abs
            self.ax_ir.set_xlim(-half_ms, half_ms)
            self.line_ir_peak.set_xdata([peak_t, peak_t])
            lbl = f'{peak_t:.2f} ms' if abs(peak_t) > 0.05 else ''

        xmin, xmax = self.ax_ir.get_xlim()
        mask_t = (t_plot >= xmin) & (t_plot <= xmax)
        if mask_t.sum() > 2:
            self.line_ir.set_data(t_plot[mask_t], ir[mask_t])
        self.txt_ir_delay.set_text(lbl)
        if abs(peak_t) > 0.01 and self.on_ir_peak_detected:
            self.on_ir_peak_detected(peak_t)
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
        if getattr(self, '_view_mag_mag', False):
            self.line_ph2.set_data(f, mag_db[mask])
        else:
            ph_wrap = ((phase_deg[mask] + 180.0) % 360.0) - 180.0
            self.line_ph2.set_data(f, ph_wrap)

    def update_avg(self, freqs, mag_db_avg, phase_deg_avg, gamma2, coh_thresh=0.5):
        """Actualiza la línea de promedio CH1+CH2."""
        if freqs is None:
            self.line_tf_avg.set_data([], [])
            self.line_ph_avg.set_data([], [])
            self._last_avg_freqs     = None
            self._last_avg_mag_db    = None
            self._last_avg_phase_deg = None
            self._last_avg_gamma2    = None
            return
        mask = (freqs >= 20) & (freqs <= 20000)
        f    = freqs[mask]
        self.line_tf_avg.set_data(f, mag_db_avg[mask])
        # Phase sin gate de coherencia — promedio siempre completo
        ph_wrap = ((phase_deg_avg[mask] + 180.0) % 360.0) - 180.0
        self.line_ph_avg.set_data(f, ph_wrap)
        # Store for trace capture
        self._last_avg_freqs     = f
        self._last_avg_mag_db    = mag_db_avg[mask]
        self._last_avg_phase_deg = ph_wrap
        self._last_avg_gamma2    = gamma2[mask] if gamma2 is not None else None

    def update_ir_range(self, ms):
        """Cambia el rango visible ±ms/2. El eje siempre es simétrico alrededor de 0."""
        half = ms / 2.0
        self.ax_ir.set_xlim(-half, half)
        self.draw_idle()

    def set_delay_ref(self, delay_ms):
        """
        Registra el retardo medido por Find Delay y activa el modo centrado.
        Desde este momento la IR muestra el pico en t=0 con zoom ±50 ms.
        """
        self._delay_ref_ms = delay_ms
        self._ir_centered  = True   # activar modo centrado
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

        # Phase envuelta completa — sin gate de coherencia (se dibuja siempre)
        ph_wrap = ((phase_deg[mask] + 180.0) % 360.0) - 180.0
        line_ph, = self.ax_ph.semilogx(
            f, ph_wrap, color=color, lw=1.0, alpha=0.70, ls='--', zorder=2)
        line_ph.set_visible(True)   # explicit: axis may be hidden at save time
        self._trace_ph_lines.append(line_ph)

        # Coherencia γ² — línea punteada sobre ax_coh
        coh_vals = gamma2[mask].clip(0.0, 1.0)
        line_coh, = self.ax_coh.semilogx(
            f, coh_vals, color=color, lw=0.8, alpha=0.55, ls=':', zorder=3)
        self._trace_coh_lines.append(line_coh)

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
            _all = (self._trace_tf_lines, self._trace_ph_lines,
                    self._trace_coh_lines, self._trace_ir_lines)
            for lst in _all:
                if idx < len(lst):
                    lst[idx].remove()
                    lst.pop(idx)
            self.draw_idle()

    def set_trace_visible(self, idx, visible):
        """Muestra u oculta la traza idx en todos los paneles."""
        if 0 <= idx < len(self._trace_tf_lines):
            _all = (self._trace_tf_lines, self._trace_ph_lines,
                    self._trace_coh_lines, self._trace_ir_lines)
            for lst in _all:
                if idx < len(lst):
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
        self.fig.subplots_adjust(left=0.04, right=0.96, top=0.995, bottom=0.025)

        setup_smaart_axis(self.ax, bg=BG_PLOT,
                          show_xlabels=True, show_xlabel=True)
        self.ax.set_ylim(-80, 6)
        self.ax.set_autoscale_on(False)        # eje Y fijo siempre
        self.ax.set_ylabel('Level (dB)', fontsize=6, color=TEXT_MID, labelpad=0)
        # Grilla horizontal cada 6 dB — estilo SMAART
        _grid_dbs = [-60, -54, -48, -42, -36, -30, -24, -18, -12, -6, 0]
        for _db in _grid_dbs:
            _col = '#232e23' if _db == 0 else '#181e18'
            _lw  = 0.6 if _db == 0 else 0.4
            self.ax.axhline(_db, color=_col, lw=_lw, ls='-', zorder=0)
        # Yticks alineados con las líneas de grid (cada 6 dB)
        self.ax.set_yticks(_grid_dbs)
        self.ax.tick_params(axis='y', labelsize=6, colors=TEXT_MID, length=0)
        for sp in self.ax.spines.values():
            sp.set_color(BORDER)

        # Línea AVG (opcional, no ligada a engines) — sin label para no mostrar leyenda permanente
        self.line_y_avg, = self.ax.semilogx([], [], color=CYAN, lw=1.5, alpha=0.90, ls='--')
        # Las líneas por engine se crean dinámicamente con add_sp_engine_line()

        # Crosshair cursor — visible, color claro
        self._cxh, = self.ax.plot([], [], color='#7fb87f', lw=0.8, ls='--',
                                   alpha=0.0, zorder=20)
        self._cyh, = self.ax.plot([], [], color='#7fb87f', lw=0.6, ls=':',
                                   alpha=0.0, zorder=20)

        # Sin leyenda permanente — el texto "AVG" solo aparece mientras hay datos activos
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

        # Botón engrane — oculto (el gear está en el header de CanvasOverlay)
        self._cfg_btn = QPushButton('⚙', self)
        self._cfg_btn.setFixedSize(24, 24)
        self._cfg_btn.setStyleSheet(_btn_ss)
        self._cfg_btn.clicked.connect(
            lambda: self.on_config_clicked() if self.on_config_clicked else None)
        self._cfg_btn.hide()   # no duplicar el gear del header
        self.on_config_clicked = None   # MainWindow lo enlaza

        self._position_overlay()

        # ── QLabel cursor overlay (idéntico al del Spectrograph) ──────
        self._freq_lbl = QLabel('', self)
        self._freq_lbl.setStyleSheet(
            'QLabel{color:#ffffff;background:rgba(0,0,0,175);'
            'padding:2px 7px;font-size:11px;font-family:Menlo,Monaco,monospace;'
            'border:1px solid #444;border-radius:3px;}')
        self._freq_lbl.hide()
        self.mpl_connect('axes_leave_event', self._on_axes_leave)
        self.setMouseTracking(True)

    def _on_axes_leave(self, event):
        self._freq_lbl.hide()
        if self.on_cursor_update:
            self.on_cursor_update('—')

    def _position_overlay(self):
        """Reposiciona los botones al redimensionar el canvas."""
        w, h = self.width(), self.height()
        if w < 30 or h < 30:
            return

        # subplots_adjust: right=0.935, bottom=0.07
        # En Qt: x_right = w*0.935,  y_ax_bottom = h*(1-0.07) = h*0.93
        rw = max(self._res_btn.sizeHint().width(), 84)
        rh = 22
        self._res_btn.resize(rw, rh)

        x_res = int(w * 0.935) - rw - 4
        y_res = int(h * 0.93) - rh - 2   # justo encima del borde inferior del ax
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
            self._last_lx    = lev
        else:
            fc  = self._centers
            lev = self._cpb(freqs, Gxx, fc, self._edges)
            self._last_freqs = fc
            self._last_lx    = lev

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

    def update_ch2_avg(self, lev_y2, lev_avg, freqs=None):
        """Actualiza la línea AVG del spectrum."""
        fc = freqs if freqs is not None else self._centers
        if lev_avg is not None and len(lev_avg) == len(fc):
            self.line_y_avg.set_data(fc, lev_avg)
        else:
            self.line_y_avg.set_data([], [])

    # ── Trazas almacenadas ────────────────────────────────────────────

    def store_trace(self, lev_x, lev_y, color, lev_freqs=None):
        """Dibuja trazas REF (--) y MED (:) como referencia estática."""
        # Usar el eje de frecuencias guardado con la traza; fallback a _centers
        if lev_freqs is not None and len(lev_freqs) == len(lev_x):
            fc = lev_freqs
        else:
            fc = self._centers
        # Truncar al mínimo para evitar shape mismatch
        n = min(len(fc), len(lev_x), len(lev_y))
        fc, lev_x, lev_y = fc[:n], lev_x[:n], lev_y[:n]
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
            if hasattr(self, '_freq_lbl'):
                self._freq_lbl.hide()
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
                info = f'SP  │  {fmt_freq(x)}'
        else:
            info = f'SP  │  {fmt_freq(x)}'

        # ── QLabel flotante junto al cursor ──────────────────────────
        if hasattr(self, '_freq_lbl'):
            x_qt = max(4, min(int(event.x) + 14, self.width() - 200))
            y_qt = max(4, min(int(self.height() - event.y) - 28, self.height() - 28))
            self._freq_lbl.setText(info)
            self._freq_lbl.adjustSize()
            self._freq_lbl.move(x_qt, y_qt)
            self._freq_lbl.show()
            self._freq_lbl.raise_()

        if self.on_cursor_update:
            self.on_cursor_update(info)
        self.draw_idle()


# ── Canvas de Espectrograma ───────────────────────────────────────────

class SpectrogramCanvas(FigureCanvas):
    """
    Espectrograma scrolling waterfall estilo SMAART:
      X = frecuencia (Hz, escala log)
      Y = tiempo — nuevo abajo, viejo arriba (↑ scroll hacia arriba)
      Color = nivel dBFS

    Optimizaciones:
      1. Ring buffer doble (2×N_TIME) — cero copias en el hot path,
         sin np.roll() en cada frame.
      2. Matplotlib blitting — solo redibuja el QuadMesh, no ejes/ticks.
         Ejes se dibujan una sola vez y se guardan como background.
    """

    N_TIME = 120   # filas de tiempo visibles (reducido para más velocidad)

    # Rango de color por defecto (dBFS)
    VMIN_DEFAULT = -90
    VMAX_DEFAULT =  -6

    def __init__(self):
        self.fig = Figure(facecolor=BG_PANEL)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # Ring buffer: el doble de N_TIME para slicing sin copias
        self._buf   = None   # shape (2 * N_TIME, N_FREQ)
        self._ptr   = 0      # write pointer (0..N_TIME-1)
        self._freqs = None
        self._mesh   = None
        self._cbar   = None
        self._cbar_ax = None  # fixed-position colorbar axes (cax approach)
        self._bg     = None   # background cacheado para blit
        self._vmin  = self.VMIN_DEFAULT
        self._vmax  = self.VMAX_DEFAULT
        # Cursor — QLabel Qt overlay (sin tocar el blit de matplotlib)
        self.on_cursor_update = None
        self._build()
        # Overlay de frecuencia: widget Qt flotante sobre el canvas
        self._freq_lbl = QLabel('', self)
        self._freq_lbl.setStyleSheet(
            'QLabel{color:#ffffff;background:rgba(0,0,0,175);'
            'padding:2px 7px;font-size:11px;font-family:Menlo,Monaco,monospace;'
            'border:1px solid #444;border-radius:3px;}')
        self._freq_lbl.hide()
        self.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.mpl_connect('axes_leave_event',    self._on_axes_leave)
        self.setMouseTracking(True)

    def _build(self):
        self.ax = self.fig.add_subplot(111, facecolor=BG_PLOT)
        self.fig.subplots_adjust(left=0.04, right=0.96, top=0.995, bottom=0.025)
        self.ax.set_xlabel('Frequency (Hz)', fontsize=6, color=TEXT_MID)
        self.ax.set_xscale('log')
        self.ax.set_xlim(20, 20000)
        self._apply_xticks()
        self._apply_xgrid()
        for sp in self.ax.spines.values():
            sp.set_color(BORDER)
        self.ax.tick_params(colors=TEXT_MID, labelsize=6)
        self.ax.set_yticks([])
        self.fig.patch.set_facecolor(BG_PANEL)
        self.draw()

    def _apply_xticks(self):
        _ft = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        self.ax.set_xticks(_ft)
        self.ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: fmt_freq(x)))
        self.ax.xaxis.set_minor_locator(ticker.NullLocator())

    def _apply_xgrid(self):
        """Grid vertical en frecuencias principales — visible sobre el mesh."""
        self.ax.set_axisbelow(False)
        self.ax.grid(True, which='major', axis='x',
                     color='#ffffff', alpha=0.08, linewidth=0.6, linestyle='--')

    def set_color_range(self, vmin: float, vmax: float):
        """Actualiza el rango de color dBFS en vivo."""
        self._vmin = vmin
        self._vmax = vmax
        if self._mesh is not None:
            self._mesh.set_clim(vmin, vmax)
            if self._cbar is not None:
                self._cbar.update_normal(self._mesh)
            # Invalidar bg para que el colorbar se redibuje
            self._bg = None
            self.draw_idle()

    def resizeEvent(self, event):
        """Al redimensionar, invalida el background cacheado."""
        super().resizeEvent(event)
        self._bg = None

    def update_spectrogram(self, freqs, Gxx):
        """
        Recibe PSD lineal → convierte a dBFS y escribe en el ring buffer.
        Dirección SMAART: nueva fila en la parte INFERIOR, scroll hacia arriba.
        """
        mask = (freqs >= 20) & (freqs <= 20000)
        f    = freqs[mask]
        row  = (10.0 * np.log10(Gxx[mask] + 1e-12)).astype(np.float32)

        # Si cambia el número de bins (ej. cambio de fs/nperseg) → reconstruir
        if self._buf is None or self._buf.shape[1] != len(f):
            self._buf   = np.full((2 * self.N_TIME, len(f)), -80.0, dtype=np.float32)
            self._ptr   = 0
            self._freqs = f
            self._mesh  = None
            self._bg    = None

        # ── Ring buffer — sin np.roll, sin allocación ─────────────────
        # Escribir la nueva fila en las dos mitades del buffer doble
        self._buf[self._ptr]              = row
        self._buf[self._ptr + self.N_TIME] = row
        self._ptr = (self._ptr + 1) % self.N_TIME

        # Vista de N_TIME filas: ptr es la más antigua → ptr+N_TIME-1 la más nueva
        # Esto da: display[0] = más antigua (arriba), display[-1] = más nueva (abajo)
        # → comportamiento SMAART: nuevo abajo, viejo arriba
        display = self._buf[self._ptr : self._ptr + self.N_TIME]

        if self._mesh is None:
            self._rebuild_mesh(f, display)
            return

        # ── Fast blit — solo redibuja el mesh, no los ejes ───────────
        self._mesh.set_array(display)

        if self._bg is None:
            # Primera vez o después de resize: capturar background
            self.draw()
            self._bg = self.copy_from_bbox(self.ax.bbox)
        else:
            self.restore_region(self._bg)
            self.ax.draw_artist(self._mesh)
            self.blit(self.ax.bbox)

    def _rebuild_mesh(self, f, display):
        """Construye pcolormesh desde cero (primer frame o cambio de resolución)."""
        self.ax.cla()
        self.ax.set_facecolor('#000000')   # fondo negro puro para mejor contraste
        self.ax.set_xlabel('Frequency (Hz)', fontsize=6, color=TEXT_MID)

        t = np.arange(self.N_TIME)  # Y: 0=más antiguo (arriba), N-1=más nuevo (abajo)
        self._mesh = self.ax.pcolormesh(
            f, t, display,
            cmap=_SGRAM_CMAP,
            vmin=self._vmin,
            vmax=self._vmax,
            shading='nearest',
            animated=True,   # necesario para blit correcto
        )
        self.ax.set_xscale('log')
        self.ax.set_xlim(20, 20000)
        # SMAART: nuevo abajo, viejo arriba — ylim invertido
        self.ax.set_ylim(self.N_TIME - 1, 0)
        self.ax.set_yticks([])
        self._apply_xticks()
        self._apply_xgrid()
        for sp in self.ax.spines.values():
            sp.set_color(BORDER)
        self.ax.tick_params(colors=TEXT_MID, labelsize=6)

        # Colorbar lateral con escala dBFS — usa cax fijo para no mover el eje X
        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
            self._cbar = None
        if self._cbar_ax is not None:
            try:
                self._cbar_ax.remove()
            except Exception:
                pass
            self._cbar_ax = None
        # Fixed-position colorbar axes: [left, bottom, width, height] in fig coords.
        # Main ax ends at right=0.945 (canonical margin); colorbar at 0.950–0.962.
        # Height = top(0.995) − bottom(0.025) = 0.970
        self._cbar_ax = self.fig.add_axes([0.965, 0.025, 0.012, 0.970])
        self._cbar = self.fig.colorbar(
            self._mesh, cax=self._cbar_ax,
            ticks=[self._vmin,
                   (self._vmin + self._vmax) / 2,
                   self._vmax],
        )
        self._cbar.ax.tick_params(colors=TEXT_MID, labelsize=6)
        self._cbar.set_label('dBFS', color=TEXT_MID, fontsize=6)
        # Etiquetas de colorbar con valores dBFS
        self._cbar.ax.yaxis.set_tick_params(color=TEXT_MID)
        for t_lbl in self._cbar.ax.get_yticklabels():
            t_lbl.set_color(TEXT_MID)

        self.fig.subplots_adjust(left=0.04, right=0.96, top=0.995, bottom=0.025)

        self.draw()
        self._bg = self.copy_from_bbox(self.ax.bbox)

    def _on_mouse_move(self, event):
        """Cursor: QLabel Qt flotante — sin tocar el blit del mesh."""
        if event.inaxes is not self.ax or self._mesh is None:
            self._freq_lbl.hide()
            return
        f_hz = event.xdata
        if f_hz is None or f_hz <= 0:
            self._freq_lbl.hide()
            return

        # ── Frecuencia ──────────────────────────────────────────────
        freq_str = fmt_freq(f_hz)

        # ── dBFS en la celda bajo el cursor ─────────────────────────
        db_str = ''
        if self._buf is not None and self._freqs is not None and len(self._freqs) > 1:
            try:
                display  = self._buf[self._ptr : self._ptr + self.N_TIME]
                row_idx  = int(round(event.ydata))
                row_idx  = max(0, min(self.N_TIME - 1, row_idx))
                col_idx  = int(np.searchsorted(self._freqs, f_hz))
                col_idx  = max(0, min(len(self._freqs) - 1, col_idx))
                db_val   = float(display[row_idx, col_idx])
                db_str   = f'   {db_val:+.1f} dBFS'
            except Exception:
                pass

        info = f'{freq_str}{db_str}'

        # ── Posicionar QLabel (coordenadas Qt: origen arriba-izq) ───
        x_qt = int(event.x) + 14
        y_qt = int(self.height() - event.y) - 28
        x_qt = max(4, min(x_qt, self.width()  - 180))
        y_qt = max(4, min(y_qt, self.height() - 28))
        self._freq_lbl.setText(info)
        self._freq_lbl.adjustSize()
        self._freq_lbl.move(x_qt, y_qt)
        self._freq_lbl.show()
        self._freq_lbl.raise_()

        # ── Status bar ──────────────────────────────────────────────
        if self.on_cursor_update:
            self.on_cursor_update(f'SGram  {info}')

    def _on_axes_leave(self, event):
        """Ocultar QLabel al salir del eje — no toca el blit."""
        self._freq_lbl.hide()
        if self.on_cursor_update:
            self.on_cursor_update('—')

    def clear(self):
        self._buf  = None
        self._mesh = None
        self._bg   = None
        self._ptr  = 0
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
                 on_select=None, on_normalize=None, on_active_changed=None,
                 on_color_changed=None, on_trace_toggle=None, on_config=None,
                 name: str = '', parent=None):
        super().__init__(parent)
        self._color              = color
        self._number             = number
        self._name               = name.strip() or str(number)
        self._delay_comp_ms      = 0.0
        self._active             = False   # inicia inactivo; el usuario presiona ▶
        self._selected           = False
        self._trace_visible      = True    # dot toggle: muestra/oculta trazo
        self._on_active_changed  = on_active_changed   # callback(idx, active)
        self._on_color_changed   = on_color_changed    # callback(idx, color_hex)
        self._on_trace_toggle    = on_trace_toggle     # callback(idx, visible)
        self._on_config_cb       = on_config           # callback(idx)
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

        # Círculo de color — click = trace ON/OFF · right-click = color picker
        dot = QPushButton('●')
        dot.setStyleSheet(
            f'QPushButton{{color:{color};font-size:22px;background:transparent;'
            f'border:none;padding:0;margin:0;}}'
            f'QPushButton:hover{{color:#ffffff;}}')
        dot.setFixedSize(26, 26)
        dot.setCursor(Qt.CursorShape.PointingHandCursor)
        dot.setToolTip('Click: mostrar/ocultar trazo · Clic derecho: cambiar color')
        dot.clicked.connect(self._on_toggle_trace)
        dot.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        dot.customContextMenuRequested.connect(lambda _: self._on_pick_color())
        self._dot = dot
        r1.addWidget(dot)

        # Nombre (user-given name, or number as fallback)
        self._name_lbl = QLabel(self._name)
        self._name_lbl.setStyleSheet(
            f'color:#cccccc;font-size:13px;font-weight:bold;'
            f'background:transparent;')
        self._name_lbl.setWordWrap(False)
        r1.addWidget(self._name_lbl, stretch=1)

        # Botón ▶ (play/pause por engine)
        self.btn_play = QPushButton('▶')
        self.btn_play.setFixedSize(34, 28)
        self.btn_play.setCheckable(True)
        self.btn_play.setChecked(False)   # inicia sin marcar (inactivo)
        self.btn_play.setStyleSheet(
            f'QPushButton{{font-size:14px;padding:0;border:none;'
            f'background:transparent;color:{color};}}'
            f'QPushButton:checked{{color:{color};}}'
            f'QPushButton:!checked{{color:#444444;}}')
        self.btn_play.clicked.connect(self._on_toggle_active)
        r1.addWidget(self.btn_play)

        # Botón ⚙ — abre configuración del engine
        btn_cfg = QPushButton('⚙')
        btn_cfg.setFixedSize(22, 22)
        btn_cfg.setToolTip('Configurar engine')
        btn_cfg.setStyleSheet(
            f'QPushButton{{font-size:12px;padding:0;border:1px solid #2a2a2a;'
            f'background:transparent;color:#555;border-radius:3px;}}'
            f'QPushButton:hover{{color:{color};border-color:{color};}}')
        btn_cfg.clicked.connect(lambda: self._on_config_cb(self._number - 1)
                                if self._on_config_cb else None)
        r1.addWidget(btn_cfg)

        # Botón ✕ — elimina engine
        btn_rm = QPushButton('✕')
        btn_rm.setFixedSize(18, 22)
        btn_rm.setToolTip('Eliminar engine')
        btn_rm.setStyleSheet(
            'QPushButton{font-size:10px;padding:0;border:none;'
            'background:transparent;color:#3a3a3a;}'
            'QPushButton:hover{color:#cc4444;}')
        btn_rm.clicked.connect(on_remove)
        r1.addWidget(btn_rm)

        card_lay.addLayout(r1)

        # ── Fila 2: M + R bars (ambas verdes) ───────────────────────────
        _BAR_GREEN = '#4ab46a'
        _BAR_CSS   = (f'QProgressBar{{background:#222;border:none;border-radius:2px;}}'
                      f'QProgressBar::chunk{{background:{_BAR_GREEN};border-radius:2px;}}')

        r2 = QHBoxLayout(); r2.setSpacing(4)

        bars_col = QVBoxLayout(); bars_col.setSpacing(4)

        row_m = QHBoxLayout(); row_m.setSpacing(4)
        lbl_m = QLabel('M')
        lbl_m.setFixedWidth(10)
        lbl_m.setStyleSheet(f'color:{TEXT_DIM};font-size:8px;background:transparent;')
        self.bar_m = QProgressBar()
        self.bar_m.setRange(0, 100); self.bar_m.setValue(0)
        self.bar_m.setTextVisible(False); self.bar_m.setFixedHeight(6)
        self.bar_m.setStyleSheet(_BAR_CSS)
        row_m.addWidget(lbl_m); row_m.addWidget(self.bar_m)
        bars_col.addLayout(row_m)

        row_r = QHBoxLayout(); row_r.setSpacing(4)
        lbl_r = QLabel('R')
        lbl_r.setFixedWidth(10)
        lbl_r.setStyleSheet(f'color:{TEXT_DIM};font-size:8px;background:transparent;')
        self.bar_r = QProgressBar()
        self.bar_r.setRange(0, 100); self.bar_r.setValue(0)
        self.bar_r.setTextVisible(False); self.bar_r.setFixedHeight(6)
        self.bar_r.setStyleSheet(_BAR_CSS)
        row_r.addWidget(lbl_r); row_r.addWidget(self.bar_r)
        bars_col.addLayout(row_r)

        r2.addLayout(bars_col, stretch=1)

        self.lbl_db = QLabel('—')   # oculto — referencia interna para compatibilidad
        self.lbl_db.hide()

        card_lay.addLayout(r2)

        # ── Fila 3: delay info ────────────────────────────────────────
        r3 = QHBoxLayout(); r3.setSpacing(6)

        self.lbl_delay = QLabel('—')
        self.lbl_delay.setStyleSheet(
            'color:#8ab48a;font-size:12px;background:transparent;'
            'font-family:monospace;')
        self.lbl_delay.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        r3.addWidget(self.lbl_delay, stretch=1)

        # Botón delay finder — círculo pequeño entre delay y norm
        btn_find = QPushButton('●')
        btn_find.setFixedSize(18, 18)
        btn_find.setStyleSheet(
            'QPushButton{font-size:10px;padding:0;border:1px solid #2a5a5a;'
            'color:#3a9a9a;background:transparent;border-radius:9px;}'
            'QPushButton:hover{border-color:#4acaca;color:#4acaca;}')
        btn_find.setToolTip('Delay Finder')
        btn_find.clicked.connect(on_find_delay)
        r3.addWidget(btn_find)

        # Botón NORM (normaliza la TF a 0 dB en banda de referencia)
        if on_normalize is not None:
            btn_norm = QPushButton('norm')
            btn_norm.setFixedHeight(18)
            btn_norm.setStyleSheet(
                'QPushButton{font-size:9px;padding:2px 7px;border:1px solid #2a4a3a;'
                'color:#3a6a4a;background:transparent;border-radius:3px;}'
                'QPushButton:hover{border-color:#4a9a6a;color:#4a9a6a;}')
            btn_norm.clicked.connect(on_normalize)
            r3.addWidget(btn_norm)

        card_lay.addLayout(r3)

        # ── Layout raíz ──────────────────────────────────────────────
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 4)
        root.setSpacing(0)
        root.addWidget(self._card)

    # ── Slots ─────────────────────────────────────────────────────────

    def _on_toggle_trace(self):
        """Dot click — apaga/prende el trazo en el canvas."""
        self._trace_visible = not self._trace_visible
        # Dim dot color when trace is off — no GraphicsEffect (evita freeze)
        dim_color = self._color if self._trace_visible else '#333333'
        self._dot.setStyleSheet(
            f'QPushButton{{color:{dim_color};font-size:22px;background:transparent;'
            f'border:none;padding:0;margin:0;}}'
            f'QPushButton:hover{{color:{self._color};}}')
        if self._on_trace_toggle is not None:
            self._on_trace_toggle(self._number - 1, self._trace_visible)

    def _on_toggle_active(self, checked: bool):
        self._active = checked
        # Sync button visual state (in case called directly instead of via click)
        self.btn_play.setChecked(checked)
        self.btn_play.setText('⏸' if checked else '▶')
        if self._on_active_changed is not None:
            self._on_active_changed(self._number - 1, checked)   # idx=0-based

    def _on_body_click(self):
        if self._on_select_cb:
            self._on_select_cb()

    def _on_pick_color(self):
        """Abre QColorDialog para que el usuario elija el color del engine."""
        from PyQt6.QtWidgets import QColorDialog
        from PyQt6.QtGui import QColor as _QColor
        c = QColorDialog.getColor(_QColor(self._color), self, 'Color del Engine')
        if c.isValid():
            self._apply_color(c.name())
            if self._on_color_changed is not None:
                self._on_color_changed(self._number - 1, c.name())

    def _apply_color(self, color: str):
        """Actualiza todos los elementos visuales del engine con el nuevo color."""
        self._color = color
        # Dot
        self._dot.setStyleSheet(
            f'QPushButton{{color:{color};font-size:22px;background:transparent;'
            f'border:none;padding:0;margin:0;}}'
            f'QPushButton:hover{{color:#ffffff;}}')
        # Card — respeta si está seleccionado o no
        if self._selected:
            self._card.setStyleSheet(
                f'QFrame#tfcard{{background:#1c2222;border:2px solid {color};'
                f'border-radius:6px;}}')
        else:
            self._card.setStyleSheet(
                f'QFrame#tfcard{{background:#161919;border:2px solid {color}55;'
                f'border-radius:6px;}}')
        # Botón play (barra M queda verde fija, no cambia con color del engine)
        self.btn_play.setStyleSheet(
            f'QPushButton{{font-size:11px;padding:0;border:none;'
            f'background:transparent;color:{color};}}'
            f'QPushButton:checked{{color:{color};}}'
            f'QPushButton:!checked{{color:#444444;}}')

    def set_selected(self, selected: bool):
        self._selected = selected
        if selected:
            self._card.setStyleSheet(
                f'QFrame#tfcard{{background:#1c2222;border:2px solid {self._color};'
                f'border-radius:6px;}}')
        else:
            self._card.setStyleSheet(
                f'QFrame#tfcard{{background:#161919;border:2px solid {self._color}55;'
                f'border-radius:6px;}}')

    # ── API pública ───────────────────────────────────────────────────

    def update_levels(self, m_db: float, r_db: float):
        def _pct(db): return max(0, min(100, int((db + 80) / 80 * 100)))
        def _color(db):
            if db >= -3.0:   return '#ef5350'   # rojo — peak
            if db >= -12.0:  return '#ffb74d'   # amarillo — warning
            return '#4ab46a'                     # verde — normal
        def _bar_css(db):
            c = _color(db)
            return (f'QProgressBar{{background:#222;border:none;border-radius:2px;}}'
                    f'QProgressBar::chunk{{background:{c};border-radius:2px;}}')
        self.bar_m.setValue(_pct(m_db)); self.bar_m.setStyleSheet(_bar_css(m_db))
        self.bar_r.setValue(_pct(r_db)); self.bar_r.setStyleSheet(_bar_css(r_db))
        self.lbl_db.setText(f'{m_db:.2f}')

    def set_delay(self, ms: float):
        self._delay_comp_ms = ms
        sign = '+' if ms >= 0 else ''
        self.lbl_delay.setText(f'{sign}{ms:.2f} ms')

    def reset_delay(self):
        self._delay_comp_ms = 0.0
        self.lbl_delay.setText('—')

    def show_detected_delay(self, ms: float):
        """Display IR-detected delay in the card label (read-only — does NOT apply compensation)."""
        sign = '+' if ms >= 0 else ''
        self.lbl_delay.setText(f'{sign}{ms:.2f} ms')

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

    _VIEWS = ['Magnitude', 'Phase', 'RTA', 'Spectrograph']

    def __init__(self, on_close, parent=None):
        super().__init__(parent)
        self._current_view = 'RTA'
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
            if view == 'Phase':
                self.canvas_meas.set_view_mode('phase_only')
            else:
                self.canvas_meas.set_view_mode('tf_phase')
        elif view == 'RTA':
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
        self._mic_cal         = None   # tuple (freqs, dB_correction) o None — kept for backward compat
        self._mic_cal_name    = ''     # nombre del archivo cargado
        self._channel_cal: dict = {}   # {ch_number (1-based): {'freqs': np.array, 'db': np.array, 'name': str}}
        self._channel_spl_offset: dict = {}  # {ch (1-based): float dB offset (dBFS→dBSPL)}
        self._target_curves: list = _make_builtin_target_curves()  # list[TargetCurve]
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

        # ── App icon — Dock + macOS title bar proxy icon ──────────────
        from PyQt6.QtGui import QIcon as _QIcon
        _logo = os.path.join(os.path.dirname(__file__), 'logo_512.png')
        if not os.path.exists(_logo):
            _logo = os.path.join(os.path.dirname(__file__), 'logo.png')
        if os.path.exists(_logo):
            _icon = _QIcon(_logo)
            self.setWindowIcon(_icon)
            QApplication.instance().setWindowIcon(_icon)
            # macOS: proxy icon in title bar (small icon next to title)
            self.setWindowFilePath(_logo)

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
        self._on_view_mode_changed('Magnitude')

    # ── Central widget ────────────────────────────────────────────────

    def _build_central(self):
        central = QWidget()
        self.setCentralWidget(central)
        # Outer layout: workspace tab bar spans full width at the very top,
        # then the three-column layout (left panel + center + right panel) below
        self._central_outer = QVBoxLayout(central)
        self._central_outer.setContentsMargins(0, 0, 0, 0)
        self._central_outer.setSpacing(0)
        _inner_container = QWidget()
        root = QHBoxLayout(_inner_container)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Panel izquierdo: guardar TXT (colapsable) ──
        root.addWidget(self._build_collapsible_save_panel())

        # ── Centro: info bar + splitter de paneles ──
        center = QWidget()
        cv = QVBoxLayout(center)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(0)
        self._info_bar_widget = self._build_info_bar()
        cv.addWidget(self._info_bar_widget)

        # ── Canvases ──────────────────────────────────────────────────
        self.canvas_meas  = MeasurementCanvas()   # IR + Phase + TF
        self.canvas_meas.on_cursor_update = self._update_cursor
        self.canvas_spec  = SpectrumCanvas()       # RTA / Spectrum
        self.canvas_spec.on_cursor_update  = self._update_cursor
        # ⚙ botón del canvas → abre el panel de settings
        self.canvas_spec.on_config_clicked = self._on_toggle_settings

        # Botón de suavizado del canvas TF → sync con cmb_smooth del panel
        self.canvas_meas.on_smooth_changed  = self._set_smooth_fraction
        # Panel labels inside canvas open the view-switcher dropdown
        self.canvas_meas.on_panel_menu      = self._show_panel_menu_from_canvas
        # Corner buttons — gear opens Graph Settings, X/+ toggles panels
        self.canvas_meas.on_gear_clicked       = lambda: self._show_graph_settings_dialog(start_tab=1)
        self.canvas_meas.on_close_mag_panel    = self._panel_close_mag
        self.canvas_meas.on_close_ph_panel     = self._panel_close_ph
        self.canvas_meas.on_toggle_ir          = self._toggle_ir_panel
        self.canvas_meas.on_panel1_changed     = lambda t: self._set_panel_type(0, t)
        self.canvas_meas.on_panel2_changed     = lambda t: self._set_panel_type(1, t)
        self.canvas_meas.on_ir_peak_detected   = self._on_ir_peak_detected
        self.canvas_meas.on_engine_select_menu = self._show_engine_select_menu
        self.canvas_sgram = SpectrogramCanvas()    # Spectrogram

        self._current_view = 'magnitude'   # 'magnitude'|'phase'|'spectrum'|'spectrograph'
        self._panel1_type  = 'magnitude'   # type shown in slot 0 (top)
        self._panel2_type  = 'phase'       # type shown in slot 1 (bottom)
        self._secondary_panel: 'SecondaryPanel | None' = None

        # Dropdown — mismas opciones en los 3 wrappers
        _OPTS = ['Magnitude', 'Phase', 'RTA', 'Spectrograph']

        self._wrap_meas  = CanvasOverlay(self.canvas_meas,  _OPTS, 'Magnitude',
                                         self._on_view_mode_changed,
                                         self._show_measurement_config,
                                         on_add=self._on_add_panel,
                                         on_focus=lambda: self._on_focus_canvas('magnitude'))
        self._wrap_spec  = CanvasOverlay(self.canvas_spec,  _OPTS, 'RTA',
                                         self._on_view_mode_changed,
                                         self._show_measurement_config,
                                         on_add=self._on_add_panel,
                                         on_focus=lambda: self._on_focus_canvas('spectrum'))
        self._wrap_sgram = CanvasOverlay(self.canvas_sgram, _OPTS, 'Spectrograph',
                                         self._on_view_mode_changed,
                                         self._show_measurement_config,
                                         on_add=self._on_add_panel,
                                         on_focus=lambda: self._on_focus_canvas('spectrum'))

        # MeasurementCanvas tiene labels propios dentro — ocultar la barra superior
        self._wrap_meas._btn.setVisible(False)
        self._wrap_meas._gear.setVisible(False)
        # Quitar el "+" exterior de TODOS los wrappers (ahora vive dentro de cada panel)
        self._wrap_meas._btn_add.setVisible(False)
        self._wrap_spec._btn_add.setVisible(False)
        self._wrap_sgram._btn_add.setVisible(False)

        # Slot principal — QStackedWidget garantiza que solo el widget activo
        # ocupa el 100% del espacio disponible (sin espacio residual para los ocultos)
        from PyQt6.QtWidgets import QStackedWidget
        self._main_area = QStackedWidget()
        self._main_area.addWidget(self._wrap_meas)
        self._main_area.addWidget(self._wrap_spec)
        self._main_area.addWidget(self._wrap_sgram)
        self._main_area.setCurrentWidget(self._wrap_meas)

        # Splitter vertical — soporta 1 o 2 paneles
        self._panel_splitter = QSplitter(Qt.Orientation.Vertical)
        self._panel_splitter.setHandleWidth(1)
        self._panel_splitter.setStyleSheet(
            'QSplitter::handle{background:#0d0d0d;}'
            'QSplitter::handle:hover{background:#3a5a3a;}')
        self._panel_splitter.addWidget(self._main_area)

        cv.addWidget(self._panel_splitter, stretch=1)
        root.addWidget(center, stretch=1)

        # ── Derecha: panel settings ──
        root.addWidget(self._build_collapsible_settings())

        # ── Assemble outer layout ──
        self._central_outer.addWidget(_inner_container, stretch=1)

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
        bar.setFixedHeight(30)
        h = QHBoxLayout(bar)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)

        # Workspace tab bar lives here (left side of info bar)
        self._ws_tab_bar = QTabBar()
        self._ws_tab_bar.setExpanding(False)
        self._ws_tab_bar.setTabsClosable(True)
        self._ws_tab_bar.setMovable(True)
        self._ws_tab_bar.setStyleSheet(
            'QTabBar{'
            '  background:transparent;'
            '}'
            'QTabBar::tab{'
            '  background:#3a3a3a; color:#aaaaaa;'
            '  border:1px solid #555555;'
            '  border-bottom:none;'
            '  border-top-left-radius:5px;'
            '  border-top-right-radius:5px;'
            '  padding:0px 20px 0px 14px; margin-right:2px; min-width:90px;'
            '  height:28px;'
            '  font-size:11px; font-weight:bold; letter-spacing:0.5px;'
            '}'
            'QTabBar::tab:selected{'
            '  background:#1c1c1c; color:#ffffff;'
            '  border-color:#666666;'
            '  border-bottom:2px solid #1c1c1c;'
            '}'
            'QTabBar::tab:hover:!selected{'
            '  background:#444444; color:#dddddd;'
            '}'
            'QTabBar::close-button{'
            '  subcontrol-position:right; subcontrol-origin:padding;'
            '  width:10px; height:10px; margin-right:2px;'
            '}'
        )
        self._ws_tab_bar.tabCloseRequested.connect(self._ws_close_tab)
        self._ws_tab_bar.currentChanged.connect(self._ws_on_tab_changed)
        self._ws_tab_bar.mouseDoubleClickEvent = self._ws_rename_tab_evt
        h.addWidget(self._ws_tab_bar)

        _ws_add_btn = QPushButton('+')
        _ws_add_btn.setFixedSize(26, 26)
        _ws_add_btn.setToolTip('New session tab')
        _ws_add_btn.setStyleSheet(
            'QPushButton{background:transparent;color:#555555;border:none;'
            'font-size:18px;font-weight:normal;padding:0;}'
            'QPushButton:hover{color:#e2e2e2;background:#1c1c1c;border-radius:3px;}'
        )
        _ws_add_btn.clicked.connect(self._ws_add_new)
        h.addWidget(_ws_add_btn)

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
        panel.setStyleSheet('background:#1a1a1a;')
        v = QVBoxLayout(panel)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        # ── Header SMAART-style ───────────────────────────────────────
        hdr_w = QWidget()
        hdr_w.setFixedHeight(34)
        hdr_w.setStyleSheet('background:#252525;border-bottom:1px solid #333;')
        hdr = QHBoxLayout(hdr_w)
        hdr.setContentsMargins(8, 0, 6, 0)
        hdr.setSpacing(4)
        title = QLabel('Transfer Function')
        title.setStyleSheet(
            'color:#e0e0e0;font-size:12px;font-weight:bold;background:transparent;')
        hdr.addWidget(title, stretch=1)
        btn_menu_tf = QPushButton('≡')
        btn_menu_tf.setFixedSize(24, 24)
        btn_menu_tf.setStyleSheet(
            'QPushButton{font-size:16px;color:#888;border:none;background:transparent;padding:0;}'
            'QPushButton:hover{color:#ccc;}')
        btn_menu_tf.setToolTip('Options')
        btn_menu_tf.clicked.connect(lambda: self._show_panel_menu(btn_menu_tf, 'tf'))
        hdr.addWidget(btn_menu_tf)
        v.addWidget(hdr_w)

        # ── Search row ────────────────────────────────────────────────
        search_w = QWidget()
        search_w.setFixedHeight(32)
        search_w.setStyleSheet('background:#1a1a1a;border-bottom:1px solid #2a2a2a;')
        sr = QHBoxLayout(search_w)
        sr.setContentsMargins(6, 4, 6, 4)
        sr.setSpacing(4)
        lbl_mag = QLabel('🔍')
        lbl_mag.setStyleSheet('color:#666;font-size:11px;background:transparent;')
        sr.addWidget(lbl_mag)
        self._tf_search = QLineEdit()
        self._tf_search.setPlaceholderText('Search…')
        self._tf_search.setStyleSheet(
            f'QLineEdit{{background:#111;border:none;border-radius:3px;'
            f'color:{TEXT_MID};font-size:11px;padding:1px 4px;}}')
        self._tf_search.textChanged.connect(self._filter_tf_traces)
        sr.addWidget(self._tf_search, stretch=1)
        v.addWidget(search_w)

        # ── Session Data folder header ─────────────────────────────────
        folder_w = QWidget()
        folder_w.setFixedHeight(28)
        folder_w.setStyleSheet('background:#1a1a1a;')
        fl = QHBoxLayout(folder_w)
        fl.setContentsMargins(8, 0, 8, 0)
        fl.setSpacing(6)
        lbl_folder = QLabel('📁')
        lbl_folder.setStyleSheet('font-size:13px;background:transparent;')
        fl.addWidget(lbl_folder)
        lbl_session = QLabel('Session Data')
        lbl_session.setStyleSheet('color:#cccccc;font-size:11px;background:transparent;')
        fl.addWidget(lbl_session, stretch=1)
        btn_save_trace = QPushButton('+')
        btn_save_trace.setFixedSize(18, 18)
        btn_save_trace.setToolTip(f'Save trace (max {MAX_TRACES})')
        btn_save_trace.setStyleSheet(
            'QPushButton{color:#888;font-size:14px;border:none;background:transparent;padding:0;}'
            'QPushButton:hover{color:#4aaa6a;}')
        btn_save_trace.clicked.connect(self._capture_trace_dialog)
        fl.addWidget(btn_save_trace)
        v.addWidget(folder_w)

        sep_line = QFrame(); sep_line.setFrameShape(QFrame.Shape.HLine)
        sep_line.setStyleSheet('color:#2a2a2a;')
        v.addWidget(sep_line)

        # ── Lista de trazas ───────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            'QScrollArea{border:none;background:#1a1a1a;}'
            'QScrollBar:vertical{background:#111;width:6px;border-radius:3px;}'
            'QScrollBar::handle:vertical{background:#333;border-radius:3px;}'
            'QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;}')

        inner = QWidget()
        inner.setStyleSheet('background:#1a1a1a;')
        self._trace_rows_layout = QVBoxLayout(inner)
        self._trace_rows_layout.setContentsMargins(0, 2, 0, 2)
        self._trace_rows_layout.setSpacing(0)

        self._lbl_no_data_tf = QLabel('   No Data')
        self._lbl_no_data_tf.setStyleSheet(
            'color:#444;font-size:11px;padding:8px 0;background:transparent;')
        self._trace_rows_layout.addWidget(self._lbl_no_data_tf)
        self._trace_rows_layout.addStretch()

        self._trace_rows_widget = inner
        scroll.setWidget(inner)
        v.addWidget(scroll, stretch=1)
        return panel

    def _build_collapsible_save_panel(self):
        """Panel izquierdo colapsable — stack 0=TF, 1=Spectrum."""
        from PyQt6.QtWidgets import QStackedWidget
        container = QWidget()
        h = QHBoxLayout(container)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)

        self._save_stack = QStackedWidget()
        self._save_stack.setFixedWidth(180)
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
        panel.setStyleSheet('background:#1a1a1a;')
        v = QVBoxLayout(panel)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        # ── Header SMAART-style ───────────────────────────────────────
        hdr_w = QWidget()
        hdr_w.setFixedHeight(34)
        hdr_w.setStyleSheet('background:#252525;border-bottom:1px solid #333;')
        hdr = QHBoxLayout(hdr_w)
        hdr.setContentsMargins(8, 0, 6, 0)
        hdr.setSpacing(4)
        title = QLabel('Spectrum')
        title.setStyleSheet(
            'color:#e0e0e0;font-size:12px;font-weight:bold;background:transparent;')
        hdr.addWidget(title, stretch=1)
        btn_menu_sp = QPushButton('≡')
        btn_menu_sp.setFixedSize(24, 24)
        btn_menu_sp.setStyleSheet(
            'QPushButton{font-size:16px;color:#888;border:none;background:transparent;padding:0;}'
            'QPushButton:hover{color:#ccc;}')
        btn_menu_sp.setToolTip('Options')
        btn_menu_sp.clicked.connect(lambda: self._show_panel_menu(btn_menu_sp, 'sp'))
        hdr.addWidget(btn_menu_sp)
        v.addWidget(hdr_w)

        # ── Search row ────────────────────────────────────────────────
        search_w = QWidget()
        search_w.setFixedHeight(32)
        search_w.setStyleSheet('background:#1a1a1a;border-bottom:1px solid #2a2a2a;')
        sr = QHBoxLayout(search_w)
        sr.setContentsMargins(6, 4, 6, 4)
        sr.setSpacing(4)
        lbl_mag = QLabel('🔍')
        lbl_mag.setStyleSheet('color:#666;font-size:11px;background:transparent;')
        sr.addWidget(lbl_mag)
        self._sp_search = QLineEdit()
        self._sp_search.setPlaceholderText('Search…')
        self._sp_search.setStyleSheet(
            f'QLineEdit{{background:#111;border:none;border-radius:3px;'
            f'color:{TEXT_MID};font-size:11px;padding:1px 4px;}}')
        self._sp_search.textChanged.connect(self._filter_sp_traces)
        sr.addWidget(self._sp_search, stretch=1)
        v.addWidget(search_w)

        # ── Session Data folder header ─────────────────────────────────
        folder_w = QWidget()
        folder_w.setFixedHeight(28)
        folder_w.setStyleSheet('background:#1a1a1a;')
        fl = QHBoxLayout(folder_w)
        fl.setContentsMargins(8, 0, 8, 0)
        fl.setSpacing(6)
        lbl_folder = QLabel('📁')
        lbl_folder.setStyleSheet('font-size:13px;background:transparent;')
        fl.addWidget(lbl_folder)
        lbl_session = QLabel('Session Data')
        lbl_session.setStyleSheet('color:#cccccc;font-size:11px;background:transparent;')
        fl.addWidget(lbl_session, stretch=1)
        btn_save_sp_trace = QPushButton('+')
        btn_save_sp_trace.setFixedSize(18, 18)
        btn_save_sp_trace.setToolTip('Save spectrum trace')
        btn_save_sp_trace.setStyleSheet(
            'QPushButton{color:#888;font-size:14px;border:none;background:transparent;padding:0;}'
            'QPushButton:hover{color:#4aaa6a;}')
        btn_save_sp_trace.clicked.connect(self._save_spectrum_trace)
        fl.addWidget(btn_save_sp_trace)
        v.addWidget(folder_w)

        sep_line = QFrame(); sep_line.setFrameShape(QFrame.Shape.HLine)
        sep_line.setStyleSheet('color:#2a2a2a;')
        v.addWidget(sep_line)

        # ── Lista de trazas ───────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            'QScrollArea{border:none;background:#1a1a1a;}'
            'QScrollBar:vertical{background:#111;width:6px;border-radius:3px;}'
            'QScrollBar::handle:vertical{background:#333;border-radius:3px;}'
            'QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;}')

        inner = QWidget()
        inner.setStyleSheet('background:#1a1a1a;')
        self._sp_trace_rows_layout = QVBoxLayout(inner)
        self._sp_trace_rows_layout.setContentsMargins(0, 2, 0, 2)
        self._sp_trace_rows_layout.setSpacing(0)

        self._lbl_no_data_sp = QLabel('   No Data')
        self._lbl_no_data_sp.setStyleSheet(
            'color:#444;font-size:11px;padding:8px 0;background:transparent;')
        self._sp_trace_rows_layout.addWidget(self._lbl_no_data_sp)
        self._sp_trace_rows_layout.addStretch()

        self._sp_trace_rows_widget = inner
        scroll.setWidget(inner)
        v.addWidget(scroll, stretch=1)
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
        idx   = getattr(self, '_selected_engine_idx', 0)
        color = ENGINE_PALETTE[idx % len(ENGINE_PALETTE)] if ENGINE_PALETTE else TEXT_HI
        self.lbl_cursor_info.setStyleSheet(
            f'color:{color};font-size:12px;font-family:Menlo,Monaco,monospace;'
            f'background:transparent;padding:0 8px;')
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
        outer.setFixedWidth(200)
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

        # ── View panel ───────────────────────────────────────────────
        view_container = QWidget()
        view_container.setStyleSheet(f'background:{BG_SETTINGS};')
        view_lay = QVBoxLayout(view_container)
        view_lay.setContentsMargins(6, 4, 6, 4)
        view_lay.setSpacing(3)
        self._build_view_block(view_lay)
        ov.addWidget(view_container)

        # Separador
        _s3 = QFrame(); _s3.setFrameShape(QFrame.Shape.HLine)
        _s3.setStyleSheet(f'color:{BORDER};')
        ov.addWidget(_s3)

        # ── Signal Generator fijo abajo ───────────────────────────────
        sg_container = QWidget()
        sg_container.setStyleSheet(f'background:{BG_SETTINGS};')
        sg_lay = QVBoxLayout(sg_container)
        sg_lay.setContentsMargins(6, 4, 6, 6)
        sg_lay.setSpacing(4)
        self._build_signal_generator_block(sg_lay)
        ov.addWidget(sg_container)

        return outer

    # ── View block ────────────────────────────────────────────────────

    def _build_view_block(self, layout):
        """Panel View colapsable — se abre/cierra con el botón View."""

        _ss_key = (f'QPushButton{{background:#252525;color:{TEXT_MID};'
                   f'border:1px solid #333;border-radius:3px;'
                   f'font-size:10px;font-weight:bold;min-width:20px;'
                   f'padding:1px 4px;}}'
                   f'QPushButton:hover{{background:#2a382a;color:{GREEN};'
                   f'border-color:{GREEN};}}')
        _ss_btn = (f'QPushButton{{background:#1e1e1e;color:{TEXT_MID};'
                   f'border:1px solid #333;border-radius:3px;'
                   f'font-size:10px;padding:3px 10px;}}'
                   f'QPushButton:hover{{background:#252525;color:{TEXT_HI};}}')
        _ss_ir  = (f'QPushButton{{background:#1e261e;color:#88bb88;'
                   f'border:1px solid #3a4a3a;border-radius:3px;'
                   f'font-size:10px;font-weight:bold;padding:2px 8px;}}'
                   f'QPushButton:checked{{background:#2a3a2a;color:{GREEN};'
                   f'border-color:{GREEN};}}'
                   f'QPushButton:hover{{border-color:{GREEN};}}')
        _ss_ico = (f'QPushButton{{background:#1a1a1a;color:{TEXT_MID};'
                   f'border:1px solid #333;border-radius:3px;font-size:9px;'
                   f'padding:2px;}}'
                   f'QPushButton:hover{{background:#222;border-color:#555;}}')

        # ── Outer frame (siempre visible) ─────────────────────────────
        frame = QFrame()
        frame.setStyleSheet(
            f'QFrame{{background:#161616;border:1px solid #2a2a2a;border-radius:5px;}}')
        fv = QVBoxLayout(frame)
        fv.setContentsMargins(6, 4, 6, 4)
        fv.setSpacing(4)

        # ── Toggle button "View" (header siempre visible) ─────────────
        btn_toggle = QPushButton('▶  View')
        btn_toggle.setCheckable(True)
        btn_toggle.setChecked(False)   # empieza cerrado
        btn_toggle.setFixedHeight(26)
        btn_toggle.setStyleSheet(
            f'QPushButton{{background:#1a1a1a;color:{TEXT_MID};'
            f'border:none;border-radius:4px;'
            f'font-size:11px;font-weight:bold;text-align:left;padding:0 8px;}}'
            f'QPushButton:checked{{color:{GREEN};}}'
            f'QPushButton:hover{{color:{TEXT_HI};background:#222;}}')
        fv.addWidget(btn_toggle)

        # ── Content widget (colapsable) ───────────────────────────────
        content = QWidget()
        content.setVisible(False)   # empieza oculto
        cv = QVBoxLayout(content)
        cv.setContentsMargins(2, 2, 2, 2)
        cv.setSpacing(5)

        # Tab: workspace selector
        tab_row = QHBoxLayout(); tab_row.setSpacing(4)
        tab_lbl = QLabel('Tab:')
        tab_lbl.setStyleSheet(f'color:{TEXT_MID};font-size:10px;background:transparent;')
        tab_lbl.setFixedWidth(28)
        if not hasattr(self, '_cmb_view_tab'):
            self._cmb_view_tab = QComboBox()
        self._cmb_view_tab.setFixedHeight(22)
        self._cmb_view_tab.setStyleSheet(
            f'font-size:10px;color:{TEXT_HI};background:#222;'
            f'border:1px solid #333;border-radius:3px;padding:0 4px;')
        tab_row.addWidget(tab_lbl)
        tab_row.addWidget(self._cmb_view_tab, stretch=1)
        cv.addLayout(tab_row)

        # Layout icons + Live IR
        ico_row = QHBoxLayout(); ico_row.setSpacing(3)
        ico_single = QPushButton('▣'); ico_single.setFixedSize(28, 22)
        ico_split  = QPushButton('▤'); ico_split.setFixedSize(28, 22)
        for b in (ico_single, ico_split):
            b.setStyleSheet(_ss_ico)

        btn_live_ir = QPushButton('Live IR')
        btn_live_ir.setCheckable(True)
        btn_live_ir.setChecked(getattr(self, '_ir_visible', True))
        btn_live_ir.setFixedHeight(22)
        btn_live_ir.setStyleSheet(_ss_ir)
        btn_live_ir.clicked.connect(lambda checked: self._toggle_ir_panel())

        if not hasattr(self, '_btn_live_ir_list'):
            self._btn_live_ir_list = []
        self._btn_live_ir_list.append(btn_live_ir)

        ico_row.addWidget(ico_single)
        ico_row.addWidget(ico_split)
        ico_row.addStretch()
        ico_row.addWidget(btn_live_ir)
        cv.addLayout(ico_row)

        # Separador "Presets"
        sep_row = QHBoxLayout(); sep_row.setSpacing(4)
        sep_l = QFrame(); sep_l.setFrameShape(QFrame.Shape.HLine)
        sep_r = QFrame(); sep_r.setFrameShape(QFrame.Shape.HLine)
        for s in (sep_l, sep_r): s.setStyleSheet('color:#333;')
        sep_lbl = QLabel('Presets')
        sep_lbl.setStyleSheet(
            f'color:{TEXT_MID};font-size:9px;background:transparent;')
        sep_row.addWidget(sep_l, stretch=1)
        sep_row.addWidget(sep_lbl)
        sep_row.addWidget(sep_r, stretch=1)
        cv.addLayout(sep_row)

        # Lista de presets
        _PRESETS = [
            ('S', 'RTA',                  lambda: self._on_view_mode_changed('RTA')),
            ('T', 'Transfer Function',    lambda: self._on_view_mode_changed('Magnitude')),
            ('1', 'Spectrograph',         lambda: self._on_view_mode_changed('Spectrograph')),
            ('2', 'Magnitude / Phase',    lambda: self._on_view_mode_changed('Phase')),
            ('3', 'RTA / RTA',            lambda: self._on_view_mode_changed('RTA')),
            ('4', 'RTA / Spectrograph',   lambda: self._on_view_mode_changed('Spectrograph')),
            ('5', 'Magnitude / Magnitude',lambda: self._on_view_mode_changed('Magnitude')),
            ('6', 'Magnitude / Phase',    lambda: self._on_view_mode_changed('Phase')),
            ('7', 'IR Log',               lambda: self._on_view_mode_changed('Magnitude')),
            ('8', 'TF Only',              lambda: self._on_view_mode_changed('Magnitude')),
            ('9', '- Empty -',            None),
            ('0', 'Multi-RTA',            lambda: self._on_view_mode_changed('RTA')),
        ]

        for key, name, action in _PRESETS:
            row = QHBoxLayout(); row.setSpacing(5)
            btn_k = QPushButton(key)
            btn_k.setFixedSize(20, 18)
            btn_k.setStyleSheet(_ss_key)
            lbl = QLabel(name)
            lbl.setStyleSheet(
                f'color:{TEXT_HI if action else TEXT_DIM};'
                f'font-size:10px;background:transparent;')
            if action:
                btn_k.clicked.connect(action)
                lbl.mousePressEvent = lambda e, a=action: a()
            row.addWidget(btn_k)
            row.addWidget(lbl, stretch=1)
            cv.addLayout(row)

        # Capture / Manage
        bot = QHBoxLayout(); bot.setSpacing(4)
        btn_capture = QPushButton('Capture')
        btn_manage  = QPushButton('Manage')
        for b in (btn_capture, btn_manage):
            b.setFixedHeight(24)
            b.setStyleSheet(_ss_btn)
        btn_capture.clicked.connect(self._capture_trace_dialog)
        btn_manage.clicked.connect(
            lambda: self.sb.showMessage('Manage traces — coming soon', 2000))
        bot.addWidget(btn_capture, stretch=1)
        bot.addWidget(btn_manage, stretch=1)
        cv.addLayout(bot)

        fv.addWidget(content)
        layout.addWidget(frame)

        # ── Toggle logic ──────────────────────────────────────────────
        def _toggle_view(checked):
            content.setVisible(checked)
            btn_toggle.setText('▼  View' if checked else '▶  View')

        btn_toggle.toggled.connect(_toggle_view)

        # Poblar workspace tabs
        self._refresh_view_tab_combo()

    def _refresh_view_tab_combo(self):
        """Sincroniza el combo Tab: con los workspaces actuales."""
        if not hasattr(self, '_cmb_view_tab'):
            return
        self._cmb_view_tab.blockSignals(True)
        self._cmb_view_tab.clear()
        for ws in getattr(self, '_workspaces', []):
            self._cmb_view_tab.addItem(ws.get('name', 'Workspace'))
        idx = getattr(self, '_current_ws_idx', 0)
        if idx < self._cmb_view_tab.count():
            self._cmb_view_tab.setCurrentIndex(idx)
        self._cmb_view_tab.blockSignals(False)
        # Conectar signal solo la primera vez (evitar múltiples conexiones)
        if not getattr(self, '_view_tab_combo_connected', False):
            self._cmb_view_tab.currentIndexChanged.connect(self._ws_load)
            self._view_tab_combo_connected = True

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

        # Fila 2: Output ▼ | - | +
        r2 = QHBoxLayout(); r2.setSpacing(4)

        if not hasattr(self, '_btn_noise_out'):
            self._btn_noise_out = QPushButton('Output ▾')
            self._btn_noise_out.setObjectName('btn_noise_out')
        self._btn_noise_out.setFixedHeight(26)
        self._btn_noise_out.setStyleSheet(
            f'QPushButton{{font-size:11px;color:{TEXT_HI};background:#222;'
            f'border:1px solid #333;border-radius:4px;padding:0 6px;text-align:left;}}'
            f'QPushButton:hover{{background:#2a2a2a;border-color:#555;}}')
        self._btn_noise_out.clicked.connect(self._show_noise_dev_menu)
        r2.addWidget(self._btn_noise_out, stretch=2)

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
        self._avg_values = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        for v in self._avg_values: self.cmb_avg.addItem(str(v))
        self.cmb_avg.setCurrentIndex(4)   # 16 default
        self.cmb_avg.setFixedWidth(68)
        self.cmb_avg.currentIndexChanged.connect(self._on_avg_changed)
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

        # Botón AVG — creación con estilo prominente (se añade al layout abajo)
        _avg_on_style  = (f'QPushButton{{font-size:12px;color:#6ade8a;font-weight:bold;'
                          f'background:#1a3a1a;border:2px solid #4ab46a;'
                          f'border-radius:5px;padding:8px 0;letter-spacing:1px;}}'
                          f'QPushButton:hover{{background:#1e4a1e;}}')
        _avg_off_style = (f'QPushButton{{font-size:12px;color:{TEXT_DIM};'
                          f'background:#111a11;border:2px solid #2a5a2a;'
                          f'border-radius:5px;padding:8px 0;letter-spacing:1px;}}'
                          f'QPushButton:hover{{background:#141e14;color:{TEXT_MID};}}')
        self.btn_show_avg = QPushButton('● AVG')
        self.btn_show_avg.setCheckable(True)
        self.btn_show_avg.setChecked(False)
        self._show_avg      = False
        self._tf_avg_config = None          # TFAvgConfig or None
        self._sp_avg_config = None          # SpectrumAvgConfig or None
        self.btn_show_avg.setStyleSheet(_avg_off_style)
        def _toggle_avg(checked):
            self._show_avg = checked
            self.btn_show_avg.setStyleSheet(_avg_on_style if checked else _avg_off_style)
            if not checked:
                self.canvas_meas.update_avg(None, None, None, None)
                if self._secondary_panel is not None:
                    self._secondary_panel.canvas_meas.update_avg(None, None, None, None)
            idx = getattr(self, '_selected_engine_idx', 0)
            self.canvas_meas.highlight_engine(idx, show_avg=checked)
            if self._secondary_panel is not None:
                self._secondary_panel.canvas_meas.highlight_engine(idx, show_avg=checked)
        self.btn_show_avg.clicked.connect(_toggle_avg)

        # Right-click → configure average
        self.btn_show_avg.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        def _avg_ctx_menu(pos):
            m = QMenu(self)
            m.setStyleSheet(
                'QMenu{background:#1e221e;color:#cccccc;border:1px solid #3a3a3a;}'
                'QMenu::item{padding:5px 20px 5px 10px;}'
                'QMenu::item:selected{background:#2a382a;color:#88cc88;}'
            )
            act_new  = m.addAction('New TF Avg…')
            act_edit = m.addAction('Edit Avg…')
            act_edit.setEnabled(self._tf_avg_config is not None)
            act_clear = m.addAction('Clear Avg Config')
            act_clear.setEnabled(self._tf_avg_config is not None)
            chosen = m.exec(self.btn_show_avg.mapToGlobal(pos))
            if chosen == act_new:
                self._open_tf_avg_dialog(edit=False)
            elif chosen == act_edit:
                self._open_tf_avg_dialog(edit=True)
            elif chosen == act_clear:
                self._tf_avg_config = None
                self.btn_show_avg.setText('● AVG')
        self.btn_show_avg.customContextMenuRequested.connect(_avg_ctx_menu)
        # (btn_show_avg se añade al layout en la zona inferior — ver abajo)

        # cmb_smooth — mantenido por compatibilidad, sin fila visible en UI
        self.cmb_smooth = QComboBox()
        for _s in ['OFF', '1/3', '1/6', '1/12', '1/24']:
            self.cmb_smooth.addItem(_s)
        self.cmb_smooth.setCurrentIndex(3)   # 1/12 oct por defecto
        self._smooth_values = [0, 3, 6, 12, 24]

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
        # (g2_row se añade al layout en la zona inferior — ver abajo)
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

        # ── ZONA INFERIOR: γ² + AVG prominente ───────────────────────────
        layout.addWidget(sep())
        layout.addLayout(g2_row)
        layout.addWidget(self.btn_show_avg)

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
        sec_sp.setText(0, 'RTA')
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
            eng = self._add_tf_engine(ch_m=ch_m, ch_r=ch_r, color=color, name=name)
            self._select_engine(len(self._tf_engines) - 1)
            self._on_channels()

            if _action[0] == 'close':
                break

    # ── Gestión dinámica de TF engines ───────────────────────────────

    def _add_tf_engine(self, ch_m: int = None, ch_r: int = 4,
                       color: str = None, name: str = '') -> 'TFEngine':
        idx   = len(self._tf_engines)
        color = color or ENGINE_PALETTE[idx % len(ENGINE_PALETTE)]
        ch_m  = ch_m if ch_m is not None else (idx + 1)

        # Callbacks (capturan referencia a la lista en ese momento)
        def _on_ch(): self._on_channels()
        def _on_rm(): self._remove_tf_engine(self._tf_engines.index(eng))
        def _on_fd(): self._on_find_delay_for_engine(self._tf_engines.index(eng))
        def _on_sel(): self._select_engine(self._tf_engines.index(eng))
        def _on_nrm(): self._normalize_engine(self._tf_engines.index(eng))
        def _on_act(ei, active): self._on_engine_active_changed(ei, active)
        def _on_col(ei, col): self._on_engine_color_changed(ei, col)
        def _on_tr(ei, visible): self.canvas_meas.set_trace_visible(ei, visible)
        def _on_cfg(ei): self._open_tf_engine_config(ei)

        eng = TFEngine(idx + 1, color, _on_ch, _on_rm, _on_fd,
                       on_select=_on_sel, on_normalize=_on_nrm,
                       on_active_changed=_on_act, on_color_changed=_on_col,
                       on_trace_toggle=_on_tr, on_config=_on_cfg,
                       name=name, parent=None)
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

        # Si hay más de 2 engines, agregar línea dinámica al canvas
        if len(self._tf_engines) > 2:
            self.canvas_meas.add_tf_engine_line(color)
            if self._secondary_panel is not None and hasattr(self._secondary_panel, 'canvas_meas'):
                self._secondary_panel.canvas_meas.add_tf_engine_line(color)

        # Sincronizar colores del canvas con los engines registrados
        colors = [e._color for e in self._tf_engines]
        self.canvas_meas.set_engine_colors(colors)
        self.canvas_meas.highlight_engine(
            min(getattr(self, '_selected_engine_idx', 0), len(self._tf_engines) - 1),
            show_avg=self._show_avg)

        return eng

    def _remove_tf_engine(self, idx: int):
        if idx < 0 or idx >= len(self._tf_engines):
            return
        eng = self._tf_engines.pop(idx)
        self._engines_layout.removeWidget(eng)
        eng.setParent(None)
        eng.deleteLater()
        # Quitar buffer de medición y RunningTF correspondiente
        if idx < len(self.engine.buf_meas_list):
            self.engine.remove_meas_channel(idx)
        if hasattr(self, '_running_tfs') and idx < len(self._running_tfs):
            self._running_tfs.pop(idx)
        # Si era engine 3+, quitar línea del canvas
        if idx >= 2:
            self.canvas_meas.remove_tf_engine_line(idx)
            if self._secondary_panel is not None and hasattr(self._secondary_panel, 'canvas_meas'):
                self._secondary_panel.canvas_meas.remove_tf_engine_line(idx)

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
        self._sp_eng_rows = []
        for i, eng in enumerate(self._sp_engines):
            def _on_sp_tr(ei, visible, _c=i):
                self.canvas_spec.set_trace_visible(ei, visible)
            row = SpectrumEngineRow(
                idx=i,
                name=eng.get('name', f'Analog {eng.get("ch", i+1)}'),
                color=eng.get('color', ENGINE_PALETTE[i % len(ENGINE_PALETTE)]),
                on_config=self._open_sp_engine_config,
                on_del=self._remove_spectrum_engine,
                on_play=self._toggle_sp_engine_active,
                active=eng.get('active', True),
                selected=(i == getattr(self, '_sp_eng_selected', 0)),
                on_trace_toggle=_on_sp_tr,
            )
            self._sp_eng_rows.append(row)
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
        if idx < 0 or idx >= len(self._sp_engines):
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
        """Toggle active/paused state of a Spectrum engine."""
        if idx >= len(self._sp_engines):
            return
        self._sp_engines[idx]['active'] = not self._sp_engines[idx].get('active', True)
        self._rebuild_sp_engine_rows()
        if self._sp_engines[idx]['active']:
            # Reset RunningTF so it starts clean
            self._sp_engines[idx]['rtf'].reset()
            # Ensure audio stream is running (needed if no TF engines active)
            if not self.engine.running:
                try:
                    self.engine.start()
                    self._post_start_check()
                except Exception as exc:
                    self.sb.showMessage(f'⚠  Error opening stream: {exc}', 8000)
                    return
            if not self.timer.isActive():
                self.timer.start()
                self._silent_ticks = 0
                QTimer.singleShot(3000, self._check_signal_present)
            self._set_running()

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

    def _open_tf_engine_config(self, idx: int):
        """Dialog rápido para editar nombre y canales de un TF engine."""
        if idx < 0 or idx >= len(self._tf_engines):
            return
        eng = self._tf_engines[idx]

        from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout,
                                     QLabel, QLineEdit, QSpinBox, QPushButton,
                                     QGridLayout)
        dlg = QDialog(self)
        dlg.setWindowTitle(f'Configure TF Engine {idx + 1}')
        dlg.setModal(True)
        dlg.setFixedWidth(300)
        dlg.setStyleSheet(
            f'QDialog{{background:#1a1a1a;color:#cccccc;font-size:11px;}}'
            f'QLabel{{color:#888;background:transparent;}}'
            f'QLineEdit{{background:#252525;color:#cccccc;border:1px solid #3a3a3a;'
            f'border-radius:3px;padding:4px 6px;}}'
            f'QSpinBox{{background:#252525;color:#cccccc;border:1px solid #3a3a3a;'
            f'border-radius:3px;padding:3px 4px;}}'
            f'QPushButton{{background:#2e2e2e;color:#cccccc;border:1px solid #444;'
            f'border-radius:3px;padding:5px 14px;}}'
            f'QPushButton:hover{{border-color:#4ab46a;color:#4ab46a;}}')

        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(18, 16, 18, 14)
        lay.setSpacing(12)

        grid = QGridLayout(); grid.setSpacing(8); grid.setColumnMinimumWidth(0, 60)
        grid.setColumnStretch(1, 1)

        txt_name = QLineEdit(eng._name)
        grid.addWidget(QLabel('Name'), 0, 0); grid.addWidget(txt_name, 0, 1)

        spn_m = QSpinBox(); spn_m.setRange(1, 32); spn_m.setValue(eng.spn_m.value())
        grid.addWidget(QLabel('Meas ch'), 1, 0); grid.addWidget(spn_m, 1, 1)

        spn_r = QSpinBox(); spn_r.setRange(1, 32); spn_r.setValue(eng.spn_r.value())
        grid.addWidget(QLabel('Ref ch'), 2, 0); grid.addWidget(spn_r, 2, 1)

        lay.addLayout(grid)

        btns = QHBoxLayout(); btns.setSpacing(8)
        btn_cancel = QPushButton('Cancel'); btn_save = QPushButton('Save')
        btns.addWidget(btn_cancel); btns.addWidget(btn_save)
        lay.addLayout(btns)

        btn_cancel.clicked.connect(dlg.reject)
        def _save():
            eng._name = txt_name.text().strip() or eng._name
            eng._name_lbl.setText(eng._name)
            eng.spn_m.setValue(spn_m.value())
            eng.spn_r.setValue(spn_r.value())
            self._on_channels()
            self._save_prefs()
            dlg.accept()
        btn_save.clicked.connect(_save)
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

    def _ws_current_state(self, save_engines: bool = False) -> dict:
        """
        Captura el estado actual como dict de workspace.

        save_engines=False (auto-save): tf_engines se preserva desde el JSON
          cargado — los engines vivos NO se persisten automáticamente.
        save_engines=True  (Cmd+S):     tf_engines se toma del estado live.
        """
        name = ''
        if hasattr(self, '_ws_tab_bar') and self._ws_tab_bar.count() > self._current_ws_idx:
            name = self._ws_tab_bar.tabText(self._current_ws_idx)

        if save_engines:
            tf_eng_list = [
                {
                    'ch_m':        eng.spn_m.value(),
                    'ch_r':        eng.spn_r.value(),
                    'delay_ms':    eng._delay_comp_ms,
                    'gain_offset': eng._gain_offset_db,
                    'color':       eng._color,
                }
                for eng in self._tf_engines
            ]
        else:
            # Preservar lo que había en el JSON — no auto-guardar engines
            existing = (self._workspaces[self._current_ws_idx]
                        if self._current_ws_idx < len(self._workspaces) else {})
            tf_eng_list = existing.get('tf_engines', [])

        return {
            'name': name or f'Workspace {self._current_ws_idx + 1}',
            'dev_in':     self.engine.dev_in,
            'dev_out':    self.engine.dev_out,
            'ch_ref':     self.engine.ch_ref,
            'ch_spl':     self.engine.ch_spl,
            'n_averages': self.engine.n_averages,
            'nperseg':    self.engine.nperseg,
            'smooth':     getattr(self, '_smooth_fraction', 0.0),
            'tf_engines': tf_eng_list,
            'sp_engines': [
                {
                    'ch':    e.get('ch', 1),
                    'color': e.get('color', ENGINE_PALETTE[0]),
                    'name':  e.get('name', 'Analog'),
                }
                for e in self._sp_engines
            ],
        }

    def _ws_save_current(self, save_engines: bool = False):
        """
        Guarda el estado actual en _workspaces[_current_ws_idx].
        save_engines=False → tf_engines no se toca (auto-save normal).
        save_engines=True  → tf_engines se actualiza desde el estado live.
        """
        if not self._workspaces:
            return
        if self._current_ws_idx < len(self._workspaces):
            self._workspaces[self._current_ws_idx] = self._ws_current_state(
                save_engines=save_engines)

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
            self.canvas_meas.highlight_engine(0, show_avg=self._show_avg)
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
        self._refresh_view_tab_combo()

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
        self._refresh_view_tab_combo()

    def _ws_on_tab_changed(self, idx: int):
        """Callback cuando el usuario cambia de tab."""
        if self._ws_loading:
            return
        if idx == self._current_ws_idx:
            return
        self._ws_save_current()
        self._ws_load(idx)
        self._save_prefs()
        self._refresh_view_tab_combo()

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

    def _graph_settings_dict(self) -> dict:
        """Devuelve un dict serializable con todos los graph settings actuales."""
        gs = {}
        # ── TF General ─────────────────────────────────────────────────
        gs['tf_smooth_bpo']       = getattr(self, '_tf_smooth_bpo',       3)
        gs['link_smooth']         = getattr(self, '_link_smooth',         True)
        gs['link_ph_mag_smooth']  = getattr(self, '_link_ph_mag_smooth',  True)
        gs['link_plot_zoom']      = getattr(self, '_link_plot_zoom',      False)
        gs['plot_proportional']   = getattr(self, '_plot_proportional',   False)
        gs['instantaneous']       = getattr(self, '_instantaneous',       False)
        gs['tf_line_thick_fg']    = getattr(self, '_tf_line_thick_fg',    2)
        gs['tf_line_thick_bg']    = getattr(self, '_tf_line_thick_bg',    2)
        gs['tf_mag_avg_mode']     = getattr(self, '_tf_mag_avg_mode',     'Polar')
        gs['tf_y_grid']           = getattr(self, '_tf_y_grid',           3)
        gs['tf_y_scroll']         = getattr(self, '_tf_y_scroll',         3)
        gs['tf_y_zoom']           = getattr(self, '_tf_y_zoom',           3)
        # ── TF Axis ranges ─────────────────────────────────────────────
        try:
            yl = self.canvas_meas.ax_tf.get_ylim()
            gs['tf_mag_min'] = float(yl[0]); gs['tf_mag_max'] = float(yl[1])
        except Exception: pass
        try:
            yp = self.canvas_meas.ax_ph.get_ylim()
            gs['tf_ph_min']  = float(yp[0]); gs['tf_ph_max']  = float(yp[1])
        except Exception: pass
        try:
            xl = self.canvas_meas.ax_tf.get_xlim()
            gs['tf_f_lo'] = float(xl[0]);  gs['tf_f_hi'] = float(xl[1])
        except Exception: pass
        # ── TF Phase flags ─────────────────────────────────────────────
        gs['unwrap_phase']        = getattr(self, '_unwrap_phase',        False)
        gs['phase_as_gd']         = getattr(self, '_phase_as_gd',         False)
        # ── TF Coherence ───────────────────────────────────────────────
        gs['coh_visible']         = bool(self.canvas_meas.ax_coh.get_visible()) \
                                    if hasattr(self, 'canvas_meas') else True
        gs['coh_quarter_height']  = getattr(self, '_coh_quarter_height',  False)
        gs['coh_squared']         = getattr(self, '_coh_squared',         True)
        gs['coh_blanking_pct']    = getattr(self, '_coh_blanking_pct',    10)
        gs['mtw_coh']             = getattr(self, '_mtw_coh',             True)
        gs['coh_thick_follows_fg']= getattr(self, '_coh_thick_follows_fg',True)
        gs['coh_line_thick']      = getattr(self, '_coh_line_thick',      4)
        gs['peak_hold']           = getattr(self, '_peak_hold',           False)
        # ── IR tab ─────────────────────────────────────────────────────
        gs['ir_show_peak']        = getattr(self, '_ir_show_peak',        True)
        gs['ir_optimize']         = getattr(self, '_ir_optimize',         True)
        gs['ir_overlap_pct']      = getattr(self, '_ir_overlap_pct',      50)
        gs['ir_mag_thresh_db']    = getattr(self, '_ir_mag_thresh_db',    0)
        gs['ir_line_thick_fg']    = getattr(self, '_ir_line_thick_fg',    1)
        gs['ir_line_thick_bg']    = getattr(self, '_ir_line_thick_bg',    2)
        gs['ir_hpf_en']           = getattr(self, '_ir_hpf_en',           False)
        gs['ir_hpf_hz']           = getattr(self, '_ir_hpf_hz',           100)
        gs['ir_lpf_en']           = getattr(self, '_ir_lpf_en',           False)
        gs['ir_lpf_hz']           = getattr(self, '_ir_lpf_hz',           24000)
        gs['ir_trace_ctrl']       = getattr(self, '_ir_trace_ctrl',       0)
        gs['ir_freq_scale']       = getattr(self, '_ir_freq_scale',       '1/3 Octave')
        gs['fft_window_type']     = getattr(self, '_fft_window_type',     'Hann')
        gs['hist_mode']           = getattr(self, '_hist_mode',           0)
        try:
            gs['ir_visible']      = bool(self._ir_visible)
        except Exception: pass
        # ── IR axis ────────────────────────────────────────────────────
        try:
            yi = self.canvas_meas.ax_ir.get_ylim()
            gs['ir_mag_min'] = float(yi[0]); gs['ir_mag_max'] = float(yi[1])
        except Exception: pass
        # ── Spectrum ───────────────────────────────────────────────────
        try:
            ys = self.canvas_spec.ax.get_ylim()
            gs['spec_mag_min'] = float(ys[0]); gs['spec_mag_max'] = float(ys[1])
        except Exception: pass
        # ── Spectrogram ────────────────────────────────────────────────
        try:
            gs['sg_vmin'] = float(self.canvas_sgram._vmin)
            gs['sg_vmax'] = float(self.canvas_sgram._vmax)
        except Exception: pass
        return gs

    def _apply_graph_settings(self, gs: dict):
        """Restaura todos los graph settings desde un dict (cargado de prefs)."""
        import matplotlib.ticker as _mtick
        # ── TF flags ───────────────────────────────────────────────────
        self._tf_smooth_bpo       = gs.get('tf_smooth_bpo',       3)
        self._link_smooth         = gs.get('link_smooth',         True)
        self._link_ph_mag_smooth  = gs.get('link_ph_mag_smooth',  True)
        self._link_plot_zoom      = gs.get('link_plot_zoom',      False)
        self._plot_proportional   = gs.get('plot_proportional',   False)
        self._instantaneous       = gs.get('instantaneous',       False)
        self._tf_line_thick_fg    = gs.get('tf_line_thick_fg',    2)
        self._tf_line_thick_bg    = gs.get('tf_line_thick_bg',    2)
        self._tf_mag_avg_mode     = gs.get('tf_mag_avg_mode',     'Polar')
        self._tf_y_grid           = gs.get('tf_y_grid',           3)
        self._tf_y_scroll         = gs.get('tf_y_scroll',         3)
        self._tf_y_zoom           = gs.get('tf_y_zoom',           3)
        self._unwrap_phase        = gs.get('unwrap_phase',        False)
        self._phase_as_gd         = gs.get('phase_as_gd',         False)
        self._coh_quarter_height  = gs.get('coh_quarter_height',  False)
        self._coh_squared         = gs.get('coh_squared',         True)
        self._coh_blanking_pct    = gs.get('coh_blanking_pct',    10)
        self._mtw_coh             = gs.get('mtw_coh',             True)
        self._coh_thick_follows_fg= gs.get('coh_thick_follows_fg',True)
        self._coh_line_thick      = gs.get('coh_line_thick',      4)
        self._peak_hold           = gs.get('peak_hold',           False)
        # ── TF axis ────────────────────────────────────────────────────
        try:
            if 'tf_mag_min' in gs and 'tf_mag_max' in gs:
                self.canvas_meas.ax_tf.set_ylim(gs['tf_mag_min'], gs['tf_mag_max'])
                self.canvas_meas.ax_tf.set_autoscale_on(False)
        except Exception: pass
        try:
            if 'tf_ph_min' in gs and 'tf_ph_max' in gs:
                self.canvas_meas.ax_ph.set_ylim(gs['tf_ph_min'], gs['tf_ph_max'])
        except Exception: pass
        try:
            if 'tf_f_lo' in gs and 'tf_f_hi' in gs:
                self.canvas_meas.ax_tf.set_xlim(gs['tf_f_lo'], gs['tf_f_hi'])
                self.canvas_meas.ax_ph.set_xlim(gs['tf_f_lo'], gs['tf_f_hi'])
        except Exception: pass
        # ── Coherence visibility ───────────────────────────────────────
        try:
            _cv = gs.get('coh_visible', True)
            self.canvas_meas.ax_coh.set_visible(_cv)
            self.canvas_meas.line_coh.set_visible(_cv)
        except Exception: pass
        # ── TF line thickness ──────────────────────────────────────────
        try:
            fg = self._tf_line_thick_fg
            self.canvas_meas.line_tf.set_linewidth(fg)
            self.canvas_meas.line_tf2.set_linewidth(max(1, fg - 1))
            self.canvas_meas.line_ph.set_linewidth(fg)
            self.canvas_meas.line_ph2.set_linewidth(max(1, fg - 1))
            self.canvas_meas.line_tf_avg.set_linewidth(self._tf_line_thick_bg)
            coh_lw = fg if self._coh_thick_follows_fg else self._coh_line_thick
            self.canvas_meas.line_coh.set_linewidth(coh_lw)
        except Exception: pass
        # ── TF Y-grid ──────────────────────────────────────────────────
        try:
            self.canvas_meas.ax_tf.yaxis.set_major_locator(
                _mtick.MultipleLocator(self._tf_y_grid))
        except Exception: pass
        # ── Coherence blanking threshold ───────────────────────────────
        try:
            if hasattr(self, 'spn_thresh'):
                self.spn_thresh.setValue(self._coh_blanking_pct / 100.0)
        except Exception: pass
        # ── Peak hold ──────────────────────────────────────────────────
        try:
            self._toggle_peak_hold(self._peak_hold)
        except Exception: pass
        # ── IR flags ───────────────────────────────────────────────────
        self._ir_show_peak        = gs.get('ir_show_peak',        True)
        self._ir_optimize         = gs.get('ir_optimize',         True)
        self._ir_overlap_pct      = gs.get('ir_overlap_pct',      50)
        self._ir_mag_thresh_db    = gs.get('ir_mag_thresh_db',    0)
        self._ir_line_thick_fg    = gs.get('ir_line_thick_fg',    1)
        self._ir_line_thick_bg    = gs.get('ir_line_thick_bg',    2)
        self._ir_hpf_en           = gs.get('ir_hpf_en',           False)
        self._ir_hpf_hz           = gs.get('ir_hpf_hz',           100)
        self._ir_lpf_en           = gs.get('ir_lpf_en',           False)
        self._ir_lpf_hz           = gs.get('ir_lpf_hz',           24000)
        self._ir_trace_ctrl       = gs.get('ir_trace_ctrl',       0)
        self._ir_freq_scale       = gs.get('ir_freq_scale',       '1/3 Octave')
        self._fft_window_type     = gs.get('fft_window_type',     'Hann')
        self._hist_mode           = gs.get('hist_mode',           0)
        # ── IR peak marker ─────────────────────────────────────────────
        try:
            self.canvas_meas.line_ir_peak.set_visible(self._ir_show_peak)
        except Exception: pass
        # ── IR line thickness ──────────────────────────────────────────
        try:
            self.canvas_meas.line_ir.set_linewidth(self._ir_line_thick_fg)
        except Exception: pass
        # ── IR axis ────────────────────────────────────────────────────
        try:
            if 'ir_mag_min' in gs and 'ir_mag_max' in gs:
                self.canvas_meas.ax_ir.set_ylim(gs['ir_mag_min'], gs['ir_mag_max'])
                self.canvas_meas.ax_ir.set_autoscale_on(False)
        except Exception: pass
        # ── IR visibility ──────────────────────────────────────────────
        try:
            _irv = gs.get('ir_visible', True)
            if _irv != getattr(self, '_ir_visible', True):
                self._toggle_ir_panel()
        except Exception: pass
        # ── Spectrum axis ──────────────────────────────────────────────
        try:
            if 'spec_mag_min' in gs and 'spec_mag_max' in gs:
                self.canvas_spec.ax.set_ylim(gs['spec_mag_min'], gs['spec_mag_max'])
                self.canvas_spec.draw_idle()
        except Exception: pass
        # ── Spectrogram ────────────────────────────────────────────────
        try:
            if 'sg_vmin' in gs and 'sg_vmax' in gs:
                self.canvas_sgram.set_color_range(gs['sg_vmin'], gs['sg_vmax'])
                if hasattr(self, '_spn_sg_floor'):
                    self._spn_sg_floor.setValue(int(gs['sg_vmin']))
                    self._spn_sg_ceil.setValue(int(gs['sg_vmax']))
        except Exception: pass
        # ── FFT window ─────────────────────────────────────────────────
        try:
            self._restart_engine_window(self._fft_window_type)
        except Exception: pass

    def _save_workspace_explicit(self):
        """
        Save explícito del workspace (Ctrl+Shift+W).
        Único momento en que tf_engines se escribe al JSON.
        """
        if self._ws_loading:
            return
        import json, os
        self._ws_save_current(save_engines=True)   # ← incluye tf_engines live
        prefs = {
            'dev_in':  self.engine.dev_in,
            'dev_out': self.engine.dev_out,
            'ch_ref':  self.engine.ch_ref,
            'ch_spl':  self.engine.ch_spl,
            'current_ws':    self._current_ws_idx,
            'workspaces':    self._workspaces,
            'graph_settings': self._graph_settings_dict(),
        }
        try:
            with open(self._prefs_path(), 'w') as f:
                json.dump(prefs, f, indent=2)
            # Feedback visual breve en la barra de título
            orig = self.windowTitle()
            self.setWindowTitle(f'{orig}  ✓ Workspace guardado')
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(1800, lambda: self.setWindowTitle(orig))
        except Exception:
            pass

    def _save_prefs(self):
        if self._ws_loading:
            return
        import json, os
        self._ws_save_current()          # save_engines=False → no toca tf_engines
        prefs = {
            'dev_in':  self.engine.dev_in,
            'dev_out': self.engine.dev_out,
            'ch_ref':  self.engine.ch_ref,
            'ch_spl':  self.engine.ch_spl,
            'current_ws':    self._current_ws_idx,
            'workspaces':    self._workspaces,
            'graph_settings': self._graph_settings_dict(),
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

        saved_graph_settings = {}

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

                # ── Graph Settings ─────────────────────────────────────
                saved_graph_settings = prefs.get('graph_settings', {})

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

        # ── Restaurar graph settings (después de canvas construido) ───
        if saved_graph_settings:
            # QTimer para asegurar que el canvas esté completamente inicializado
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(200, lambda: self._apply_graph_settings(saved_graph_settings))

    # ── Sub-panel SPECTRUM ────────────────────────────────────────────

    def _build_settings_spectrum(self):
        w = QWidget()
        w.setStyleSheet(f'background:{BG_SETTINGS};')
        layout = QVBoxLayout(w)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(5)

        def rl(t): return lbl(t, color=TEXT_MID, size=9)

        # ── Header discreto ──────────────────────────────────────────
        h_lbl = QLabel('RTA')
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
        self._avg_spec_values = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        for _av in self._avg_spec_values:
            self.cmb_avg_spec.addItem(str(_av))
        self.cmb_avg_spec.setCurrentIndex(4)   # 16 default
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
        self.spn_ch_spec_sg.valueChanged.connect(
            lambda v: (setattr(self.engine, 'ch_spec', v),
                       self.spn_ch_spec.setValue(v)))
        gsg.addWidget(self.spn_ch_spec_sg, 0, 1)

        gsg.addWidget(rl('TIME WIN'), 1, 0)
        lbl_tw = QLabel('auto')
        lbl_tw.setStyleSheet(f'color:{TEXT_MID};font-size:9px;background:transparent;')
        gsg.addWidget(lbl_tw, 1, 1)
        layout.addWidget(grp_sg)

        # COLOR RANGE — floor / ceiling en dBFS
        grp_color = QGroupBox('COLOR RANGE')
        gc = QGridLayout(grp_color)
        gc.setContentsMargins(5, 6, 5, 4); gc.setSpacing(3); gc.setColumnStretch(1, 1)

        gc.addWidget(rl('FLOOR dBFS'), 0, 0)
        self._spn_sg_floor = QSpinBox()
        self._spn_sg_floor.setRange(-120, -20)
        self._spn_sg_floor.setValue(SpectrogramCanvas.VMIN_DEFAULT)
        self._spn_sg_floor.setSuffix(' dB')

        gc.addWidget(rl('CEIL dBFS'), 1, 0)
        self._spn_sg_ceil = QSpinBox()
        self._spn_sg_ceil.setRange(-60, 0)
        self._spn_sg_ceil.setValue(SpectrogramCanvas.VMAX_DEFAULT)
        self._spn_sg_ceil.setSuffix(' dB')

        def _update_sg_range():
            vmin = self._spn_sg_floor.value()
            vmax = self._spn_sg_ceil.value()
            if vmax <= vmin:
                vmax = vmin + 6
                self._spn_sg_ceil.setValue(vmax)
            self.canvas_sgram.set_color_range(vmin, vmax)
            if hasattr(self, '_secondary_panel'):
                self._secondary_panel.canvas_sgram.set_color_range(vmin, vmax)

        self._spn_sg_floor.valueChanged.connect(lambda _: _update_sg_range())
        self._spn_sg_ceil.valueChanged.connect(lambda _: _update_sg_range())

        gc.addWidget(self._spn_sg_floor, 0, 1)
        gc.addWidget(self._spn_sg_ceil,  1, 1)

        # Botón reset al rango default
        btn_sg_reset = QPushButton('Reset')
        btn_sg_reset.setFixedHeight(18)
        btn_sg_reset.setStyleSheet(
            f'QPushButton{{background:transparent;color:{TEXT_DIM};'
            f'border:1px solid #2a2a2a;border-radius:2px;font-size:8px;}}'
            f'QPushButton:hover{{color:{CYAN};border-color:{CYAN};}}')
        def _reset_sg_range():
            self._spn_sg_floor.setValue(SpectrogramCanvas.VMIN_DEFAULT)
            self._spn_sg_ceil.setValue(SpectrogramCanvas.VMAX_DEFAULT)
        btn_sg_reset.clicked.connect(_reset_sg_range)
        gc.addWidget(btn_sg_reset, 2, 0, 1, 2)

        layout.addWidget(grp_color)

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

    def _sync_splitter_for_ir(self):
        """Adjust the two-slot splitter proportions based on how many panels
        are visible inside canvas_meas (slot1).

        3 panels (IR + TF + Phase): slot1 = 72%, slot2 = 28%
        2 panels (IR + TF  or  IR + Phase): slot1 = 62%, slot2 = 38%
        1 panel  (TF only  or  Phase only): slot1 = 55%, slot2 = 45%

        Has no effect when slot2 is not present (single-canvas layouts handle
        proportions purely via matplotlib axis positioning).
        """
        if not hasattr(self, '_slot2_area') or not self._slot2_area.isVisible():
            return
        sizes = self._panel_splitter.sizes()
        total = sum(sizes) if sizes else self._panel_splitter.height()
        if total <= 0:
            return

        ir_on = getattr(self, '_ir_visible', False)
        # IR on → canvas_meas needs 57.5% (IR 15% + TF/Phase 42.5% of total)
        # IR off → equal 50/50 split
        ratio = 0.575 if ir_on else 0.50
        self._panel_splitter.setSizes([int(total * ratio), int(total * (1 - ratio))])

    def _toggle_ir_panel(self, *args):
        """Cmd+I — toggle IR independientemente del layout Magnitude/Phase."""
        canvas  = self.canvas_meas
        current = getattr(canvas, '_current_view_mode', 'tf_phase')
        # Map every mode to its IR-toggled counterpart
        _toggle_map = {
            'tf_phase':               'tf_phase_ir',
            'tf_phase_ir':            'tf_phase',
            'magnitude_only':         'magnitude_ir',
            'magnitude_ir':           'magnitude_only',
            'phase_only':             'phase_ir',
            'phase_ir':               'phase_only',
            'magnitude_magnitude':    'magnitude_magnitude_ir',
            'magnitude_magnitude_ir': 'magnitude_magnitude',
        }
        target = _toggle_map.get(current)
        if target:
            canvas.set_view_mode(target)
            self._ir_visible = canvas._ir_visible
        if hasattr(self, '_act_live_ir'):
            self._act_live_ir.setChecked(self._ir_visible)
        # Reajustar splitter: 60/40 con IR, 50/50 sin IR
        from PyQt6.QtCore import QTimer as _QT
        _QT.singleShot(30, self._sync_splitter_for_ir)

    def _show_panel_menu_from_canvas(self, name: str):
        """Global view switch triggered from inside-panel dropdown."""
        self._on_view_mode_changed(name)

    # ── Corner X/+ panel toggle callbacks ─────────────────────────────

    def _panel_close_mag(self):
        """✕ or ＋ on Magnitude corner — NEVER touches IR."""
        mode = getattr(self.canvas_meas, '_current_view_mode', 'tf_phase')
        ir   = getattr(self.canvas_meas, '_ir_visible', False)
        if mode in ('tf_phase', 'tf_phase_ir'):
            # ✕ → remove Magnitude, keep Phase (IR state preserved)
            self.canvas_meas.set_view_mode('phase_ir' if ir else 'phase_only')
        elif mode in ('magnitude_only', 'magnitude_ir'):
            # ＋ → add Phase back (IR state preserved)
            self.canvas_meas.set_view_mode('tf_phase_ir' if ir else 'tf_phase')

    def _panel_close_ph(self):
        """✕ or ＋ on Phase corner — NEVER touches IR."""
        mode = getattr(self.canvas_meas, '_current_view_mode', 'tf_phase')
        ir   = getattr(self.canvas_meas, '_ir_visible', False)
        if mode in ('tf_phase', 'tf_phase_ir'):
            # ✕ → remove Phase, keep Magnitude (IR state preserved)
            self.canvas_meas.set_view_mode('magnitude_ir' if ir else 'magnitude_only')
        elif mode in ('phase_only', 'phase_ir'):
            # ＋ → add Magnitude back (IR state preserved)
            self.canvas_meas.set_view_mode('tf_phase_ir' if ir else 'tf_phase')

    # ── Two-slot panel layout ──────────────────────────────────────────

    def _set_panel_type(self, slot: int, canvas_type: str):
        """Change what's shown in panel slot 0 (top) or 1 (bottom)."""
        if slot == 0:
            self._panel1_type = canvas_type.lower()
        else:
            self._panel2_type = canvas_type.lower()
        self._configure_panels()

    def _configure_panels(self):
        """
        Reconfigure the visual layout based on _panel1_type and _panel2_type.

        MeasurementCanvas handles Magnitude+Phase internally (2 subplots).
        SpectrumCanvas handles RTA. SpectrogramCanvas handles Spectrograph.

        When both slots use TF content (magnitude/phase): single MeasurementCanvas
        with appropriate view mode, no QSplitter needed.
        When slots have different canvas types: reparent wraps into two QSplitter slots.
        """
        p1 = self._panel1_type
        p2 = self._panel2_type

        _wrap_for = {
            'magnitude':    self._wrap_meas,
            'phase':        self._wrap_meas,
            'rta':          self._wrap_spec,
            'spectrograph': self._wrap_sgram,
        }
        wrap1 = _wrap_for.get(p1)
        wrap2 = _wrap_for.get(p2)

        if wrap1 is wrap2:
            # Same underlying canvas for both slots → single canvas, internal subplots
            self._ensure_in_main_area(wrap1)
            self._hide_slot2_area()
            if wrap1 is self._wrap_meas:
                ir = getattr(self.canvas_meas, '_ir_visible', False)
                if p1 == 'magnitude' and p2 == 'magnitude':
                    # Both slots want magnitude → dual-magnitude internal mode
                    self.canvas_meas.set_view_mode(
                        'magnitude_magnitude_ir' if ir else 'magnitude_magnitude')
                elif p1 == 'magnitude' and p2 == 'phase':
                    self.canvas_meas.set_view_mode('tf_phase_ir' if ir else 'tf_phase')
                elif p1 == 'phase' and p2 == 'magnitude':
                    self.canvas_meas.set_view_mode('tf_phase_ir' if ir else 'tf_phase')
                elif p1 == 'phase' and p2 == 'phase':
                    # Both slots want phase → show phase_only (can't duplicate phase easily)
                    self.canvas_meas.set_view_mode('phase_ir' if ir else 'phase_only')
                elif p1 == 'magnitude':
                    self.canvas_meas.set_view_mode('magnitude_ir' if ir else 'magnitude_only')
                elif p1 == 'phase':
                    self.canvas_meas.set_view_mode('phase_ir' if ir else 'phase_only')
            # For rta+rta or spectrograph+spectrograph: just show the canvas full screen

        elif wrap1 is not None and wrap2 is not None and wrap1 is not wrap2:
            # Different canvas types: slot1 in _main_area, slot2 in _slot2_area
            self._ensure_in_main_area(wrap1)
            self._ensure_in_slot2_area(wrap2)
            # Set MeasurementCanvas mode for whichever slot uses it
            ir = getattr(self.canvas_meas, '_ir_visible', False)
            if wrap1 is self._wrap_meas:
                mode_map = {'magnitude': 'magnitude_ir' if ir else 'magnitude_only',
                            'phase':     'phase_ir'     if ir else 'phase_only'}
                self.canvas_meas.set_view_mode(mode_map.get(p1, 'magnitude_only'))
            elif wrap2 is self._wrap_meas:
                mode_map = {'magnitude': 'magnitude_ir' if ir else 'magnitude_only',
                            'phase':     'phase_ir'     if ir else 'phase_only'}
                self.canvas_meas.set_view_mode(mode_map.get(p2, 'phase_only'))

        elif wrap1 is not None:
            # Only slot1
            self._ensure_in_main_area(wrap1)
            self._hide_slot2_area()
            if wrap1 is self._wrap_meas:
                ir = getattr(self.canvas_meas, '_ir_visible', False)
                m = {'magnitude': 'magnitude_ir' if ir else 'magnitude_only',
                     'phase':     'phase_ir'     if ir else 'phase_only'}
                self.canvas_meas.set_view_mode(m.get(p1, 'magnitude_only'))

        # Sync slot types to canvas_meas so menus show correct checkmarks
        self.canvas_meas._slot1_type = self._panel1_type
        self.canvas_meas._slot2_type = self._panel2_type or 'phase'
        # Update panel label text on MeasurementCanvas panels
        self._update_panel_labels_from_types()
        # Re-adjust splitter ratio for new panel count
        from PyQt6.QtCore import QTimer as _QT
        _QT.singleShot(40, self._sync_splitter_for_ir)

    def _ensure_in_main_area(self, wrap):
        """Move wrap to _main_area (QStackedWidget) if not already there, show it.
        Also routes the wrap's dropdown to call _set_panel_type(0, ...) for slot1."""
        if self._main_area.indexOf(wrap) == -1:
            self._main_area.addWidget(wrap)
        self._main_area.setCurrentWidget(wrap)
        # Route slot1 dropdown: non-meas canvases need their _on_select updated
        # _wrap_meas uses internal _mag_lbl which already routes to on_panel1_changed
        if wrap is self._wrap_spec:
            wrap._btn.setText('RTA ▾')
            wrap._btn.setVisible(True)
            wrap._on_select = lambda n: self._set_panel_type(0, n.lower())
        elif wrap is self._wrap_sgram:
            wrap._btn.setText('Spectrograph ▾')
            wrap._btn.setVisible(True)
            wrap._on_select = lambda n: self._set_panel_type(0, n.lower())
        elif wrap is self._wrap_meas:
            wrap._btn.setVisible(False)   # meas uses internal _mag_lbl overlay

    def _ensure_in_slot2_area(self, wrap):
        """Move wrap to _slot2_area (create it if needed), show it."""
        if not hasattr(self, '_slot2_area'):
            from PyQt6.QtWidgets import QVBoxLayout as _VBL
            self._slot2_area = QWidget()
            _lay = _VBL(self._slot2_area)
            _lay.setContentsMargins(0, 0, 0, 0)
            _lay.setSpacing(0)
            self._panel_splitter.addWidget(self._slot2_area)
            self._panel_splitter.setStretchFactor(0, 1)
            self._panel_splitter.setStretchFactor(1, 1)

        all_wraps = (self._wrap_meas, self._wrap_spec, self._wrap_sgram)
        if wrap.parent() is not self._slot2_area:
            self._slot2_area.layout().addWidget(wrap)
        for w in all_wraps:
            if w.parent() is self._slot2_area:
                w.setVisible(w is wrap)
        self._slot2_area.setVisible(True)

        # Corregir label Y callback del wrap en slot2
        if wrap is self._wrap_spec:
            wrap._btn.setText('RTA ▾')
            wrap._btn.setVisible(True)
            wrap._on_select = lambda n: self._set_panel_type(1, n.lower())
        elif wrap is self._wrap_sgram:
            wrap._btn.setText('Spectrograph ▾')
            wrap._btn.setVisible(True)
            wrap._on_select = lambda n: self._set_panel_type(1, n.lower())
        elif wrap is self._wrap_meas:
            wrap._btn.setVisible(False)   # meas usa _ph_lbl interno para slot2

        # Ajustar splitter respetando IR: 60/40 si IR ON, 50/50 si IR OFF
        from PyQt6.QtCore import QTimer as _QT
        _QT.singleShot(60, self._sync_splitter_for_ir)

    def _hide_slot2_area(self):
        """Hide slot2 and return any wraps in it back to the main QStackedWidget."""
        if not hasattr(self, '_slot2_area'):
            return
        all_wraps = (self._wrap_meas, self._wrap_spec, self._wrap_sgram)
        for w in all_wraps:
            if w.parent() is self._slot2_area:
                # Re-add to the QStackedWidget (reparents automatically)
                self._main_area.addWidget(w)
        self._slot2_area.setVisible(False)

    def _on_ir_peak_detected(self, delay_ms: float):
        """Called on every refresh when the IR argmax is computed.
        Updates the SELECTED engine card's delay label (display only — no compensation applied).
        The value mirrors what is shown on the IR canvas."""
        sel_idx = getattr(self, '_selected_engine_idx', 0)
        if sel_idx < len(self._tf_engines):
            self._tf_engines[sel_idx].show_detected_delay(delay_ms)

    def _update_panel_labels_from_types(self):
        """Update the panel label text based on current panel types."""
        if hasattr(self.canvas_meas, '_mag_lbl'):
            p1, p2 = self._panel1_type, self._panel2_type
            lbl1 = 'RTA' if p1 == 'rta' else p1.capitalize()
            lbl2 = 'RTA' if p2 == 'rta' else (p2.capitalize() if p2 else '')
            if p1 in ('magnitude', 'phase'):
                self.canvas_meas._mag_lbl.setText(f'{lbl1} ▾')
            if p2 and p2 in ('magnitude', 'phase'):
                self.canvas_meas._ph_lbl.setText(f'{lbl2} ▾')

    def _show_engine_select_menu(self, anchor_widget):
        """Dropdown to pick active engine — shown from top-right corner of panel."""
        if not self._tf_engines:
            return
        _ss = (
            f'QMenu{{background:#1a1a1a;color:#cccccc;border:1px solid #3a3a3a;'
            'font-size:11px;padding:2px 0;}}'
            f'QMenu::item{{padding:5px 20px 5px 10px;}}'
            f'QMenu::item:selected{{background:#2a382a;color:#88cc88;}}'
            f'QMenu::item:checked{{font-weight:bold;}}'
        )
        from PyQt6.QtWidgets import QMenu
        menu = QMenu(self)
        menu.setStyleSheet(_ss)
        sel = getattr(self, '_selected_engine_idx', 0)
        for i, eng in enumerate(self._tf_engines):
            name = getattr(eng, 'name', None) or f'TF {eng._number}'
            act  = menu.addAction(name)
            act.setCheckable(True)
            act.setChecked(i == sel)
            act.triggered.connect(lambda checked, idx=i: self._select_engine(idx))
        menu.exec(anchor_widget.mapToGlobal(anchor_widget.rect().bottomLeft()))

    # ── Engine label update helper ─────────────────────────────────────

    def _refresh_engine_label(self):
        """Push the selected engine name to the canvas corner labels."""
        idx = getattr(self, '_selected_engine_idx', 0)
        if self._tf_engines and 0 <= idx < len(self._tf_engines):
            eng  = self._tf_engines[idx]
            name = getattr(eng, 'name', None) or f'TF {eng._number}'
        else:
            name = '—'
        self.canvas_meas.update_engine_label(name)

    def _switch_to_tf(self):
        self._on_view_mode_changed('Magnitude')

    def _switch_to_spectrum(self):
        self._on_view_mode_changed('RTA')

    def _toggle_spec_panel(self):
        if self._current_view == 'magnitude':
            self._switch_to_spectrum()
        else:
            self._switch_to_tf()

    # ── Callbacks de los dropdowns de panel ───────────────────────────

    def _on_view_mode_changed(self, mode: str):
        """
        Cambia la vista principal y el panel de settings de la derecha.

        Transfer Function → canvas_meas (IR + TF + Phase + Coh completo), settings TF (idx 0)
        Magnitude         → canvas_meas solo TF + Coh (sin IR, sin Phase), settings TF (idx 0)
        Phase             → canvas_meas solo Phase expandido,              settings TF (idx 0)
        Spectrum          → canvas_spec (RTA),                             settings Spec (idx 1)
        Spectrograph      → canvas_sgram,                                  settings Spec (idx 1)
        """
        # Collapsar el layout de dos-slots (panel dropdowns) de vuelta a slot único
        self._hide_slot2_area()
        # Restaurar callbacks globales de spec/sgram (en vista global S/T usan _on_view_mode_changed)
        self._wrap_spec._on_select  = self._on_view_mode_changed
        self._wrap_sgram._on_select = self._on_view_mode_changed
        # En vista global, ocultar _btn de spec/sgram NO — deben ser visibles para cambiar vista
        self._wrap_spec._btn.setVisible(True)
        self._wrap_sgram._btn.setVisible(True)
        # Sincronizar _panel1/2_type con la nueva vista global
        _type_map = {
            'Magnitude':   ('magnitude', 'phase'),
            'Phase':       ('phase',     None),
            'RTA':         ('rta',       None),
            'Spectrograph':('spectrograph', None),
        }
        _t = _type_map.get(mode, ('magnitude', 'phase'))
        self._panel1_type = _t[0]
        self._panel2_type = _t[1]

        # Sincronizar _current en todos los wraps.
        # NO cambiar el texto del _btn aquí — cada wrap mantiene su propio label
        # (_wrap_spec siempre dice "RTA ▾", _wrap_sgram "Spectrograph ▾", etc.)
        for w in (self._wrap_meas, self._wrap_spec, self._wrap_sgram):
            w._current = mode

        if mode == 'Magnitude':
            self._main_area.setCurrentWidget(self._wrap_meas)
            self.canvas_meas.set_view_mode('tf_phase')   # TF + Phase + Coh (sin IR)
            self._current_view = 'magnitude'
            settings_idx = 0

        elif mode == 'Phase':
            self._main_area.setCurrentWidget(self._wrap_meas)
            self.canvas_meas.set_view_mode('phase_only')
            self._current_view = 'phase'
            settings_idx = 0
            # Forzar sync de color: Phase debe tener exactamente el mismo color que TF
            _colors = [e._color for e in self._tf_engines]
            self.canvas_meas.set_engine_colors(_colors)
            self.canvas_meas.highlight_engine(
                getattr(self, '_selected_engine_idx', 0),
                show_avg=self._show_avg)
            # draw() síncrono para asegurar que las trazas guardadas aparezcan
            self.canvas_meas.draw()

        elif mode == 'RTA':
            self._main_area.setCurrentWidget(self._wrap_spec)
            self.canvas_meas.set_view_mode('tf_phase')   # restaurar TF+Phase si venía de Phase-only
            self._current_view = 'spectrum'
            settings_idx = 1

        elif mode == 'Spectrograph':
            self._main_area.setCurrentWidget(self._wrap_sgram)
            self.canvas_meas.set_view_mode('tf_phase')
            self._current_view = 'spectrograph'
            settings_idx = 1

        else:
            self._main_area.setCurrentWidget(self._wrap_meas)
            self.canvas_meas.set_view_mode('tf_phase')   # default: TF + Phase
            self._current_view = 'magnitude'
            settings_idx = 0

        # Reset panel labels después de cambio global de vista
        self._update_panel_labels_from_types()

        if hasattr(self, '_settings_stack'):
            self._settings_stack.setCurrentIndex(settings_idx)

        # Sincronizar panel SAVE izquierdo: 0=TF/Phase, 1=Spectrum/Sgram
        if hasattr(self, '_save_stack'):
            self._save_stack.setCurrentIndex(
                0 if self._current_view in ('magnitude', 'phase') else 1)

    def _on_focus_canvas(self, canvas_type: str):
        """
        Called when the user clicks on a canvas area.
        Switches the left/right side panels to match the clicked canvas type
        without changing the visible canvas itself.
          canvas_type = 'magnitude'  → TF/IR settings + TF save panel
          canvas_type = 'spectrum'   → Spectrum settings + Spectrum save panel
        """
        if canvas_type == 'magnitude':
            settings_idx = 0
            save_idx = 0
        else:  # 'spectrum' / 'spectrograph'
            settings_idx = 1
            save_idx = 1
        if hasattr(self, '_settings_stack'):
            self._settings_stack.setCurrentIndex(settings_idx)
        if hasattr(self, '_save_stack'):
            self._save_stack.setCurrentIndex(save_idx)

    def _on_add_panel(self):
        """Add or remove secondary panel (split view)."""
        if self._secondary_panel is not None:
            # Remove secondary panel — window stays same size, splitter fills naturally
            self._secondary_panel.setParent(None)
            self._secondary_panel.deleteLater()
            self._secondary_panel = None
            return

        # Create secondary panel
        self._secondary_panel = SecondaryPanel(on_close=self._on_add_panel)
        # Share cursor updates
        self._secondary_panel.canvas_spec.on_cursor_update  = self._update_cursor
        self._secondary_panel.canvas_meas.on_cursor_update  = self._update_cursor
        self._secondary_panel.canvas_sgram.on_cursor_update = self._update_cursor

        # Replay any traces already saved before this panel was opened
        cm2 = self._secondary_panel.canvas_meas
        for tr in self._traces:
            cm2.store_trace(
                tr.freqs, tr.mag_db, tr.phase_deg, tr.gamma2, tr.ir,
                tr.delay_ref_ms, tr.color, tr.coh_thresh
            )

        # Add to splitter
        self._panel_splitter.addWidget(self._secondary_panel)

        # Equal stretch so both panels resize proportionally with window
        self._panel_splitter.setStretchFactor(0, 1)
        self._panel_splitter.setStretchFactor(1, 1)

        # 50/50 split after repaint (deferred so splitter has final geometry)
        from PyQt6.QtCore import QTimer as _QT
        def _do_equal_split():
            total = self._panel_splitter.height()
            half  = max(total // 2, 200)
            self._panel_splitter.setSizes([half, total - half])
        _QT.singleShot(0, _do_equal_split)

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
            dlg.setMinimumWidth(400)
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
            cmb_dev.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            cmb_dev.setMinimumWidth(220)
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
                grid.setSpacing(6)
                grid.setColumnStretch(0, 0)
                grid.setColumnStretch(1, 1)

                grid.addWidget(lbl('CH  MEAS', color=TEXT_MID, size=9), 0, 0)
                spn_m = QSpinBox()
                spn_m.setRange(1, 32); spn_m.setValue(eng.spn_m.value())
                spn_m.setMinimumWidth(64); spn_m.setFixedHeight(24)
                grid.addWidget(spn_m, 0, 1)
                spn_m_list.append(spn_m)

                grid.addWidget(lbl('CH  REF', color=TEXT_MID, size=9), 1, 0)
                spn_r = QSpinBox()
                spn_r.setRange(1, 32); spn_r.setValue(eng.spn_r.value())
                spn_r.setMinimumWidth(64); spn_r.setFixedHeight(24)
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
            self.lbl_spl_max.setText('Max  —')

    def _show_spl_settings(self):
        """Redirect — replaced by the unified Amplitude Calibration dialog."""
        self._show_spl_calibration_dialog()

    def _show_spl_calibration_dialog(self):
        """
        SMAART-style Amplitude Calibration dialog.
        Input Device / Input Channel selector, live VU bar, Calibrator Level, Calibrate / OK.
        """
        from PyQt6.QtCore import QTimer as _QT, QSize as _QSize
        from PyQt6.QtGui  import QPainter, QColor, QLinearGradient, QFont as _QFont
        from PyQt6.QtWidgets import QDoubleSpinBox, QSizePolicy
        import numpy as _np_cal

        # ── Custom VU scale (painted tick labels) ─────────────────────
        class _VUScale(QWidget):
            _MARKS  = [-90, -60, -48, -36, -24, -12, -9, -6]
            _DB_MIN = -90.0
            _DB_MAX =   0.0
            def sizeHint(self): return _QSize(360, 16)
            def paintEvent(self, _e):
                p  = QPainter(self)
                w  = self.width();  h = self.height()
                rng = self._DB_MAX - self._DB_MIN
                fn = _QFont(); fn.setPixelSize(9)
                p.setFont(fn)
                p.setPen(QColor('#888888'))
                for db in self._MARKS:
                    x = int((db - self._DB_MIN) / rng * w)
                    p.drawText(x - 14, 0, 28, h,
                               Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
                               str(db))

        # ── Custom VU bar (gradient fill + tick lines) ─────────────────
        class _VUBar(QWidget):
            _DB_MIN = -90.0
            _DB_MAX =   0.0
            def __init__(self):
                super().__init__()
                self._level = -90.0
                self.setFixedHeight(22)
                self.setMinimumWidth(260)
            def set_level(self, db: float):
                self._level = db; self.update()
            def sizeHint(self): return _QSize(360, 22)
            def paintEvent(self, _e):
                p   = QPainter(self)
                w, h = self.width(), self.height()
                rng  = self._DB_MAX - self._DB_MIN
                # background
                p.fillRect(0, 0, w, h, QColor('#111111'))
                # fill
                frac   = max(0.0, min(1.0, (self._level - self._DB_MIN) / rng))
                fill_w = int(frac * w)
                if fill_w > 1:
                    g = QLinearGradient(0, 0, w, 0)
                    g.setColorAt(0.00, QColor('#0d500d'))
                    g.setColorAt(0.60, QColor('#22cc22'))   # ≈ -36 dBFS
                    g.setColorAt(0.86, QColor('#cccc00'))   # ≈ -12 dBFS
                    g.setColorAt(0.93, QColor('#cc7700'))   # ≈  -6 dBFS
                    g.setColorAt(1.00, QColor('#cc2222'))   #    0 dBFS
                    p.fillRect(0, 2, fill_w, h - 4, g)
                # tick lines
                p.setPen(QColor('#2a2a2a'))
                for db in [-60, -48, -36, -24, -12, -9, -6]:
                    x = int((db - self._DB_MIN) / rng * w)
                    p.drawLine(x, 1, x, h - 1)

        # ── Dialog shell ───────────────────────────────────────────────
        dlg = QDialog(self)
        dlg.setWindowTitle('Amplitude Calibration')
        dlg.setFixedSize(520, 430)
        _ss = (
            f'QDialog{{background:#1c1c1c;color:{TEXT_HI};font-size:11px;}}'
            f'QLabel{{background:transparent;color:{TEXT_HI};}}'
            f'QComboBox{{background:#2a2a2a;color:{TEXT_HI};border:1px solid #555;'
            f'border-radius:3px;padding:4px 8px;font-size:11px;}}'
            f'QComboBox::drop-down{{border:none;width:18px;}}'
            f'QDoubleSpinBox{{background:#2a2a2a;color:{TEXT_HI};border:1px solid #555;'
            f'border-radius:3px;padding:3px 6px;font-size:12px;}}'
            f'QPushButton{{background:#2e2e2e;color:{TEXT_HI};border:1px solid #555;'
            f'border-radius:4px;padding:6px 28px;font-size:11px;}}'
            f'QPushButton:hover{{background:#3a3a3a;border-color:#888;}}'
            f'QPushButton:disabled{{color:#444;border-color:#333;}}')
        dlg.setStyleSheet(_ss)

        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(22, 18, 22, 16)
        lay.setSpacing(10)

        # ── Input Device ──────────────────────────────────────────────
        form = QFormLayout(); form.setSpacing(8)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        cmb_device = QComboBox(); cmb_device.setMinimumWidth(240)
        all_devs   = getattr(self, '_cached_all_devices', [])
        input_devs = [d for d in all_devs if int(d.get('in', 0)) > 0]
        for d in input_devs:
            cmb_device.addItem(d['name'], d['id'])
        for i, d in enumerate(input_devs):
            if d['id'] == self.engine.dev_in:
                cmb_device.setCurrentIndex(i); break
        form.addRow('Input Device:', cmb_device)

        # ── Input Channel ─────────────────────────────────────────────
        cmb_channel = QComboBox(); cmb_channel.setMinimumWidth(140)

        def _fill_channels():
            cmb_channel.clear()
            dev_id   = cmb_device.currentData()
            dev_info = next((d for d in all_devs if d['id'] == dev_id), None)
            n_ch     = int(dev_info.get('in', 8)) if dev_info else 8
            for i in range(n_ch):
                cmb_channel.addItem(f'Input {i + 1}', i + 1)
            ch_spl = getattr(self.engine, 'ch_spl', 1)
            cmb_channel.setCurrentIndex(max(0, ch_spl - 1))

        _fill_channels()
        cmb_device.currentIndexChanged.connect(lambda _: _fill_channels())
        form.addRow('Input Channel:', cmb_channel)
        lay.addLayout(form)

        # ── Separator ─────────────────────────────────────────────────
        _sep = QFrame(); _sep.setFrameShape(QFrame.Shape.HLine)
        _sep.setStyleSheet('color:#333;max-height:1px;')
        lay.addWidget(_sep)

        # ── VU meter row ──────────────────────────────────────────────
        vu_form = QFormLayout(); vu_form.setSpacing(4)
        vu_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        vu_container = QWidget()
        vc = QVBoxLayout(vu_container); vc.setContentsMargins(0,0,0,0); vc.setSpacing(2)
        vu_scale = _VUScale(); vc.addWidget(vu_scale)
        vu_bar   = _VUBar();   vc.addWidget(vu_bar)
        vu_form.addRow('Input Level:', vu_container)

        # Numeric readout
        rdout_row = QHBoxLayout(); rdout_row.setSpacing(6)
        lbl_dbfs_val = QLabel('—')
        lbl_dbfs_val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_dbfs_val.setStyleSheet(
            'background:#111;color:#eeeeee;border:1px solid #444;border-radius:3px;'
            'padding:2px 6px;font-size:12px;font-family:Menlo,monospace;'
            'min-width:72px;max-width:72px;')
        rdout_row.addStretch()
        rdout_row.addWidget(lbl_dbfs_val)
        rdout_row.addWidget(QLabel('dBFS'))
        rdout_row.addStretch()
        vu_form.addRow('', rdout_row)
        lay.addLayout(vu_form)

        # ── Calibrator Level ──────────────────────────────────────────
        cal_form = QFormLayout(); cal_form.setSpacing(8)
        cal_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        spn_cal = QDoubleSpinBox()
        spn_cal.setRange(60.0, 140.0); spn_cal.setDecimals(1)
        spn_cal.setValue(94.0); spn_cal.setFixedWidth(82)
        cal_row = QHBoxLayout(); cal_row.setSpacing(6)
        cal_row.addWidget(spn_cal)
        cal_row.addWidget(QLabel('dB')); cal_row.addStretch()
        cal_form.addRow('Calibrator Level:', cal_row)
        lay.addLayout(cal_form)

        # ── Separator 2 ───────────────────────────────────────────────
        _sep2 = QFrame(); _sep2.setFrameShape(QFrame.Shape.HLine)
        _sep2.setStyleSheet('color:#333;max-height:1px;')
        lay.addWidget(_sep2)

        # ── SPL Meter settings ────────────────────────────────────────
        spl_form = QFormLayout(); spl_form.setSpacing(8)
        spl_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        spn_offset = QDoubleSpinBox()
        spn_offset.setRange(-60.0, 60.0); spn_offset.setSingleStep(0.5)
        spn_offset.setDecimals(1); spn_offset.setSuffix(' dB')
        spn_offset.setValue(getattr(self, '_spl_offset_db', 0.0))
        spn_offset.setFixedWidth(90)
        spl_form.addRow('Cal. Offset:', spn_offset)

        spn_warn = QDoubleSpinBox()
        spn_warn.setRange(0.0, 160.0); spn_warn.setSingleStep(1.0)
        spn_warn.setDecimals(1); spn_warn.setSuffix(' dB')
        spn_warn.setValue(getattr(self, '_spl_warn_db', 100.0))
        spn_warn.setFixedWidth(90)
        spl_form.addRow('Warn Threshold:', spn_warn)

        spn_clip = QDoubleSpinBox()
        spn_clip.setRange(0.0, 160.0); spn_clip.setSingleStep(1.0)
        spn_clip.setDecimals(1); spn_clip.setSuffix(' dB')
        spn_clip.setValue(getattr(self, '_spl_clip_db', 118.0))
        spn_clip.setFixedWidth(90)
        spl_form.addRow('Clip Threshold:', spn_clip)

        lay.addLayout(spl_form)

        # ── Status label ──────────────────────────────────────────────
        lbl_status = QLabel('')
        lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_status.setStyleSheet(f'color:{GREEN};font-size:10px;font-weight:bold;')
        lay.addWidget(lbl_status)
        lay.addStretch()

        # ── Bottom buttons ────────────────────────────────────────────
        btn_row = QHBoxLayout(); btn_row.setSpacing(8)
        btn_calibrate = QPushButton('Calibrate')
        btn_ok        = QPushButton('OK')
        btn_row.addWidget(btn_calibrate)
        btn_row.addStretch()
        btn_row.addWidget(btn_ok)
        lay.addLayout(btn_row)

        # ── Live meter (reads engine buffers — no new stream) ─────────
        _live = _QT(dlg); _live.setInterval(80)

        def _read_ch_buf():
            ch_1   = cmb_channel.currentData() or 1
            ch_spl = getattr(self.engine, 'ch_spl', -1)
            ch_ref = getattr(self.engine, 'ch_ref', -1)
            try:
                if ch_1 == ch_spl:
                    return self.engine.get_buffer_spl()
                if ch_1 == ch_ref:
                    return self.engine.buf_ref.read(self.engine.nperseg)
                for idx, mch in enumerate(getattr(self.engine, '_ch_meas_list', [])):
                    if mch == ch_1:
                        return self.engine.buf_meas_list[idx].read(self.engine.nperseg)
            except Exception:
                pass
            return None

        def _upd_live():
            if not self.engine.running:
                vu_bar.set_level(-90.0); lbl_dbfs_val.setText('—'); return
            buf = _read_ch_buf()
            if buf is None or len(buf) == 0:
                return
            rms = float(_np_cal.sqrt(_np_cal.mean(buf.astype(float) ** 2)))
            db  = 20.0 * _np_cal.log10(max(rms, 1e-10))
            vu_bar.set_level(db)
            lbl_dbfs_val.setText(f'{db:.1f}')

        _live.timeout.connect(_upd_live); _live.start()
        dlg.finished.connect(_live.stop)

        # ── Calibrate action ──────────────────────────────────────────
        _samples = []

        def _do_calibrate():
            if not self.engine.running:
                lbl_status.setText('⚠  Start the audio engine first (▶)')
                lbl_status.setStyleSheet('color:#ffaa44;font-size:10px;font-weight:bold;')
                return
            _samples.clear()
            btn_calibrate.setEnabled(False)
            lbl_status.setText('Measuring…  (2 s)')
            lbl_status.setStyleSheet(f'color:{GREEN};font-size:10px;font-weight:bold;')

            _ct = _QT(dlg); _ct.setInterval(100)
            def _collect():
                buf = _read_ch_buf()
                if buf is not None and len(buf) > 0:
                    rms = float(_np_cal.sqrt(_np_cal.mean(buf.astype(float) ** 2)))
                    if rms > 1e-10:
                        _samples.append(rms)
            _ct.timeout.connect(_collect); _ct.start()

            def _finish():
                _ct.stop(); btn_calibrate.setEnabled(True)
                if not _samples:
                    lbl_status.setText('⚠  No signal — check engine and channel')
                    lbl_status.setStyleSheet('color:#ff6644;font-size:10px;font-weight:bold;')
                    return
                avg_rms  = float(_np_cal.mean(_samples))
                avg_dbfs = 20.0 * _np_cal.log10(max(avg_rms, 1e-10))
                cal_spl  = spn_cal.value()
                offset   = cal_spl - avg_dbfs
                ch_1     = cmb_channel.currentData() or 1
                self._channel_spl_offset[ch_1] = offset
                self._spl_offset_db = offset
                self._spl_max_db    = -200.0
                if hasattr(self, 'lbl_spl_max'):
                    self.lbl_spl_max.setText('Max  —')
                lbl_status.setText(
                    f'✓  {avg_dbfs:.1f} dBFS = {cal_spl:.1f} dB SPL   '
                    f'(offset {offset:+.1f} dB)')
                lbl_status.setStyleSheet(f'color:{GREEN};font-size:10px;font-weight:bold;')
                self._save_prefs()
                self.sb.showMessage(
                    f'✓  Calibrated Ch {ch_1}: {avg_dbfs:.1f} dBFS = {cal_spl:.1f} dBSPL', 5000)

            _QT.singleShot(2100, _finish)

        def _on_ok():
            # Save selected channel as the active SPL channel
            ch_sel = cmb_channel.currentData()
            if ch_sel is not None:
                self.engine.ch_spl = int(ch_sel)
            # Save manual offset and thresholds
            self._spl_offset_db = spn_offset.value()
            self._spl_warn_db   = spn_warn.value()
            self._spl_clip_db   = spn_clip.value()
            self._save_prefs()
            dlg.accept()

        btn_calibrate.clicked.connect(_do_calibrate)
        btn_ok.clicked.connect(_on_ok)
        dlg.exec()

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
        # Sincronizar colores y destacar en canvas principal Y secundario
        colors = [eng._color for eng in self._tf_engines]
        self.canvas_meas.set_engine_colors(colors)
        self.canvas_meas.highlight_engine(idx, show_avg=self._show_avg)
        if self._secondary_panel is not None:
            cm2 = self._secondary_panel.canvas_meas
            cm2.set_engine_colors(colors)
            cm2.highlight_engine(idx, show_avg=self._show_avg)
        # Update engine name in panel corners
        self._refresh_engine_label()

    def _on_engine_active_changed(self, idx: int, active: bool):
        """
        Callback del ▶/⏸ por engine.
        - Activar (▶): abre el stream si no está corriendo, inicia el timer.
        - Pausar (⏸): limpia las líneas de ese engine. Los otros siguen midiendo.
        """
        def _clear_eng(canvas, i):
            if i == 0:
                canvas.line_ir.set_data([], [])
                canvas.line_ir_peak.set_xdata([0, 0])
            if i < len(canvas._lines_tf):
                canvas._lines_tf[i].set_data([], [])
                canvas._lines_ph[i].set_data([], [])
            canvas.draw_idle()

        if active:
            # ── El engine quiere medir → asegurar que el stream esté corriendo ──
            if not self.engine.running:
                try:
                    self.engine.start()
                    self._post_start_check()
                except Exception as exc:
                    # Si el stream no pudo arrancar, revertir el botón
                    if idx < len(self._tf_engines):
                        self._tf_engines[idx]._active = False
                        self._tf_engines[idx].btn_play.setChecked(False)
                        self._tf_engines[idx].btn_play.setText('▶')
                    self.sb.showMessage(f'⚠  Error opening stream: {exc}', 8000)
                    return
            if not self.timer.isActive():
                self.timer.start()
                self._silent_ticks = 0
                QTimer.singleShot(3000, self._check_signal_present)
            self._set_running()   # actualizar UI del botón principal
        else:
            # ── El engine se pausa → limpiar sus líneas ──
            _clear_eng(self.canvas_meas, idx)
            if self._secondary_panel is not None:
                _clear_eng(self._secondary_panel.canvas_meas, idx)
            self.canvas_meas.update_avg(None, None, None, None)
            if self._secondary_panel is not None:
                self._secondary_panel.canvas_meas.update_avg(None, None, None, None)
            # Si TODOS los engines están en pausa, detener el timer (pero no el stream)
            if not any(eng.active for eng in self._tf_engines):
                self.timer.stop()
                self._set_stopped()

    def _on_engine_color_changed(self, idx: int, color: str):
        """Callback del color picker del engine — sincroniza canvas y paneles."""
        colors = [eng._color for eng in self._tf_engines]
        self.canvas_meas.set_engine_colors(colors)
        self.canvas_meas.highlight_engine(
            getattr(self, '_selected_engine_idx', 0),
            show_avg=self._show_avg)
        if self._secondary_panel is not None:
            cm2 = self._secondary_panel.canvas_meas
            cm2.set_engine_colors(colors)
            cm2.highlight_engine(
                getattr(self, '_selected_engine_idx', 0),
                show_avg=self._show_avg)

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
            self.sb.showMessage('⚠  No data yet to normalize', 4000)
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
        Dialog de configuración de I/O.
        - Selección exclusiva (radio) del dispositivo activo en Input y Output.
        - Apply confirma la selección y reinicia el stream si estaba corriendo.
        - Cancel descarta cambios.
        """
        dlg = QDialog(self)
        dlg.setWindowTitle('I-O Config')
        dlg.resize(860, 560)
        dlg.setStyleSheet(f'background:#1e1e1e;color:{TEXT_HI};font-size:11px;')

        root = QVBoxLayout(dlg)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # Obtener dispositivos — reutilizar la lista cacheada en _populate_devices()
        # para no llamar sd.query_devices() desde el hilo principal mientras el
        # stream WASAPI está corriendo (causaría freeze en Windows).
        all_devs = getattr(self, '_cached_all_devices', None) or AudioEngine.list_devices()

        _tbl_ss = (
            f'QTableWidget{{background:#111;color:{TEXT_HI};gridline-color:#2a2a2a;'
            f'border:1px solid #2a2a2a;font-size:11px;selection-background-color:#1e3a4a;}}'
            f'QTableWidget::item{{padding:4px 6px;}}'
            f'QHeaderView::section{{background:#2a2a2a;color:{TEXT_MID};'
            f'padding:4px 6px;border:none;border-right:1px solid #1a1a1a;font-size:10px;}}')
        _btn_ss = (
            f'QPushButton{{background:#2a2a2a;color:{TEXT_MID};border:1px solid #3a3a3a;'
            f'border-radius:4px;padding:5px 18px;font-size:11px;min-width:72px;}}'
            f'QPushButton:hover{{background:#333;color:{TEXT_HI};}}'
            f'QPushButton[apply="1"]{{background:#1e3a1e;color:{GREEN};'
            f'border-color:{GREEN};font-weight:bold;}}'
            f'QPushButton[apply="1"]:hover{{background:#244a24;}}')

        # ── Selección temporal (antes de Apply) ───────────────────────
        _sel = {
            'in':  self.engine.dev_in,
            'out': self.engine.dev_out,
        }

        # Snapshot de calibraciones por canal — se actualiza en la UI del Input tab
        _ch_cal_local = dict(self._channel_cal)

        # ── Tabs Input / Output ───────────────────────────────────────
        tab_widget = QTabWidget()
        tab_widget.setStyleSheet(
            f'QTabWidget::pane{{border:1px solid #333;background:#1a1a1a;}}'
            f'QTabBar::tab{{background:#252525;color:{TEXT_MID};padding:6px 22px;'
            f'border:1px solid #333;border-bottom:none;border-radius:3px 3px 0 0;}}'
            f'QTabBar::tab:selected{{background:#1a1a1a;color:{TEXT_HI};}}')
        root.addWidget(tab_widget, stretch=1)

        def _make_panel(mode):
            devs = [d for d in all_devs if d[mode] > 0]
            panel = QWidget()
            pv = QVBoxLayout(panel)
            pv.setContentsMargins(6, 8, 6, 6)
            pv.setSpacing(6)

            lbl = QLabel(
                f'Select the active {"input" if mode=="in" else "output"} device:')
            lbl.setStyleSheet(f'color:{TEXT_MID};font-size:10px;background:transparent;')
            pv.addWidget(lbl)

            tbl = QTableWidget(len(devs), 4)
            tbl.setStyleSheet(_tbl_ss)
            tbl.setHorizontalHeaderLabels(['●', 'Device', 'Channels', 'Sample Rate'])
            tbl.horizontalHeader().setSectionResizeMode(0, tbl.horizontalHeader().ResizeMode.Fixed)
            tbl.horizontalHeader().setSectionResizeMode(1, tbl.horizontalHeader().ResizeMode.Stretch)
            tbl.horizontalHeader().setSectionResizeMode(2, tbl.horizontalHeader().ResizeMode.Fixed)
            tbl.horizontalHeader().setSectionResizeMode(3, tbl.horizontalHeader().ResizeMode.Fixed)
            tbl.setColumnWidth(0, 34)
            tbl.setColumnWidth(2, 80)
            tbl.setColumnWidth(3, 100)
            tbl.setSelectionBehavior(tbl.SelectionBehavior.SelectRows)
            tbl.setEditTriggers(tbl.EditTrigger.NoEditTriggers)
            tbl.verticalHeader().setVisible(False)
            tbl.setSelectionMode(tbl.SelectionMode.SingleSelection)

            radio_grp = []   # lista de QRadioButton

            for row, d in enumerate(devs):
                rb = QRadioButton()
                rb.setChecked(d['id'] == _sel[mode])
                rb.setStyleSheet('QRadioButton{background:transparent;}')
                cw = QWidget(); cl = QHBoxLayout(cw)
                cl.setContentsMargins(6, 0, 0, 0); cl.addWidget(rb)
                tbl.setCellWidget(row, 0, cw)
                radio_grp.append((rb, d['id']))

                tbl.setItem(row, 1, QTableWidgetItem(d['name']))
                tbl.setItem(row, 2, QTableWidgetItem(str(d[mode])))
                tbl.setItem(row, 3, QTableWidgetItem(f"{d['fs']} Hz"))

                # Seleccionar la fila del dispositivo activo
                if d['id'] == _sel[mode]:
                    tbl.selectRow(row)

                # Clic en fila → activa radio
                def _on_row(row=row, radio_grp=radio_grp, mode=mode):
                    for rb2, did in radio_grp:
                        rb2.setChecked(False)
                    radio_grp[row][0].setChecked(True)
                    _sel[mode] = radio_grp[row][1]

                rb.toggled.connect(
                    lambda checked, row=row, radio_grp=radio_grp, mode=mode:
                        _sel.__setitem__(mode, radio_grp[row][1]) if checked else None)

            tbl.cellClicked.connect(
                lambda r, c, rg=radio_grp, mode=mode:
                    [rb2.setChecked(i == r) for i, (rb2, _) in enumerate(rg)]
                    or _sel.__setitem__(mode, rg[r][1]))

            pv.addWidget(tbl, stretch=1)

            # Info del dispositivo seleccionado
            lbl_cur = QLabel()
            lbl_cur.setStyleSheet(
                f'color:{GREEN};font-size:10px;background:transparent;padding:2px 0;')
            cur_name = next((d['name'] for d in devs if d['id'] == _sel[mode]), '—')
            lbl_cur.setText(f'Active: {cur_name}')
            pv.addWidget(lbl_cur)

            if mode == 'in':
                sep_ch = QFrame(); sep_ch.setFrameShape(QFrame.Shape.HLine)
                sep_ch.setStyleSheet('color:#2a2a2a;'); pv.addWidget(sep_ch)

                lbl_ch = QLabel('Channel Setup — Mic Correction Curves')
                lbl_ch.setStyleSheet(
                    f'color:{ACCENT};font-size:9px;letter-spacing:1px;font-weight:bold;'
                    f'background:transparent;padding:2px 0;')
                pv.addWidget(lbl_ch)

                # Determine max channels for selected device
                sel_dev = next((d for d in devs if d['id'] == _sel['in']), None)
                n_ch = min(sel_dev['in'], 16) if sel_dev else 8

                ch_tbl = QTableWidget(n_ch, 4)
                ch_tbl.setStyleSheet(_tbl_ss)
                ch_tbl.setHorizontalHeaderLabels(['Ch', 'Level', 'Channel Name', 'Mic Correction Curve'])
                hh = ch_tbl.horizontalHeader()
                hh.setSectionResizeMode(0, hh.ResizeMode.Fixed)
                hh.setSectionResizeMode(1, hh.ResizeMode.Fixed)
                hh.setSectionResizeMode(2, hh.ResizeMode.Stretch)
                hh.setSectionResizeMode(3, hh.ResizeMode.Fixed)
                ch_tbl.setColumnWidth(0, 32)
                ch_tbl.setColumnWidth(1, 110)
                ch_tbl.setColumnWidth(3, 190)
                ch_tbl.setSelectionMode(ch_tbl.SelectionMode.NoSelection)
                ch_tbl.setEditTriggers(ch_tbl.EditTrigger.NoEditTriggers)
                ch_tbl.verticalHeader().setVisible(False)
                ch_tbl.setFixedHeight(min(n_ch * 28 + 28, 220))

                # Level bar widgets — one QProgressBar per channel
                _level_bars = []   # indexed by ch_idx (0-based)

                _pb_ss = (
                    'QProgressBar{background:#0a0a0a;border:1px solid #222;border-radius:2px;}'
                    'QProgressBar::chunk{background:qlineargradient('
                    'x1:0,y1:0,x2:1,y2:0,'
                    'stop:0 #1a6a1a,stop:0.6 #2aaa2a,stop:0.85 #aaaa00,stop:1 #aa2222);'
                    'border-radius:1px;}')

                def _build_ch_table(n):
                    ch_tbl.setRowCount(n)
                    _level_bars.clear()
                    for ci in range(n):
                        ch_num = ci + 1
                        ch_tbl.setItem(ci, 0, QTableWidgetItem(str(ch_num)))

                        # Level meter bar
                        pb = QProgressBar()
                        pb.setRange(0, 100)
                        pb.setValue(0)
                        pb.setTextVisible(False)
                        pb.setFixedHeight(14)
                        pb.setStyleSheet(_pb_ss)
                        _level_bars.append(pb)
                        pw = QWidget(); pl = QHBoxLayout(pw)
                        pl.setContentsMargins(4, 4, 4, 4); pl.addWidget(pb)
                        ch_tbl.setCellWidget(ci, 1, pw)

                        ch_tbl.setItem(ci, 2, QTableWidgetItem(f'Input {ch_num}'))

                        cal = _ch_cal_local.get(ch_num)
                        btn_label = f'✓ {cal["name"]}' if cal else 'None  ＋'
                        btn_cal_ch = QPushButton(btn_label)
                        btn_cal_ch.setStyleSheet(
                            f'QPushButton{{font-size:9px;padding:1px 6px;border:none;'
                            f'color:{"#4aaa4a" if cal else TEXT_MID};background:transparent;text-align:left;}}'
                            f'QPushButton:hover{{color:{TEXT_HI};}}')

                        def _make_cal_handler(ch=ch_num, btn=btn_cal_ch):
                            def _handler():
                                if _ch_cal_local.get(ch):
                                    _ch_cal_local.pop(ch, None)
                                    btn.setText('None  ＋')
                                    btn.setStyleSheet(
                                        f'QPushButton{{font-size:9px;padding:1px 6px;border:none;'
                                        f'color:{TEXT_MID};background:transparent;text-align:left;}}'
                                        f'QPushButton:hover{{color:{TEXT_HI};}}')
                                    return
                                path, _ = QFileDialog.getOpenFileName(
                                    dlg, f'Load Mic Correction — Ch {ch}',
                                    os.path.expanduser('~'),
                                    'Calibration files (*.txt *.csv);;All files (*)')
                                if not path:
                                    return
                                try:
                                    _, data = _load_curve_file(path)
                                    name = os.path.basename(path)
                                    _ch_cal_local[ch] = {'freqs': data[:, 0], 'db': data[:, 1], 'name': name}
                                    btn.setText(f'✓ {name}')
                                    btn.setStyleSheet(
                                        f'QPushButton{{font-size:9px;padding:1px 6px;border:none;'
                                        f'color:#4aaa4a;background:transparent;text-align:left;}}'
                                        f'QPushButton:hover{{color:#ff6666;}}')
                                except Exception as e:
                                    QMessageBox.critical(dlg, 'Error', str(e))
                            return _handler

                        btn_cal_ch.clicked.connect(_make_cal_handler())
                        ch_tbl.setCellWidget(ci, 3, btn_cal_ch)

                _build_ch_table(n_ch)
                pv.addWidget(ch_tbl)

                # ── Live level meter: reads engine buffers (no extra stream) ──
                # CoreAudio/WASAPI cannot open two InputStream on the same device.
                # We read the already-captured buffers from the running engine.
                import numpy as _np_io

                _meter_timer = QTimer(dlg)
                _meter_timer.setInterval(100)

                def _update_meters():
                    if not self.engine.running:
                        return
                    nperseg = self.engine.nperseg
                    ch_spl  = getattr(self.engine, 'ch_spl',  -1)
                    ch_ref  = getattr(self.engine, 'ch_ref',  -1)
                    ch_meas = list(getattr(self.engine, '_ch_meas_list', []))
                    for ci, pb in enumerate(_level_bars):
                        ch_1 = ci + 1  # 1-indexed channel number
                        buf  = None
                        try:
                            if ch_1 == ch_spl:
                                buf = self.engine.buf_spl.read(nperseg)
                            elif ch_1 == ch_ref:
                                buf = self.engine.buf_ref.read(nperseg)
                            else:
                                for midx, mch in enumerate(ch_meas):
                                    if mch == ch_1:
                                        buf = self.engine.buf_meas_list[midx].read(nperseg)
                                        break
                        except Exception:
                            pass
                        if buf is not None and len(buf) > 0:
                            rms = float(_np_io.sqrt(_np_io.mean(buf.astype(float) ** 2)))
                            db  = 20.0 * _np_io.log10(max(rms, 1e-10))
                            pct = int(max(0.0, min(100.0, (db + 60.0) * 100.0 / 60.0)))
                            pb.setValue(pct)
                        else:
                            pb.setValue(0)

                def _cleanup():
                    _meter_timer.stop()

                _meter_timer.timeout.connect(_update_meters)
                _meter_timer.start()
                dlg.finished.connect(_cleanup)

            return panel

        tab_widget.addTab(_make_panel('in'),  'Input')
        tab_widget.addTab(_make_panel('out'), 'Output')

        # ── Botones Apply / Cancel ────────────────────────────────────
        btn_apply  = QPushButton('Apply')
        btn_cancel = QPushButton('Cancel')
        btn_apply.setProperty('apply', '1')
        for b in (btn_apply, btn_cancel):
            b.setStyleSheet(_btn_ss)

        def _apply():
            # ── Aplicar selección de dispositivos ────────────────────────
            self.engine.dev_in  = _sel['in']
            self.engine.dev_out = _sel['out']

            # Actualizar combos de device en la UI principal
            for i, did in enumerate(getattr(self, '_dev_in_ids', [])):
                if did == _sel['in']:
                    self.cmb_dev_in.blockSignals(True)
                    self.cmb_dev_in.setCurrentIndex(i)
                    self.cmb_dev_in.blockSignals(False)
                    break
            for i, did in enumerate(getattr(self, '_dev_out_ids', [])):
                if did == _sel['out']:
                    self.cmb_dev_out.blockSignals(True)
                    self.cmb_dev_out.setCurrentIndex(i)
                    self.cmb_dev_out.blockSignals(False)
                    break

            # Sincronizar Signal Generator Output con el nuevo dev_out
            out_name = next(
                (d['name'] for d in all_devs if d['id'] == _sel['out']), 'Output')
            self._noise_dev_list = [(_sel['out'], out_name)]
            if hasattr(self, '_btn_noise_out'):
                short = out_name[:22] + '…' if len(out_name) > 22 else out_name
                self._btn_noise_out.setText(f'{short} ▾')

            # ── Parar todo limpiamente ────────────────────────────────────
            # Cambiar el dispositivo WASAPI en caliente es inestable en Windows.
            # Detenemos stream, ruido y engines; el usuario reinicia con ▶.
            self.timer.stop()
            if self.engine.running:
                self.engine.stop()
            self.engine.noise_on = False
            if hasattr(self, 'btn_noise_p'):
                self.btn_noise_p.setChecked(False)
            for eng in getattr(self, '_tf_engines', []):
                eng._active = False
                eng.btn_play.setChecked(False)
                eng.btn_play.setText('▶')
            self._set_stopped()

            # Apply per-channel calibrations from dialog
            self._channel_cal.update(_ch_cal_local)

            self._save_prefs()
            self.sb.showMessage(
                '✓  Device updated — press ▶ to start', 5000)
            dlg.accept()

        btn_apply.clicked.connect(_apply)
        btn_cancel.clicked.connect(dlg.reject)

        bot = QHBoxLayout()
        bot.addStretch()
        bot.addWidget(btn_cancel)
        bot.addWidget(btn_apply)
        root.addLayout(bot)

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

        # ── ARCHIVO ───────────────────────────────────────────────────
        fm = mb.addMenu('File')
        fm.addAction('Save Workspace',       self._save_workspace_explicit).setShortcut('Ctrl+Shift+W')
        fm.addSeparator()
        fm.addAction('Export IR…',           self._save_ir_txt).setShortcut('Ctrl+Shift+I')
        fm.addAction('Export TF…',           self._save_tf_txt).setShortcut('Ctrl+Shift+T')
        fm.addAction('Export Phase…',        self._save_ph_txt).setShortcut('Ctrl+Shift+P')
        fm.addAction('Export Spectrum…',     self._save_sp_txt).setShortcut('Ctrl+Shift+S')
        fm.addSeparator()
        fm.addAction('Save Graph (PNG)…',    self._save_graph_png).setShortcut('Ctrl+G')
        fm.addSeparator()
        fm.addAction('Load Mic Calibration…', self._load_mic_cal).setShortcut('Ctrl+K')
        fm.addAction('Clear Mic Calibration', self._clear_mic_cal)
        fm.addSeparator()
        fm.addAction('Quit', self.close).setShortcut('Ctrl+Q')

        # ── CONFIG ───────────────────────────────────────────────────
        cm = mb.addMenu('Config')

        cm.addAction('Manage Configurations…',       self._show_manage_configs)
        cm.addSeparator()

        cm.addAction('I-O Config…',                  self._show_io_config).setShortcut('Alt+A')
        cm.addAction('Measurement Config…',          self._show_measurement_config).setShortcut('Alt+G')
        cm.addAction('SPL Config…',                  self._show_spl_calibration_dialog).setShortcut('Ctrl+Shift+E')
        cm.addAction('Command Bar Config…',          self._show_command_bar_config)
        cm.addSeparator()

        cm.addAction('New Spectrum Measurement…',    self._new_spectrum_measurement).setShortcut('Ctrl+S')
        cm.addAction('New Spectrum Average…',        self._new_spectrum_avg)
        cm.addAction('New TF…',                      self._show_new_tf_dialog).setShortcut('Ctrl+T')
        cm.addAction('New TF Average…',              self._new_tf_avg)
        cm.addSeparator()

        cm.addAction('New Tab',                      self._new_tab)
        cm.addAction('Duplicate Tab',                self._duplicate_tab)
        cm.addAction('Delete Tab',                   self._delete_tab)
        cm.addAction('Move Tab',                     self._move_tab)
        cm.addSeparator()

        cm.addAction('Amplitude Calibration…',       self._show_amplitude_cal)

        # ── OPTIONS ──────────────────────────────────────────────────
        om = mb.addMenu('Options')

        om.addAction('Preferences…',                 self._show_preferences).setShortcut('Alt+O')
        om.addSeparator()

        # Graph Settings submenu
        gs_menu = om.addMenu('Graph Settings')
        gs_menu.addAction('Spectrum…',               self._show_spectrum_settings).setShortcut('Alt+S')
        gs_menu.addAction('Transfer Function…',      self._show_tf_settings).setShortcut('Alt+T')
        gs_menu.addAction('Impulse Response…',       self._show_ir_settings).setShortcut('Alt+I')

        # Measurement Settings submenu
        ms_menu = om.addMenu('Measurement Settings')
        ms_menu.addAction('I-O Config…',             self._show_io_config)
        ms_menu.addAction('Measurement Config…',     self._show_measurement_config)
        ms_menu.addSeparator()
        ms_menu.addAction('Find Delay  [D]',         self._on_find_delay)
        ms_menu.addAction('Reset Delay  [R]',        self._on_delay_reset)
        ms_menu.addSeparator()
        ms_menu.addAction('SPL Config…',             self._show_spl_calibration_dialog)
        ms_menu.addAction('Amplitude Calibration…',  self._show_amplitude_cal)

        om.addSeparator()
        om.addAction('Signal Generator…',            self._show_signal_generator_dialog).setShortcut('Alt+N')
        om.addSeparator()
        om.addAction('Target Curves…',               self._show_target_curves_dialog).setShortcut('Alt+X')
        om.addAction('Optimize to Target…',          self._show_optimize_dialog).setShortcut('Alt+Z')
        om.addAction('Weighting Curves…',            self._show_weighting_dialog)
        om.addSeparator()
        om.addAction('Load Mic Correction…',         self._load_mic_cal)
        om.addAction('Clear Mic Correction',         self._clear_mic_cal)

        # ── VIEW ──────────────────────────────────────────────────────
        vm = mb.addMenu('View')

        # ── Modos de medición ─────────────────────────────────────────
        from PyQt6.QtGui import QAction as _QA
        self._act_rt  = _QA('Real Time Mode',         self, checkable=True, checked=True)
        self._act_ir  = _QA('Impulse Response Mode', self, checkable=True)
        self._act_spl = _QA('SPL Mode',              self, checkable=True)
        # 'R' está reservado para Reset Delay (Command) → usamos Ctrl+1/2/3
        self._act_rt.setShortcut('Ctrl+1')
        self._act_ir.setShortcut('Ctrl+2')
        self._act_spl.setShortcut('Ctrl+3')

        def _set_mode(act, others):
            def _f(checked):
                if not checked:
                    act.setChecked(True); return
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
        vp_menu.addAction('Magnitude',  self._switch_to_tf)
        vp_menu.addAction('RTA',        self._switch_to_spectrum)
        vm.addSeparator()

        # ── Windows / panels ──────────────────────────────────────────
        _act_client = vm.addAction('Client Window…',
                                   lambda: self.sb.showMessage('Client Window — coming soon', 2000))
        _act_client.setShortcut('Alt+R')

        inp_menu = vm.addMenu('Input Meters')
        inp_menu.addAction('Show Input Meters',
                           lambda: self.sb.showMessage('Input Meters — coming soon', 2000))

        vm.addAction('SPL Meters  [E]', self._toggle_spl_meters)
        vm.addSeparator()

        # ── Graph toggles ─────────────────────────────────────────────
        self._act_live_ir = _QA('Live IR  [Cmd+I]', self, checkable=True, checked=False)
        self._act_live_ir.setShortcut('Ctrl+I')   # Ctrl = Cmd en macOS
        self._act_live_ir.triggered.connect(self._toggle_ir_panel)
        vm.addAction(self._act_live_ir)

        self._act_peak = _QA('Peak Hold', self, checkable=True)
        self._act_peak.setShortcut('Alt+P')
        self._act_peak.triggered.connect(self._toggle_peak_hold)
        vm.addAction(self._act_peak)

        self._act_coh = _QA('Coherence', self, checkable=True, checked=True)
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

        # ── UI bars / panels ──────────────────────────────────────────
        self._act_spl_bar  = _QA('SPL/Data Bar',             self, checkable=True, checked=True)
        self._act_ctrl_bar = _QA('Control Bar',              self, checkable=True, checked=True)
        self._act_cmd_bar  = _QA('Command Bar',              self, checkable=True)
        self._act_tab_bar  = _QA('Tab Bar',                  self, checkable=True, checked=True)
        self._act_spl_met  = _QA('SPL Meter',                self, checkable=True, checked=True)
        self._act_compact  = _QA('Compact Generator',        self, checkable=True)

        self._act_spl_bar.setShortcut('B')
        self._act_ctrl_bar.setShortcut('O')
        self._act_cmd_bar.setShortcut('U')
        self._act_tab_bar.setShortcut('A')
        self._act_spl_met.setShortcut('Alt+K')

        self._act_spl_bar.triggered.connect( lambda c: self._toggle_ui_bar('spl_bar',  c))
        self._act_ctrl_bar.triggered.connect(lambda c: self._toggle_ui_bar('ctrl_bar', c))
        self._act_spl_met.triggered.connect( lambda c: self._toggle_ui_bar('spl_meter',c))
        # Tab Bar → muestra/oculta la fila del workspace tab bar
        self._act_tab_bar.triggered.connect(
            lambda c: self._ws_row.setVisible(c) if hasattr(self, '_ws_row') else None)
        # Command Bar / Compact — sin widget real, solo actualiza estado
        self._act_cmd_bar.triggered.connect(
            lambda c: self.sb.showMessage(f'Barra de Comandos {"visible" if c else "oculta"}', 2000))
        self._act_compact.triggered.connect(
            lambda c: self.sb.showMessage(f'Generador Compacto {"activado" if c else "desactivado"}', 2000))

        for a in (self._act_spl_bar, self._act_ctrl_bar, self._act_cmd_bar,
                  self._act_tab_bar, self._act_spl_met, self._act_compact):
            vm.addAction(a)
        vm.addSeparator()

        # Toggle SPL/Clock — 'K' en _sc(), sin shortcut aquí
        vm.addAction('SPL/Reloj  [K]',            self._toggle_spl_clock)
        vm.addAction('Barra SPL/Data',            lambda: self._act_spl_bar.trigger()).setShortcut('Alt+E')
        vm.addSeparator()

        vm.addAction('Panel de Ajustes',          self._on_toggle_settings).setShortcut('Ctrl+Right')
        vm.addAction('Save Panel',                self._on_toggle_save).setShortcut('Ctrl+Left')

        # ── COMANDO ──────────────────────────────────────────────────
        # NOTA: Space, D, F, G, P ya están registrados como ApplicationShortcut
        # vía _sc() — los QAction NO llevan setShortcut() para evitar
        # "Ambiguous shortcut" (Qt dispararía el slot dos veces).
        # El hint [X] en el texto es solo informativo.
        xm = mb.addMenu('Comando')
        xm.addAction('Iniciar  [Space]',          self._on_start)
        xm.addAction('Detener',                   self._on_stop)
        xm.addSeparator()
        xm.addAction('Buscar Retardo  [D]',       self._on_find_delay)
        xm.addAction('Resetear Retardo  [R]',     self._on_delay_reset).setShortcut('R')
        xm.addAction('Congelar  [F]',             lambda: self.btn_freeze_p.click())
        xm.addSeparator()
        xm.addAction('Capturar Traza  [P]',       self._capture_trace_dialog)
        xm.addAction('Activar/Desact. Ruido  [G]',lambda: self.btn_noise_p.click())

        # ── AYUDA ────────────────────────────────────────────────────
        hm = mb.addMenu('Ayuda')
        hm.addAction('Atajos de Teclado', self._show_shortcuts)
        hm.addAction('Acerca de Coherence…', self._show_about)

        # ── Shortcuts globales — ApplicationShortcut para que funcionen ──
        # aunque el canvas matplotlib tenga el foco de teclado.
        # En macOS: Qt Ctrl = Command(⌘), Qt Meta = Control(⌃)
        from PyQt6.QtCore import Qt as _Qt

        def _sc(key, slot):
            s = QShortcut(QKeySequence(key), self)
            s.setContext(_Qt.ShortcutContext.ApplicationShortcut)
            s.activated.connect(slot)
            return s

        # Multi-key shortcuts — estos NO son interceptados por matplotlib
        _sc('Ctrl+I', self._toggle_ir_panel)

        # Teclas de una sola letra — se registran via _install_key_filter()
        # (event filter a nivel QApplication, más confiable que QShortcut con
        #  ApplicationShortcut cuando matplotlib FigureCanvasQTAgg tiene el foco)
        self._install_key_filter()

    def _install_key_filter(self):
        """Event filter instalado en QApplication — intercepta teclas de una sola
        letra ANTES de que matplotlib FigureCanvasQTAgg las consuma.

        matplotlib tiene atajos propios ('p'=pan, 's'=save, 'g'=grid, 'r'=reset
        view, 'c'=back…) que consume a nivel de widget (keyPressEvent). Los
        QShortcut con ApplicationShortcut se procesan DESPUÉS del dispatch al
        widget focalizado, por lo que también son bloqueados. El único método
        infalible es un QObject.eventFilter en QApplication, que se llama ANTES
        de cualquier dispatch.
        """
        from PyQt6.QtCore import QObject, QEvent, Qt as _Qt2
        from PyQt6.QtWidgets import (QApplication, QLineEdit, QTextEdit,
                                     QSpinBox, QDoubleSpinBox, QComboBox)

        _TEXT_WIDGETS = (QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox)

        bindings = {
            _Qt2.Key.Key_Space: self._on_start,
            _Qt2.Key.Key_D:     self._on_find_delay,
            _Qt2.Key.Key_F:     lambda: self.btn_freeze_p.click(),
            _Qt2.Key.Key_G:     lambda: self.btn_noise_p.click(),
            _Qt2.Key.Key_P:     self._capture_trace_dialog,
            _Qt2.Key.Key_S:     self._switch_to_spectrum,
            _Qt2.Key.Key_T:     self._switch_to_tf,
            _Qt2.Key.Key_C:     self._toggle_coherence,
            _Qt2.Key.Key_E:     self._toggle_spl_meters,
            _Qt2.Key.Key_K:     self._toggle_spl_clock,
            _Qt2.Key.Key_R:     self._on_delay_reset,
            _Qt2.Key.Key_V:     self._clear_averaging_buffers,   # SMAART: Clear Averaging Buffer
            _Qt2.Key.Key_B:     self._toggle_data_bar,           # SMAART: Toggle Data Bar
        }

        class _AppKeyFilter(QObject):
            def eventFilter(self_, obj, event):
                if event.type() == QEvent.Type.KeyPress:
                    key = event.key()
                    if key in bindings:
                        fw = QApplication.focusWidget()
                        if not isinstance(fw, _TEXT_WIDGETS):
                            bindings[key]()
                            return True   # consume — no llega a matplotlib
                return False

        self._app_key_filter = _AppKeyFilter(self)
        QApplication.instance().installEventFilter(self._app_key_filter)

    def _build_statusbar(self):
        self.sb = QStatusBar()
        self.setStatusBar(self.sb)
        self.sb.showMessage('Coherence v0.2  —  GPL v3')

    # ── Populate devices ──────────────────────────────────────────────

    def _populate_devices(self):
        devices = AudioEngine.list_devices()
        self._cached_all_devices = devices   # reutilizado en _show_io_config sin rellamar al API
        self._dev_in_ids    = []
        self._dev_out_ids   = []
        self._dev_noise_ids = []
        self._noise_dev_list = []   # [(dev_id, name), …] para el menú del generador
        _initial_noise_name  = 'Output'

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
                    # Solo el dispositivo configurado en I-O Config aparece en Signal Generator
                    self._noise_dev_list.append((d['id'], d['name']))
                    _initial_noise_name = d['name']

        self.cmb_dev_in.currentIndexChanged.connect(self._on_dev_in)
        self.cmb_dev_out.currentIndexChanged.connect(self._on_dev_out)

        # Actualizar label del botón Output con el dispositivo actual
        if hasattr(self, '_btn_noise_out'):
            short = _initial_noise_name[:22] + '…' if len(_initial_noise_name) > 22 else _initial_noise_name
            self._btn_noise_out.setText(f'{short} ▾')

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
        """
        Main ▶ — activates all engines.
        TF engines open the stream via _on_engine_active_changed.
        Spectrum-only workspaces (no TF engines) start the stream directly here.
        """
        for eng in self._tf_engines:
            if not eng.active:
                eng._on_toggle_active(True)

        # Spectrum-only mode: no TF engines to delegate to → start directly
        if not self._tf_engines:
            if not self.engine.running:
                try:
                    self.engine.start()
                except Exception as exc:
                    self.sb.showMessage(f'⚠  Error opening stream: {exc}', 8000)
                    return
            if not self.timer.isActive():
                self.timer.start()
                self._silent_ticks = 0
                QTimer.singleShot(3000, self._check_signal_present)
            self._set_running()

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
        dlg.setWindowTitle('No input signal')
        dlg.setIcon(QMessageBox.Icon.Warning)
        dlg.setText(
            '<b>Audio stream is open but no signal is arriving.</b><br><br>'
            'On macOS this happens when Python does not have microphone permission.<br><br>'
            '<b>How to fix it:</b><br>'
            '1. Open <b>System Settings → Privacy &amp; Security → Microphone</b><br>'
            '2. Enable permission for <b>Python</b> (or Terminal / your app).<br>'
            '3. Restart Coherence.<br><br>'
            'If permission is already enabled, verify that the channel selected '
            'in TF Engine Config matches the physical channel on your interface.'
        )
        dlg.setStandardButtons(QMessageBox.StandardButton.Ok)
        dlg.setStyleSheet(
            f'QMessageBox{{background:{BG_APP};color:{TEXT_HI};font-size:11px;}}'
            f'QPushButton{{background:#2a2a2a;color:{TEXT_HI};padding:4px 12px;'
            f'border:1px solid {BORDER};border-radius:3px;}}'
        )
        dlg.exec()

    def _on_stop(self):
        """
        Main ⏹ — deactivates all engines.
        Spectrum-only workspaces stop the stream directly.
        """
        for eng in self._tf_engines:
            if eng.active:
                eng._on_toggle_active(False)

        # Spectrum-only mode: stop stream directly
        if not self._tf_engines:
            self.timer.stop()
            self.engine.stop()
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

    def _post_start_check(self):
        """
        Called after engine.start() succeeds.
        If the engine fell back to a different device, update the UI.
        """
        fb = getattr(self.engine, '_fallback_device_name', None)
        if fb:
            self.sb.showMessage(
                f'⚠  Interface not found — using "{fb}" (built-in). '
                'Change device in Options → I-O Config.', 10000)
            # Sync combobox to actual device
            for i, did in enumerate(self._dev_in_ids):
                if did == self.engine.dev_in:
                    self.cmb_dev_in.blockSignals(True)
                    self.cmb_dev_in.setCurrentIndex(i)
                    self.cmb_dev_in.blockSignals(False)
                    break
            # Clear the flag so we don't re-warn
            del self.engine._fallback_device_name

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
            self.sb.showMessage('▶  Stream restarted', 3000)
        except Exception as exc:
            self.timer.stop()
            self._set_stopped()
            self.sb.showMessage(f'⚠  Error restarting stream: {exc}', 10000)

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
        if checked and self.engine.dev_out < 0:
            self.sb.showMessage(
                '⚠  Configure audio device in I-O Config before using the generator', 5000)
            for b in self._all_noise_btns:
                b.setChecked(False)
            self._show_io_config()
            return

        # Start engine first so has_output is set correctly
        if checked and not self.engine.running:
            try:
                self.engine.start()
            except Exception as exc:
                self.sb.showMessage(f'⚠  Error opening stream: {exc}', 6000)
                for b in self._all_noise_btns:
                    b.setChecked(False)
                return

        # Check if stream actually has output capability
        if checked and not getattr(self.engine, 'has_output', True):
            QMessageBox.warning(
                self, 'No Output Device',
                'The audio engine is running in <b>input-only mode</b>.<br><br>'
                'The signal generator needs an output device.<br>'
                'Go to <b>Options → I-O Config</b> and select a valid output device,<br>'
                'then restart the engine (▶).')
            for b in self._all_noise_btns:
                b.setChecked(False)
            return

        self.engine.noise_on = checked
        # Sync state across all generator buttons
        for b in self._all_noise_btns:
            b.setChecked(checked)
        if not checked:
            # Generator off: close stream if measurement is also stopped
            if not self.timer.isActive():
                self.engine.stop()

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

    def _show_noise_dev_menu(self):
        """Muestra el menú de dispositivos de salida al pulsar el botón Output ▾."""
        menu = QMenu(self)
        menu.setStyleSheet(
            f'QMenu{{background:#1e221e;color:{TEXT_HI};border:1px solid #3a3a3a;'
            f'font-size:11px;padding:2px 0;}}'
            f'QMenu::item{{padding:5px 28px 5px 12px;}}'
            f'QMenu::item:selected{{background:#2a382a;color:{GREEN};}}'
            f'QMenu::item:checked{{font-weight:bold;}}'
        )

        # Configure…
        act_cfg = menu.addAction('Configure…')
        act_cfg.triggered.connect(self._show_io_config)
        menu.addSeparator()

        # Dispositivos de salida
        for dev_id, name in getattr(self, '_noise_dev_list', []):
            act = menu.addAction(name)
            act.setCheckable(True)
            act.setChecked(dev_id == self.engine.dev_out)
            act.triggered.connect(
                lambda checked, did=dev_id, n=name: self._set_noise_dev(did, n))

        menu.exec(self._btn_noise_out.mapToGlobal(
            self._btn_noise_out.rect().bottomLeft()))

    def _set_noise_dev(self, dev_id: int, name: str):
        """Aplica el dispositivo de salida seleccionado."""
        self.engine.dev_out = dev_id
        # Actualizar label del botón con el nombre corto del dispositivo
        short = name[:22] + '…' if len(name) > 22 else name
        if hasattr(self, '_btn_noise_out'):
            self._btn_noise_out.setText(f'{short} ▾')
        if self.engine.running:
            self._safe_restart()

    def _on_noise_dev(self, idx):
        # Mantenido por compatibilidad (ya no se usa con el nuevo menú)
        pass

    # ── Window type helper ───────────────────────────────────────────────
    _WINDOW_MAP = {
        'Hann':        lambda n: np.hanning(n),
        'Hamming':     lambda n: np.hamming(n),
        'Blackman':    lambda n: np.blackman(n),
        'Flat Top':    lambda n: np.array([
            0.21557895 - 0.41663158*np.cos(2*np.pi*k/(n-1))
            + 0.27726316*np.cos(4*np.pi*k/(n-1))
            - 0.08357895*np.cos(6*np.pi*k/(n-1))
            + 0.00694737*np.cos(8*np.pi*k/(n-1))
            for k in range(n)], dtype=np.float64),
        'Rectangular': lambda n: np.ones(n, dtype=np.float64),
    }

    def _restart_engine_window(self, window_name: str):
        """Cambia la ventana FFT de todos los TF engines sin reiniciar el stream."""
        import numpy as _np
        win_fn = self._WINDOW_MAP.get(window_name, self._WINDOW_MAP['Hann'])
        engines = []
        if hasattr(self, 'engine') and hasattr(self.engine, '_win'):
            engines.append(self.engine)
        for row in getattr(self, '_tf_engines', []):
            eng = getattr(row, 'engine', None) or getattr(row, '_engine', None)
            if eng and hasattr(eng, '_win'):
                engines.append(eng)
        for eng in engines:
            try:
                n   = eng.nperseg
                win = win_fn(n).astype(_np.float64)
                eng._win  = win
                eng._wpow = float(_np.sum(win ** 2))
            except Exception:
                pass
        self.sb.showMessage(f'Data Window: {window_name}', 3000)

    def _clear_averaging_buffers(self):
        """V — Clear Averaging Buffer (igual que SMAART).
        Resetea los RunningTF de todos los engines TF y Spectrum."""
        # TF engines
        if hasattr(self, '_running_tfs'):
            for rtf in self._running_tfs:
                try: rtf.reset()
                except Exception: pass
        # Spectrum engines
        for eng in getattr(self, '_sp_engines', []):
            try: eng['rtf'].reset()
            except Exception: pass
        # Limpiar canvas TF
        if hasattr(self, 'canvas_meas'):
            self.canvas_meas.clear()
        self.sb.showMessage('Averaging buffer cleared', 2000)

    def _toggle_data_bar(self):
        """B — Toggle Data Bar (igual que SMAART).
        Muestra/oculta el panel lateral de engines y settings."""
        if hasattr(self, '_settings_widget'):
            self._settings_widget.setVisible(not self._settings_widget.isVisible())

    def _on_delay_reset(self):
        """Reset global (tecla R) → resetea todos los engines y vuelve a modo raw."""
        self._delay_comp_ms = 0.0
        self.lbl_delay_full.setText('—')
        self.canvas_meas._delay_ref_ms = 0.0
        self.canvas_meas._ir_centered  = False   # volver a eje de tiempo real
        self.canvas_meas.ax_ir.axvline  # marcador queda en 0 hasta próximo refresh
        self.canvas_meas.draw_idle()
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
            # Meter de nivel en la fila del engine
            _rows = getattr(self, '_sp_eng_rows', [])
            if i < len(_rows):
                _rms_db = 20.0 * np.log10(float(np.sqrt(np.mean(sig ** 2))) + 1e-9)
                _rows[i].update_level(_rms_db)
            # ── Panel secundario — Spectrum ───────────────────────────
            if (self._secondary_panel is not None and
                    self._secondary_panel._current_view == 'RTA'):
                sp2 = self._secondary_panel.canvas_spec
                # Agregar líneas de engine al canvas secundario si faltan
                while len(sp2._sp_eng_lines) <= eng['canvas_idx']:
                    sp2.add_sp_engine_line(eng.get('color', ENGINE_PALETTE[i % len(ENGINE_PALETTE)]))
                sp2.update_sp_engine(eng['canvas_idx'], rtf.freqs, rtf.Gxx)
            need_draw = True

        # ── Spectrum Avg ─────────────────────────────────────────────
        _sp_cfg = getattr(self, '_sp_avg_config', None)
        if _sp_cfg is not None and self._sp_engines:
            eps = 1e-12
            _contribs = _sp_cfg.contributors if _sp_cfg.contributors else list(range(len(self._sp_engines)))
            _contribs = [i for i in _contribs if i < len(self._sp_engines) and self._sp_engines[i].get('active', True)]
            _lev_arrays = []
            _fc_ref     = None
            for i in _contribs:
                eng = self._sp_engines[i]
                rtf = eng.get('rtf')
                if rtf is None or not rtf._ready:
                    continue
                if self.canvas_spec._bpo == 0:
                    mask = (rtf.freqs >= 20) & (rtf.freqs <= 20000)
                    _fc_ref = rtf.freqs[mask]
                    _lev_arrays.append(10.0 * np.log10(rtf.Gxx[mask] + eps))
                else:
                    _lev_arrays.append(self.canvas_spec._cpb(
                        rtf.freqs, rtf.Gxx,
                        self.canvas_spec._centers, self.canvas_spec._edges))
                    _fc_ref = self.canvas_spec._centers
            if _lev_arrays:
                if _sp_cfg.avg_type == 'dB':
                    _sp_avg = np.mean(_lev_arrays, axis=0)
                else:  # Power
                    _sp_avg = 10.0 * np.log10(np.mean([10.0 ** (l / 10.0) for l in _lev_arrays], axis=0) + eps)
                self.canvas_spec.update_ch2_avg(None, _sp_avg, freqs=_fc_ref)
                need_draw = True
            else:
                self.canvas_spec.update_ch2_avg(None, None)
        elif _sp_cfg is None:
            self.canvas_spec.update_ch2_avg(None, None)

        if need_draw:
            self.canvas_spec.draw_idle()
            if (self._secondary_panel is not None and
                    self._secondary_panel._current_view == 'RTA'):
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

    def _open_sp_avg_dialog(self, edit: bool = False):
        """Open the TFAvgDialog (Spectrum mode) to create or edit a SpectrumAvgConfig."""
        eng_names = [e.get('name', f'Spectrum {i+1}') for i, e in enumerate(self._sp_engines)]
        if not eng_names:
            QMessageBox.information(self, 'No Engines',
                                    'Add at least one Spectrum engine first.')
            return
        cfg = self._sp_avg_config if edit else None
        while True:
            dlg = TFAvgDialog(eng_names, parent=self, config=cfg, show_coh_weighted=False)
            dlg.setWindowTitle('Edit Spectrum Average' if edit else 'New Spectrum Average')
            if dlg.exec() == QDialog.DialogCode.Accepted:
                raw = dlg.get_config()
                if raw:
                    self._sp_avg_config = SpectrumAvgConfig(
                        name=raw.name, avg_type=raw.avg_type,
                        contributors=raw.contributors)
            if getattr(dlg, '_create_another', False):
                cfg = None; continue
            break

    def _open_tf_avg_dialog(self, edit: bool = False):
        """Open the TFAvgDialog to create or edit a TFAvgConfig."""
        eng_names = []
        for i, eng in enumerate(self._tf_engines):
            n = getattr(eng, '_name', None) or f'TF {i + 1}'
            eng_names.append(n)
        if not eng_names:
            QMessageBox.information(self, 'No Engines',
                                    'Add at least one TF engine before creating an average.')
            return

        cfg = self._tf_avg_config if edit else None
        while True:
            dlg = TFAvgDialog(eng_names, parent=self, config=cfg)
            if dlg.exec() == QDialog.DialogCode.Accepted:
                new_cfg = dlg.get_config()
                if new_cfg:
                    self._tf_avg_config = new_cfg
                    self.btn_show_avg.setText(f'● {new_cfg.name}')
                    # Auto-enable AVG
                    self.btn_show_avg.setChecked(True)
                    self._show_avg = True
                if getattr(dlg, '_create_another', False):
                    cfg = None   # reset for next dialog
                    continue
            break

    def _on_avg_changed(self, idx: int):
        """Cambia n_averages y resetea los acumuladores RunningTF.

        Sin reset, el nuevo alpha tarda decenas de bloques en desplazar
        la historia vieja — el cambio es casi imperceptible en tiempo real.
        Con reset, el nuevo valor de averaging tiene efecto en el
        siguiente bloque que se pushea.
        """
        n = self._avg_values[idx]
        self.engine.n_averages = n
        # Aplicar alpha nuevo y limpiar acumuladores de todos los RunningTF
        for rtf in getattr(self, '_running_tfs', []):
            rtf.set_n_averages(n)
            rtf.reset()

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

        # ── Detectar qué canvases necesitan refresh ───────────────────
        # Incluye: vista global (view) + panel dropdown slots + secondary panel
        _need_spec  = (view == 'spectrum')
        _need_sgram = (view == 'spectrograph')

        # Panel dropdown (dos-slots): _panel1_type / _panel2_type
        if getattr(self, '_panel1_type', None) == 'rta' \
                or getattr(self, '_panel2_type', None) == 'rta':
            _need_spec = True
        if getattr(self, '_panel1_type', None) == 'spectrograph' \
                or getattr(self, '_panel2_type', None) == 'spectrograph':
            _need_sgram = True

        # Panel secundario (ventana flotante)
        if self._secondary_panel is not None:
            sv = self._secondary_panel._current_view
            if sv == 'RTA':
                _need_spec = True
            elif sv == 'Spectrograph':
                _need_sgram = True

        if _need_spec:
            self._refresh_spectrum()
        if _need_sgram:
            self._refresh_spectrogram()

        # Si la vista principal ES solo spectrum/spectrograph, no continuar con TF
        if view == 'spectrum' or view == 'spectrograph':
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
        _SPL_FONT = 'font-size:42px;font-weight:bold;background:transparent;letter-spacing:-1px;'
        if not self.engine.running:
            # Stream caído: resetear display
            _dim = f'color:{TEXT_DIM};{_SPL_FONT}'
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
            _spl_ch = getattr(self.engine, 'ch_spl', 1)
            _spl_cal = self._channel_spl_offset.get(_spl_ch, 0.0)
            spl_db = rms_spl_f + self._spl_offset_db + _spl_cal

            # Clamp to physically plausible SPL range before display / max tracking
            spl_db_clamped = max(-20.0, min(160.0, spl_db))

            if spl_db_clamped >= self._spl_clip_db:
                spl_color = '#ef5350'
            elif spl_db_clamped >= self._spl_warn_db:
                spl_color = '#ffb74d'
            else:
                spl_color = '#66bb6a'

            bar_pct   = max(0, min(100, int((spl_db_clamped - 50) / 80.0 * 100)))
            spl_style = f'color:{spl_color};{_SPL_FONT}'
            bar_style = (f'QProgressBar{{background:#111;border:none;border-radius:3px;}}'
                         f'QProgressBar::chunk{{background:{spl_color};border-radius:3px;}}')
            # Actualizar max — solo si el valor calibrado es razonable (offset OK)
            if -20.0 <= spl_db <= 160.0 and spl_db > getattr(self, '_spl_max_db', -200.0):
                self._spl_max_db = spl_db
                if hasattr(self, 'lbl_spl_max'):
                    self.lbl_spl_max.setText(f'Max  {spl_db:.1f}')
            for _lv, _br in zip(
                    getattr(self, '_spl_labels', [self.lbl_spl_val]),
                    getattr(self, '_spl_bars',   [self.bar_spl])):
                _lv.setText(f'{spl_db_clamped:.1f}')
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
        _eng0_active = self._tf_engines[0].active

        rtf0 = self._running_tfs[0] if self._running_tfs else None

        freqs = gamma2 = mag_db = phase_deg = gxx_db = None

        if rtf0 is not None:
            # Solo acumular datos si el engine está activo (pause = no push)
            if _signal_ok and _eng0_active:
                rtf0.push(x, y0)
            if rtf0.ready:
                freqs, gamma2, mag_db, phase_deg, gxx_db = rtf0.get_tf(
                    delay_comp_s    = delay0 / 1000.0,
                    smooth_fraction = smooth_frac,
                )
            elif _eng0_active:
                # Engine activo pero sin datos aún → esperar
                return
            # Si engine0 inactivo y rtf0 sin datos, continuar para procesar eng1
        else:
            if _eng0_active:
                if not _signal_ok:
                    return
                freqs, gamma2, mag_db, phase_deg, gxx_db, _ = compute_analysis(
                    x, y0, nperseg=self.engine.nperseg, fs=self.engine.fs,
                    smooth_fraction=smooth_frac, delay_comp_s=delay0/1000.0)

        # Aplicar offset y calibración solo si engine0 activo y tiene datos
        if _eng0_active and freqs is not None:
            mag_db = mag_db + self._tf_engines[0]._gain_offset_db
            _ch0 = self._tf_engines[0].spn_m.value() if hasattr(self._tf_engines[0], 'spn_m') else 1
            _cal0 = self._channel_cal.get(_ch0)
            if _cal0 is not None:
                mag_db = mag_db - np.interp(freqs, _cal0['freqs'], _cal0['db'], left=0.0, right=0.0)

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
            _ch1 = self._tf_engines[1].spn_m.value() if len(self._tf_engines) > 1 and hasattr(self._tf_engines[1], 'spn_m') else 2
            _cal1 = self._channel_cal.get(_ch1)
            if mag_db2 is not None and _cal1 is not None:
                mag_db2 = mag_db2 - np.interp(freqs2, _cal1['freqs'], _cal1['db'], left=0.0, right=0.0)

        # ── Promedio de TODOS los engines activos (N engines) ────────────
        # Solo incluir engines que estén activos Y con datos
        _all_lin = []
        _all_ph  = []
        _all_g2  = []
        if _eng0_active and mag_db is not None:
            _all_lin.append(10.0 ** (mag_db / 20.0))
            _all_ph.append(phase_deg)
            _all_g2.append(gamma2)
        if mag_db2 is not None:
            _all_lin.append(10.0 ** (mag_db2 / 20.0))
            _all_ph.append(phase_deg2)
            _all_g2.append(gamma2_2)

        # Engines 2+ (si existen y están activos)
        for _ei in range(2, len(self._tf_engines)):
            if not self._tf_engines[_ei].active:
                self.canvas_meas.update_engine_n(_ei, None, None, None)
                if self._secondary_panel is not None and hasattr(self._secondary_panel, 'canvas_meas'):
                    self._secondary_panel.canvas_meas.update_engine_n(_ei, None, None, None)
                continue
            _y_ei    = self.engine.get_buffer_meas(_ei)
            _delay_ei = self._tf_engines[_ei]._delay_comp_ms
            _rtf_ei   = self._running_tfs[_ei] if _ei < len(self._running_tfs) else None
            if _rtf_ei is not None:
                if _signal_ok:
                    _rtf_ei.push(x, _y_ei)
                if _rtf_ei.ready:
                    _freqs_ei, _g2ei, _mei, _phei, _ = _rtf_ei.get_tf(
                        delay_comp_s=_delay_ei / 1000.0,
                        smooth_fraction=smooth_frac)
                    _gain_ei = self._tf_engines[_ei]._gain_offset_db
                    _mei = _mei + _gain_ei
                    _ch_ei = self._tf_engines[_ei].spn_m.value() if hasattr(self._tf_engines[_ei], 'spn_m') else (_ei + 1)
                    _cal_ei = self._channel_cal.get(_ch_ei)
                    if _cal_ei is not None:
                        _mei = _mei - np.interp(_freqs_ei, _cal_ei['freqs'], _cal_ei['db'], left=0.0, right=0.0)
                    _all_lin.append(10.0 ** (_mei / 20.0))
                    _all_ph.append(_phei)
                    _all_g2.append(_g2ei)
                    self.canvas_meas.update_engine_n(_ei, _freqs_ei, _mei, _phei)
                    if self._secondary_panel is not None and hasattr(self._secondary_panel, 'canvas_meas'):
                        self._secondary_panel.canvas_meas.update_engine_n(_ei, _freqs_ei, _mei, _phei)
                else:
                    self.canvas_meas.update_engine_n(_ei, None, None, None)
                    if self._secondary_panel is not None and hasattr(self._secondary_panel, 'canvas_meas'):
                        self._secondary_panel.canvas_meas.update_engine_n(_ei, None, None, None)
            else:
                self.canvas_meas.update_engine_n(_ei, None, None, None)
                if self._secondary_panel is not None and hasattr(self._secondary_panel, 'canvas_meas'):
                    self._secondary_panel.canvas_meas.update_engine_n(_ei, None, None, None)

        mag_avg = ph_avg = g2_avg = None
        _cfg = getattr(self, '_tf_avg_config', None)
        if _cfg is not None and len(_all_lin) >= 1:
            # ── TFAvgConfig: selective contributors, avg type, coh-weighted ──
            _contribs = _cfg.contributors if _cfg.contributors else list(range(len(_all_lin)))
            _contribs = [i for i in _contribs if i < len(_all_lin)]
            if _contribs:
                _sel_lin = [_all_lin[i] for i in _contribs]
                _sel_ph  = [_all_ph[i]  for i in _contribs]
                _sel_g2  = [_all_g2[i]  for i in _contribs]
                if _cfg.coh_weighted:
                    _weights = [np.clip(g, 0.0, 1.0) for g in _sel_g2]
                    _w_sum   = np.sum(_weights, axis=0) + eps
                    if _cfg.avg_type == 'dB':
                        _sel_db  = [20.0 * np.log10(l + eps) for l in _sel_lin]
                        mag_avg  = np.sum([d * w for d, w in zip(_sel_db, _weights)], axis=0) / _w_sum
                    else:
                        mag_avg  = 20.0 * np.log10(
                            np.sum([l * w for l, w in zip(_sel_lin, _weights)], axis=0) / _w_sum + eps)
                    ph_avg   = np.sum([p * w for p, w in zip(_sel_ph, _weights)], axis=0) / _w_sum
                    g2_avg   = np.sum([g * w for g, w in zip(_sel_g2, _weights)], axis=0) / _w_sum
                else:
                    if _cfg.avg_type == 'dB':
                        _sel_db  = [20.0 * np.log10(l + eps) for l in _sel_lin]
                        mag_avg  = np.mean(_sel_db, axis=0)
                    else:
                        mag_avg  = 20.0 * np.log10(np.mean(_sel_lin, axis=0) + eps)
                    ph_avg   = np.mean(_sel_ph, axis=0)
                    g2_avg   = np.mean(_sel_g2, axis=0)
        elif len(_all_lin) > 1:
            # Default: simple linear average of all active engines
            mag_avg = 20.0 * np.log10(np.mean(_all_lin, axis=0) + eps)
            ph_avg  = np.mean(_all_ph, axis=0)
            g2_avg  = np.mean(_all_g2, axis=0)

        # ── IR via IFFT coherence-weighted — usa el engine seleccionado ─
        sel_idx  = getattr(self, '_selected_engine_idx', 0)
        sel_idx  = min(sel_idx, len(self._running_tfs) - 1)
        rtf_sel  = self._running_tfs[sel_idx] if self._running_tfs else rtf0
        y_sel    = self.engine.get_buffer_meas(sel_idx)
        _sel_active = (sel_idx < len(self._tf_engines) and
                       self._tf_engines[sel_idx].active)
        # Pushear solo si: el selected engine es diferente de eng0 (ya pushado)
        # Y si el selected engine está activo
        if _signal_ok and rtf_sel is not None and rtf_sel != rtf0 and _sel_active:
            rtf_sel.push(x, y_sel)

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

        # ── Broadband Filter (HPF/LPF) del IR Tab ────────────────────
        if getattr(self, '_ir_hpf_en', False) or getattr(self, '_ir_lpf_en', False):
            try:
                from scipy.signal import butter, sosfilt
                _fs_ir = float(self.engine.fs)
                _nyq   = _fs_ir / 2.0
                sos_stages = []
                if getattr(self, '_ir_hpf_en', False):
                    _hpf = min(self._ir_hpf_hz, _nyq * 0.95) / _nyq
                    if 0 < _hpf < 1:
                        sos_stages.append(butter(4, _hpf, btype='high', output='sos'))
                if getattr(self, '_ir_lpf_en', False):
                    _lpf = min(self._ir_lpf_hz, _nyq * 0.98) / _nyq
                    if 0 < _lpf < 1:
                        sos_stages.append(butter(4, _lpf, btype='low', output='sos'))
                for sos in sos_stages:
                    ir = sosfilt(sos, ir).astype(np.float32)
                # Re-normalizar tras filtrado
                _pk = np.max(np.abs(ir))
                if _pk > 0:
                    ir = ir / _pk
            except Exception:
                pass

        self.canvas_meas._fs      = self.engine.fs
        self.canvas_meas._nperseg = self.engine.nperseg

        # ── Update canvas — solo si el engine tiene datos ─────────────
        if _eng0_active and freqs is not None:
            mask    = (freqs >= 20) & (freqs <= 20000)
            avg_coh = float(np.mean(gamma2[mask]))
            self.canvas_meas.update_plots(
                freqs, gamma2, mag_db, phase_deg, gxx_db, ir,
                coh_thresh    = thresh,
                unwrap_phase  = getattr(self, '_unwrap_phase',  False),
                coh_squared   = getattr(self, '_coh_squared',   True),
                phase_as_gd   = getattr(self, '_phase_as_gd',   False),
            )
        else:
            # Engine 0 en pausa — limpiar sus líneas TF/Phase
            self.canvas_meas.line_tf.set_data([], [])
            self.canvas_meas.line_ph.set_data([], [])
            # Mostrar IR del engine SELECCIONADO si tiene datos activos,
            # o limpiar si no hay señal útil
            if ir is not None and _sel_active:
                self.canvas_meas.update_ir_only(ir)
            else:
                self.canvas_meas.line_ir.set_data([], [])
                self.canvas_meas.line_ir_peak.set_xdata([0, 0])
                self.canvas_meas.draw_idle()
            # Calcular avg_coh desde engine 1 si está disponible
            avg_coh = 0.0
            if freqs2 is not None and gamma2_2 is not None:
                _m2 = (freqs2 >= 20) & (freqs2 <= 20000)
                avg_coh = float(np.mean(gamma2_2[_m2]))
        if freqs2 is not None and self._show_ch2:
            self.canvas_meas.update_ch2(freqs2, mag_db2, phase_deg2,
                                        gamma2_2, coh_thresh=thresh)
        else:
            self.canvas_meas.update_ch2(None, None, None, None)

        _avg_freqs_main = freqs if freqs is not None else freqs2
        if mag_avg is not None and self._show_avg and _avg_freqs_main is not None:
            self.canvas_meas.update_avg(_avg_freqs_main, mag_avg, ph_avg,
                                        g2_avg, coh_thresh=thresh)
        else:
            self.canvas_meas.update_avg(None, None, None, None)

        # ── Panel secundario — Magnitude / Phase ──────────────────────
        if (self._secondary_panel is not None and
                self._secondary_panel._current_view in ('Magnitude', 'Phase')):
            cm2 = self._secondary_panel.canvas_meas
            cm2._fs = self.engine.fs
            if _eng0_active and freqs is not None:
                cm2.update_plots(freqs, gamma2, mag_db, phase_deg,
                                 gxx_db, ir, coh_thresh=thresh)
            else:
                cm2.line_tf.set_data([], [])
                cm2.line_ph.set_data([], [])
                if ir is not None and _sel_active:
                    cm2.update_ir_only(ir)
                else:
                    cm2.line_ir.set_data([], [])
                    cm2.line_ir_peak.set_xdata([0, 0])
                    cm2.draw_idle()
            if freqs2 is not None:
                cm2.update_ch2(freqs2, mag_db2, phase_deg2,
                               gamma2_2, coh_thresh=thresh)
            else:
                cm2.update_ch2(None, None, None, None)
            _avg_freqs = freqs if freqs is not None else freqs2
            if mag_avg is not None and self._show_avg and _avg_freqs is not None:
                cm2.update_avg(_avg_freqs, mag_avg, ph_avg, g2_avg, coh_thresh=thresh)
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
            self, 'Save TXT', default_name, 'Text files (*.txt);;All files (*)')
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
        self.sb.showMessage(f'Saved: {os.path.basename(path)}', 4000)

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
        """Open I-O Config so user can load per-channel mic correction."""
        self._show_io_config()

    def _clear_mic_cal(self):
        """Clear all channel mic corrections."""
        self._channel_cal.clear()
        self.sb.showMessage('All mic calibrations cleared', 3000)

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
            1: (self.canvas_spec,  'RTA'),
            2: (self.canvas_sgram, 'Spectrogram'),
        }
        canvas, name = canvas_map.get(tab, (self.canvas_meas, 'graph'))
        ts   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(os.path.expanduser('~/Desktop'),
                            f'coherence_{name}_{ts}.png')
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Graph', path, 'PNG (*.png);;PDF (*.pdf)')
        if not path:
            return
        canvas.fig.savefig(path, dpi=150, facecolor=BG_PANEL,
                           bbox_inches='tight')
        self.sb.showMessage(f'Graph saved: {os.path.basename(path)}', 4000)

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
            if hasattr(self.canvas_meas, '_coh_fill') and self.canvas_meas._coh_fill is not None:
                self.canvas_meas._coh_fill.set_visible(checked)
            self.canvas_meas.draw_idle()
        self.sb.showMessage(f'Coherence {"ON" if checked else "OFF"}', 2000)

    def _show_input_meters(self):
        """Muestra un diálogo con los niveles de entrada en tiempo real."""
        self.sb.showMessage('Input Meters — coming soon', 3000)

    def _toggle_spl_meters(self):
        """E — toggle del SPL meter en el panel."""
        if hasattr(self, '_spl_labels'):
            visible = not self._spl_labels[0].isVisible()
            for lbl in self._spl_labels:
                lbl.parentWidget().setVisible(visible) if lbl.parentWidget() else lbl.setVisible(visible)
        self.sb.showMessage('SPL Meters toggled', 2000)

    def _toggle_ui_bar(self, bar: str, checked: bool):
        """Toggle de barras de UI — spl_bar, ctrl_bar, spl_meter."""
        if bar == 'spl_bar':
            # Data/SPL Meter Bar: las barras QProgressBar de cada canvas
            for b in getattr(self, '_spl_bars', []):
                b.setVisible(checked)
                p = b.parent()
                if p:
                    p.setVisible(checked)
        elif bar == 'ctrl_bar':
            # Control Bar: la barra de info/status en la parte superior
            if hasattr(self, '_info_bar_widget'):
                self._info_bar_widget.setVisible(checked)
        elif bar == 'spl_meter':
            # SPL Meter: los labels de nivel superpuestos en los canvases
            if hasattr(self, '_spl_labels'):
                for lbl in self._spl_labels:
                    p = lbl.parent()
                    if p:
                        p.setVisible(checked)
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
        self._open_sp_avg_dialog(edit=False)

    def _new_tf_avg(self):
        self._open_tf_avg_dialog(edit=False)

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
        self._show_spl_calibration_dialog()

    # ── Options menu — dialogs ────────────────────────────────────────

    @staticmethod
    def _dlg_style_str():
        """Stylesheet común para todos los dialogs de Options."""
        return (
            f'QDialog{{background:{BG_APP};color:{TEXT_HI};font-size:11px;}}'
            f'QGroupBox{{color:{TEXT_MID};font-size:10px;border:1px solid #333;'
            f'border-radius:4px;margin-top:8px;padding-top:6px;}}'
            f'QGroupBox::title{{subcontrol-origin:margin;left:8px;}}'
            f'QLabel{{color:{TEXT_MID};}}'
            f'QCheckBox{{color:{TEXT_HI};}}'
            f'QCheckBox::indicator{{width:14px;height:14px;}}'
            f'QComboBox,QSpinBox,QDoubleSpinBox{{background:#252525;color:{TEXT_HI};'
            f'border:1px solid #444;border-radius:3px;padding:2px 4px;}}'
            f'QPushButton{{background:#2a2a2a;color:{TEXT_HI};padding:4px 14px;'
            f'border:1px solid #444;border-radius:3px;}}'
            f'QPushButton:hover{{background:#333;}}'
        )

    def _show_preferences(self):
        """Preferences dialog — configuración general de la app."""
        from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout,
                                     QFormLayout, QGroupBox, QCheckBox,
                                     QSpinBox, QDoubleSpinBox, QComboBox,
                                     QDialogButtonBox, QLabel)
        dlg = QDialog(self)
        dlg.setWindowTitle('Preferences')
        dlg.setMinimumWidth(400)
        dlg.setStyleSheet(self._dlg_style_str())
        lay = QVBoxLayout(dlg)
        lay.setSpacing(10)

        # ── Display ───────────────────────────────────────────────────
        grp_disp = QGroupBox('Display')
        fd = QFormLayout(grp_disp)
        fd.setSpacing(6)
        self._pref_peak_hold = QCheckBox('Peak Hold')
        self._pref_peak_hold.setChecked(getattr(self, '_peak_hold', False))
        self._pref_peak_hold.toggled.connect(
            lambda c: (setattr(self, '_peak_hold', c),
                       self._toggle_peak_hold(c)))
        fd.addRow(self._pref_peak_hold)

        self._pref_show_coh = QCheckBox('Show Coherence Overlay')
        self._pref_show_coh.setChecked(True)
        self._pref_show_coh.toggled.connect(
            lambda c: self.canvas_meas.ax_coh.set_visible(c) or
                      self.canvas_meas.draw_idle())
        fd.addRow(self._pref_show_coh)

        self._pref_show_ir = QCheckBox('Show Live IR Panel  (Cmd+I)')
        self._pref_show_ir.setChecked(getattr(self, '_ir_visible', True))
        self._pref_show_ir.toggled.connect(
            lambda c: self._toggle_ir_panel() if c != self._ir_visible else None)
        fd.addRow(self._pref_show_ir)
        lay.addWidget(grp_disp)

        # ── Refresh ───────────────────────────────────────────────────
        grp_ref = QGroupBox('Refresh Rate')
        fr = QFormLayout(grp_ref)
        fr.setSpacing(6)
        spn_fps = QSpinBox()
        spn_fps.setRange(1, 60)
        spn_fps.setValue(max(1, 1000 // max(self.timer.interval(), 1)))
        spn_fps.setSuffix('  fps')
        spn_fps.valueChanged.connect(
            lambda v: self.timer.setInterval(max(16, 1000 // v)))
        fr.addRow(QLabel('Update Rate:'), spn_fps)
        lay.addWidget(grp_ref)

        # ── Buttons ───────────────────────────────────────────────────
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(dlg.accept)
        lay.addWidget(bb)
        dlg.exec()

    # ── Graph Settings — dialogs por tipo de gráfico (estilo SMAART) ──────

    def _show_spectrum_settings(self):
        """Graph Settings abriendo en tab Spectrum (⌥S)."""
        self._show_graph_settings_dialog(start_tab=0)

    def _show_tf_settings(self):
        """Graph Settings abriendo en tab Transfer Function (⌥T)."""
        self._show_graph_settings_dialog(start_tab=1)

    def _show_ir_settings(self):
        """Graph Settings abriendo en tab Impulse Response (⌥I)."""
        self._show_graph_settings_dialog(start_tab=2)

    def _show_graph_settings_dialog(self, start_tab: int = 0):
        """
        Dialog unificado Graph Settings — 3 tabs estilo SMAART:
          0: Spectrum  |  1: Transfer Function  |  2: Impulse Response
        """
        from PyQt6.QtWidgets import (
            QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
            QGroupBox, QTabWidget, QWidget, QScrollArea,
            QLabel, QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox,
            QPushButton, QSizePolicy,
        )

        dlg = QDialog(self)
        dlg.setWindowTitle('Graph Settings')
        dlg.setMinimumWidth(520)
        dlg.setStyleSheet(
            self._dlg_style_str() +
            # Estilo de tabs estilo SMAART oscuro
            f'QTabWidget::pane{{border:1px solid #333;background:{BG_APP};}}'
            f'QTabBar::tab{{background:#1a1a1a;color:{TEXT_MID};'
            f'  padding:5px 16px;border:1px solid #333;border-bottom:none;'
            f'  font-size:11px; margin-right:2px;}}'
            f'QTabBar::tab:selected{{background:{BG_APP};color:{TEXT_HI};'
            f'  border-bottom:1px solid {BG_APP};}}'
            f'QGroupBox{{color:{TEXT_MID};font-size:10px;border:1px solid #333;'
            f'  border-radius:4px;margin-top:10px;padding-top:8px;padding-bottom:4px;}}'
            f'QGroupBox::title{{subcontrol-origin:margin;left:8px;padding:0 4px;}}'
        )

        root = QVBoxLayout(dlg)
        root.setContentsMargins(10, 10, 10, 8)
        root.setSpacing(8)

        tabs = QTabWidget()
        root.addWidget(tabs)

        # ── helpers internos ───────────────────────────────────────────
        def _lbl(t, dim=False):
            l = QLabel(t)
            l.setStyleSheet(f'color:{"#666" if dim else TEXT_MID};font-size:10px;background:transparent;')
            return l

        def _chk(text, checked=False, enabled=True):
            c = QCheckBox(text)
            c.setChecked(checked)
            c.setEnabled(enabled)
            return c

        def _spn(lo, hi, val, suffix='', decimals=0):
            s = QSpinBox() if decimals == 0 else QDoubleSpinBox()
            s.setRange(int(lo), int(hi))
            s.setValue(val)
            if suffix: s.setSuffix(suffix)
            if decimals and hasattr(s, 'setDecimals'): s.setDecimals(decimals)
            s.setFixedWidth(68)
            return s

        def _cmb(opts, current=0):
            c = QComboBox()
            c.addItems(opts)
            c.setCurrentIndex(current)
            return c

        def _scroll_tab():
            """Devuelve (tab_widget, inner_layout) listos para agregar groups."""
            tab = QWidget()
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(scroll.Shape.NoFrame)
            scroll.setStyleSheet('QScrollArea{background:transparent;}')
            inner = QWidget()
            inner.setStyleSheet('background:transparent;')
            lay = QVBoxLayout(inner)
            lay.setContentsMargins(6, 6, 6, 6)
            lay.setSpacing(8)
            scroll.setWidget(inner)
            tab_lay = QVBoxLayout(tab)
            tab_lay.setContentsMargins(0, 0, 0, 0)
            tab_lay.addWidget(scroll)
            return tab, lay

        # ══════════════════════════════════════════════════════════════
        # TAB 0 — SPECTRUM
        # ══════════════════════════════════════════════════════════════
        tab_sp, sp_lay = _scroll_tab()
        tabs.addTab(tab_sp, 'RTA')

        # ── General Settings ──────────────────────────────────────────
        grp_gen = QGroupBox('General Settings')
        gg = QGridLayout(grp_gen)
        gg.setContentsMargins(10, 14, 10, 8); gg.setSpacing(7)
        gg.setColumnStretch(1, 1); gg.setColumnStretch(3, 1)

        # Frequency Scale → BPO del espectro
        _sp_bpo_map = {'1/12 Octave': 12, '1/6 Octave': 6,
                       '1/3 Octave': 3,  '1 Octave': 1, 'Broadband (lin)': 0}
        _cur_bpo = getattr(self, '_sp_bpo', [3, 12, 24])
        _cur_bpo_val = _cur_bpo[0] if isinstance(_cur_bpo, list) else _cur_bpo
        _bpo_keys = list(_sp_bpo_map.keys())
        _bpo_idx  = next((i for i, v in enumerate(_sp_bpo_map.values())
                         if v == _cur_bpo_val), 2)
        cmb_fscale = _cmb(_bpo_keys, _bpo_idx)

        cmb_weight = _cmb(['None', 'A-weight', 'C-weight', 'Z-weight'], 0)
        chk_disp_w = _chk('Display Weighting', False)
        chk_link_b = _chk('Link Plot Banding', True)
        chk_link_z = _chk('Link Plot Zoom', False)

        gg.addWidget(_lbl('Frequency Scale:'), 0, 0, Qt.AlignmentFlag.AlignRight)
        gg.addWidget(cmb_fscale,               0, 1)
        gg.addWidget(chk_link_b,               0, 2, 1, 2)

        gg.addWidget(_lbl('Weighting:'),       1, 0, Qt.AlignmentFlag.AlignRight)
        gg.addWidget(cmb_weight,               1, 1)
        gg.addWidget(chk_link_z,               1, 2, 1, 2)

        gg.addWidget(chk_disp_w,               2, 0, 1, 2)
        sp_lay.addWidget(grp_gen)

        # ── RTA Display Settings ──────────────────────────────────────
        grp_rta = QGroupBox('RTA Display Settings')
        gr = QGridLayout(grp_rta)
        gr.setContentsMargins(10, 14, 10, 8); gr.setSpacing(7)
        gr.setColumnStretch(1, 1); gr.setColumnStretch(3, 1)

        try:    _yl_sp = self.canvas_spec.ax.get_ylim()
        except: _yl_sp = (-100, 0)

        cmb_bdata     = _cmb(['Bars', 'Lines', 'Both'], 0)
        chk_track_pk  = _chk('', False)
        chk_thd       = _chk('', False, enabled=False)
        chk_cal_lvl   = _chk('', False)
        spn_mag_max   = _spn(-60, 20,   int(_yl_sp[1]))
        spn_mag_min   = _spn(-160, -20, int(_yl_sp[0]))
        cmb_thick_fg  = _cmb(['1','2','3','4','5'], 1)
        cmb_thick_bg  = _cmb(['1','2','3','4','5'], 0)
        chk_peak_hold = _chk('', getattr(self, '_peak_hold', False))
        cmb_peak_type = _cmb(['Infinite', 'Decay'], 0)
        spn_hold_s    = _spn(0, 60,  1)
        chk_averaged  = _chk('', True)
        spn_y_zoom    = _spn(1, 30,  3)
        spn_y_scroll  = _spn(1, 30,  3)
        spn_y_grid    = _spn(3, 48, 12)

        r = 0
        gr.addWidget(_lbl('Banded Data:'),           r, 0, Qt.AlignmentFlag.AlignRight); gr.addWidget(cmb_bdata,    r, 1)
        gr.addWidget(_lbl('Peak Hold:'),              r, 2, Qt.AlignmentFlag.AlignRight); gr.addWidget(chk_peak_hold,r, 3); r+=1

        gr.addWidget(_lbl('Track Peak:'),             r, 0, Qt.AlignmentFlag.AlignRight); gr.addWidget(chk_track_pk, r, 1)
        gr.addWidget(_lbl('Peak Type:'),              r, 2, Qt.AlignmentFlag.AlignRight); gr.addWidget(cmb_peak_type,r, 3); r+=1

        gr.addWidget(_lbl('Show THD:'),               r, 0, Qt.AlignmentFlag.AlignRight); gr.addWidget(chk_thd,      r, 1)
        _h_row = QHBoxLayout()
        _h_row.addWidget(spn_hold_s); _h_row.addWidget(_lbl('Seconds'))
        gr.addWidget(_lbl('Hold:'),                   r, 2, Qt.AlignmentFlag.AlignRight); gr.addLayout(_h_row,       r, 3); r+=1

        gr.addWidget(_lbl('Plot Calibrated Level:'),  r, 0, Qt.AlignmentFlag.AlignRight); gr.addWidget(chk_cal_lvl,  r, 1)
        gr.addWidget(_lbl('Averaged:'),               r, 2, Qt.AlignmentFlag.AlignRight); gr.addWidget(chk_averaged, r, 3); r+=1

        _mag_box = QHBoxLayout()
        _mag_box.addWidget(spn_mag_max); _mag_box.addWidget(_lbl('Max'))
        gr.addWidget(_lbl('Magnitude Range (dB):'),   r, 0, Qt.AlignmentFlag.AlignRight); gr.addLayout(_mag_box,     r, 1)
        gr.addWidget(_lbl('Y-Zoom Increment (dB):'),  r, 2, Qt.AlignmentFlag.AlignRight); gr.addWidget(spn_y_zoom,   r, 3); r+=1

        _mag_min_box = QHBoxLayout()
        _mag_min_box.addWidget(spn_mag_min); _mag_min_box.addWidget(_lbl('Min'))
        gr.addLayout(_mag_min_box,                    r, 1)
        gr.addWidget(_lbl('Y-Scroll Increment (dB):'),r, 2, Qt.AlignmentFlag.AlignRight); gr.addWidget(spn_y_scroll, r, 3); r+=1

        _thick_fg_box = QHBoxLayout()
        _thick_fg_box.addWidget(cmb_thick_fg); _thick_fg_box.addWidget(_lbl('Foreground'))
        gr.addWidget(_lbl('Line Thickness:'),         r, 0, Qt.AlignmentFlag.AlignRight); gr.addLayout(_thick_fg_box,r, 1)
        gr.addWidget(_lbl('Y-Grid Interval (dB):'),   r, 2, Qt.AlignmentFlag.AlignRight); gr.addWidget(spn_y_grid,   r, 3); r+=1

        _thick_bg_box = QHBoxLayout()
        _thick_bg_box.addWidget(cmb_thick_bg); _thick_bg_box.addWidget(_lbl('Background'))
        gr.addLayout(_thick_bg_box,                   r, 1); r+=1

        sp_lay.addWidget(grp_rta)

        # ── Spectrograph Settings ──────────────────────────────────────
        grp_sgr = QGroupBox('Spectrograph Settings')
        gsg = QGridLayout(grp_sgr)
        gsg.setContentsMargins(10, 14, 10, 8); gsg.setSpacing(7)
        gsg.setColumnStretch(1, 1); gsg.setColumnStretch(3, 1)

        _sg_vmin = getattr(self.canvas_sgram, '_vmin', SpectrogramCanvas.VMIN_DEFAULT)
        _sg_vmax = getattr(self.canvas_sgram, '_vmax', SpectrogramCanvas.VMAX_DEFAULT)
        _sg_n    = SpectrogramCanvas.N_TIME

        spn_slice_h   = _spn(1, 20,   4)
        spn_slices    = _spn(100, 5000, _sg_n)
        spn_dr_max    = _spn(-60, 0,   int(_sg_vmax))
        spn_dr_min    = _spn(-160, -20, int(_sg_vmin))
        chk_gray      = _chk('Grayscale', False)

        # Cálculo dinámico de memoria
        def _mem_str():
            n_sl   = spn_slices.value()
            n_freq = 512   # bins típicos
            mb = (n_sl * n_freq * 4 * 2) / (1024 * 1024)
            return f'{mb:.0f} Mb'

        lbl_mem = _lbl(_mem_str())
        spn_slices.valueChanged.connect(lambda _: lbl_mem.setText(_mem_str()))

        r = 0
        gsg.addWidget(_lbl('Slice Height:'),           r, 0, Qt.AlignmentFlag.AlignRight); gsg.addWidget(spn_slice_h, r, 1)
        _dr_max_box = QHBoxLayout()
        _dr_max_box.addWidget(spn_dr_max); _dr_max_box.addWidget(_lbl('Max'))
        gsg.addWidget(_lbl('Dynamic Range (dB FS):'),  r, 2, Qt.AlignmentFlag.AlignRight); gsg.addLayout(_dr_max_box, r, 3); r+=1

        gsg.addWidget(_lbl('Slices in History:'),      r, 0, Qt.AlignmentFlag.AlignRight); gsg.addWidget(spn_slices,  r, 1)
        _dr_min_box = QHBoxLayout()
        _dr_min_box.addWidget(spn_dr_min); _dr_min_box.addWidget(_lbl('Min:'))
        gsg.addLayout(_dr_min_box,                     r, 3); r+=1

        gsg.addWidget(_lbl('Min Memory Required:'),    r, 0, Qt.AlignmentFlag.AlignRight); gsg.addWidget(lbl_mem,     r, 1)
        gsg.addWidget(chk_gray,                        r, 2, 1, 2); r+=1

        sp_lay.addWidget(grp_sgr)
        sp_lay.addStretch()

        # ══════════════════════════════════════════════════════════════
        # TAB 1 — TRANSFER FUNCTION  (estilo SMAART exacto)
        # ══════════════════════════════════════════════════════════════
        tab_tf, tf_lay = _scroll_tab()
        tabs.addTab(tab_tf, 'Transfer Function')

        # Estado actual de flags
        _yl_tf  = self.canvas_meas.ax_tf.get_ylim()
        _yp_tf  = self.canvas_meas.ax_ph.get_ylim()
        _xl_tf  = self.canvas_meas.ax_tf.get_xlim()
        _cur_bpo_tf = getattr(self, '_tf_smooth_bpo', 3)
        _bpo_tf_idx = next((i for i, v in enumerate(_sp_bpo_map.values())
                            if v == _cur_bpo_tf), 2)

        # ── General Settings ──────────────────────────────────────────
        grp_tf_gen = QGroupBox('General Settings')
        g_gen = QGridLayout(grp_tf_gen)
        g_gen.setContentsMargins(10, 14, 10, 8); g_gen.setSpacing(7)
        g_gen.setColumnStretch(1, 1); g_gen.setColumnStretch(3, 1)

        cmb_tf_fscale = _cmb(list(_sp_bpo_map.keys()), _bpo_tf_idx)
        chk_tf_ir     = _chk('Display Live IR',              getattr(self, '_ir_visible', True))
        cmb_tf_fg     = _cmb(['1','2','3','4','5'],          getattr(self, '_tf_line_thick_fg', 2) - 1)
        cmb_tf_bg     = _cmb(['1','2','3','4','5'],          getattr(self, '_tf_line_thick_bg', 2) - 1)
        chk_lk_smooth = _chk('Link Global Plot Smoothing',   getattr(self, '_link_smooth', True))
        chk_lk_phm    = _chk('Link Phase & Mag. Smoothing',  getattr(self, '_link_ph_mag_smooth', True))
        chk_lk_zoom   = _chk('Link Plot Zoom',               getattr(self, '_link_plot_zoom', False))
        chk_prop      = _chk('Plot Proportional Graphs',     getattr(self, '_plot_proportional', False))
        chk_instant   = _chk('Plot Instantaneous Response',  getattr(self, '_instantaneous', False))

        # Live IR toggle inmediato
        chk_tf_ir.toggled.connect(
            lambda c: self._toggle_ir_panel() if c != getattr(self, '_ir_visible', True) else None)

        r = 0
        g_gen.addWidget(_lbl('Frequency Scale:'),         r, 0, Qt.AlignmentFlag.AlignRight)
        g_gen.addWidget(cmb_tf_fscale,                    r, 1)
        g_gen.addWidget(chk_lk_smooth,                    r, 2, 1, 2); r += 1

        g_gen.addWidget(chk_tf_ir,                        r, 0, 1, 2)
        g_gen.addWidget(chk_lk_phm,                       r, 2, 1, 2); r += 1

        _tf_fg_row = QHBoxLayout()
        _tf_fg_row.addWidget(cmb_tf_fg); _tf_fg_row.addWidget(_lbl('Foreground'))
        g_gen.addWidget(_lbl('Line Thickness:'),           r, 0, Qt.AlignmentFlag.AlignRight)
        g_gen.addLayout(_tf_fg_row,                        r, 1)
        g_gen.addWidget(chk_lk_zoom,                       r, 2, 1, 2); r += 1

        _tf_bg_row = QHBoxLayout()
        _tf_bg_row.addWidget(cmb_tf_bg); _tf_bg_row.addWidget(_lbl('Background'))
        g_gen.addLayout(_tf_bg_row,                        r, 1)
        g_gen.addWidget(chk_prop,                          r, 2, 1, 2); r += 1

        g_gen.addWidget(chk_instant,                       r, 2, 1, 2); r += 1
        tf_lay.addWidget(grp_tf_gen)

        # ── Magnitude ─────────────────────────────────────────────────
        grp_tf_mag = QGroupBox('Magnitude')
        g_mag = QGridLayout(grp_tf_mag)
        g_mag.setContentsMargins(10, 14, 10, 8); g_mag.setSpacing(7)
        g_mag.setColumnStretch(1, 1); g_mag.setColumnStretch(3, 1)

        spn_tf_mhi    = _spn(-30,  60, int(_yl_tf[1]))
        spn_tf_mlo    = _spn(-120,  0, int(_yl_tf[0]))
        chk_tf_peak   = _chk('', getattr(self, '_peak_hold', False))
        cmb_tf_avgmode= _cmb(['Polar','Log-Magnitude','Power'],
                              {'Polar':0,'Log-Magnitude':1,'Power':2}.get(
                                  getattr(self,'_tf_mag_avg_mode','Polar'), 0))
        cmb_tf_weight = _cmb(['None','A-weight','C-weight','Z-weight'], 0)
        chk_tf_disp_w = _chk('Display Weighting', False)
        spn_tf_ygrid  = _spn(1, 48, getattr(self, '_tf_y_grid', 3))
        spn_tf_yscroll= _spn(1, 30, getattr(self, '_tf_y_scroll', 3))
        spn_tf_yzoom  = _spn(1, 30, getattr(self, '_tf_y_zoom', 3))

        r = 0
        _mag_max_row = QHBoxLayout()
        _mag_max_row.addWidget(spn_tf_mhi); _mag_max_row.addWidget(_lbl('Max'))
        _mag_min_row = QHBoxLayout()
        _mag_min_row.addWidget(spn_tf_mlo); _mag_min_row.addWidget(_lbl('Min'))

        g_mag.addWidget(_lbl('Mag Range (dB):'),           r, 0, Qt.AlignmentFlag.AlignRight)
        g_mag.addLayout(_mag_max_row,                       r, 1)
        g_mag.addWidget(_lbl('Y-Grid Interval (dB):'),     r, 2, Qt.AlignmentFlag.AlignRight)
        g_mag.addWidget(spn_tf_ygrid,                       r, 3); r += 1

        g_mag.addLayout(_mag_min_row,                       r, 1)
        g_mag.addWidget(_lbl('Y-Scroll Increment (dB):'),  r, 2, Qt.AlignmentFlag.AlignRight)
        g_mag.addWidget(spn_tf_yscroll,                     r, 3); r += 1

        g_mag.addWidget(_lbl('Track Peak:'),                r, 0, Qt.AlignmentFlag.AlignRight)
        g_mag.addWidget(chk_tf_peak,                        r, 1)
        g_mag.addWidget(_lbl('Y-Zoom Increment (dB):'),    r, 2, Qt.AlignmentFlag.AlignRight)
        g_mag.addWidget(spn_tf_yzoom,                       r, 3); r += 1

        g_mag.addWidget(_lbl('Mag Averaging Display:'),    r, 0, Qt.AlignmentFlag.AlignRight)
        g_mag.addWidget(cmb_tf_avgmode,                     r, 1); r += 1

        g_mag.addWidget(_lbl('Weighting:'),                r, 0, Qt.AlignmentFlag.AlignRight)
        g_mag.addWidget(cmb_tf_weight,                      r, 1); r += 1

        g_mag.addWidget(chk_tf_disp_w,                     r, 0, 1, 2); r += 1
        tf_lay.addWidget(grp_tf_mag)

        # ── Phase ─────────────────────────────────────────────────────
        grp_tf_ph = QGroupBox('Phase')
        g_ph = QGridLayout(grp_tf_ph)
        g_ph.setContentsMargins(10, 14, 10, 8); g_ph.setSpacing(7)
        g_ph.setColumnStretch(1, 1); g_ph.setColumnStretch(3, 1)

        chk_unwrap  = _chk('Unwrap Phase',       getattr(self, '_unwrap_phase', False))
        chk_gd      = _chk('Phase As Group Delay', getattr(self, '_phase_as_gd', False))
        spn_ph_max  = _spn(0,    720, int(_yp_tf[1]))
        spn_ph_min  = _spn(-720,   0, int(_yp_tf[0]))

        # Unwrap y Group Delay se aplican en tiempo real vía flags
        chk_unwrap.toggled.connect(lambda c: setattr(self, '_unwrap_phase', c))
        chk_gd.toggled.connect(    lambda c: setattr(self, '_phase_as_gd',  c))

        r = 0
        g_ph.addWidget(chk_unwrap,  r, 0); g_ph.addWidget(chk_gd, r, 2, 1, 2); r += 1

        _ph_max_row = QHBoxLayout()
        _ph_max_row.addWidget(spn_ph_max); _ph_max_row.addWidget(_lbl('Max'))
        _ph_min_row = QHBoxLayout()
        _ph_min_row.addWidget(spn_ph_min); _ph_min_row.addWidget(_lbl('Min'))

        g_ph.addWidget(_lbl('Range:'),    r, 0, Qt.AlignmentFlag.AlignRight)
        g_ph.addLayout(_ph_max_row,        r, 1)
        g_ph.addLayout(_ph_min_row,        r, 3); r += 1
        tf_lay.addWidget(grp_tf_ph)

        # ── Coherence ─────────────────────────────────────────────────
        grp_tf_coh = QGroupBox('Coherence')
        g_coh = QGridLayout(grp_tf_coh)
        g_coh.setContentsMargins(10, 14, 10, 8); g_coh.setSpacing(7)
        g_coh.setColumnStretch(1, 1); g_coh.setColumnStretch(3, 1)

        _coh_blank_pct = int(getattr(self, '_coh_blanking_pct',
                             getattr(self, 'spn_thresh', None) and
                             int(self.spn_thresh.value() * 100) or 10))
        chk_show_coh   = _chk('Show Coherence',               self.canvas_meas.ax_coh.get_visible())
        chk_coh_qh     = _chk('1/4 Height',                   getattr(self, '_coh_quarter_height', False))
        chk_coh_sq     = _chk('Squared Coh',                  getattr(self, '_coh_squared', True))
        spn_coh_blank  = _spn(0, 100, _coh_blank_pct)
        spn_coh_blank.setSuffix('%')
        chk_mtw        = _chk('Always Display MTW Coherence',  getattr(self, '_mtw_coh', True))
        chk_coh_fol_fg = _chk('Line Thickness Follows Foreground',
                               getattr(self, '_coh_thick_follows_fg', True))
        spn_coh_thick  = _spn(1, 8, getattr(self, '_coh_line_thick', 4))

        # Line Thickness se desactiva si "Follows Foreground" está activado
        spn_coh_thick.setEnabled(not chk_coh_fol_fg.isChecked())
        chk_coh_fol_fg.toggled.connect(lambda c: spn_coh_thick.setEnabled(not c))

        # Show Coherence → toggle inmediato
        chk_show_coh.toggled.connect(lambda c: (
            self.canvas_meas.line_coh.set_visible(c),
            self.canvas_meas.ax_coh.set_visible(c),
            self.canvas_meas.draw_idle()))

        r = 0
        g_coh.addWidget(chk_show_coh,    r, 0, 1, 2)
        g_coh.addWidget(_lbl('Coh Blanking Threshold (%):'), r, 2, Qt.AlignmentFlag.AlignRight)
        g_coh.addWidget(spn_coh_blank,   r, 3); r += 1

        g_coh.addWidget(chk_coh_qh,      r, 0, 1, 2)
        g_coh.addWidget(chk_mtw,          r, 2, 1, 2); r += 1

        g_coh.addWidget(chk_coh_sq,      r, 0, 1, 2)
        g_coh.addWidget(chk_coh_fol_fg,  r, 2, 1, 2); r += 1

        g_coh.addWidget(_lbl('Line Thickness:'), r, 2, Qt.AlignmentFlag.AlignRight)
        g_coh.addWidget(spn_coh_thick,   r, 3); r += 1
        tf_lay.addWidget(grp_tf_coh)
        tf_lay.addStretch()

        # ══════════════════════════════════════════════════════════════
        # TAB 2 — IMPULSE RESPONSE  (estilo SMAART exacto)
        # ══════════════════════════════════════════════════════════════
        tab_ir, ir_lay = _scroll_tab()
        tabs.addTab(tab_ir, 'Impulse Response')

        # ── General Settings ──────────────────────────────────────────
        grp_ir_gen = QGroupBox('General Settings')
        g_ir_gen = QGridLayout(grp_ir_gen)
        g_ir_gen.setContentsMargins(10, 14, 10, 8); g_ir_gen.setSpacing(7)
        g_ir_gen.setColumnStretch(1, 1); g_ir_gen.setColumnStretch(3, 1)

        chk_ir_peak  = _chk('Show IR Peak',    getattr(self, '_ir_show_peak',    True))
        chk_ir_optim = _chk('Optimize Graphing', getattr(self, '_ir_optimize',   True))
        spn_ir_over  = _spn(0, 95, getattr(self, '_ir_overlap_pct', 50))
        spn_ir_magth = _spn(-120, 0, getattr(self, '_ir_mag_thresh_db', 0))
        cmb_ir_fg    = _cmb(['1','2','3','4','5'],
                            getattr(self, '_ir_line_thick_fg', 4) - 1)
        cmb_ir_bg    = _cmb(['1','2','3','4','5'],
                            getattr(self, '_ir_line_thick_bg', 2) - 1)

        # Show IR Peak → toggle marcador de pico inmediatamente
        chk_ir_peak.toggled.connect(
            lambda c: (setattr(self, '_ir_show_peak', c),
                       self.canvas_meas.line_ir_peak.set_visible(c),
                       self.canvas_meas.draw_idle()))

        r = 0
        g_ir_gen.addWidget(chk_ir_peak,              r, 0)
        g_ir_gen.addWidget(_lbl('Overlap (%):'),     r, 2, Qt.AlignmentFlag.AlignRight)
        g_ir_gen.addWidget(spn_ir_over,               r, 3); r += 1

        g_ir_gen.addWidget(chk_ir_optim,             r, 0)
        g_ir_gen.addWidget(_lbl('Mag Threshold (dB FS):'), r, 2, Qt.AlignmentFlag.AlignRight)
        g_ir_gen.addWidget(spn_ir_magth,              r, 3); r += 1

        _ir_fg_row = QHBoxLayout()
        _ir_fg_row.addWidget(cmb_ir_fg); _ir_fg_row.addWidget(_lbl('Foreground'))
        g_ir_gen.addWidget(_lbl('Line Thickness:'),  r, 0, Qt.AlignmentFlag.AlignRight)
        g_ir_gen.addLayout(_ir_fg_row,               r, 1); r += 1

        _ir_bg_row = QHBoxLayout()
        _ir_bg_row.addWidget(cmb_ir_bg); _ir_bg_row.addWidget(_lbl('Background'))
        g_ir_gen.addLayout(_ir_bg_row,               r, 1); r += 1
        ir_lay.addWidget(grp_ir_gen)

        # ── Broadband Filter Settings ─────────────────────────────────
        grp_ir_filt = QGroupBox('Broadband Filter Settings')
        g_ir_filt = QGridLayout(grp_ir_filt)
        g_ir_filt.setContentsMargins(10, 14, 10, 8); g_ir_filt.setSpacing(7)
        g_ir_filt.setColumnStretch(1, 0); g_ir_filt.setColumnStretch(2, 1)
        g_ir_filt.setColumnStretch(3, 0); g_ir_filt.setColumnStretch(4, 1)

        chk_hpf  = _chk('High Pass Filter (Hz):', getattr(self, '_ir_hpf_en',  True))
        spn_hpf  = _spn(20, 2000, getattr(self, '_ir_hpf_hz', 100))
        chk_lpf  = _chk('Low Pass Filter (Hz):', getattr(self, '_ir_lpf_en',  True))
        spn_lpf  = _spn(1000, 48000, getattr(self, '_ir_lpf_hz', 24000))

        # Habilitar/deshabilitar spinbox según checkbox
        spn_hpf.setEnabled(chk_hpf.isChecked())
        spn_lpf.setEnabled(chk_lpf.isChecked())
        chk_hpf.toggled.connect(spn_hpf.setEnabled)
        chk_lpf.toggled.connect(spn_lpf.setEnabled)

        g_ir_filt.addWidget(chk_hpf, 0, 0); g_ir_filt.addWidget(spn_hpf, 0, 1)
        g_ir_filt.addWidget(chk_lpf, 0, 2); g_ir_filt.addWidget(spn_lpf, 0, 3)
        ir_lay.addWidget(grp_ir_filt)

        # ── Frequency Plot Settings ───────────────────────────────────
        grp_ir_freq = QGroupBox('Frequency Plot Settings')
        g_ir_freq = QGridLayout(grp_ir_freq)
        g_ir_freq.setContentsMargins(10, 14, 10, 8); g_ir_freq.setSpacing(7)
        g_ir_freq.setColumnStretch(1, 1); g_ir_freq.setColumnStretch(3, 1)

        _ir_ylim = self.canvas_meas.ax_ir.get_ylim()
        spn_ir_mhi = _spn(-60, 30,  max(6,  int(_ir_ylim[1])))
        spn_ir_mlo = _spn(-120, 0, min(-36, int(_ir_ylim[0])))

        # Trace Controls — radio buttons
        from PyQt6.QtWidgets import QRadioButton, QButtonGroup
        _ir_tc_grp  = QButtonGroup()
        rad_smooth  = QRadioButton('Smoothing')
        rad_banding = QRadioButton('Banding')
        rad_smooth.setStyleSheet(f'color:{TEXT_MID};font-size:11px;')
        rad_banding.setStyleSheet(f'color:{TEXT_MID};font-size:11px;')
        _ir_tc_grp.addButton(rad_smooth, 0)
        _ir_tc_grp.addButton(rad_banding, 1)
        _tc_cur = getattr(self, '_ir_trace_ctrl', 0)
        (rad_smooth if _tc_cur == 0 else rad_banding).setChecked(True)

        _tc_widget = QWidget()
        _tc_lay = QVBoxLayout(_tc_widget)
        _tc_lay.setContentsMargins(0,0,0,0); _tc_lay.setSpacing(2)
        _tc_lay.addWidget(rad_smooth)
        _tc_lay.addWidget(rad_banding)

        _bpo_tf_ir_idx = next((i for i, k in enumerate(_sp_bpo_map.keys())
                               if k == getattr(self, '_ir_freq_scale', '1/3 Octave')), 2)
        cmb_ir_fscale = _cmb(list(_sp_bpo_map.keys()), _bpo_tf_ir_idx)

        _mag_ir_max_row = QHBoxLayout()
        _mag_ir_max_row.addWidget(spn_ir_mhi); _mag_ir_max_row.addWidget(_lbl('Max'))
        _mag_ir_min_row = QHBoxLayout()
        _mag_ir_min_row.addWidget(spn_ir_mlo); _mag_ir_min_row.addWidget(_lbl('Min'))

        r = 0
        g_ir_freq.addWidget(_lbl('Magnitude Range (dB):'), r, 0, Qt.AlignmentFlag.AlignRight)
        g_ir_freq.addLayout(_mag_ir_max_row,                r, 1)
        g_ir_freq.addWidget(_lbl('Trace Controls:'),        r, 2, Qt.AlignmentFlag.AlignRight)
        g_ir_freq.addWidget(_tc_widget,                     r, 3, 2, 1); r += 1
        g_ir_freq.addLayout(_mag_ir_min_row,                r, 1); r += 1

        g_ir_freq.addWidget(_lbl('Frequency Scale:'),      r, 2, Qt.AlignmentFlag.AlignRight)
        g_ir_freq.addWidget(cmb_ir_fscale,                  r, 3); r += 1
        ir_lay.addWidget(grp_ir_freq)

        # ── Spectrograph Settings ─────────────────────────────────────
        grp_ir_sgram = QGroupBox('Spectrograph Settings')
        g_ir_sgram = QGridLayout(grp_ir_sgram)
        g_ir_sgram.setContentsMargins(10, 14, 10, 8); g_ir_sgram.setSpacing(7)
        g_ir_sgram.setColumnStretch(1, 1); g_ir_sgram.setColumnStretch(3, 1)

        _sg_is_gray = (hasattr(self, 'canvas_sgram') and
                       self.canvas_sgram._mesh is not None and
                       getattr(self.canvas_sgram._mesh, 'cmap', None) is not None and
                       self.canvas_sgram._mesh.cmap.name == 'gray_r')
        chk_ir_gray = _chk('Grayscale', _sg_is_gray)
        _win_opts   = ['Hann', 'Hamming', 'Blackman', 'Flat Top', 'Rectangular']
        _win_cur    = getattr(self, '_fft_window_type', 'Hann')
        cmb_ir_win  = _cmb(_win_opts,
                            _win_opts.index(_win_cur) if _win_cur in _win_opts else 0)

        # Grayscale → toggle inmediato en el spectrogram
        def _toggle_gray_ir(c):
            if hasattr(self, 'canvas_sgram') and self.canvas_sgram._mesh is not None:
                self.canvas_sgram._mesh.set_cmap('gray_r' if c else _SGRAM_CMAP)
                self.canvas_sgram._bg = None
                self.canvas_sgram.draw_idle()
        chk_ir_gray.toggled.connect(_toggle_gray_ir)

        g_ir_sgram.addWidget(chk_ir_gray,          0, 0, 1, 2)
        g_ir_sgram.addWidget(_lbl('Data Window:'), 0, 2, Qt.AlignmentFlag.AlignRight)
        g_ir_sgram.addWidget(cmb_ir_win,            0, 3)
        ir_lay.addWidget(grp_ir_sgram)

        # ── Histogram Settings ────────────────────────────────────────
        grp_ir_hist = QGroupBox('Histogram Settings')
        g_ir_hist = QGridLayout(grp_ir_hist)
        g_ir_hist.setContentsMargins(10, 14, 10, 8); g_ir_hist.setSpacing(7)

        _hist_grp  = QButtonGroup()
        rad_bars   = QRadioButton('Bars')
        rad_lines  = QRadioButton('Lines')
        rad_bars.setStyleSheet(f'color:{TEXT_MID};font-size:11px;')
        rad_lines.setStyleSheet(f'color:{TEXT_MID};font-size:11px;')
        _hist_grp.addButton(rad_bars, 0)
        _hist_grp.addButton(rad_lines, 1)
        (rad_bars if getattr(self, '_hist_mode', 0) == 0 else rad_lines).setChecked(True)

        g_ir_hist.addWidget(_lbl('Plot as:'), 0, 0, Qt.AlignmentFlag.AlignRight)
        g_ir_hist.addWidget(rad_bars,          0, 1)
        g_ir_hist.addWidget(rad_lines,         0, 2)
        ir_lay.addWidget(grp_ir_hist)
        ir_lay.addStretch()

        # ══════════════════════════════════════════════════════════════
        # BOTTOM BUTTONS
        # ══════════════════════════════════════════════════════════════
        btn_row = QHBoxLayout()
        btn_meas = QPushButton('Measurement Settings')
        btn_meas.setStyleSheet(
            f'QPushButton{{background:#1a1a1a;color:{TEXT_MID};padding:5px 12px;'
            f'border:1px solid #444;border-radius:3px;font-size:10px;}}'
            f'QPushButton:hover{{color:{TEXT_HI};background:#222;}}')
        btn_meas.clicked.connect(self._show_measurement_config)
        btn_ok = QPushButton('OK')
        btn_ok.setDefault(True)
        btn_ok.setStyleSheet(
            f'QPushButton{{background:#1a3a1a;color:{TEXT_HI};padding:5px 20px;'
            f'border:1px solid #2a5a2a;border-radius:3px;font-size:10px;}}'
            f'QPushButton:hover{{background:#1e4a1e;}}')
        btn_ok.clicked.connect(dlg.accept)
        btn_row.addWidget(btn_meas)
        btn_row.addStretch()
        btn_row.addWidget(btn_ok)
        root.addLayout(btn_row)

        # ── Apply on OK ────────────────────────────────────────────────
        def _on_ok():
            import matplotlib.ticker as _mtick

            # ── SPECTRUM tab ───────────────────────────────────────────
            try:
                mlo, mhi = spn_mag_min.value(), spn_mag_max.value()
                if mlo < mhi:
                    self.canvas_spec.ax.set_ylim(mlo, mhi)
                    self.canvas_spec.draw_idle()
            except Exception: pass

            try:
                dr_lo, dr_hi = spn_dr_min.value(), spn_dr_max.value()
                if dr_lo < dr_hi:
                    self.canvas_sgram.set_color_range(dr_lo, dr_hi)
                    if hasattr(self, '_spn_sg_floor'):
                        self._spn_sg_floor.setValue(dr_lo)
                        self._spn_sg_ceil.setValue(dr_hi)
            except Exception: pass

            try:
                if chk_gray.isChecked() and self.canvas_sgram._mesh is not None:
                    self.canvas_sgram._mesh.set_cmap('gray_r')
                    self.canvas_sgram._bg = None
                    self.canvas_sgram.draw_idle()
                elif not chk_gray.isChecked() and self.canvas_sgram._mesh is not None:
                    self.canvas_sgram._mesh.set_cmap(_SGRAM_CMAP)
                    self.canvas_sgram._bg = None
                    self.canvas_sgram.draw_idle()
            except Exception: pass

            # ── TRANSFER FUNCTION tab ──────────────────────────────────

            # General — flags de estado
            _bpo_keys_list = list(_sp_bpo_map.keys())
            _bpo_vals_list = list(_sp_bpo_map.values())
            _new_bpo = _bpo_vals_list[cmb_tf_fscale.currentIndex()]
            self._tf_smooth_bpo      = _new_bpo
            self._link_smooth        = chk_lk_smooth.isChecked()
            self._link_ph_mag_smooth = chk_lk_phm.isChecked()
            self._link_plot_zoom     = chk_lk_zoom.isChecked()
            self._plot_proportional  = chk_prop.isChecked()
            self._instantaneous      = chk_instant.isChecked()

            # Line thickness — foreground
            _fg_thick = int(cmb_tf_fg.currentText())
            _bg_thick = int(cmb_tf_bg.currentText())
            self._tf_line_thick_fg = _fg_thick
            self._tf_line_thick_bg = _bg_thick
            try:
                self.canvas_meas.line_tf.set_linewidth(_fg_thick)
                self.canvas_meas.line_tf2.set_linewidth(max(1, _fg_thick - 1))
                self.canvas_meas.line_ph.set_linewidth(_fg_thick)
                self.canvas_meas.line_ph2.set_linewidth(max(1, _fg_thick - 1))
                self.canvas_meas.line_tf_avg.set_linewidth(_bg_thick)
                for tl in getattr(self.canvas_meas, '_trace_tf_lines', []):
                    tl.set_linewidth(_bg_thick)
            except Exception: pass

            # Mag averaging mode
            self._tf_mag_avg_mode = cmb_tf_avgmode.currentText()

            # Smoothing — sync con cmb_smooth si link está activo
            if self._link_smooth and _new_bpo > 0:
                try: self._set_smooth_fraction(_new_bpo)
                except Exception: pass

            # Magnitude range
            try:
                mlo_tf, mhi_tf = spn_tf_mlo.value(), spn_tf_mhi.value()
                if mlo_tf < mhi_tf:
                    self.canvas_meas.ax_tf.set_ylim(mlo_tf, mhi_tf)
                    self.canvas_meas.ax_tf.set_autoscale_on(False)
            except Exception: pass

            # Y-Grid Interval
            try:
                _grid_v = spn_tf_ygrid.value()
                self._tf_y_grid = _grid_v
                self.canvas_meas.ax_tf.yaxis.set_major_locator(
                    _mtick.MultipleLocator(_grid_v))
            except Exception: pass

            # Y-Scroll / Y-Zoom
            self._tf_y_scroll = spn_tf_yscroll.value()
            self._tf_y_zoom   = spn_tf_yzoom.value()

            # Track Peak
            ph = chk_tf_peak.isChecked() or chk_peak_hold.isChecked()
            setattr(self, '_peak_hold', ph)
            self._toggle_peak_hold(ph)

            # Phase range
            try:
                plo_tf, phi_tf = spn_ph_min.value(), spn_ph_max.value()
                if plo_tf < phi_tf:
                    self.canvas_meas.ax_ph.set_ylim(plo_tf, phi_tf)
            except Exception: pass

            # Unwrap / Group Delay — ya aplicados via toggled.connect
            # Coherence blanking threshold
            try:
                _blank_pct = spn_coh_blank.value()
                self._coh_blanking_pct = _blank_pct
                if hasattr(self, 'spn_thresh'):
                    self.spn_thresh.setValue(_blank_pct / 100.0)
            except Exception: pass

            # Coherence flags
            self._coh_quarter_height = chk_coh_qh.isChecked()
            self._coh_squared        = chk_coh_sq.isChecked()
            self._mtw_coh            = chk_mtw.isChecked()
            self._coh_thick_follows_fg = chk_coh_fol_fg.isChecked()
            _coh_thick = _fg_thick if self._coh_thick_follows_fg else spn_coh_thick.value()
            self._coh_line_thick     = _coh_thick
            try:
                self.canvas_meas.line_coh.set_linewidth(_coh_thick)
            except Exception: pass

            # ── IMPULSE RESPONSE tab ───────────────────────────────────

            # General — flags
            self._ir_show_peak    = chk_ir_peak.isChecked()
            self._ir_optimize     = chk_ir_optim.isChecked()
            self._ir_overlap_pct  = spn_ir_over.value()
            self._ir_mag_thresh_db = spn_ir_magth.value()
            _ir_fg = int(cmb_ir_fg.currentText())
            _ir_bg = int(cmb_ir_bg.currentText())
            self._ir_line_thick_fg = _ir_fg
            self._ir_line_thick_bg = _ir_bg
            try:
                self.canvas_meas.line_ir.set_linewidth(_ir_fg)
                for tl in getattr(self.canvas_meas, '_trace_ir_lines', []):
                    tl.set_linewidth(_ir_bg)
            except Exception: pass

            # Broadband filter
            self._ir_hpf_en  = chk_hpf.isChecked()
            self._ir_hpf_hz  = spn_hpf.value()
            self._ir_lpf_en  = chk_lpf.isChecked()
            self._ir_lpf_hz  = spn_lpf.value()

            # Freq Plot — magnitude range de IR
            try:
                ir_mhi, ir_mlo = spn_ir_mhi.value(), spn_ir_mlo.value()
                if ir_mlo < ir_mhi:
                    self.canvas_meas.ax_ir.set_ylim(ir_mlo, ir_mhi)
                    self.canvas_meas.ax_ir.set_autoscale_on(False)
            except Exception: pass
            self._ir_trace_ctrl  = _ir_tc_grp.checkedId()
            self._ir_freq_scale  = cmb_ir_fscale.currentText()

            # Spectrograph — grayscale ya se aplica en tiempo real (toggled.connect)
            # Data Window → re-inicia el engine si cambió
            _new_win = cmb_ir_win.currentText()
            if _new_win != getattr(self, '_fft_window_type', 'Hann'):
                self._fft_window_type = _new_win
                # Re-start measurement engine con nueva ventana
                try: self._restart_engine_window(_new_win)
                except Exception: pass

            # Histogram
            self._hist_mode = _hist_grp.checkedId()

            # Redibuja
            try: self.canvas_meas.draw_idle()
            except Exception: pass

            # ── Persistir inmediatamente ───────────────────────────────
            try: self._save_prefs()
            except Exception: pass

        btn_ok.clicked.connect(_on_ok)

        tabs.setCurrentIndex(start_tab)
        dlg.exec()

    def _show_freq_range_dialog(self):
        """Ajustar rango de frecuencias de los gráficos."""
        from PyQt6.QtWidgets import (QDialog, QFormLayout, QVBoxLayout,
                                     QDoubleSpinBox, QDialogButtonBox, QLabel,
                                     QHBoxLayout)
        dlg = QDialog(self)
        dlg.setWindowTitle('Frequency Range')
        dlg.setStyleSheet(
            f'QDialog{{background:{BG_APP};color:{TEXT_HI};font-size:11px;}}'
            f'QLabel{{color:{TEXT_MID};}}'
            f'QDoubleSpinBox{{background:#252525;color:{TEXT_HI};'
            f'border:1px solid #444;border-radius:3px;padding:2px 6px;}}'
            f'QPushButton{{background:#2a2a2a;color:{TEXT_HI};padding:4px 14px;'
            f'border:1px solid #444;border-radius:3px;}}'
            f'QPushButton:hover{{background:#333;}}')
        lay = QVBoxLayout(dlg)
        form = QFormLayout(); form.setSpacing(8)

        cur_xlim = self.canvas_meas.ax_tf.get_xlim()
        spn_lo = QDoubleSpinBox(); spn_lo.setRange(1, 1000); spn_lo.setDecimals(0)
        spn_lo.setValue(cur_xlim[0]); spn_lo.setSuffix('  Hz')
        spn_hi = QDoubleSpinBox(); spn_hi.setRange(1000, 96000); spn_hi.setDecimals(0)
        spn_hi.setValue(cur_xlim[1]); spn_hi.setSuffix('  Hz')
        form.addRow(QLabel('Low:'),  spn_lo)
        form.addRow(QLabel('High:'), spn_hi)
        lay.addLayout(form)

        def _apply():
            lo, hi = spn_lo.value(), spn_hi.value()
            if lo < hi:
                for c in (self.canvas_meas, ):
                    c.ax_tf.set_xlim(lo, hi)
                    c.ax_ph.set_xlim(lo, hi)
                    c.draw_idle()
                if hasattr(self, 'canvas_spec'):
                    self.canvas_spec.ax.set_xlim(lo, hi)
                    self.canvas_spec.draw_idle()

        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply |
            QDialogButtonBox.StandardButton.Close)
        bb.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(_apply)
        bb.rejected.connect(dlg.accept)
        lay.addWidget(bb)
        dlg.exec()

    def _show_mag_range_dialog(self):
        """Ajustar rango de magnitud (dB) del gráfico TF."""
        from PyQt6.QtWidgets import (QDialog, QFormLayout, QVBoxLayout,
                                     QDoubleSpinBox, QDialogButtonBox, QLabel)
        dlg = QDialog(self)
        dlg.setWindowTitle('Magnitude Range')
        dlg.setStyleSheet(
            f'QDialog{{background:{BG_APP};color:{TEXT_HI};font-size:11px;}}'
            f'QLabel{{color:{TEXT_MID};}}'
            f'QDoubleSpinBox{{background:#252525;color:{TEXT_HI};'
            f'border:1px solid #444;border-radius:3px;padding:2px 6px;}}'
            f'QPushButton{{background:#2a2a2a;color:{TEXT_HI};padding:4px 14px;'
            f'border:1px solid #444;border-radius:3px;}}'
            f'QPushButton:hover{{background:#333;}}')
        lay = QVBoxLayout(dlg)
        form = QFormLayout(); form.setSpacing(8)

        cur_ylim = self.canvas_meas.ax_tf.get_ylim()
        spn_lo = QDoubleSpinBox(); spn_lo.setRange(-120, 0); spn_lo.setDecimals(0)
        spn_lo.setValue(cur_ylim[0]); spn_lo.setSuffix('  dB')
        spn_hi = QDoubleSpinBox(); spn_hi.setRange(-30, 60); spn_hi.setDecimals(0)
        spn_hi.setValue(cur_ylim[1]); spn_hi.setSuffix('  dB')
        form.addRow(QLabel('Bottom:'), spn_lo)
        form.addRow(QLabel('Top:'),    spn_hi)
        lay.addLayout(form)

        def _apply():
            lo, hi = spn_lo.value(), spn_hi.value()
            if lo < hi:
                self.canvas_meas.ax_tf.set_ylim(lo, hi)
                self.canvas_meas.ax_tf.set_autoscale_on(False)
                self.canvas_meas.draw_idle()

        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply |
            QDialogButtonBox.StandardButton.Close)
        bb.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(_apply)
        bb.rejected.connect(dlg.accept)
        lay.addWidget(bb)
        dlg.exec()

    def _show_phase_range_dialog(self):
        """Ajustar rango de fase del gráfico TF."""
        from PyQt6.QtWidgets import (QDialog, QFormLayout, QVBoxLayout,
                                     QDoubleSpinBox, QDialogButtonBox, QLabel)
        dlg = QDialog(self)
        dlg.setWindowTitle('Phase Range')
        dlg.setStyleSheet(
            f'QDialog{{background:{BG_APP};color:{TEXT_HI};font-size:11px;}}'
            f'QLabel{{color:{TEXT_MID};}}'
            f'QDoubleSpinBox{{background:#252525;color:{TEXT_HI};'
            f'border:1px solid #444;border-radius:3px;padding:2px 6px;}}'
            f'QPushButton{{background:#2a2a2a;color:{TEXT_HI};padding:4px 14px;'
            f'border:1px solid #444;border-radius:3px;}}'
            f'QPushButton:hover{{background:#333;}}')
        lay = QVBoxLayout(dlg)
        form = QFormLayout(); form.setSpacing(8)

        cur_ylim = self.canvas_meas.ax_ph.get_ylim()
        spn_lo = QDoubleSpinBox(); spn_lo.setRange(-360, 0); spn_lo.setDecimals(0)
        spn_lo.setValue(cur_ylim[0]); spn_lo.setSuffix('  °')
        spn_hi = QDoubleSpinBox(); spn_hi.setRange(0, 360); spn_hi.setDecimals(0)
        spn_hi.setValue(cur_ylim[1]); spn_hi.setSuffix('  °')
        form.addRow(QLabel('Bottom:'), spn_lo)
        form.addRow(QLabel('Top:'),    spn_hi)
        lay.addLayout(form)

        def _apply():
            lo, hi = spn_lo.value(), spn_hi.value()
            if lo < hi:
                self.canvas_meas.ax_ph.set_ylim(lo, hi)
                self.canvas_meas.draw_idle()

        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply |
            QDialogButtonBox.StandardButton.Close)
        bb.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(_apply)
        bb.rejected.connect(dlg.accept)
        lay.addWidget(bb)
        dlg.exec()

    def _toggle_smooth_visible(self):
        """Toggle visibilidad del control de smoothing en el panel."""
        if hasattr(self, 'cmb_smooth'):
            vis = not self.cmb_smooth.isVisible()
            self.cmb_smooth.setVisible(vis)

    def _show_signal_generator_dialog(self):
        """Abre/enfoca el panel de Signal Generator en el dock lateral."""
        # El generador está en el panel lateral — hacer scroll hasta él
        # o mostrar un mini-dialog con los controles principales
        from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout,
                                     QFormLayout, QGroupBox, QDialogButtonBox,
                                     QLabel, QSlider, QComboBox, QDoubleSpinBox)
        dlg = QDialog(self)
        dlg.setWindowTitle('Signal Generator')
        dlg.setMinimumWidth(320)
        dlg.setStyleSheet(
            f'QDialog{{background:{BG_APP};color:{TEXT_HI};font-size:11px;}}'
            f'QGroupBox{{color:{TEXT_MID};font-size:10px;border:1px solid #333;'
            f'border-radius:4px;margin-top:8px;padding-top:6px;}}'
            f'QGroupBox::title{{subcontrol-origin:margin;left:8px;}}'
            f'QLabel{{color:{TEXT_MID};}}'
            f'QComboBox,QDoubleSpinBox{{background:#252525;color:{TEXT_HI};'
            f'border:1px solid #444;border-radius:3px;padding:2px 6px;}}'
            f'QPushButton{{background:#2a2a2a;color:{TEXT_HI};padding:4px 14px;'
            f'border:1px solid #444;border-radius:3px;}}'
            f'QPushButton:hover{{background:#333;}}')
        lay = QVBoxLayout(dlg)

        grp = QGroupBox('Generator Settings')
        glay = QFormLayout(grp)
        glay.setSpacing(8)

        # Signal type
        cmb_type = QComboBox()
        for t in ['Pink Noise', 'White Noise', 'Sine Tone', 'Sweep']:
            cmb_type.addItem(t)
        type_map = {'pink': 0, 'white': 1, 'tone': 2, 'sweep': 3}
        cmb_type.setCurrentIndex(type_map.get(self.engine.signal_type, 0))
        cmb_type.currentIndexChanged.connect(
            lambda i: setattr(self.engine, 'signal_type',
                              ['pink', 'white', 'tone', 'sweep'][i]))
        glay.addRow(QLabel('Signal Type:'), cmb_type)

        # Gain
        spn_gain = QDoubleSpinBox()
        spn_gain.setRange(-60, 0); spn_gain.setDecimals(1)
        spn_gain.setValue(20.0 * np.log10(max(self.engine.gain, 1e-9)))
        spn_gain.setSuffix('  dBFS')
        spn_gain.valueChanged.connect(
            lambda v: setattr(self.engine, 'gain', 10.0 ** (v / 20.0)))
        glay.addRow(QLabel('Level:'), spn_gain)

        lay.addWidget(grp)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(dlg.accept)
        lay.addWidget(bb)
        dlg.exec()

    def _show_target_curves_dialog(self):
        """
        SMAART-style Target Curves dialog.
        Tabs: Spectrum / Transfer Function.
        Table: ⠿  Show  Color  Size  Offset  Tolerance+/-  Name
        """
        from PyQt6.QtWidgets import (QTabWidget, QTableWidget, QTableWidgetItem,
                                      QHeaderView, QCheckBox, QDoubleSpinBox as _DSPB,
                                      QColorDialog)
        from PyQt6.QtGui import QColor
        import numpy as _np_tc

        # ── Dialog shell ───────────────────────────────────────────────
        dlg = QDialog(self)
        dlg.setWindowTitle('Target Curves')
        dlg.setFixedSize(760, 440)
        _ss = (
            f'QDialog{{background:#1c1c1c;color:{TEXT_HI};font-size:11px;}}'
            f'QLabel{{background:transparent;color:{TEXT_HI};}}'
            f'QTabWidget::pane{{border:1px solid #333;background:#1a1a1a;}}'
            f'QTabBar::tab{{background:#252525;color:{TEXT_MID};padding:5px 22px;'
            f'border:1px solid #333;border-bottom:none;border-radius:3px 3px 0 0;font-size:11px;}}'
            f'QTabBar::tab:selected{{background:#1a1a1a;color:{TEXT_HI};}}'
            f'QTableWidget{{background:#161616;color:{TEXT_HI};gridline-color:#2a2a2a;'
            f'border:none;font-size:11px;selection-background-color:#2a3a2a;}}'
            f'QHeaderView::section{{background:#222;color:{TEXT_MID};border:none;'
            f'border-right:1px solid #333;padding:4px 6px;font-size:10px;letter-spacing:1px;}}'
            f'QComboBox{{background:#2a2a2a;color:{TEXT_HI};border:1px solid #444;'
            f'border-radius:2px;padding:1px 4px;font-size:11px;}}'
            f'QDoubleSpinBox{{background:#1e1e1e;color:{TEXT_HI};border:1px solid #383838;'
            f'border-radius:2px;padding:1px 4px;font-size:10px;}}'
            f'QCheckBox::indicator{{width:14px;height:14px;}}'
            f'QPushButton{{background:#2a2a2a;color:{TEXT_HI};border:1px solid #555;'
            f'border-radius:3px;padding:5px 20px;font-size:11px;}}'
            f'QPushButton:hover{{background:#363636;border-color:#777;}}')
        dlg.setStyleSheet(_ss)

        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(0, 0, 0, 12)
        lay.setSpacing(0)

        # ── Tab widget ─────────────────────────────────────────────────
        tabs = QTabWidget()
        lay.addWidget(tabs, stretch=1)

        # ── Transfer Function tab ──────────────────────────────────────
        tf_widget = QWidget()
        tf_lay = QVBoxLayout(tf_widget)
        tf_lay.setContentsMargins(0, 0, 0, 0)
        tf_lay.setSpacing(0)

        table = QTableWidget()
        table.setColumnCount(7)
        table.setHorizontalHeaderLabels(['', 'Show', 'Color', 'Size', 'Offset', 'Tolerance +/-', 'Name'])
        table.verticalHeader().setVisible(False)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setShowGrid(True)
        table.horizontalHeader().setHighlightSections(False)
        # Column widths
        table.setColumnWidth(0, 24)   # drag
        table.setColumnWidth(1, 50)   # show
        table.setColumnWidth(2, 62)   # color
        table.setColumnWidth(3, 64)   # size
        table.setColumnWidth(4, 100)  # offset
        table.setColumnWidth(5, 150)  # tolerance
        table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setDefaultSectionSize(30)
        tf_lay.addWidget(table)

        tabs.addTab(tf_widget, 'Transfer Function')

        # ── Spectrum tab (placeholder) ─────────────────────────────────
        sp_widget = QWidget()
        sp_lay = QVBoxLayout(sp_widget)
        lbl_sp = QLabel('Spectrum target curves — use Transfer Function curves\nfor TF / Magnitude overlay')
        lbl_sp.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_sp.setStyleSheet(f'color:{TEXT_MID};font-size:11px;')
        sp_lay.addWidget(lbl_sp)
        tabs.addTab(sp_widget, 'RTA')
        tabs.setCurrentIndex(0)   # default TF

        # ── Apply helper ───────────────────────────────────────────────
        def _apply():
            self.canvas_meas.update_target_curves(self._target_curves)
            if self._secondary_panel is not None:
                self._secondary_panel.canvas_meas.update_target_curves(self._target_curves)

        # ── Build / rebuild table rows ─────────────────────────────────
        _building = [False]   # guard against recursive signals

        def _rebuild():
            _building[0] = True
            table.setRowCount(0)
            for ri, tc in enumerate(self._target_curves):
                table.insertRow(ri)
                _fill_row(ri, tc)
            _building[0] = False

        def _fill_row(row, tc):
            # col 0 — drag handle
            lbl_dh = QLabel('⠿')
            lbl_dh.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl_dh.setStyleSheet('color:#444;font-size:16px;')
            table.setCellWidget(row, 0, lbl_dh)

            # col 1 — Show checkbox (centered)
            chk_show = QCheckBox()
            chk_show.setChecked(tc.visible)
            wrap1 = QWidget(); wl1 = QHBoxLayout(wrap1)
            wl1.setContentsMargins(0,0,0,0); wl1.addWidget(chk_show, 0, Qt.AlignmentFlag.AlignCenter)
            table.setCellWidget(row, 1, wrap1)

            def _on_show(state, r=row, chk=chk_show):
                if _building[0]: return
                self._target_curves[r].visible = chk.isChecked()
                _apply()

            chk_show.stateChanged.connect(_on_show)

            # col 2 — Color swatch button
            btn_col = QPushButton()
            btn_col.setStyleSheet(
                f'background:{tc.color};border:1px solid #777;border-radius:2px;min-height:18px;')
            btn_col.setFixedHeight(22)

            def _on_color(_, r=row, b=btn_col):
                if _building[0]: return
                current_tc = self._target_curves[r]
                c = QColorDialog.getColor(QColor(current_tc.color), dlg, 'Choose Color')
                if c.isValid():
                    current_tc.color = c.name()
                    b.setStyleSheet(
                        f'background:{c.name()};border:1px solid #777;border-radius:2px;min-height:18px;')
                    _apply()

            btn_col.clicked.connect(_on_color)
            table.setCellWidget(row, 2, btn_col)

            # col 3 — Size combobox
            cmb_size = QComboBox()
            cmb_size.addItems(['2', '3'])
            cmb_size.setCurrentText(str(tc.line_width))

            def _on_size(v, r=row):
                if _building[0]: return
                self._target_curves[r].line_width = int(v)
                _apply()

            cmb_size.currentTextChanged.connect(_on_size)
            table.setCellWidget(row, 3, cmb_size)

            # col 4 — Offset spinbox
            spn_off = _DSPB()
            spn_off.setRange(-30.0, 30.0); spn_off.setDecimals(2)
            spn_off.setSuffix(' dB'); spn_off.setValue(tc.offset_db)
            spn_off.setButtonSymbols(_DSPB.ButtonSymbols.NoButtons)
            spn_off.setStyleSheet(
                'QDoubleSpinBox{background:#1e1e1e;color:#ddd;border:none;'
                'padding:2px 4px;font-size:10px;}')

            def _on_offset(v, r=row):
                if _building[0]: return
                self._target_curves[r].offset_db = v
                _apply()

            spn_off.valueChanged.connect(_on_offset)
            table.setCellWidget(row, 4, spn_off)

            # col 5 — Tolerance: [checkbox] [spinbox dB]
            tol_w = QWidget()
            tol_l = QHBoxLayout(tol_w)
            tol_l.setContentsMargins(6, 0, 6, 0); tol_l.setSpacing(4)
            chk_tol = QCheckBox()
            chk_tol.setChecked(tc.tolerance)
            spn_tol = _DSPB()
            spn_tol.setRange(0.1, 20.0); spn_tol.setDecimals(1)
            spn_tol.setSuffix(' dB')
            _tol_default = tc.tolerance_db if tc.tolerance_db > 0 else 3.0
            if tc.tolerance_db <= 0:
                tc.tolerance_db = _tol_default   # sync model to display default
            spn_tol.setValue(_tol_default)
            spn_tol.setEnabled(tc.tolerance)
            spn_tol.setButtonSymbols(_DSPB.ButtonSymbols.NoButtons)
            spn_tol.setStyleSheet(
                'QDoubleSpinBox{background:#1e1e1e;color:#ddd;border:1px solid #383838;'
                'border-radius:2px;padding:1px 4px;font-size:10px;}')
            tol_l.addWidget(chk_tol); tol_l.addWidget(spn_tol)

            def _on_tol_check(state, r=row, sp=spn_tol):
                if _building[0]: return
                chkd = (state == Qt.CheckState.Checked.value)
                self._target_curves[r].tolerance    = chkd
                self._target_curves[r].tolerance_db = sp.value()  # sync current spinbox value
                sp.setEnabled(chkd)
                _apply()

            def _on_tol_val(v, r=row):
                if _building[0]: return
                self._target_curves[r].tolerance_db = v
                _apply()

            chk_tol.stateChanged.connect(_on_tol_check)
            spn_tol.valueChanged.connect(_on_tol_val)
            table.setCellWidget(row, 5, tol_w)

            # col 6 — Name (QTableWidgetItem, not editable)
            item_name = QTableWidgetItem(tc.name)
            item_name.setFlags(item_name.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if tc.builtin:
                item_name.setForeground(QColor('#888888'))
            table.setItem(row, 6, item_name)

        _rebuild()

        # ── Bottom button row ──────────────────────────────────────────
        btn_bar = QWidget(); btn_bar.setFixedHeight(48)
        btn_bar.setStyleSheet('background:#1c1c1c;')
        bb_lay = QHBoxLayout(btn_bar)
        bb_lay.setContentsMargins(12, 8, 12, 8); bb_lay.setSpacing(8)

        btn_import = QPushButton('Import…')
        btn_delete = QPushButton('Delete')
        btn_close  = QPushButton('Close')

        bb_lay.addWidget(btn_import)
        bb_lay.addWidget(btn_delete)
        bb_lay.addStretch()
        bb_lay.addWidget(btn_close)
        lay.addWidget(btn_bar)

        # ── Import ─────────────────────────────────────────────────────
        def _do_import():
            path, _ = QFileDialog.getOpenFileName(
                dlg, 'Import Target Curve', os.path.expanduser('~'),
                'Curve files (*.crv *.txt *.csv *.tsv);;'
                'SMAART curves (*.crv);;'
                'Text / CSV (*.txt *.csv *.tsv);;'
                'All files (*)')
            if not path:
                return
            try:
                meta_name, data = _load_curve_file(path)
            except Exception as e:
                QMessageBox.critical(dlg, 'Error loading curve', str(e))
                return
            color = TARGET_PALETTE[len(self._target_curves) % len(TARGET_PALETTE)]
            # Prefer name from file metadata, fall back to filename
            name  = meta_name or os.path.splitext(os.path.basename(path))[0]
            tc    = TargetCurve(
                name=name, freqs=data[:, 0], mag_db=data[:, 1],
                color=color, visible=True, line_width=2)
            self._target_curves.append(tc)
            _rebuild()
            _apply()
            self.sb.showMessage(
                f'✓  Target: {name}  ({len(data)} pts, '
                f'{data[0,0]:.0f}–{data[-1,0]:.0f} Hz)', 5000)

        # ── Delete ─────────────────────────────────────────────────────
        def _do_delete():
            row = table.currentRow()
            if row < 0 or row >= len(self._target_curves):
                return
            tc = self._target_curves[row]
            if tc.builtin:
                # Factory curves: just hide rather than delete
                tc.visible = False
                _rebuild(); _apply()
                self.sb.showMessage('Built-in curves cannot be deleted — hidden instead.', 3000)
                return
            self._target_curves.pop(row)
            _rebuild(); _apply()

        btn_import.clicked.connect(_do_import)
        btn_delete.clicked.connect(_do_delete)
        btn_close.clicked.connect(dlg.accept)

        dlg.exec()

    def _show_optimize_dialog(self):
        """
        Intelligent optimization: compare current TF measurement to a target curve,
        compute the delta, and suggest parametric EQ corrections.
        """
        import numpy as np

        # Validate prerequisites
        if not self._target_curves:
            QMessageBox.information(self, 'No Target Curve',
                'Enable a target curve first via Options → Target Curves…')
            return
        if self.canvas_meas._last_freqs is None:
            QMessageBox.information(self, 'No Measurement',
                'Run a Transfer Function measurement first.')
            return

        freqs_m  = self.canvas_meas._last_freqs
        mag_m    = self.canvas_meas._last_mag_db

        if freqs_m is None or mag_m is None:
            QMessageBox.information(self, 'No Data', 'No magnitude data available.')
            return

        # Use first visible target curve (or let user pick if multiple visible)
        visible_tcs = [t for t in self._target_curves if t.visible]
        if not visible_tcs:
            visible_tcs = list(self._target_curves)  # fallback: all
        if len(visible_tcs) == 1:
            tc = visible_tcs[0]
        else:
            from PyQt6.QtWidgets import QInputDialog
            names = [t.name for t in visible_tcs]
            name, ok = QInputDialog.getItem(self, 'Select Target', 'Target curve:', names, 0, False)
            if not ok:
                return
            tc = next(t for t in visible_tcs if t.name == name)

        freqs_t = tc.freqs
        mag_t   = tc.mag_db + tc.offset_db

        # Interpolate target to measurement frequencies
        mag_t_interp = np.interp(freqs_m, freqs_t, mag_t)
        delta = mag_t_interp - mag_m   # positive = need boost, negative = need cut

        # Smooth delta for stable peak detection
        try:
            from scipy.ndimage import uniform_filter1d
            delta_s = uniform_filter1d(delta, size=max(3, len(delta) // 80))
        except ImportError:
            delta_s = delta

        # Detect peaks and valleys in delta (= EQ bands needed)
        try:
            from scipy.signal import find_peaks
            peak_idx,  _ = find_peaks( delta_s, height=1.0, distance=max(5, len(delta)//60), prominence=1.2)
            valley_idx,_ = find_peaks(-delta_s, height=1.0, distance=max(5, len(delta)//60), prominence=1.2)
        except ImportError:
            peak_idx = valley_idx = np.array([], dtype=int)

        def _estimate_q(freqs, delta_smooth, idx, gain):
            """Estimate Q from bandwidth at gain/2."""
            half = abs(gain) / 2.0
            direction = 1 if gain > 0 else -1
            mask = (direction * delta_smooth) >= half
            idxs = np.where(mask)[0]
            if len(idxs) < 2:
                return 2.0
            bw_hz = freqs[idxs[-1]] - freqs[idxs[0]]
            q = freqs[idx] / bw_hz if bw_hz > 0 else 2.0
            return max(0.4, min(q, 12.0))

        suggestions = []
        for i in peak_idx:
            g = float(delta_s[i])
            q = _estimate_q(freqs_m, delta_s, i, g)
            suggestions.append({'freq': float(freqs_m[i]), 'gain': g, 'q': q})
        for i in valley_idx:
            g = float(-delta_s[i])   # negative
            q = _estimate_q(freqs_m, -delta_s, i, g)
            suggestions.append({'freq': float(freqs_m[i]), 'gain': g, 'q': q})
        suggestions.sort(key=lambda x: x['freq'])

        # Also compute 1/3-octave ISO average corrections
        _ISO = np.array([20,25,31.5,40,50,63,80,100,125,160,200,250,315,400,
                         500,630,800,1000,1250,1600,2000,2500,3150,4000,5000,
                         6300,8000,10000,12500,16000,20000], dtype=float)
        iso_gains = []
        for fc in _ISO:
            fl, fh = fc / 2**(1/6), fc * 2**(1/6)
            mask = (freqs_m >= fl) & (freqs_m <= fh)
            if mask.any():
                iso_gains.append((fc, float(np.mean(delta[mask]))))

        # ── Build the dialog ────────────────────────────────────────────
        dlg = QDialog(self)
        dlg.setWindowTitle(f'Optimize: Measurement → {tc.name}')
        dlg.resize(700, 560)
        dlg.setStyleSheet(
            f'QDialog{{background:{BG_APP};color:{TEXT_HI};font-size:11px;}}'
            f'QLabel{{color:{TEXT_MID};background:transparent;}}'
            f'QTableWidget{{background:#111;color:{TEXT_HI};gridline-color:#222;'
            f'border:1px solid #2a2a2a;font-size:10px;}}'
            f'QHeaderView::section{{background:#222;color:{TEXT_MID};'
            f'border:none;padding:3px 6px;font-size:9px;letter-spacing:1px;}}'
            f'QPushButton{{background:#252525;color:{TEXT_HI};border:1px solid #444;'
            f'border-radius:3px;padding:4px 14px;font-size:10px;}}'
            f'QPushButton:hover{{background:#333;}}')

        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(8)

        hdr = QLabel(f'OPTIMIZATION  ·  {tc.name}')
        hdr.setStyleSheet(f'color:{ACCENT};font-size:9px;letter-spacing:2px;font-weight:bold;background:transparent;')
        lay.addWidget(hdr)

        # ── Delta miniplot ──────────────────────────────────────────────
        try:
            import matplotlib
            matplotlib.use('QtAgg')
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

            fig, ax = plt.subplots(figsize=(6, 2.2), dpi=80)
            fig.patch.set_facecolor('#111111')
            ax.set_facecolor('#111111')
            ax.plot(freqs_m, delta, color='#444', linewidth=0.8, alpha=0.6)
            ax.plot(freqs_m, delta_s, color=tc.color, linewidth=1.4)
            ax.axhline(0, color='#3a3a3a', linewidth=0.8, linestyle='--')
            ax.fill_between(freqs_m, 0, delta_s,
                            where=(delta_s > 0), color='#2a4a2a', alpha=0.4)
            ax.fill_between(freqs_m, 0, delta_s,
                            where=(delta_s < 0), color='#4a2a2a', alpha=0.4)
            ax.set_xscale('log')
            ax.set_xlim(20, 20000)
            ax.set_xlabel('Frequency (Hz)', color='#666', fontsize=7)
            ax.set_ylabel('Correction (dB)', color='#666', fontsize=7)
            ax.tick_params(colors='#555', labelsize=7)
            ax.grid(True, which='both', color='#222', linewidth=0.5)
            ax.set_title('Required correction (Target − Measurement)', color='#888', fontsize=8)
            fig.tight_layout()

            canvas_mini = FigureCanvasQTAgg(fig)
            canvas_mini.setFixedHeight(180)
            lay.addWidget(canvas_mini)
            plt.close(fig)
        except Exception:
            pass   # matplotlib unavailable

        # ── Tabs: Parametric EQ / 1/3-Oct GEQ ─────────────────────────
        tabs = QTabWidget()
        tabs.setStyleSheet(
            f'QTabBar::tab{{background:#252525;color:{TEXT_MID};padding:4px 16px;'
            f'border:1px solid #333;border-bottom:none;border-radius:3px 3px 0 0;}}'
            f'QTabBar::tab:selected{{background:#1a1a1a;color:{TEXT_HI};}}'
            f'QTabWidget::pane{{border:1px solid #333;background:#1a1a1a;}}')
        lay.addWidget(tabs, stretch=1)

        # Tab 1 — Parametric EQ
        peq_w = QWidget()
        peq_lay = QVBoxLayout(peq_w)
        peq_lay.setContentsMargins(6, 6, 6, 6)

        if suggestions:
            peq_tbl = QTableWidget(len(suggestions), 3)
            peq_tbl.setHorizontalHeaderLabels(['Frequency (Hz)', 'Gain (dB)', 'Q'])
            peq_tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            peq_tbl.verticalHeader().setVisible(False)
            peq_tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            peq_tbl.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
            for row_i, s in enumerate(suggestions):
                it_f = QTableWidgetItem(f'{s["freq"]:.0f} Hz')
                it_g = QTableWidgetItem(f'{s["gain"]:+.1f} dB')
                it_q = QTableWidgetItem(f'{s["q"]:.2f}')
                it_g.setForeground(QColor('#4aaa4a' if s['gain'] > 0 else '#ff6666'))
                for it in (it_f, it_g, it_q):
                    it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                peq_tbl.setItem(row_i, 0, it_f)
                peq_tbl.setItem(row_i, 1, it_g)
                peq_tbl.setItem(row_i, 2, it_q)
            peq_lay.addWidget(peq_tbl)
        else:
            lbl_no = QLabel('No significant corrections needed (delta < 1 dB everywhere).')
            lbl_no.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl_no.setStyleSheet(f'color:{GREEN};font-size:11px;')
            peq_lay.addWidget(lbl_no)

        tabs.addTab(peq_w, 'Parametric EQ')

        # Tab 2 — 1/3-Oct GEQ
        geq_w = QWidget()
        geq_lay = QVBoxLayout(geq_w)
        geq_lay.setContentsMargins(6, 6, 6, 6)

        geq_tbl = QTableWidget(len(iso_gains), 2)
        geq_tbl.setHorizontalHeaderLabels(['1/3-Oct Center (Hz)', 'Correction (dB)'])
        geq_tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        geq_tbl.verticalHeader().setVisible(False)
        geq_tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        for row_i, (fc, gain) in enumerate(iso_gains):
            it_f = QTableWidgetItem(f'{int(fc)} Hz')
            it_g = QTableWidgetItem(f'{gain:+.1f} dB')
            it_g.setForeground(QColor('#4aaa4a' if gain > 0 else '#ff6666'))
            for it in (it_f, it_g):
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            geq_tbl.setItem(row_i, 0, it_f)
            geq_tbl.setItem(row_i, 1, it_g)
        geq_lay.addWidget(geq_tbl)

        tabs.addTab(geq_w, '1/3-Oct GEQ')

        # ── Buttons ─────────────────────────────────────────────────────
        btn_row_h = QHBoxLayout()
        btn_export = QPushButton('Export .txt…')
        btn_delta  = QPushButton('Show Delta on Graph')
        btn_close  = QPushButton('Close')

        def _export():
            path, _ = QFileDialog.getSaveFileName(
                dlg, 'Export EQ Corrections', os.path.expanduser('~'),
                'Text files (*.txt);;All files (*)')
            if not path:
                return
            lines = [f'# Coherence — Optimization: Measurement vs "{tc.name}"',
                     f'# Generated: {__import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M")}',
                     '', '## PARAMETRIC EQ CORRECTIONS', 'Freq (Hz)  Gain (dB)  Q']
            for s in suggestions:
                lines.append(f'{s["freq"]:.0f}  {s["gain"]:+.1f}  {s["q"]:.2f}')
            lines += ['', '## 1/3-OCTAVE GEQ CORRECTIONS', 'Freq (Hz)  Gain (dB)']
            for fc, g in iso_gains:
                lines.append(f'{int(fc)}  {g:+.1f}')
            with open(path, 'w') as f:
                f.write('\n'.join(lines))
            self.sb.showMessage(f'✓  Exported: {os.path.basename(path)}', 4000)

        def _show_delta():
            color    = '#FF9F0A'
            dname    = f'Δ {tc.name}'
            delta_tc = TargetCurve(
                name=dname, freqs=freqs_m, mag_db=mag_m + delta_s,
                color=color, visible=True, line_width=2)
            self._target_curves.append(delta_tc)
            self.canvas_meas.update_target_curves(self._target_curves)
            if self._secondary_panel is not None:
                self._secondary_panel.canvas_meas.update_target_curves(self._target_curves)
            self.sb.showMessage('Delta curve added to Magnitude graph', 3000)
            btn_delta.setEnabled(False)

        btn_export.clicked.connect(_export)
        btn_delta.clicked.connect(_show_delta)
        btn_close.clicked.connect(dlg.accept)

        btn_row_h.addWidget(btn_export)
        btn_row_h.addWidget(btn_delta)
        btn_row_h.addStretch()
        btn_row_h.addWidget(btn_close)
        lay.addLayout(btn_row_h)

        dlg.exec()

    def _show_weighting_dialog(self):
        """Selección de curva de ponderación A/C/Z."""
        from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QButtonGroup,
                                     QRadioButton, QDialogButtonBox, QLabel)
        dlg = QDialog(self)
        dlg.setWindowTitle('Weighting Curves')
        dlg.setStyleSheet(
            f'QDialog{{background:{BG_APP};color:{TEXT_HI};font-size:11px;}}'
            f'QLabel{{color:{TEXT_MID};}}'
            f'QRadioButton{{color:{TEXT_HI};}}'
            f'QPushButton{{background:#2a2a2a;color:{TEXT_HI};padding:4px 14px;'
            f'border:1px solid #444;border-radius:3px;}}'
            f'QPushButton:hover{{background:#333;}}')
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel('Apply frequency weighting to SPL and spectrum:'))

        grp   = QButtonGroup(dlg)
        cur   = getattr(self, '_weighting', 'Z')
        for w in ('Z  (Flat / unweighted)', 'A  (dBA — vocal intelligibility)',
                  'C  (dBC — low frequency)'):
            rb = QRadioButton(w)
            rb.setChecked(w[0] == cur)
            grp.addButton(rb)
            lay.addWidget(rb)

        def _apply():
            btn = grp.checkedButton()
            if btn:
                self._weighting = btn.text()[0]
                self.sb.showMessage(f'Weighting: {self._weighting}', 2000)

        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply |
            QDialogButtonBox.StandardButton.Close)
        bb.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(_apply)
        bb.rejected.connect(dlg.accept)
        lay.addWidget(bb)
        dlg.exec()

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


    def _capture_trace_dialog(self):
        """Muestra el dialog 'Capture Trace' — nombre + selector de color + fuente (Current / AVG)."""
        if len(self._traces) >= MAX_TRACES:
            self.sb.showMessage(
                f'⚠  Maximum {MAX_TRACES} traces. Delete one before saving.', 4000)
            return

        # Check data availability
        has_current = self.canvas_meas._last_freqs is not None
        has_avg     = self.canvas_meas._last_avg_freqs is not None
        if not has_current and not has_avg:
            self.sb.showMessage('⚠  No data — start the measurement first.', 3000)
            return

        # Nombre y color por defecto
        ws_name = (self._workspaces[self._current_ws_idx].get('name', 'Workspace')
                   if self._current_ws_idx < len(self._workspaces) else 'Workspace')
        default_name  = f'{ws_name}-{len(self._traces) + 1}'
        default_color = TRACE_PALETTE[self._trace_color_idx % len(TRACE_PALETTE)]
        _picked_color = [default_color]   # lista mutable para capturar en closures

        dlg = QDialog(self)
        dlg.setWindowTitle('Save Trace')
        dlg.setModal(True)
        dlg.setFixedSize(400, 230)
        dlg.setStyleSheet(
            'QDialog{background:#1a1a1a;}'
            f'QLabel{{color:{TEXT_HI};font-size:12px;background:transparent;}}'
            'QLineEdit{background:#e8e8e8;color:#111;border:none;'
            'border-radius:3px;padding:4px 8px;font-size:12px;}'
            f'QRadioButton{{color:{TEXT_HI};font-size:12px;background:transparent;}}'
            f'QPushButton#ok_btn{{background:#2a3a2a;color:{TEXT_HI};'
            'border:1px solid #4a6a4a;border-radius:4px;'
            'font-size:11px;padding:5px 22px;}}'
            f'QPushButton#ok_btn:hover{{background:#3a4a3a;border-color:#6a9a6a;}}'
            f'QPushButton#cancel_btn{{background:#2e2e2e;color:{TEXT_HI};'
            'border:1px solid #444;border-radius:4px;'
            'font-size:11px;padding:5px 22px;}}'
            f'QPushButton#cancel_btn:hover{{background:#3a3a3a;border-color:#666;}}')

        root = QVBoxLayout(dlg)
        root.setContentsMargins(20, 18, 20, 14)
        root.setSpacing(10)

        # Source row (Current / AVG)
        from PyQt6.QtWidgets import QButtonGroup, QRadioButton
        row_src = QHBoxLayout()
        row_src.addWidget(QLabel('Source:'))
        rb_current = QRadioButton('Current')
        rb_avg     = QRadioButton('AVG')
        rb_current.setEnabled(has_current)
        rb_avg.setEnabled(has_avg)
        # Default: prefer AVG if available, else Current
        if has_avg:
            rb_avg.setChecked(True)
        else:
            rb_current.setChecked(True)
        grp_src = QButtonGroup(dlg)
        grp_src.addButton(rb_current)
        grp_src.addButton(rb_avg)
        row_src.addWidget(rb_current)
        row_src.addWidget(rb_avg)
        row_src.addStretch()
        root.addLayout(row_src)

        # Name row
        row_name = QHBoxLayout()
        row_name.addWidget(QLabel('Name:'))
        edit = QLineEdit(default_name)
        edit.selectAll()
        row_name.addWidget(edit, stretch=1)
        root.addLayout(row_name)

        # Description row
        row_desc = QHBoxLayout()
        row_desc.addWidget(QLabel('Description:'))
        edit_desc = QLineEdit()
        edit_desc.setPlaceholderText('Optional notes…')
        row_desc.addWidget(edit_desc, stretch=1)
        root.addLayout(row_desc)

        # Color row
        row_color = QHBoxLayout()
        row_color.addWidget(QLabel('Color:'))
        btn_color = QPushButton()
        btn_color.setFixedSize(34, 24)
        btn_color.setStyleSheet(
            f'background:{default_color};border:1px solid #666;border-radius:3px;')
        btn_color.setToolTip('Pick color for magnitude and phase')

        def _pick_color():
            from PyQt6.QtWidgets import QColorDialog
            from PyQt6.QtGui import QColor as _QColor
            c = QColorDialog.getColor(_QColor(_picked_color[0]), dlg, 'Trace Color')
            if c.isValid():
                _picked_color[0] = c.name()
                btn_color.setStyleSheet(
                    f'background:{c.name()};border:1px solid #666;border-radius:3px;')

        btn_color.clicked.connect(_pick_color)
        row_color.addWidget(btn_color)
        row_color.addStretch()
        root.addLayout(row_color)

        # OK / Cancel buttons
        btns = QHBoxLayout()
        btns.setSpacing(8)
        btns.addStretch()
        btn_ok     = QPushButton('OK');     btn_ok.setObjectName('ok_btn')
        btn_cancel = QPushButton('Cancel'); btn_cancel.setObjectName('cancel_btn')
        btn_ok.setDefault(True)
        btns.addWidget(btn_ok)
        btns.addWidget(btn_cancel)
        root.addLayout(btns)

        btn_ok.clicked.connect(dlg.accept)
        btn_cancel.clicked.connect(dlg.reject)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        name   = edit.text().strip() or default_name
        color  = _picked_color[0]
        source = 'avg' if rb_avg.isChecked() else 'current'
        self._save_trace(name=name, color=color, source=source)

    def _save_trace(self, name: str = '', color: str = None, source: str = 'current'):
        """Captura el estado actual (o AVG) y lo guarda como traza estática de referencia.
        Siempre guarda magnitud Y fase con el mismo color (están enlazadas).
        source: 'current' | 'avg'
        """
        if len(self._traces) >= MAX_TRACES:
            self.sb.showMessage(
                f'⚠  Maximum {MAX_TRACES} traces. Delete one before saving.', 4000)
            return

        use_avg = (source == 'avg') and (self.canvas_meas._last_avg_freqs is not None)

        if use_avg:
            f        = self.canvas_meas._last_avg_freqs
            mag_db   = self.canvas_meas._last_avg_mag_db
            phase_deg = self.canvas_meas._last_avg_phase_deg
            gamma2   = self.canvas_meas._last_avg_gamma2
            ir       = None   # AVG trace has no IR
        else:
            f         = self.canvas_meas._last_freqs
            mag_db    = self.canvas_meas._last_mag_db
            phase_deg = self.canvas_meas._last_phase_deg
            gamma2    = self.canvas_meas._last_gamma2
            ir        = self.canvas_meas._last_ir

        if f is None:
            self.sb.showMessage('⚠  No data — start the measurement first.', 3000)
            return

        if color is None:
            color = TRACE_PALETTE[self._trace_color_idx % len(TRACE_PALETTE)]
        self._trace_color_idx += 1

        if not name:
            name = f'T{len(self._traces) + 1}'
        thresh    = self.spn_thresh.value()

        # Niveles del spectrum (CPB o FFT raw según resolución activa)
        lev_x     = self.canvas_spec._last_lx
        lev_y     = self.canvas_spec._last_ly
        lev_freqs = self.canvas_spec._last_freqs
        if lev_x is None or lev_y is None or lev_freqs is None:
            lev_freqs = self.canvas_spec._centers if len(
                self.canvas_spec._centers) > 0 else np.array([1000.0])
            lev_x = np.full(len(lev_freqs), -80.0)
            lev_y = np.full(len(lev_freqs), -80.0)
        # Asegurar coherencia de longitudes
        min_len = min(len(lev_freqs), len(lev_x), len(lev_y))
        lev_freqs = lev_freqs[:min_len]
        lev_x     = lev_x[:min_len]
        lev_y     = lev_y[:min_len]

        tr = TraceData(
            name         = name,
            color        = color,
            freqs        = f.copy(),
            mag_db       = mag_db.copy(),
            phase_deg    = phase_deg.copy(),
            gamma2       = gamma2.copy() if gamma2 is not None else np.zeros_like(f),
            ir           = (ir.copy() if ir is not None else None),
            delay_ref_ms = self.canvas_meas._delay_ref_ms,
            coh_thresh   = thresh,
            lev_x        = lev_x.copy(),
            lev_y        = lev_y.copy(),
            lev_freqs    = lev_freqs.copy(),
        )
        self._traces.append(tr)

        # Dibujar en todos los canvas de medición (principal + panel secundario si existe)
        self.canvas_meas.store_trace(
            tr.freqs, tr.mag_db, tr.phase_deg, tr.gamma2, tr.ir,
            tr.delay_ref_ms, tr.color, tr.coh_thresh
        )
        if self._secondary_panel is not None:
            self._secondary_panel.canvas_meas.store_trace(
                tr.freqs, tr.mag_db, tr.phase_deg, tr.gamma2, tr.ir,
                tr.delay_ref_ms, tr.color, tr.coh_thresh
            )
        self.canvas_spec.store_trace(tr.lev_x, tr.lev_y, tr.color, tr.lev_freqs)

        self._rebuild_trace_panel()
        src_label = 'AVG' if use_avg else 'Current'
        self.sb.showMessage(f'✓  Trace saved: {name}  [{src_label}]  (Magnitude + Phase + Coherence)', 3000)

    def _delete_trace(self, idx):
        """Elimina la traza idx de canvas y lista."""
        if 0 <= idx < len(self._traces):
            self._traces.pop(idx)
            self.canvas_meas.remove_trace(idx)
            self.canvas_spec.remove_trace(idx)
            if self._secondary_panel is not None:
                self._secondary_panel.canvas_meas.remove_trace(idx)
            self._rebuild_trace_panel()

    def _toggle_trace_visible(self, idx, checked):
        """Muestra u oculta la traza idx."""
        if 0 <= idx < len(self._traces):
            self._traces[idx].visible = checked
            self.canvas_meas.set_trace_visible(idx, checked)
            self.canvas_spec.set_trace_visible(idx, checked)
            if self._secondary_panel is not None:
                self._secondary_panel.canvas_meas.set_trace_visible(idx, checked)

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

    # ── Panel ≡ menu (SMAART-style) ───────────────────────────────────────

    def _show_panel_menu(self, btn: 'QPushButton', panel: str):
        """Show the SMAART-style ≡ context menu below the button."""
        menu = QMenu(self)
        menu.setStyleSheet(
            'QMenu{'
            '  background:#2d2d2d;color:#cccccc;border:1px solid #444;'
            '  font-size:13px;padding:4px 0;'
            '}'
            'QMenu::item{padding:5px 24px 5px 16px;}'
            'QMenu::item:selected{background:#3a5a8a;color:#ffffff;}'
            'QMenu::item:disabled{color:#555;}'
            'QMenu::separator{height:1px;background:#444;margin:3px 0;}'
        )

        def _act(label, shortcut='', enabled=True, fn=None):
            txt = f'{label}\t{shortcut}' if shortcut else label
            a = menu.addAction(txt)
            a.setEnabled(enabled)
            if fn and enabled:
                a.triggered.connect(fn)
            return a

        # Sort By submenu
        sort_menu = menu.addMenu('Sort By')
        sort_menu.setStyleSheet(menu.styleSheet())
        for s in ('Name', 'Date', 'Color'):
            sort_menu.addAction(s)

        _act('Hide All', 'H', fn=self._hide_all_traces if panel == 'tf' else self._hide_all_sp_traces)
        menu.addSeparator()

        _act('New Session Folder',      fn=self._new_session_folder)
        _act('Set Folder Root…',        fn=self._set_folder_root)
        _act('New Folder',              fn=self._new_folder)
        _act('New Folder from Selection', enabled=False)
        menu.addSeparator()

        _act('Save As',   enabled=False)
        _act('Recapture', 'Space',
             fn=self._capture_trace_dialog if panel == 'tf' else self._save_spectrum_trace)
        _act('Rename',    enabled=False)
        _act('Average',
             fn=self._open_tf_avg_dialog if panel == 'tf' else self._open_sp_avg_dialog)
        _act('Assign Random Color', enabled=False)
        _act('Open File Location',  enabled=False)
        menu.addSeparator()

        _act('Import Trace…', fn=self._import_trace)
        _act('Import ASCII…', fn=self._import_ascii)
        _act('Export To ASCII',            enabled=False)
        _act('Copy To ASCII',   '©',       enabled=False)
        _act('Export As Weighting Curve',  enabled=False)
        _act('Create DSP Channel from Selected', enabled=False)
        menu.addSeparator()

        _act('Toggle Data/SPL Meter Bar', 'E', fn=self._toggle_spl_bar)
        menu.addSeparator()

        _act('Show Archive',         fn=self._show_archive)
        _act('Refresh Data Library', 'R', fn=self._refresh_data_library)

        # Show below the button
        menu.exec(btn.mapToGlobal(btn.rect().bottomLeft()))

    # ── Stubs for new panel menu actions ─────────────────────────────────

    def _hide_all_traces(self):
        for tr in self._traces:
            tr.visible = False
        self._rebuild_trace_panel()
        self._schedule_refresh()

    def _hide_all_sp_traces(self):
        for tr in getattr(self, '_sp_traces', []):
            tr.visible = False
        self._rebuild_trace_panel()

    def _new_session_folder(self):
        name, ok = QInputDialog.getText(self, 'New Session Folder', 'Folder name:')
        if ok and name.strip():
            QMessageBox.information(self, 'New Session Folder',
                                    f'"{name}" (not yet persisted to disk)')

    def _set_folder_root(self):
        path = QFileDialog.getExistingDirectory(self, 'Set Folder Root')
        if path:
            QMessageBox.information(self, 'Set Folder Root', path)

    def _new_folder(self):
        name, ok = QInputDialog.getText(self, 'New Folder', 'Folder name:')
        if ok and name.strip():
            QMessageBox.information(self, 'New Folder',
                                    f'"{name}" (not yet persisted to disk)')

    def _import_trace(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Import Trace', '', 'Trace files (*.tf *.txt *.csv);;All files (*)')
        if path:
            QMessageBox.information(self, 'Import Trace',
                                    f'Import from:\n{path}\n(coming soon)')

    def _import_ascii(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Import ASCII', '', 'ASCII files (*.txt *.csv);;All files (*)')
        if path:
            QMessageBox.information(self, 'Import ASCII',
                                    f'Import ASCII from:\n{path}\n(coming soon)')

    def _toggle_spl_bar(self):
        # Placeholder — will show/hide the SPL meter bar when implemented
        pass

    def _show_archive(self):
        QMessageBox.information(self, 'Show Archive', 'Archive (coming soon)')

    def _refresh_data_library(self):
        self._rebuild_trace_panel()

    def _open_sp_avg_dialog(self, edit: bool = False):
        cfg = getattr(self, '_sp_avg_config', None)
        sp_names = [e.get('name', f'Engine {i+1}')
                    for i, e in enumerate(self._sp_engines)]
        dlg = TFAvgDialog(sp_names, parent=self,
                          config=cfg, show_coh_weighted=False)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._sp_avg_config = dlg.result_config()

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
        """Save spectrum trace with name / description / color dialog."""
        lev_x = self.canvas_spec._last_lx
        if lev_x is None:
            self.sb.showMessage('⚠  No spectrum data — start the measurement first.', 3000)
            return
        lev_y = self.canvas_spec._last_ly
        if lev_y is None:
            lev_y = lev_x

        if len(self._sp_traces) >= MAX_TRACES:
            self.sb.showMessage(f'⚠  Maximum {MAX_TRACES} traces. Delete one before saving.', 4000)
            return

        default_name  = f'Sp-{len(self._sp_traces) + 1}'
        default_color = TRACE_PALETTE[len(self._sp_traces) % len(TRACE_PALETTE)]
        _picked_color = [default_color]

        dlg = QDialog(self)
        dlg.setWindowTitle('Save Spectrum Trace')
        dlg.setModal(True)
        dlg.setFixedSize(400, 190)
        dlg.setStyleSheet(
            'QDialog{background:#1a1a1a;}'
            f'QLabel{{color:{TEXT_HI};font-size:12px;background:transparent;}}'
            'QLineEdit{background:#e8e8e8;color:#111;border:none;'
            'border-radius:3px;padding:4px 8px;font-size:12px;}'
            f'QPushButton#ok_btn{{background:#2a3a2a;color:{TEXT_HI};'
            'border:1px solid #4a6a4a;border-radius:4px;'
            'font-size:11px;padding:5px 22px;}}'
            f'QPushButton#ok_btn:hover{{background:#3a4a3a;border-color:#6a9a6a;}}'
            f'QPushButton#cancel_btn{{background:#2e2e2e;color:{TEXT_HI};'
            'border:1px solid #444;border-radius:4px;'
            'font-size:11px;padding:5px 22px;}}'
            f'QPushButton#cancel_btn:hover{{background:#3a3a3a;border-color:#666;}}')

        root = QVBoxLayout(dlg)
        root.setContentsMargins(20, 18, 20, 14)
        root.setSpacing(10)

        row_name = QHBoxLayout()
        row_name.addWidget(QLabel('Name:'))
        edit = QLineEdit(default_name)
        edit.selectAll()
        row_name.addWidget(edit, stretch=1)
        root.addLayout(row_name)

        row_desc = QHBoxLayout()
        row_desc.addWidget(QLabel('Description:'))
        edit_desc = QLineEdit()
        edit_desc.setPlaceholderText('Optional notes…')
        row_desc.addWidget(edit_desc, stretch=1)
        root.addLayout(row_desc)

        row_color = QHBoxLayout()
        row_color.addWidget(QLabel('Color:'))
        btn_color = QPushButton()
        btn_color.setFixedSize(34, 24)
        btn_color.setStyleSheet(
            f'background:{default_color};border:1px solid #666;border-radius:3px;')

        def _pick_color():
            from PyQt6.QtWidgets import QColorDialog
            from PyQt6.QtGui import QColor as _QC
            c = QColorDialog.getColor(_QC(_picked_color[0]), dlg, 'Trace Color')
            if c.isValid():
                _picked_color[0] = c.name()
                btn_color.setStyleSheet(
                    f'background:{c.name()};border:1px solid #666;border-radius:3px;')

        btn_color.clicked.connect(_pick_color)
        row_color.addWidget(btn_color)
        row_color.addStretch()
        root.addLayout(row_color)

        root.addStretch()

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_ok     = QPushButton('Save');   btn_ok.setObjectName('ok_btn')
        btn_cancel = QPushButton('Cancel'); btn_cancel.setObjectName('cancel_btn')
        btn_ok.clicked.connect(dlg.accept)
        btn_cancel.clicked.connect(dlg.reject)
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_ok)
        root.addLayout(btn_row)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        name  = edit.text().strip() or default_name
        desc  = edit_desc.text().strip()
        color = _picked_color[0]

        self.canvas_spec.store_trace(lev_x.copy(), lev_y.copy(), color)
        idx = len(self._sp_traces)
        self._sp_traces.append({
            'lev_x': lev_x.copy(), 'lev_y': lev_y.copy(),
            'color': color, 'name': name, 'desc': desc,
        })
        row_w = TraceRow(
            idx=idx, name=name, color=color,
            on_vis=lambda i, c: self.canvas_spec.set_trace_visible(i, c),
            on_del=self._delete_sp_trace,
        )
        row_w.setToolTip(desc)
        self._sp_trace_rows_layout.addWidget(row_w)
        self.sb.showMessage(f'✓  Spectrum trace saved: {name}', 3000)

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
        # ── Preguntar si guardar workspace antes de cerrar ────────────
        dlg = QMessageBox(self)
        dlg.setWindowTitle('Coherence')
        dlg.setText('Do you want to save the workspace before closing?')
        dlg.setIcon(QMessageBox.Icon.Question)
        dlg.setStandardButtons(
            QMessageBox.StandardButton.Save   |
            QMessageBox.StandardButton.Discard |
            QMessageBox.StandardButton.Cancel
        )
        dlg.setDefaultButton(QMessageBox.StandardButton.Save)
        dlg.button(QMessageBox.StandardButton.Save).setText('Save')
        dlg.button(QMessageBox.StandardButton.Discard).setText("Don't Save")
        dlg.button(QMessageBox.StandardButton.Cancel).setText('Cancel')
        dlg.setStyleSheet(
            f'QMessageBox{{background:{BG_APP};color:{TEXT_HI};font-size:12px;}}'
            f'QPushButton{{background:#1e221e;color:{TEXT_HI};border:1px solid #3a3a3a;'
            f'border-radius:4px;padding:5px 16px;font-size:11px;min-width:80px;}}'
            f'QPushButton:hover{{border-color:{GREEN};color:{GREEN};}}'
            f'QPushButton:default{{border-color:{GREEN};}}'
        )
        result = dlg.exec()

        if result == QMessageBox.StandardButton.Cancel:
            event.ignore()
            return
        if result == QMessageBox.StandardButton.Save:
            self._save_workspace_explicit()

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
