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
    QTabWidget, QStatusBar, QDoubleSpinBox,
    QProgressBar, QFileDialog, QDialog,
    QDialogButtonBox, QFormLayout, QScrollArea,
    QMessageBox, QCheckBox,
)
from PyQt6.QtCore  import Qt, QTimer, QDateTime
from PyQt6.QtGui   import QFont, QPalette, QColor, QKeySequence
from PyQt6.QtWidgets import QMenuBar

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

from .dsp   import compute_analysis, setup_smaart_axis, find_delay, welch_spectra
from .dsp   import BLUE, GREEN, ORANGE, PURPLE, GRAY, CYAN, RED
from .audio import AudioEngine


# ── Paleta ────────────────────────────────────────────────────────────
BG_APP      = '#111111'
BG_PANEL    = '#161616'
BG_SETTINGS = '#1e2020'
BG_PLOT     = '#0e0e0e'
BG_TOOLBAR  = '#141414'
ACCENT      = '#00bcd4'
TEXT_HI     = '#eeeeee'
TEXT_MID    = '#9e9e9e'
TEXT_DIM    = '#4a4a4a'
BORDER      = '#272727'
COH_COLOR   = '#ff5252'   # rojo para coherencia
IR_COLOR    = '#80cbc4'   # verde agua para IR

# ── Paleta de trazas almacenadas ──────────────────────────────────────
TRACE_PALETTE = [
    '#4fc3f7',   # 0 cyan claro
    '#ef9a9a',   # 1 rojo suave
    '#a5d6a7',   # 2 verde suave
    '#fff176',   # 3 amarillo
    '#ce93d8',   # 4 violeta
    '#80deea',   # 5 turquesa
    '#ffcc80',   # 6 naranja
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


# ── Canvas de Medición (IR / TF+Coh / Phase) ─────────────────────────

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
        # Listas paralelas de líneas matplotlib para trazas almacenadas
        self._trace_tf_lines: List = []
        self._trace_ph_lines: List = []
        self._trace_ir_lines: List = []
        self._build()

    def _build(self):
        from matplotlib.transforms import blended_transform_factory

        self._delay_ref_ms = 0.0

        gs = gridspec.GridSpec(
            3, 1, figure=self.fig,
            hspace=0.32,
            left=0.06, right=0.93,
            top=0.99, bottom=0.08,
            height_ratios=[0.5, 1.5, 1.0]
        )

        # ── IR ──
        self.ax_ir = self.fig.add_subplot(gs[0])
        self.ax_ir.set_facecolor(BG_PLOT)
        self.ax_ir.set_xlim(-100, 100)   # siempre fijo, simétrico, NUNCA se mueve
        self.ax_ir.set_ylim(-1.1, 1.1)
        self.ax_ir.axhline(0, color=BORDER, lw=0.6)
        self.ax_ir.axvline(0, color='#2a2a2a', lw=0.7, ls='--')   # referencia fija 0 ms
        self.ax_ir.set_ylabel('amp', fontsize=7, color=TEXT_MID)
        self.ax_ir.set_title('IR', fontsize=8,
                              fontfamily='monospace', color=TEXT_MID, loc='left', pad=2)
        self._style_ax(self.ax_ir, log=False)
        # Eje X en ms — ticks cada 50 ms (mayor) y 25 ms (menor)
        self.ax_ir.xaxis.set_major_locator(ticker.MultipleLocator(50))
        self.ax_ir.xaxis.set_minor_locator(ticker.MultipleLocator(25))
        self.ax_ir.tick_params(axis='x', labelsize=7, colors=TEXT_MID)
        self.ax_ir.set_xlabel('ms', fontsize=7, color=TEXT_MID, labelpad=1)

        self.line_ir, = self.ax_ir.plot([], [], color=IR_COLOR, lw=1.0, alpha=0.85)
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
        self.ax_tf.set_ylabel('dB', fontsize=7, color=TEXT_MID)
        self.ax_tf.set_title('TF + COH', fontsize=8,
                              fontfamily='monospace', color=TEXT_MID, loc='left', pad=2)
        self.ax_tf.axhline(0, color='#2a2a2a', lw=0.7, ls='--')

        # Alinear 0 de coherencia con 0 dB de TF
        # Fórmula: coh_min = -(tf_zero_pos / (1 - tf_zero_pos)) * coh_max
        # Con TF ylim(-30,12): pos_0dB = 30/42; coh_min = -2.5 * coh_max
        self._tf_ymin, self._tf_ymax = -30, 30
        self._coh_ymax = 1.05
        self.ax_coh.set_ylim(*self._coh_aligned_lim())
        self.ax_coh.set_ylabel('γ²', fontsize=8, color=COH_COLOR)
        self.ax_coh.tick_params(axis='y', colors=COH_COLOR, labelsize=7)
        self.ax_coh.set_yticks([0.0, 0.5, 1.0])
        self.ax_coh.spines['right'].set_color(COH_COLOR)
        self.ax_coh.spines['right'].set_alpha(0.5)
        for sp in ('top','left','bottom'):
            self.ax_coh.spines[sp].set_visible(False)

        # Línea referencia coherencia 0.9
        self.ax_coh.axhline(0.9, color=COH_COLOR, lw=0.7, ls=':', alpha=0.4)

        f0 = np.array([20.0, 20000.0])
        self.line_tf,     = self.ax_tf.semilogx(f0, [0, 0], color=BLUE,   lw=1.5, label='CH1')
        self.line_tf2,    = self.ax_tf.semilogx([], [],     color=GREEN,  lw=1.2, alpha=0.8, ls='-', label='CH2')
        self.line_tf_avg, = self.ax_tf.semilogx([], [],     color=ORANGE, lw=1.5, alpha=0.9, ls='-', label='AVG')
        self.line_coh,    = self.ax_coh.semilogx(f0, [0, 0], color=COH_COLOR, lw=1.0, alpha=0.7)
        self._coh_fill = self.ax_coh.fill_between(f0, [0, 0],
                                                   color=COH_COLOR, alpha=0.08)

        # ── Phase ──  (fase envuelta −180…+180, estilo SMAART)
        self.ax_ph = self.fig.add_subplot(gs[2])
        self.ax_ph.set_facecolor(BG_PLOT)
        self.ax_ph.set_ylim(-185, 185)
        setup_smaart_axis(self.ax_ph, bg=BG_PLOT)
        self.ax_ph.set_ylabel('deg', fontsize=7, color=TEXT_MID)
        self.ax_ph.set_title('PHASE', fontsize=8,
                              fontfamily='monospace', color=TEXT_MID, loc='left', pad=2)
        self.ax_ph.axhline(  0, color='#2a2a2a', lw=0.7, ls='--')
        self.ax_ph.axhline( 90, color='#252525', lw=0.4, ls=':')
        self.ax_ph.axhline(-90, color='#252525', lw=0.4, ls=':')
        self.ax_ph.axhline( 180, color='#252525', lw=0.5, ls=':')
        self.ax_ph.axhline(-180, color='#252525', lw=0.5, ls=':')
        self.ax_ph.set_yticks([-180, -90, 0, 90, 180])
        self.ax_ph.tick_params(axis='y', labelsize=7, colors=TEXT_MID)

        self.line_ph,     = self.ax_ph.semilogx(f0, [0, 0], color=BLUE,   lw=1.0, alpha=0.8)
        self.line_ph2,    = self.ax_ph.semilogx([], [],     color=GREEN,  lw=0.9, alpha=0.75, ls='-')
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
        ax.grid(True, which='major', lw=0.4, color='#252525')
        ax.grid(True, which='minor', lw=0.25, color='#1a1a1a', ls=':')
        for sp in ax.spines.values():
            sp.set_color(BORDER)
            sp.set_linewidth(0.6)

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

        # IR — eje X siempre -100…+100 ms (NUNCA cambia).
        # _delay_ref_ms desplaza los datos: pico queda en 0 ms cuando está seteado.
        if ir is not None and len(ir) > 0:
            t_abs = np.arange(len(ir)) / 48000.0 * 1000.0
            t_rel = t_abs - self._delay_ref_ms
            xmin, xmax = self.ax_ir.get_xlim()
            mask_t = (t_rel >= xmin) & (t_rel <= xmax)
            if mask_t.sum() > 2:
                self.line_ir.set_data(t_rel[mask_t], ir[mask_t])
            peak_t_rel = t_rel[int(np.argmax(np.abs(ir)))]
            self.line_ir_peak.set_xdata([peak_t_rel, peak_t_rel])

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
        Fija el retardo de referencia (delay finder).
        NO mueve el eje — siempre -100…+100 ms.
        Solo desplaza los DATOS para que el pico quede en 0 ms,
        y muestra '(X.XX ms)' justo encima del marcador naranja.
        """
        self._delay_ref_ms = delay_ms
        if delay_ms != 0.0:
            self.txt_ir_delay.set_text(f'({delay_ms:.2f} ms)')
        else:
            self.txt_ir_delay.set_text('')
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
        s_g2 = f'{g2:.3f}'    if g2 is not None else '—'
        return f'TF  │  {fs}  │  {s_db}  │  γ²: {s_g2}'

    def _fmt_ph(self, f_hz):
        ph = self._interp(f_hz, self._last_freqs, self._last_phase_deg)
        g2 = self._interp(f_hz, self._last_freqs, self._last_gamma2)
        fs = self._fmt_freq(f_hz)
        # Wrap for display
        s_ph = f'{((ph+180)%360)-180:.1f}°' if ph is not None else '—'
        s_g2 = f'{g2:.3f}'                  if g2 is not None else '—'
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
        self._build()

    def _build(self):
        self.ax = self.fig.add_subplot(111, facecolor=BG_PLOT)
        self.fig.subplots_adjust(left=0.07, right=0.97, top=0.95, bottom=0.09)

        setup_smaart_axis(self.ax, bg=BG_PLOT)
        self.ax.set_ylim(-80, 6)
        self.ax.set_ylabel('dBFS', fontsize=8, color=TEXT_MID)
        self.ax.set_title('SPECTRUM  1/3-oct CPB', fontsize=9,
                          fontfamily='monospace', color=TEXT_MID, loc='left')
        self.ax.axhline(-18, color='#252525', lw=0.5, ls=':', alpha=0.6)
        self.ax.axhline(-6,  color='#252525', lw=0.5, ls=':', alpha=0.6)
        self.ax.tick_params(axis='y', labelsize=7, colors=TEXT_MID)
        for sp in self.ax.spines.values():
            sp.set_color(BORDER)

        # Líneas conectadas en frecuencias centrales
        self.line_x,     = self.ax.semilogx([], [], color=BLUE,   lw=1.6, alpha=0.9,  label='REF')
        self.line_y,     = self.ax.semilogx([], [], color=ORANGE, lw=1.3, alpha=0.85, label='CH1')
        self.line_y2,    = self.ax.semilogx([], [], color=GREEN,  lw=1.2, alpha=0.80, label='CH2', ls='-')
        self.line_y_avg, = self.ax.semilogx([], [], color=CYAN,   lw=1.5, alpha=0.90, label='AVG', ls='--')
        self._fill_x = self.ax.fill_between([], [], -80, color=BLUE,   alpha=0.08)
        self._fill_y = self.ax.fill_between([], [], -80, color=ORANGE, alpha=0.06)

        # Crosshair cursor
        _ck = dict(color=TEXT_MID, lw=0.7, ls=':', alpha=0.0, zorder=20)
        self._cxh, = self.ax.plot([], [], **_ck)
        self._cyh, = self.ax.plot([], [], **_ck)

        self.ax.legend(fontsize=7, facecolor='#1a1a1a', edgecolor=BORDER,
                       labelcolor=TEXT_MID, loc='upper right', ncol=2)
        self.fig.patch.set_facecolor(BG_PANEL)
        self.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.draw()

    # ── Análisis CPB ─────────────────────────────────────────────────

    @staticmethod
    def _cpb(freqs, G):
        """
        CPB 1/3-oct con bordes contiguos (media geométrica entre centros).

        Para bandas con ≥2 bins FFT:  suma directa (precisa).
        Para bandas con 0-1 bins  :  interpola Gxx en fc × (bw/df).
            bw/df = número efectivo de bins → preserva la planitud
            del ruido rosa (∝ 1/f) incluso con FFT gruesa.

        Con nperseg=4096, Δf≈11.7 Hz; bandas <63 Hz tienen ≤1 bin,
        pero la interpolación las estima correctamente → rango <3 dB.
        """
        eps  = 1e-12
        df   = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
        levels = np.empty(len(_ISO_CENTERS))

        for i, fc in enumerate(_ISO_CENTERS):
            fl, fh = float(_ISO_EDGES[i]), float(_ISO_EDGES[i + 1])
            bw     = fh - fl
            mask   = (freqs >= fl) & (freqs < fh)
            n_bins = int(mask.sum())

            if n_bins >= 2:
                power = float(np.sum(G[mask]))
            else:
                # 0 ó 1 bin: interpolar PSD en fc, escalar por (bw/df)
                g_fc  = float(np.interp(fc, freqs, G))
                power = g_fc * (bw / df)

            levels[i] = 10.0 * np.log10(power + eps)

        return levels

    # ── Update ────────────────────────────────────────────────────────

    def update_plots(self, freqs, Gxx, Gyy):
        lev_x = self._cpb(freqs, Gxx)
        lev_y = self._cpb(freqs, Gyy)

        self._last_lx  = lev_x
        self._last_ly  = lev_y
        self._last_Gxx = Gxx
        self._last_Gyy = Gyy

        fc = _ISO_CENTERS
        self.line_x.set_data(fc, lev_x)
        self.line_y.set_data(fc, lev_y)

        self._fill_x.remove()
        self._fill_y.remove()
        self._fill_x = self.ax.fill_between(fc, lev_x, -80, color=BLUE,   alpha=0.08)
        self._fill_y = self.ax.fill_between(fc, lev_y, -80, color=ORANGE, alpha=0.06)

        self.draw_idle()

    def update_ch2_avg(self, lev_y2, lev_avg):
        """Actualiza líneas CH2 y AVG del spectrum."""
        fc = _ISO_CENTERS
        if lev_y2 is not None:
            self.line_y2.set_data(fc, lev_y2)
        else:
            self.line_y2.set_data([], [])
        if lev_avg is not None:
            self.line_y_avg.set_data(fc, lev_avg)
        else:
            self.line_y_avg.set_data([], [])

    # ── Trazas almacenadas ────────────────────────────────────────────

    def store_trace(self, lev_x, lev_y, color):
        """Dibuja trazas REF (--) y MED (:) como referencia estática."""
        fc = _ISO_CENTERS
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
        self.line_x.set_data([], [])
        self.line_y.set_data([], [])
        self._fill_x.remove()
        self._fill_y.remove()
        self._fill_x = self.ax.fill_between([], [], -80, color=BLUE,   alpha=0.08)
        self._fill_y = self.ax.fill_between([], [], -80, color=ORANGE, alpha=0.06)
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

        if self._last_lx is not None:
            lx = float(np.interp(np.log10(x),
                                  np.log10(_ISO_CENTERS), self._last_lx))
            ly = float(np.interp(np.log10(x),
                                  np.log10(_ISO_CENTERS), self._last_ly))
            info = (f'SP  │  {fmt_freq(x)}  │  '
                    f'REF: {lx:.1f} dBFS  │  MED: {ly:.1f} dBFS')
        else:
            info = '—'

        if self.on_cursor_update:
            self.on_cursor_update(info)
        self.draw_idle()


# ── Canvas de Espectrograma ───────────────────────────────────────────

class SpectrogramCanvas(FigureCanvas):
    """
    Espectrograma scrolling: frecuencia (eje Y, log) × tiempo (eje X).
    Color = nivel en dBFS.  Actualiza solo el array de datos — sin redibujar
    los ejes en cada frame para mantener fluidez.
    """

    N_TIME = 180   # columnas de tiempo (frames)

    def __init__(self):
        self.fig = Figure(facecolor=BG_PANEL)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._data  = None
        self._freqs = None
        self._img   = None
        self._built = False
        self._build()

    def _build(self):
        self.ax = self.fig.add_subplot(111, facecolor=BG_PLOT)
        self.fig.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.09)
        self.ax.set_title('SPECTROGRAM', fontsize=9,
                          fontfamily='monospace', color=TEXT_MID, loc='left')
        self.ax.set_xlabel('← tiempo', fontsize=8, color=TEXT_MID)
        self.ax.set_ylabel('Hz', fontsize=8, color=TEXT_MID)
        for sp in self.ax.spines.values():
            sp.set_color(BORDER)
        self.ax.tick_params(colors=TEXT_MID, labelsize=7)
        self.fig.patch.set_facecolor(BG_PANEL)
        self.draw()

    def update_spectrogram(self, freqs, Gxx):
        """Recibe PSD lineal, convierte a dB, desplaza buffer y redibuja."""
        mask = (freqs >= 20) & (freqs <= 20000)
        f    = freqs[mask]
        col  = 10.0 * np.log10(Gxx[mask] + 1e-12)

        if self._data is None or self._data.shape[0] != len(f):
            self._data  = np.full((len(f), self.N_TIME), -80.0, dtype=np.float32)
            self._freqs = f
            self._img   = None
            self._built = False

        # Scroll: desplaza columnas a la izquierda, agrega nueva al final
        self._data = np.roll(self._data, -1, axis=1)
        self._data[:, -1] = col

        if not self._built or self._img is None:
            self.ax.cla()
            self.ax.set_facecolor(BG_PLOT)
            self.ax.set_title('SPECTROGRAM', fontsize=9,
                              fontfamily='monospace', color=TEXT_MID, loc='left')
            self.ax.set_xlabel('← tiempo', fontsize=8, color=TEXT_MID)
            self.ax.set_ylabel('Hz', fontsize=8, color=TEXT_MID)

            # Construir eje Y con frecuencias reales del FFT
            self._img = self.ax.imshow(
                self._data,
                aspect='auto', origin='lower',
                extent=[0, self.N_TIME, 0, len(f) - 1],
                cmap='inferno', vmin=-80, vmax=-10,
                interpolation='nearest',
            )
            # Ticks en frecuencias ISO
            iso_ticks = [20, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000, 20000]
            ytick_pos = [float(np.searchsorted(f, ft)) for ft in iso_ticks if ft <= f[-1]]
            ytick_lbl = [fmt_freq(ft) for ft in iso_ticks if ft <= f[-1]]
            self.ax.set_yticks(ytick_pos)
            self.ax.set_yticklabels(ytick_lbl, fontsize=6)
            self.ax.set_xticks([])
            for sp in self.ax.spines.values():
                sp.set_color(BORDER)
            self.ax.tick_params(colors=TEXT_MID, labelsize=6)

            # Colorbar
            try:
                self._cbar = self.fig.colorbar(self._img, ax=self.ax, pad=0.01)
                self._cbar.ax.tick_params(colors=TEXT_MID, labelsize=6)
                self._cbar.set_label('dBFS', color=TEXT_MID, fontsize=7)
            except Exception:
                pass

            self.fig.subplots_adjust(left=0.08, right=0.88, top=0.95, bottom=0.06)
            self._built = True
        else:
            self._img.set_data(self._data)

        self.draw_idle()

    def clear(self):
        self._data  = None
        self._img   = None
        self._built = False
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


# ── Ventana principal ─────────────────────────────────────────────────

class MainWindow(QMainWindow):

    UPDATE_MS = 120

    def __init__(self):
        super().__init__()
        self.engine           = AudioEngine()
        self._frozen          = False
        self._delay_comp_ms   = 0.0    # retardo compensado (delay finder → fase)
        self._traces: List[TraceData] = []
        self._trace_color_idx = 0      # cicla por TRACE_PALETTE
        self._mic_cal         = None   # tuple (freqs, dB_correction) o None
        self._mic_cal_name    = ''     # nombre del archivo cargado
        self._show_ch2        = True   # mostrar 2do canal en gráficas
        self._show_avg        = True   # mostrar promedio

        self.setWindowTitle('Coherence  v0.2')
        self.setMinimumSize(1100, 720)
        self.resize(1380, 820)
        self.setStyleSheet(QSS)

        self._build_central()
        self._build_menubar()
        self._build_statusbar()
        self._populate_devices()

        self.timer = QTimer(self)
        self.timer.setInterval(self.UPDATE_MS)
        self.timer.timeout.connect(self._refresh)

        self._set_stopped()

    # ── Central widget ────────────────────────────────────────────────

    def _build_central(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Panel izquierdo: guardar TXT (colapsable) ──
        root.addWidget(self._build_collapsible_save_panel())

        # ── Centro: info bar + tabs ──
        center = QWidget()
        cv = QVBoxLayout(center)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(0)
        cv.addWidget(self._build_info_bar())

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.canvas_meas = MeasurementCanvas()
        self.canvas_meas.on_cursor_update = self._update_cursor
        self.canvas_spec = SpectrumCanvas()
        self.canvas_spec.on_cursor_update = self._update_cursor
        self.canvas_sgram = SpectrogramCanvas()
        self.tabs.addTab(self.canvas_meas,  'TRANSFER FUNCTION')
        self.tabs.addTab(self.canvas_spec,  'SPECTRUM')
        self.tabs.addTab(self.canvas_sgram, 'SPECTROGRAM')
        cv.addWidget(self.tabs, stretch=1)
        root.addWidget(center, stretch=1)

        # ── Derecha: panel settings colapsable ──
        root.addWidget(self._build_collapsible_settings())

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
        panel = QWidget()
        panel.setObjectName('save_panel')
        panel.setFixedWidth(155)
        v = QVBoxLayout(panel)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(3)

        # Título del panel
        title = QLabel('SAVE DATA')
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            f'color:{ACCENT};font-size:9px;letter-spacing:3px;font-weight:bold;'
            f'padding:4px 0 5px 0;border-bottom:1px solid {BORDER};background:transparent;'
        )
        v.addWidget(title)

        for label, tip, fn in [
            ('SAVE  IR',       'Guardar Impulse Response como .txt',  self._save_ir_txt),
            ('SAVE  TF',       'Guardar Transfer Function como .txt', self._save_tf_txt),
            ('SAVE  PHASE',    'Guardar Phase como .txt',             self._save_ph_txt),
            ('SAVE  SPECTRUM', 'Guardar Spectrum como .txt',          self._save_sp_txt),
        ]:
            btn = QPushButton(label)
            btn.setToolTip(tip)
            btn.setMinimumHeight(30)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn.clicked.connect(fn)
            v.addWidget(btn)

        # ── Sección TRAZAS ──────────────────────────────────────────
        v.addWidget(sep())

        traces_title = QLabel('TRAZAS')
        traces_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        traces_title.setStyleSheet(
            f'color:{ACCENT};font-size:9px;letter-spacing:3px;font-weight:bold;'
            f'padding:3px 0;background:transparent;'
        )
        v.addWidget(traces_title)

        btn_save_trace = QPushButton('📌  GUARDAR TRAZA')
        btn_save_trace.setToolTip(
            f'Guarda la medición actual como traza de referencia (máx {MAX_TRACES})'
        )
        btn_save_trace.setMinimumHeight(30)
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
        """Panel izquierdo colapsable (espejo del de settings)."""
        container = QWidget()
        h = QHBoxLayout(container)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)

        self._save_widget = self._build_save_panel()
        h.addWidget(self._save_widget)

        self.btn_toggle_save = QPushButton('▶')
        self.btn_toggle_save.setObjectName('btn_toggle_panel')
        self.btn_toggle_save.setFixedWidth(14)
        self.btn_toggle_save.clicked.connect(self._on_toggle_save)
        h.addWidget(self.btn_toggle_save)
        return container

    def _on_toggle_save(self):
        visible = self._save_widget.isVisible()
        self._save_widget.setVisible(not visible)
        self.btn_toggle_save.setText('◀' if visible else '▶')

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
        # Panel directo — sin scroll
        panel = QWidget()
        panel.setObjectName('settings_panel')
        panel.setFixedWidth(155)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(3)

        # Título del panel
        title_s = QLabel('SETTINGS')
        title_s.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_s.setStyleSheet(
            f'color:{ACCENT};font-size:9px;letter-spacing:3px;font-weight:bold;'
            f'padding:4px 0 5px 0;border-bottom:1px solid {BORDER};background:transparent;'
        )
        layout.addWidget(title_s)

        # ════════════════════════════════════════
        # 1. SPL / LEVEL
        # ════════════════════════════════════════
        grp_spl = QGroupBox('SPL  /  LEVEL')
        gs = QGridLayout(grp_spl)
        gs.setContentsMargins(5, 4, 5, 4)
        gs.setSpacing(2)
        gs.setColumnStretch(1, 1)

        gs.addWidget(lbl('MED', color='#ffb74d', size=9), 0, 0)
        self.lbl_spl_meas = QLabel('—')
        self.lbl_spl_meas.setStyleSheet('color:#ffb74d;font-size:13px;font-weight:bold;')
        gs.addWidget(self.lbl_spl_meas, 0, 1)
        self.bar_spl_meas = QProgressBar()
        self.bar_spl_meas.setRange(0, 100)
        self.bar_spl_meas.setValue(0)
        self.bar_spl_meas.setTextVisible(False)
        self.bar_spl_meas.setFixedHeight(5)
        gs.addWidget(self.bar_spl_meas, 1, 0, 1, 2)

        gs.addWidget(lbl('REF', color='#80cbc4', size=8), 2, 0)
        self.lbl_spl_ref = QLabel('—')
        self.lbl_spl_ref.setStyleSheet('color:#80cbc4;font-size:10px;font-weight:bold;')
        gs.addWidget(self.lbl_spl_ref, 2, 1)
        self.bar_spl_ref = QProgressBar()
        self.bar_spl_ref.setRange(0, 100)
        self.bar_spl_ref.setValue(0)
        self.bar_spl_ref.setTextVisible(False)
        self.bar_spl_ref.setFixedHeight(4)
        gs.addWidget(self.bar_spl_ref, 3, 0, 1, 2)
        layout.addWidget(grp_spl)

        # ════════════════════════════════════════
        # 2. DSP
        # ════════════════════════════════════════
        grp_dsp = QGroupBox('DSP')
        gd = QGridLayout(grp_dsp)
        gd.setContentsMargins(5, 4, 5, 4)
        gd.setSpacing(2)
        gd.setColumnStretch(1, 1)

        gd.addWidget(lbl('AVERAGES'), 0, 0)
        self.cmb_avg = QComboBox()
        self._avg_values = [1, 4, 8, 16, 32]
        for v in self._avg_values:
            self.cmb_avg.addItem(str(v))
        self.cmb_avg.setCurrentIndex(2)   # default 8
        self.cmb_avg.currentIndexChanged.connect(
            lambda i: setattr(self.engine, 'n_averages', self._avg_values[i]))
        gd.addWidget(self.cmb_avg, 0, 1)

        gd.addWidget(lbl('COH THRESH'), 1, 0)
        self.spn_thresh = QDoubleSpinBox()
        self.spn_thresh.setRange(0.0, 1.0)
        self.spn_thresh.setSingleStep(0.05)
        self.spn_thresh.setValue(0.10)
        self.spn_thresh.setDecimals(2)
        gd.addWidget(self.spn_thresh, 1, 1)

        gd.addWidget(lbl('SMOOTH'), 2, 0)
        self.cmb_smooth = QComboBox()
        for slbl in ['OFF', '1/3', '1/6', '1/12', '1/24']:
            self.cmb_smooth.addItem(slbl)
        self.cmb_smooth.setCurrentIndex(3)
        self._smooth_values = [0, 3, 6, 12, 24]
        gd.addWidget(self.cmb_smooth, 2, 1)
        layout.addWidget(grp_dsp)

        # ════════════════════════════════════════
        # 3. DISPLAY
        # ════════════════════════════════════════
        grp_disp = QGroupBox('DISPLAY')
        gdp = QGridLayout(grp_disp)
        gdp.setContentsMargins(5, 4, 5, 4)
        gdp.setSpacing(2)
        gdp.setColumnStretch(1, 1)

        gdp.addWidget(lbl('IR  ms'), 0, 0)
        self.spn_ir_ms = QSpinBox()
        self.spn_ir_ms.setRange(10, 1000)
        self.spn_ir_ms.setSingleStep(10)
        self.spn_ir_ms.setValue(200)
        self.spn_ir_ms.valueChanged.connect(lambda v: self.canvas_meas.update_ir_range(v))
        gdp.addWidget(self.spn_ir_ms, 0, 1)

        gdp.addWidget(lbl('TF MIN'), 1, 0)
        self.spn_tf_min = QSpinBox()
        self.spn_tf_min.setRange(-60, 0)
        self.spn_tf_min.setValue(-30)
        self.spn_tf_min.valueChanged.connect(
            lambda v: (self.canvas_meas.ax_tf.set_ylim(v, self.spn_tf_max.value()),
                       self.canvas_meas.align_coh_axis(v, self.spn_tf_max.value()),
                       self.canvas_meas.draw_idle()))
        gdp.addWidget(self.spn_tf_min, 1, 1)

        gdp.addWidget(lbl('TF MAX'), 2, 0)
        self.spn_tf_max = QSpinBox()
        self.spn_tf_max.setRange(0, 30)
        self.spn_tf_max.setValue(30)
        self.spn_tf_max.valueChanged.connect(
            lambda v: (self.canvas_meas.ax_tf.set_ylim(self.spn_tf_min.value(), v),
                       self.canvas_meas.align_coh_axis(self.spn_tf_min.value(), v),
                       self.canvas_meas.draw_idle()))
        gdp.addWidget(self.spn_tf_max, 2, 1)
        layout.addWidget(grp_disp)

        # ════════════════════════════════════════
        # 4. ROUTING
        # ════════════════════════════════════════
        grp_rt = QGroupBox('ROUTING')
        gr = QGridLayout(grp_rt)
        gr.setContentsMargins(5, 4, 5, 4)
        gr.setSpacing(2)
        gr.setColumnStretch(1, 1)

        gr.addWidget(lbl('DEV IN'), 0, 0)
        self.cmb_dev_in = QComboBox()
        gr.addWidget(self.cmb_dev_in, 0, 1)

        gr.addWidget(lbl('DEV OUT'), 1, 0)
        self.cmb_dev_out = QComboBox()
        gr.addWidget(self.cmb_dev_out, 1, 1)

        gr.addWidget(lbl('CH MED 1'), 2, 0)
        self.spn_ch_meas = QSpinBox()
        self.spn_ch_meas.setRange(1, 20)
        self.spn_ch_meas.setValue(1)
        self.spn_ch_meas.valueChanged.connect(self._on_channels)
        gr.addWidget(self.spn_ch_meas, 2, 1)

        gr.addWidget(lbl('CH MED 2'), 3, 0)
        self.spn_ch_meas2 = QSpinBox()
        self.spn_ch_meas2.setRange(1, 20)
        self.spn_ch_meas2.setValue(2)
        self.spn_ch_meas2.valueChanged.connect(self._on_channels)
        gr.addWidget(self.spn_ch_meas2, 3, 1)

        gr.addWidget(lbl('CH REF'), 4, 0)
        self.spn_ch_ref = QSpinBox()
        self.spn_ch_ref.setRange(1, 20)
        self.spn_ch_ref.setValue(4)
        self.spn_ch_ref.valueChanged.connect(self._on_channels)
        gr.addWidget(self.spn_ch_ref, 4, 1)

        # CH2 / AVG toggles
        row_ch2 = QHBoxLayout()
        row_ch2.setSpacing(3)
        self.chk_ch2 = QCheckBox('CH2')
        self.chk_ch2.setChecked(True)
        self.chk_ch2.stateChanged.connect(
            lambda s: setattr(self, '_show_ch2', bool(s)))
        self.chk_avg = QCheckBox('AVG')
        self.chk_avg.setChecked(True)
        self.chk_avg.stateChanged.connect(
            lambda s: setattr(self, '_show_avg', bool(s)))
        row_ch2.addWidget(self.chk_ch2)
        row_ch2.addWidget(self.chk_avg)
        gr.addLayout(row_ch2, 5, 0, 1, 2)
        layout.addWidget(grp_rt)

        # ════════════════════════════════════════
        # 5. DELAY FINDER
        # ════════════════════════════════════════
        grp_delay = QGroupBox('DELAY FINDER')
        gly = QVBoxLayout(grp_delay)
        gly.setContentsMargins(5, 4, 5, 4)
        gly.setSpacing(3)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(3)
        self.btn_delay_p = QPushButton('▶  FIND')
        self.btn_delay_p.setObjectName('btn_delay')
        self.btn_delay_p.clicked.connect(self._on_find_delay)
        btn_row.addWidget(self.btn_delay_p, stretch=3)
        self.btn_delay_reset = QPushButton('RST')
        self.btn_delay_reset.setToolTip('Reset compensación de fase a 0')
        self.btn_delay_reset.clicked.connect(self._on_delay_reset)
        btn_row.addWidget(self.btn_delay_reset, stretch=1)
        gly.addLayout(btn_row)

        self.lbl_delay_full = QLabel('—')
        self.lbl_delay_full.setObjectName('lbl_delay_val')
        self.lbl_delay_full.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_delay_full.setWordWrap(True)
        self.lbl_delay_full.setStyleSheet(f'color:{ACCENT};font-size:11px;font-weight:bold;padding:1px 0;')
        gly.addWidget(self.lbl_delay_full)
        layout.addWidget(grp_delay)

        # ════════════════════════════════════════
        # 6. TRANSPORT
        # ════════════════════════════════════════
        grp_trans = QGroupBox('TRANSPORT')
        gt = QHBoxLayout(grp_trans)
        gt.setContentsMargins(5, 4, 5, 4)
        gt.setSpacing(4)
        self.btn_start_p = QPushButton('▶  START')
        self.btn_start_p.setObjectName('btn_start')
        self.btn_start_p.clicked.connect(self._on_start)
        gt.addWidget(self.btn_start_p)
        self.btn_stop_p = QPushButton('⏹  STOP')
        self.btn_stop_p.setObjectName('btn_stop')
        self.btn_stop_p.clicked.connect(self._on_stop)
        gt.addWidget(self.btn_stop_p)
        layout.addWidget(grp_trans)

        # ════════════════════════════════════════
        # 7. NOISE GENERATOR
        # ════════════════════════════════════════
        grp_noise = QGroupBox('NOISE')
        gn = QGridLayout(grp_noise)
        gn.setContentsMargins(5, 4, 5, 4)
        gn.setSpacing(2)
        gn.setColumnStretch(1, 1)

        self.btn_noise_p = QPushButton('NOISE  ●')
        self.btn_noise_p.setObjectName('btn_noise')
        self.btn_noise_p.setCheckable(True)
        self.btn_noise_p.setChecked(True)
        self.btn_noise_p.clicked.connect(self._on_noise_panel)
        gn.addWidget(self.btn_noise_p, 0, 0, 1, 2)

        gn.addWidget(lbl('SIGNAL'), 1, 0)
        self.cmb_signal = QComboBox()
        for s in ['PINK', 'WHITE', 'TONE', 'SWEEP']:
            self.cmb_signal.addItem(s)
        self.cmb_signal.currentIndexChanged.connect(self._on_signal_type)
        gn.addWidget(self.cmb_signal, 1, 1)

        # Frecuencia del tono (solo visible cuando TONE está seleccionado)
        self.lbl_tone_freq = lbl('FREQ Hz')
        self.spn_tone_freq = QSpinBox()
        self.spn_tone_freq.setRange(20, 20000)
        self.spn_tone_freq.setSingleStep(10)
        self.spn_tone_freq.setValue(1000)
        self.spn_tone_freq.valueChanged.connect(
            lambda v: setattr(self.engine, 'tone_freq', v))
        gn.addWidget(self.lbl_tone_freq, 2, 0)
        gn.addWidget(self.spn_tone_freq, 2, 1)
        self.lbl_tone_freq.hide()
        self.spn_tone_freq.hide()

        gn.addWidget(lbl('GAIN'), 3, 0)
        self.sld_gain_p = QSlider(Qt.Orientation.Horizontal)
        self.sld_gain_p.setRange(0, 100)
        self.sld_gain_p.setValue(30)
        self.sld_gain_p.valueChanged.connect(self._on_gain_panel)
        gn.addWidget(self.sld_gain_p, 3, 1)

        self.lbl_gain_p = QLabel('-10.5 dBFS')
        self.lbl_gain_p.setStyleSheet(f'color:{ACCENT};font-size:9px;')
        gn.addWidget(self.lbl_gain_p, 4, 0, 1, 2)

        gn.addWidget(lbl('DEV OUT'), 5, 0)
        self.cmb_noise_dev = QComboBox()
        gn.addWidget(self.cmb_noise_dev, 5, 1)

        gn.addWidget(lbl('CH OUT'), 6, 0)
        self.spn_noise_ch = QSpinBox()
        self.spn_noise_ch.setRange(1, 8)
        self.spn_noise_ch.setValue(1)
        self.spn_noise_ch.valueChanged.connect(self._on_noise_channel)
        gn.addWidget(self.spn_noise_ch, 6, 1)
        layout.addWidget(grp_noise)

        # ════════════════════════════════════════
        # 8. FREEZE  (sin group box)
        # ════════════════════════════════════════
        self.btn_freeze_p = QPushButton('❄  FREEZE')
        self.btn_freeze_p.setCheckable(True)
        self.btn_freeze_p.clicked.connect(self._on_freeze_panel)
        layout.addWidget(self.btn_freeze_p)

        # ════════════════════════════════════════
        # 9. γ² AVG  +  COHERENCE (fusionados)
        # ════════════════════════════════════════
        grp_g2 = QGroupBox('γ²  AVG')
        gg = QVBoxLayout(grp_g2)
        gg.setContentsMargins(5, 4, 5, 4)
        gg.setSpacing(2)

        # Barra de coherencia promedio
        self.bar_coh = QProgressBar()
        self.bar_coh.setRange(0, 100)
        self.bar_coh.setValue(0)
        self.bar_coh.setTextVisible(False)
        self.bar_coh.setFixedHeight(5)
        gg.addWidget(self.bar_coh)

        # Valor numérico de coherencia (también sirve como lbl_coh_full)
        self.lbl_gamma2 = QLabel('—')
        self.lbl_gamma2.setStyleSheet(
            f'color:{COH_COLOR};font-size:10px;font-weight:bold;'
            f'padding:0px 0;letter-spacing:1px;'
        )
        self.lbl_gamma2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gg.addWidget(self.lbl_gamma2)

        self.lbl_gamma2_mark = QLabel('')
        self.lbl_gamma2_mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_gamma2_mark.setStyleSheet('font-size:7px;color:#555;letter-spacing:1px;')
        gg.addWidget(self.lbl_gamma2_mark)
        layout.addWidget(grp_g2)

        # lbl_coh_full apunta al mismo label que gamma2 (para _refresh)
        self.lbl_coh_full = self.lbl_gamma2

        return panel

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
        cm.addAction('Measurement Config…', self._show_meas_config).setShortcut('Ctrl+M')
        cm.addAction('Mic Calibration…',    self._load_mic_cal).setShortcut('Ctrl+K')

        # ── VIEW ─────────────────────────────────────────────────────
        vm = mb.addMenu('View')
        vm.addAction('Transfer Function  [T]',
                     lambda: self.tabs.setCurrentIndex(0)).setShortcut('T')
        vm.addAction('Spectrum  [S]',
                     lambda: self.tabs.setCurrentIndex(1)).setShortcut('S')
        vm.addAction('Spectrogram  [E]',
                     lambda: self.tabs.setCurrentIndex(2)).setShortcut('E')
        vm.addSeparator()
        vm.addAction('Toggle Settings Panel', self._on_toggle_settings).setShortcut('Ctrl+Right')
        vm.addAction('Toggle Save Panel',     self._on_toggle_save).setShortcut('Ctrl+Left')

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

        for d in devices:
            label = f"[{d['id']}] {d['name'][:22]}"
            if d['in'] >= 1:
                self.cmb_dev_in.addItem(label)
                self._dev_in_ids.append(d['id'])
                if d['id'] == self.engine.dev_in:
                    self.cmb_dev_in.setCurrentIndex(self.cmb_dev_in.count() - 1)
            if d['out'] >= 1:
                self.cmb_dev_out.addItem(label)
                self._dev_out_ids.append(d['id'])
                if d['id'] == self.engine.dev_out:
                    self.cmb_dev_out.setCurrentIndex(self.cmb_dev_out.count() - 1)
                self.cmb_noise_dev.addItem(label)
                self._dev_noise_ids.append(d['id'])
                if d['id'] == self.engine.dev_out:
                    self.cmb_noise_dev.setCurrentIndex(self.cmb_noise_dev.count() - 1)

        self.cmb_dev_in.currentIndexChanged.connect(self._on_dev_in)
        self.cmb_dev_out.currentIndexChanged.connect(self._on_dev_out)
        self.cmb_noise_dev.currentIndexChanged.connect(self._on_noise_dev)

    # ── Handlers ──────────────────────────────────────────────────────

    def _on_start(self):
        try:
            self.engine.start()
        except Exception as exc:
            self.sb.showMessage(f'⚠  Error al abrir stream: {exc}', 8000)
            return
        self.timer.start()
        self._set_running()

    def _on_stop(self):
        self.timer.stop()
        self.engine.stop()
        self.canvas_meas.clear()
        self.canvas_spec.clear()
        self._set_stopped()



    def _on_channels(self):
        self.engine.ch_meas  = self.spn_ch_meas.value()
        self.engine.ch_meas2 = self.spn_ch_meas2.value()
        self.engine.ch_ref   = self.spn_ch_ref.value()
        if self.engine.running:
            self.engine.restart()

    def _on_dev_in(self, idx):
        if 0 <= idx < len(self._dev_in_ids):
            self.engine.dev_in = self._dev_in_ids[idx]
            if self.engine.running:
                self.engine.restart()

    def _on_dev_out(self, idx):
        if 0 <= idx < len(self._dev_out_ids):
            self.engine.dev_out = self._dev_out_ids[idx]
            if self.engine.running:
                self.engine.restart()

    def _on_noise_panel(self, checked):
        self.engine.noise_on = checked
        self.btn_noise_p.setText('NOISE  ●' if checked else 'NOISE  ○')

    def _on_signal_type(self, idx):
        types = ['pink', 'white', 'tone', 'sweep']
        self.engine.signal_type = types[idx]
        is_tone = (types[idx] == 'tone')
        self.lbl_tone_freq.setVisible(is_tone)
        self.spn_tone_freq.setVisible(is_tone)

    def _on_gain_panel(self, val):
        gain = val / 100.0
        self.engine.gain = gain
        db = 20 * np.log10(gain + 1e-9)
        self.lbl_gain_p.setText(f'{db:.1f} dBFS')

    def _on_noise_channel(self, val):
        self.engine.noise_ch_start = val

    def _on_noise_dev(self, idx):
        if 0 <= idx < len(self._dev_noise_ids):
            self.engine.dev_out = self._dev_noise_ids[idx]
            if self.engine.running:
                self.engine.restart()

    def _on_delay_reset(self):
        """Elimina la compensación → fase sin compensar, IR datos desde t=0."""
        self._delay_comp_ms = 0.0
        self.lbl_delay_full.setText('—')
        self.canvas_meas.set_delay_ref(0.0)

    def _on_freeze_panel(self, checked):
        self._frozen = checked
        self.btn_freeze_p.setText('❄  FROZEN' if checked else '❄  FREEZE')

    def _on_find_delay(self):
        """
        Calcula retardo y activa la compensación de fase.

        Después de presionar:
        • La fase de la TF queda compensada → debería verse relativamente
          plana (sin la pendiente de retardo lineal).
        • La IR se recentra alrededor del pico encontrado.
        • El label muestra ms / samples / metros.

        Para RESETEAR la compensación: volver a presionar con la señal
        detenida (delay ≈ 0) o añadir un botón reset si se desea.
        """
        x, y = self.engine.get_buffers()
        delay_ms, smp = find_delay(x, y, fs=self.engine.fs)

        # ── Almacena el retardo para compensar la fase en cada refresh ──
        self._delay_comp_ms = delay_ms

        sign   = '+' if delay_ms >= 0 else ''
        dist_m = abs(delay_ms) / 1000.0 * 343.0
        text   = (f'{sign}{delay_ms:.2f} ms\n'
                  f'{smp:+d} smp\n'
                  f'{dist_m:.2f} m')
        self.lbl_delay_full.setText(text)

        # ── Activa la referencia de delay (SOLO desplaza datos, NO mueve eje) ──
        self.canvas_meas.set_delay_ref(delay_ms)

    # ── Refresh (QTimer) ──────────────────────────────────────────────

    def _refresh(self):
        # Reloj
        self.lbl_clock.setText(
            '  ' + QDateTime.currentDateTime().toString('hh:mm:ss') + '  ')

        if self._frozen:
            return
        x, y = self.engine.get_buffers()
        if np.max(np.abs(x)) < 1e-6:
            return

        # SPL RMS dBFS
        rms_ref  = 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)
        rms_meas = 20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-9)
        self.lbl_spl_ref.setText(f'{rms_ref:.1f} dBFS')
        self.lbl_spl_meas.setText(f'{rms_meas:.1f} dBFS')
        # Barra 0-100: mapear -60..0 dBFS
        bar_ref  = max(0, min(100, int((rms_ref  + 60) / 60 * 100)))
        bar_meas = max(0, min(100, int((rms_meas + 60) / 60 * 100)))
        self.bar_spl_ref.setValue(bar_ref)
        self.bar_spl_meas.setValue(bar_meas)
        # Color según nivel
        for bar, val in [(self.bar_spl_ref, bar_ref), (self.bar_spl_meas, bar_meas)]:
            if val > 90:
                bar.setObjectName('bar_clip')
            elif val > 75:
                bar.setObjectName('bar_warn')
            else:
                bar.setObjectName('')
            bar.style().unpolish(bar)
            bar.style().polish(bar)

        smooth_idx  = self.cmb_smooth.currentIndex()
        smooth_frac = self._smooth_values[smooth_idx]
        delay_s     = self._delay_comp_ms / 1000.0
        eps         = 1e-12

        freqs, gamma2, mag_db, phase_deg, gxx_db, gyy_db = compute_analysis(
            x, y,
            nperseg         = self.engine.nperseg,
            fs              = self.engine.fs,
            smooth_fraction = smooth_frac,
            delay_comp_s    = delay_s,
        )

        # ── Aplicar calibración de micrófono (si existe) ──────────────
        if self._mic_cal is not None:
            cal_f, cal_db = self._mic_cal
            correction = np.interp(freqs, cal_f, cal_db, left=0.0, right=0.0)
            mag_db = mag_db - correction   # restar corrección a la TF

        # ── 2do canal ────────────────────────────────────────────────
        y2 = self.engine.get_buffer_meas2()
        freqs2, gamma2_2, mag_db2, phase_deg2, _, _ = compute_analysis(
            x, y2,
            nperseg         = self.engine.nperseg,
            fs              = self.engine.fs,
            smooth_fraction = smooth_frac,
            delay_comp_s    = delay_s,
        )
        if self._mic_cal is not None:
            correction2 = np.interp(freqs2, cal_f, cal_db, left=0.0, right=0.0)
            mag_db2 = mag_db2 - correction2

        # ── Promedio TF (dominio lineal) ─────────────────────────────
        H1_lin   = 10.0 ** (mag_db  / 20.0)
        H2_lin   = 10.0 ** (mag_db2 / 20.0)
        mag_avg  = 20.0 * np.log10((H1_lin + H2_lin) / 2.0 + eps)
        ph_avg   = (phase_deg + phase_deg2) / 2.0
        g2_avg   = (gamma2 + gamma2_2) / 2.0

        # ── IR via IFFT + PSD para CPB ────────────────────────────────
        _, Gxx, Gyy_raw, Gxy = welch_spectra(
            x, y, nperseg=self.engine.nperseg, fs=self.engine.fs
        )
        _, _,   Gyy2_raw, _  = welch_spectra(
            x, y2, nperseg=self.engine.nperseg, fs=self.engine.fs
        )
        H  = Gxy / (Gxx + eps)
        ir = np.fft.irfft(H)
        ir = ir / (np.max(np.abs(ir)) + eps)

        mask    = (freqs >= 20) & (freqs <= 20000)
        avg_coh = float(np.mean(gamma2[mask]))
        thresh  = self.spn_thresh.value()

        # ── Update canvases según tab activo ──────────────────────────
        tab = self.tabs.currentIndex()
        if tab == 0:
            self.canvas_meas.update_plots(freqs, gamma2, mag_db, phase_deg,
                                          gxx_db, ir, coh_thresh=thresh)
            if self._show_ch2:
                self.canvas_meas.update_ch2(freqs2, mag_db2, phase_deg2,
                                            gamma2_2, coh_thresh=thresh)
            else:
                self.canvas_meas.update_ch2(None, None, None, None)
            if self._show_avg:
                self.canvas_meas.update_avg(freqs, mag_avg, ph_avg,
                                            g2_avg, coh_thresh=thresh)
            else:
                self.canvas_meas.update_avg(None, None, None, None)

        elif tab == 1:
            self.canvas_spec.update_plots(freqs, Gxx, Gyy_raw)
            lev_y2  = SpectrumCanvas._cpb(freqs2, Gyy2_raw) if self._show_ch2 else None
            lev_avg = (SpectrumCanvas._cpb(freqs, Gyy_raw) +
                       SpectrumCanvas._cpb(freqs2, Gyy2_raw)) / 2.0 if self._show_avg else None
            self.canvas_spec.update_ch2_avg(lev_y2, lev_avg)

        elif tab == 2:
            self.canvas_sgram.update_spectrogram(freqs, Gxx)

        # Toolbar live
        mark  = '✓  GOOD' if avg_coh > 0.9 else ('~  OK' if avg_coh > 0.7 else '⚠  LOW')
        color = '#66bb6a'  if avg_coh > 0.9 else ('#ffb74d' if avg_coh > 0.7 else '#ef5350')
        self.lbl_coh_full.setText(f'{avg_coh:.3f}')
        self.lbl_gamma2.setText(f'{avg_coh:.3f}')
        self.lbl_gamma2.setStyleSheet(
            f'color:{color};font-size:10px;font-weight:bold;padding:0px 0;letter-spacing:1px;'
        )
        self.lbl_gamma2_mark.setText(mark)
        self.lbl_gamma2_mark.setStyleSheet(f'font-size:7px;color:{color};letter-spacing:1px;')
        self.bar_coh.setValue(int(avg_coh * 100))

        self.sb.showMessage(
            f'CH MED={self.engine.ch_meas}  CH REF={self.engine.ch_ref}   |   '
            f'γ² avg={avg_coh:.3f}   |   '
            f'avg={self.engine.n_averages} frames   |   '
            f'Δf={self.engine.fs/self.engine.nperseg:.1f} Hz/bin'
        )

    # ── Estados ───────────────────────────────────────────────────────

    def _set_running(self):
        self.btn_start_p.setDisabled(True)
        self.btn_stop_p.setEnabled(True)

    def _set_stopped(self):
        self.btn_start_p.setEnabled(True)
        self.btn_stop_p.setDisabled(True)

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

    def closeEvent(self, event):
        self.timer.stop()
        self.engine.stop()
        event.accept()


# ── Entry point ───────────────────────────────────────────────────────

def run():
    app = QApplication(sys.argv)
    app.setApplicationName('Coherence')
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
