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

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QSlider, QLabel, QSpinBox,
    QComboBox, QFrame, QGroupBox, QSizePolicy,
    QTabWidget, QStatusBar, QDoubleSpinBox,
    QProgressBar, QFileDialog,
)
from PyQt6.QtCore  import Qt, QTimer, QDateTime
from PyQt6.QtGui   import QFont, QPalette, QColor

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
    font-size: 12px;
    font-weight: bold;
    letter-spacing: 4px;
    padding: 0 10px;
}}
#lbl_cursor_info {{
    color: {TEXT_MID};
    font-size: 10px;
    padding: 0 8px;
}}
#lbl_clock {{
    color: {TEXT_DIM};
    font-size: 10px;
    padding: 0 10px;
    letter-spacing: 1px;
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
        self.line_tf,  = self.ax_tf.semilogx(f0, [0, 0],  color=BLUE,   lw=1.5)
        self.line_coh, = self.ax_coh.semilogx(f0, [0, 0], color=COH_COLOR, lw=1.0, alpha=0.7)
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

        self.line_ph, = self.ax_ph.semilogx(f0, [0, 0], color=ORANGE, lw=1.0, alpha=0.8)

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
        self._last_lx = None
        self._last_ly = None
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
        self.line_x, = self.ax.semilogx([], [], color=BLUE,   lw=1.6, alpha=0.9,
                                         label='REF')
        self.line_y, = self.ax.semilogx([], [], color=ORANGE, lw=1.3, alpha=0.85,
                                         label='MED')
        self._fill_x = self.ax.fill_between([], [], -80, color=BLUE,   alpha=0.08)
        self._fill_y = self.ax.fill_between([], [], -80, color=ORANGE, alpha=0.06)

        # Crosshair cursor
        _ck = dict(color=TEXT_MID, lw=0.7, ls=':', alpha=0.0, zorder=20)
        self._cxh, = self.ax.plot([], [], **_ck)
        self._cyh, = self.ax.plot([], [], **_ck)

        self.ax.legend(fontsize=8, facecolor='#1a1a1a', edgecolor=BORDER,
                       labelcolor=TEXT_MID, loc='upper right')
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

        self._last_lx = lev_x
        self._last_ly = lev_y

        fc = _ISO_CENTERS
        self.line_x.set_data(fc, lev_x)
        self.line_y.set_data(fc, lev_y)

        self._fill_x.remove()
        self._fill_y.remove()
        self._fill_x = self.ax.fill_between(fc, lev_x, -80, color=BLUE,   alpha=0.08)
        self._fill_y = self.ax.fill_between(fc, lev_y, -80, color=ORANGE, alpha=0.06)

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


# ── Ventana principal ─────────────────────────────────────────────────

class MainWindow(QMainWindow):

    UPDATE_MS = 120

    def __init__(self):
        super().__init__()
        self.engine         = AudioEngine()
        self._frozen        = False
        self._delay_comp_ms = 0.0    # retardo compensado (delay finder → fase)

        self.setWindowTitle('Coherence  v0.1')
        self.setMinimumSize(1100, 720)
        self.resize(1380, 820)
        self.setStyleSheet(QSS)

        self._build_central()
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
        self.tabs.addTab(self.canvas_meas, 'MEASUREMENT')
        self.tabs.addTab(self.canvas_spec, 'SPECTRUM')
        cv.addWidget(self.tabs, stretch=1)
        root.addWidget(center, stretch=1)

        # ── Derecha: panel settings colapsable ──
        root.addWidget(self._build_collapsible_settings())

    def _build_info_bar(self):
        bar = QWidget()
        bar.setObjectName('info_bar')
        bar.setFixedHeight(26)
        h = QHBoxLayout(bar)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)

        self.lbl_app_title = QLabel('  COHERENCE  v0.1  ')
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
        panel.setFixedWidth(190)
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
        panel.setFixedWidth(190)

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

        gr.addWidget(lbl('CH MED'), 2, 0)
        self.spn_ch_meas = QSpinBox()
        self.spn_ch_meas.setRange(1, 20)
        self.spn_ch_meas.setValue(1)
        self.spn_ch_meas.valueChanged.connect(self._on_channels)
        gr.addWidget(self.spn_ch_meas, 2, 1)

        gr.addWidget(lbl('CH REF'), 3, 0)
        self.spn_ch_ref = QSpinBox()
        self.spn_ch_ref.setRange(1, 20)
        self.spn_ch_ref.setValue(4)
        self.spn_ch_ref.valueChanged.connect(self._on_channels)
        gr.addWidget(self.spn_ch_ref, 3, 1)
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
            f'color:{COH_COLOR};font-size:14px;font-weight:bold;'
            f'padding:1px 0;letter-spacing:1px;'
        )
        self.lbl_gamma2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gg.addWidget(self.lbl_gamma2)

        self.lbl_gamma2_mark = QLabel('')
        self.lbl_gamma2_mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_gamma2_mark.setStyleSheet('font-size:8px;color:#555;letter-spacing:1px;')
        gg.addWidget(self.lbl_gamma2_mark)
        layout.addWidget(grp_g2)

        # lbl_coh_full apunta al mismo label que gamma2 (para _refresh)
        self.lbl_coh_full = self.lbl_gamma2

        return panel

    def _build_statusbar(self):
        self.sb = QStatusBar()
        self.setStatusBar(self.sb)
        self.sb.showMessage('Coherence v0.1  —  GPL v3')

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
        self.engine.ch_meas = self.spn_ch_meas.value()
        self.engine.ch_ref  = self.spn_ch_ref.value()
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
        freqs, gamma2, mag_db, phase_deg, gxx_db, gyy_db = compute_analysis(
            x, y,
            nperseg        = self.engine.nperseg,
            fs             = self.engine.fs,
            smooth_fraction= smooth_frac,
            delay_comp_s   = self._delay_comp_ms / 1000.0,
        )

        # Impulse Response via IFFT de H(f)  +  potencia lineal para CPB
        eps = 1e-12
        _, Gxx, Gyy_raw, Gxy = welch_spectra(
            x, y, nperseg=self.engine.nperseg, fs=self.engine.fs
        )
        H  = Gxy / (Gxx + eps)
        ir = np.fft.irfft(H)
        ir = ir / (np.max(np.abs(ir)) + eps)

        mask = (freqs >= 20) & (freqs <= 20000)
        avg_coh = float(np.mean(gamma2[mask]))
        thresh  = self.spn_thresh.value()

        # Update canvases
        tab = self.tabs.currentIndex()
        if tab == 0:
            self.canvas_meas.update_plots(freqs, gamma2, mag_db, phase_deg,
                                     gxx_db, ir, coh_thresh=thresh)
        else:
            # Spectrum: pasar potencia lineal → CPB 1/3-oct dentro del canvas
            self.canvas_spec.update_plots(freqs, Gxx, Gyy_raw)

        # Toolbar live
        mark  = '✓  GOOD' if avg_coh > 0.9 else ('~  OK' if avg_coh > 0.7 else '⚠  LOW')
        color = '#66bb6a'  if avg_coh > 0.9 else ('#ffb74d' if avg_coh > 0.7 else '#ef5350')
        self.lbl_coh_full.setText(f'{avg_coh:.3f}')
        self.lbl_gamma2.setText(f'{avg_coh:.3f}')
        self.lbl_gamma2.setStyleSheet(
            f'color:{color};font-size:14px;font-weight:bold;padding:1px 0;letter-spacing:1px;'
        )
        self.lbl_gamma2_mark.setText(mark)
        self.lbl_gamma2_mark.setStyleSheet(f'font-size:10px;color:{color};letter-spacing:1px;')
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
