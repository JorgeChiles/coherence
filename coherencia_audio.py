"""
COHERENCIA EN AUDIO — Animación educativa
Canal: Audio Ingeniería

Muestra:
 Escena 1 → Señales correlacionadas (tiempo) + coherencia alta
 Escena 2 → Ruido degrada la coherencia
 Escena 3 → Reflexión crea dip de coherencia en frecuencia específica
 Escena 4 → Vista tipo SMAART: coherencia + magnitud + fase
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────────────
# PARÁMETROS GLOBALES
# ──────────────────────────────────────────────────────────────────────────────
FS        = 48000          # sample rate
DURATION  = 0.05           # segundos de señal visible en time-domain
N_SAMPLES = int(FS * DURATION)
t         = np.linspace(0, DURATION, N_SAMPLES)

NPERSEG   = 2048           # ventana para Welch
N_FRAMES  = 240            # frames totales
FPS       = 30

# Frecuencias de la señal de referencia (multi-tono para que se vea interesante)
FREQS_REF = [250, 1000, 4000, 8000]   # Hz
DELAY_SAMPLES = 96                     # ~2ms @ 48kHz (delay de sistema real)

# ──────────────────────────────────────────────────────────────────────────────
# COLORES — estética cinematográfica oscura
# ──────────────────────────────────────────────────────────────────────────────
BG        = '#0b0d11'
PANEL_BG  = '#10141a'
GRID_COL  = '#1e2530'
CYAN      = '#00e5ff'
AMBER     = '#ffab40'
GREEN     = '#69ff47'
MAGENTA   = '#e040fb'
WHITE     = '#e8eaf6'
DIM       = '#546e7a'
RED       = '#ff1744'

# ──────────────────────────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ──────────────────────────────────────────────────────────────────────────────

def make_reference(n_total=FS):
    """Genera señal de referencia multi-tono."""
    t_full = np.arange(n_total) / FS
    sig = np.zeros(n_total)
    for f in FREQS_REF:
        sig += np.sin(2 * np.pi * f * t_full)
    sig += 0.3 * np.random.randn(n_total)   # ruido de señal base (simula música)
    return sig / np.max(np.abs(sig))


def make_measured(ref, noise_level=0.05, reflection_level=0.0, refl_delay=400):
    """
    Genera señal medida:
    - delay de sistema
    - ruido ambiente
    - reflexión opcional (degrada coherencia en una frecuencia)
    """
    n = len(ref)
    meas = np.roll(ref, DELAY_SAMPLES)      # delay del sistema
    meas[:DELAY_SAMPLES] = 0
    meas += noise_level * np.random.randn(n)  # ruido ambiente
    if reflection_level > 0:
        refl = np.roll(ref, DELAY_SAMPLES + refl_delay) * reflection_level
        refl[:DELAY_SAMPLES + refl_delay] = 0
        meas += refl
    return meas


def welch_coherence(x, y, nperseg=NPERSEG):
    """
    Calcula coherencia de Welch manualmente con numpy.
    Retorna: freqs (Hz), coherencia [0-1], magnitud (dB), fase (grados)
    """
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    n_segs = n // nperseg
    if n_segs < 1:
        n_segs = 1
        nperseg = n

    win = np.hanning(nperseg)
    Sxy = np.zeros(nperseg // 2 + 1, dtype=complex)
    Sxx = np.zeros(nperseg // 2 + 1)
    Syy = np.zeros(nperseg // 2 + 1)

    for i in range(n_segs):
        xi = x[i * nperseg:(i + 1) * nperseg] * win
        yi = y[i * nperseg:(i + 1) * nperseg] * win
        Xi = np.fft.rfft(xi)
        Yi = np.fft.rfft(yi)
        Sxy += np.conj(Xi) * Yi
        Sxx += np.abs(Xi) ** 2
        Syy += np.abs(Yi) ** 2

    # Coherencia
    coh = np.abs(Sxy) ** 2 / (Sxx * Syy + 1e-12)
    coh = np.clip(coh, 0, 1)

    # Transfer Function H = Sxy / Sxx
    H = Sxy / (Sxx + 1e-12)
    mag_db = 20 * np.log10(np.abs(H) + 1e-12)
    phase_deg = np.angle(H, deg=True)

    freqs = np.fft.rfftfreq(nperseg, 1 / FS)
    return freqs, coh, mag_db, phase_deg


# Pre-generar señal de referencia larga
np.random.seed(42)
REF_LONG = make_reference(n_total=FS * 3)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE FIGURA
# ──────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 9), facecolor=BG)
fig.patch.set_facecolor(BG)

gs = GridSpec(3, 2, figure=fig,
              left=0.07, right=0.97,
              top=0.90, bottom=0.08,
              hspace=0.55, wspace=0.35)

ax_ref   = fig.add_subplot(gs[0, 0])   # señal referencia
ax_meas  = fig.add_subplot(gs[0, 1])   # señal medida
ax_coh   = fig.add_subplot(gs[1, :])   # coherencia (ancho completo)
ax_mag   = fig.add_subplot(gs[2, 0])   # magnitud
ax_phase = fig.add_subplot(gs[2, 1])   # fase

AXES = [ax_ref, ax_meas, ax_coh, ax_mag, ax_phase]

def style_ax(ax, title, xlabel='', ylabel='', xlim=None, ylim=None, xlog=False):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=DIM, labelsize=8)
    ax.spines[:].set_color(GRID_COL)
    ax.grid(True, color=GRID_COL, linewidth=0.5, linestyle='--', alpha=0.7)
    ax.set_title(title, color=WHITE, fontsize=9, fontweight='bold',
                 pad=6, loc='left')
    if xlabel:
        ax.set_xlabel(xlabel, color=DIM, fontsize=7)
    if ylabel:
        ax.set_ylabel(ylabel, color=DIM, fontsize=7)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if xlog:
        ax.set_xscale('log')
        ax.set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
        ax.set_xticklabels(['20', '50', '100', '200', '500', '1k',
                            '2k', '5k', '10k', '20k'], fontsize=7, color=DIM)

t_ms = t * 1000   # tiempo en ms para display

style_ax(ax_ref,   '▶  REFERENCIA  [x(t)]',
         xlabel='tiempo (ms)', ylabel='amplitud',
         xlim=(0, DURATION * 1000), ylim=(-1.5, 1.5))
style_ax(ax_meas,  '▶  MEDICIÓN  [y(t)]',
         xlabel='tiempo (ms)', ylabel='amplitud',
         xlim=(0, DURATION * 1000), ylim=(-1.5, 1.5))
style_ax(ax_coh,   '▶  COHERENCIA  γ²(f)',
         xlabel='frecuencia (Hz)', ylabel='coherencia',
         xlim=(20, 20000), ylim=(-0.05, 1.1), xlog=True)
style_ax(ax_mag,   '▶  MAGNITUD  |H(f)|',
         xlabel='frecuencia (Hz)', ylabel='dB',
         xlim=(20, 20000), ylim=(-20, 10), xlog=True)
style_ax(ax_phase, '▶  FASE  ∠H(f)',
         xlabel='frecuencia (Hz)', ylabel='grados',
         xlim=(20, 20000), ylim=(-200, 200), xlog=True)

# Líneas de referencia fijas
ax_coh.axhline(1.0, color=GREEN, linewidth=0.6, alpha=0.3, linestyle='--')
ax_coh.axhline(0.9, color=AMBER, linewidth=0.6, alpha=0.3, linestyle='--')
ax_coh.axhline(0.5, color=RED,   linewidth=0.6, alpha=0.2, linestyle='--')
ax_coh.text(22, 1.02,  '1.0 — coherencia perfecta', color=GREEN,  fontsize=6, alpha=0.6)
ax_coh.text(22, 0.92,  '0.9 — umbral aceptable',    color=AMBER,  fontsize=6, alpha=0.6)
ax_coh.text(22, 0.52,  '0.5 — zona de alerta',      color=RED,    fontsize=6, alpha=0.5)
ax_mag.axhline(0, color=DIM, linewidth=0.5, linestyle='--', alpha=0.5)
ax_phase.axhline(0, color=DIM, linewidth=0.5, linestyle='--', alpha=0.5)

# Título principal
title_text = fig.text(0.5, 0.96,
    'COHERENCIA  //  Transfer Function Analysis',
    ha='center', va='top', color=CYAN,
    fontsize=15, fontweight='bold',
    fontfamily='monospace')

subtitle_text = fig.text(0.5, 0.925,
    'Escena 1 — Condiciones ideales: señal limpia, sin reflexiones',
    ha='center', va='top', color=DIM,
    fontsize=9, fontfamily='monospace')

# Indicador de estado (esquina sup derecha)
status_box = fig.text(0.97, 0.96, '● LIVE',
    ha='right', va='top', color=GREEN,
    fontsize=8, fontfamily='monospace', fontweight='bold')

# ──────────────────────────────────────────────────────────────────────────────
# LÍNEAS ANIMADAS
# ──────────────────────────────────────────────────────────────────────────────
line_ref,   = ax_ref.plot([], [], color=CYAN,    lw=1.2, alpha=0.9)
line_meas,  = ax_meas.plot([], [], color=AMBER,  lw=1.2, alpha=0.9)
line_coh,   = ax_coh.plot([], [], color=GREEN,   lw=2.0, alpha=0.95)
fill_coh    = ax_coh.fill_between([], [], 0, color=GREEN, alpha=0.08)
line_mag,   = ax_mag.plot([], [], color=MAGENTA, lw=1.5, alpha=0.9)
line_phase, = ax_phase.plot([], [], color=CYAN,  lw=1.0, alpha=0.7)

# ──────────────────────────────────────────────────────────────────────────────
# LEYENDAS
# ──────────────────────────────────────────────────────────────────────────────
for ax, label, color in [
    (ax_ref,   'x(t) — señal referencia (salida PA)',   CYAN),
    (ax_meas,  'y(t) — señal medida (micrófono)',        AMBER),
    (ax_coh,   'γ²(f) — coherencia de Welch',            GREEN),
    (ax_mag,   '|H(f)| — función de transferencia',      MAGENTA),
    (ax_phase, '∠H(f) — fase',                           CYAN),
]:
    ax.legend([mpatches.Patch(color=color)], [label],
              loc='upper right', fontsize=6,
              facecolor=PANEL_BG, edgecolor=GRID_COL,
              labelcolor=color)

# ──────────────────────────────────────────────────────────────────────────────
# LÓGICA DE ESCENAS
# ──────────────────────────────────────────────────────────────────────────────
SCENES = [
    # (frame_start, frame_end, noise, refl_level, refl_delay, subtitle)
    (0,   60,  0.02, 0.00, 0,   'Escena 1 — Condiciones ideales: señal limpia, sin reflexiones'),
    (60,  120, 0.25, 0.00, 0,   'Escena 2 — Ruido ambiente: la coherencia cae en altas frecuencias'),
    (120, 180, 0.05, 0.60, 480, 'Escena 3 — Reflexión tardía: dip de coherencia en frecuencia específica'),
    (180, 240, 0.02, 0.00, 0,   'Escena 4 — Vista SMAART: sistema optimizado, coherencia restaurada'),
]

def get_scene_params(frame):
    for (fs, fe, noise, refl, rdel, sub) in SCENES:
        if fs <= frame < fe:
            # progreso dentro de la escena (0→1)
            prog = (frame - fs) / (fe - fs)
            return noise, refl, rdel, sub, prog
    return 0.02, 0.0, 0, SCENES[-1][5], 1.0


def smooth_interp(a, b, t):
    """Interpolación suave (ease in-out)."""
    t = t * t * (3 - 2 * t)
    return a + (b - a) * t


# Cache de señales por escena para no recalcular en cada frame
_cache = {}

def get_signals(frame):
    noise, refl, rdel, sub, prog = get_scene_params(frame)

    # Determinar escena actual
    for si, (fs, fe, n, r, rd, s) in enumerate(SCENES):
        if fs <= frame < fe:
            scene_idx = si
            break
    else:
        scene_idx = len(SCENES) - 1

    if scene_idx not in _cache:
        # Calcular señales largas para coherencia
        meas_long = make_measured(REF_LONG,
                                  noise_level=noise,
                                  reflection_level=refl,
                                  refl_delay=rdel)
        freqs, coh, mag, phase = welch_coherence(REF_LONG, meas_long)
        _cache[scene_idx] = (freqs, coh, mag, phase, noise, refl, rdel)

    freqs, coh, mag, phase, noise_c, refl_c, rdel_c = _cache[scene_idx]

    # Señales time-domain (mini segmento, se mueve con el frame)
    offset = frame * 16   # desplazamiento temporal para dar sensación de "live"
    n_vis = N_SAMPLES
    ref_vis  = REF_LONG[offset:offset + n_vis]
    meas_vis = make_measured(REF_LONG, noise_level=noise_c,
                             reflection_level=refl_c, refl_delay=rdel_c)
    meas_vis = meas_vis[offset:offset + n_vis]

    return ref_vis, meas_vis, freqs, coh, mag, phase, sub


# ──────────────────────────────────────────────────────────────────────────────
# FUNCIÓN DE ANIMACIÓN
# ──────────────────────────────────────────────────────────────────────────────
def init():
    line_ref.set_data([], [])
    line_meas.set_data([], [])
    line_coh.set_data([], [])
    line_mag.set_data([], [])
    line_phase.set_data([], [])
    return line_ref, line_meas, line_coh, line_mag, line_phase


def update(frame):
    global fill_coh

    ref_vis, meas_vis, freqs, coh, mag, phase, subtitle = get_signals(frame)

    # Filtrar a rango audible 20Hz–20kHz
    mask = (freqs >= 20) & (freqs <= 20000)
    f_plot  = freqs[mask]
    c_plot  = coh[mask]
    m_plot  = mag[mask]
    p_plot  = phase[mask]

    # Time domain
    line_ref.set_data(t_ms, ref_vis)
    line_meas.set_data(t_ms, meas_vis)

    # Frecuencia
    line_coh.set_data(f_plot, c_plot)
    line_mag.set_data(f_plot, m_plot)
    line_phase.set_data(f_plot, p_plot)

    # Fill bajo la coherencia (recrear)
    global fill_coh
    try:
        fill_coh.remove()
    except Exception:
        pass
    fill_coh = ax_coh.fill_between(f_plot, c_plot, 0,
                                    color=GREEN, alpha=0.07)

    # Color dinámico de la línea de coherencia según valor promedio
    mean_coh = np.mean(c_plot)
    if mean_coh > 0.9:
        line_coh.set_color(GREEN)
    elif mean_coh > 0.7:
        line_coh.set_color(AMBER)
    else:
        line_coh.set_color(RED)

    # Subtítulo
    subtitle_text.set_text(subtitle)

    # Parpadeo del indicador LIVE
    if frame % 20 < 10:
        status_box.set_text('● LIVE')
        status_box.set_color(GREEN)
    else:
        status_box.set_text('○ LIVE')
        status_box.set_color(DIM)

    # Valor de coherencia promedio en el panel
    ax_coh.set_title(
        f'▶  COHERENCIA  γ²(f)          '
        f'avg: {mean_coh:.3f}   '
        f'{"✓ OK" if mean_coh > 0.9 else "⚠ BAJA" if mean_coh > 0.6 else "✗ MALA"}',
        color=WHITE, fontsize=9, fontweight='bold', pad=6, loc='left'
    )

    return line_ref, line_meas, line_coh, line_mag, line_phase, fill_coh


# ──────────────────────────────────────────────────────────────────────────────
# RENDER
# ──────────────────────────────────────────────────────────────────────────────
print("Generando animación...")
ani = animation.FuncAnimation(
    fig, update,
    frames=N_FRAMES,
    init_func=init,
    interval=1000 / FPS,
    blit=False
)

# Exportar
OUTPUT_PATH = '/sessions/youthful-vibrant-mayer/mnt/Audio Engineering/coherencia_audio.mp4'
OUTPUT_PATH_GIF = '/sessions/youthful-vibrant-mayer/mnt/outputs/coherencia_audio_preview.gif'

print("Guardando MP4...")
plt.rcParams['savefig.facecolor'] = BG
try:
    writer_mp4 = animation.FFMpegWriter(fps=FPS, bitrate=3000,
                                         extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    ani.save(OUTPUT_PATH, writer=writer_mp4, dpi=150)
    print(f"✓ MP4 guardado: {OUTPUT_PATH}")
except Exception as e:
    print(f"FFMpeg no disponible ({e}), guardando GIF de preview...")
    writer_gif = animation.PillowWriter(fps=15)
    ani.save(OUTPUT_PATH_GIF, writer=writer_gif, dpi=100)
    print(f"✓ GIF guardado: {OUTPUT_PATH_GIF}")

plt.close()
print("Listo.")
