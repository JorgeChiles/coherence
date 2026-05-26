"""
coherence.dsp
─────────────
Motor DSP: Welch, coherencia, Transfer Function, eje SMAART.
"""

import numpy as np

# ── Paleta de colores (dark SMAART-style) ─────────────────────────────
BLUE   = '#4fc3f7'
ORANGE = '#ffb74d'
GREEN  = '#81c784'
RED    = '#ef5350'
PURPLE = '#ce93d8'
GRAY   = '#546e7a'
CYAN   = '#00bcd4'


# ── Welch ─────────────────────────────────────────────────────────────

def welch_spectra(x, y, nperseg=4096, overlap=0.5, fs=48000):
    """
    Calcula Gxx, Gyy, Gxy usando el método de Welch con ventana Hanning.

    Parámetros
    ----------
    x        : señal de referencia (1D array)
    y        : señal de medición   (1D array)
    nperseg  : muestras por ventana   → resolución frecuencial Δf = fs/nperseg
    overlap  : solapamiento (0.0–0.9)
    fs       : frecuencia de muestreo (Hz)

    Retorna
    -------
    freqs, Gxx, Gyy, Gxy
    """
    n    = min(len(x), len(y))
    step = int(nperseg * (1 - overlap))
    win  = np.hanning(nperseg)
    wpow = np.sum(win ** 2)
    nfft = nperseg // 2 + 1

    Gxx   = np.zeros(nfft)
    Gyy   = np.zeros(nfft)
    Gxy   = np.zeros(nfft, dtype=complex)
    count = 0

    for start in range(0, n - nperseg, step):
        xi  = x[start:start + nperseg] * win
        yi  = y[start:start + nperseg] * win
        Xi  = np.fft.rfft(xi)
        Yi  = np.fft.rfft(yi)
        Gxx += np.abs(Xi) ** 2 / wpow
        Gyy += np.abs(Yi) ** 2 / wpow
        Gxy += np.conj(Xi) * Yi / wpow
        count += 1

    freqs = np.fft.rfftfreq(nperseg, 1 / fs)
    if count == 0:
        return freqs, Gxx, Gyy, Gxy
    return freqs, Gxx / count, Gyy / count, Gxy / count


# ── Suavizado fraccionado de octava ──────────────────────────────────

def smooth_fractional_octave(freqs, H_complex, fraction=24):
    """
    Suaviza H(f) complejo con ventana Hanning de 1/fraction octava.

    Opera sobre el espectro complejo — magnitud y fase se suavizan
    coherentemente (no por separado). Este es el método correcto.

    fraction=0  → sin suavizado
    fraction=24 → 1/24 oct  (suave, conserva detalles)
    fraction=12 → 1/12 oct
    fraction=6  → 1/6  oct  (suavizado medio)
    fraction=3  → 1/3  oct  (máximo suavizado)
    """
    if fraction == 0:
        return H_complex.copy()

    valid = freqs > 0
    f     = freqs[valid]
    H     = H_complex[valid]
    width = 1.0 / fraction          # ancho de ventana en octavas
    half  = width / 2.0

    smoothed = np.empty_like(H)
    for i in range(len(f)):
        f_lo = f[i] * 2 ** (-half)
        f_hi = f[i] * 2 ** ( half)
        mask = (f >= f_lo) & (f <= f_hi)
        if mask.sum() < 2:
            smoothed[i] = H[i]
            continue
        # Ventana Hanning en escala log-frecuencia
        log_r   = np.log2(f[mask] / f[i]) / half   # −1 a +1
        weights = 0.5 * (1.0 + np.cos(np.pi * log_r))
        weights /= weights.sum()
        smoothed[i] = np.dot(weights, H[mask])

    result        = H_complex.copy()
    result[valid] = smoothed
    return result


# ── Análisis completo ─────────────────────────────────────────────────

def compute_analysis(x, y, nperseg=4096, overlap=0.5, fs=48000,
                     smooth_fraction=0, delay_comp_s=0.0):
    """
    Calcula coherencia, Transfer Function y espectros.

    Parámetros
    ----------
    smooth_fraction : 0=off, 3=1/3oct, 6=1/6oct, 12=1/12oct, 24=1/24oct
    delay_comp_s    : retardo a compensar en segundos (del delay finder).
                      Multiplica H(f) por e^(+j·2π·f·τ) para cancelar la
                      pendiente de fase lineal del retardo de propagación.
                      Sin esto, la fase gira ~360° por cada 1/(f·τ) Hz,
                      haciendo la gráfica ilegible.

    Retorna
    -------
    freqs, gamma2, mag_db, phase_deg, gxx_db, gyy_db
    """
    freqs, Gxx, Gyy, Gxy = welch_spectra(x, y, nperseg, overlap, fs)

    eps    = 1e-12
    gamma2 = np.clip(np.abs(Gxy) ** 2 / (Gxx * Gyy + eps), 0, 1)
    H      = Gxy / (Gxx + eps)

    # ── Compensación de retardo ───────────────────────────────────────
    # H(f) de un sistema con retardo τ = |H_real(f)| · e^(-j·2π·f·τ) · e^(j·φ_real)
    # Al multiplicar por e^(+j·2π·f·τ) se cancela la pendiente lineal,
    # dejando solo la fase del sistema real.
    if delay_comp_s != 0.0:
        H = H * np.exp(1j * 2.0 * np.pi * freqs * delay_comp_s)

    # ── Suavizado (sobre H complejo — magnitud y fase coherentes) ─────
    if smooth_fraction > 0:
        H = smooth_fractional_octave(freqs, H, fraction=smooth_fraction)

    mag_db    = 20 * np.log10(np.abs(H) + eps)
    # Con compensación de retardo la fase queda dentro de ±180° para la
    # mayoría de sistemas PA → se muestra envuelta (más legible)
    phase_deg = np.degrees(np.angle(H))
    gxx_db    = 10 * np.log10(Gxx + eps)
    gyy_db    = 10 * np.log10(Gyy + eps)

    return freqs, gamma2, mag_db, phase_deg, gxx_db, gyy_db


# ── Retardo por correlación cruzada ──────────────────────────────────

def find_delay(x, y, fs=48000):
    """
    Estima el retardo entre x (referencia) e y (medición)
    usando correlación cruzada.

    Retorna
    -------
    delay_ms  : retardo estimado en milisegundos
    delay_smp : retardo en muestras
    """
    n   = min(len(x), len(y))
    X   = np.fft.rfft(x[:n], n=2 * n)
    Y   = np.fft.rfft(y[:n], n=2 * n)
    cc  = np.fft.irfft(np.conj(X) * Y)
    lag = np.argmax(np.abs(cc))
    if lag > n:
        lag -= 2 * n
    delay_ms = lag / fs * 1000
    return delay_ms, lag


# ── Eje logarítmico estilo SMAART ─────────────────────────────────────

def setup_smaart_axis(ax, fmin=20, fmax=20000, bg='#0d0d0d'):
    """
    Configura un eje X logarítmico con ticks en frecuencias estándar,
    estilo SMAART oscuro.
    """
    ax.set_xscale('log')
    ax.set_xlim(fmin, fmax)

    major = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    minor = [25, 31.5, 40, 63, 80, 125, 160, 250, 315, 400,
             630, 800, 1250, 1600, 2500, 3150, 4000, 6300, 8000,
             12500, 16000]

    ax.set_xticks(major)
    ax.set_xticklabels(
        ['20', '50', '100', '200', '500', '1k', '2k', '5k', '10k', '20k'],
        fontsize=8, color='#9e9e9e'
    )
    ax.set_xticks(minor, minor=True)
    ax.set_xticklabels([], minor=True)

    ax.tick_params(axis='y', labelsize=8, colors='#9e9e9e')
    ax.tick_params(axis='x', which='both', colors='#555555')

    ax.grid(True, which='major', linestyle='-',  lw=0.5, color='#252525')
    ax.grid(True, which='minor', linestyle=':', lw=0.3, color='#1a1a1a')

    ax.set_facecolor(bg)
    for spine in ax.spines.values():
        spine.set_color('#2a2a2a')
        spine.set_linewidth(0.8)
