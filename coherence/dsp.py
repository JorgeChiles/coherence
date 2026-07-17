"""
coherence.dsp
─────────────
Motor DSP: RunningTF (promedio exponencial estilo SMAART),
suavizado fraccionado de octava, delay finder, eje SMAART.

Diferencia clave vs Welch clásico
──────────────────────────────────
Welch promedia POTENCIAS (|X|², |Y|², |X·Y|) — pierde fase en el promedio.
RunningTF promedia el ESPECTRO CRUZADO COMPLEJO Gxy = conj(X)·Y con
decaimiento exponencial. Al conservar la fase en el promedio, la TF y la
coherencia convergen mucho más rápido y la fase queda estable — igual que SMAART.

  H(f)  = Gxy / Gxx          (estimador H1 — mínima varianza con ruido en y)
  γ²(f) = |Gxy|² / (Gxx·Gyy) (coherencia real, 0→1)
"""

import numpy as np
import matplotlib.ticker as _mticker

# ── Paleta de colores (dark SMAART-style) ──────────────────────────────
BLUE   = '#4fc3f7'
ORANGE = '#ffb74d'
GREEN  = '#81c784'
RED    = '#ef5350'
PURPLE = '#ce93d8'
GRAY   = '#546e7a'
CYAN   = '#00bcd4'


# ── Suavizado fraccionado de octava (rápido) ───────────────────────────

def smooth_fractional_octave(freqs, H_complex, fraction=24):
    """
    Suaviza H(f) complejo con ventana Gaussiana de 1/fraction octava.

    Implementación vectorizada en escala log-frecuencia:
    1. Interpola H a una grilla log-uniforme de N puntos.
    2. Convoluciona con ventana Hanning de ancho proporcional.
    3. Interpola de vuelta a las frecuencias originales.

    Complejidad: O(N log N) en lugar de O(N²).

    fraction=0  → sin suavizado
    fraction=3  → 1/3  oct  (máximo suavizado)
    fraction=6  → 1/6  oct
    fraction=12 → 1/12 oct
    fraction=24 → 1/24 oct  (muy fino)
    """
    if fraction == 0 or np.sum(freqs > 0) < 4:
        return H_complex.copy()

    valid  = freqs > 0
    f_v    = freqs[valid]
    H_v    = H_complex[valid]

    # Grilla log-uniforme con la misma cantidad de puntos
    N      = len(f_v)
    f_log  = np.logspace(np.log10(f_v[0]), np.log10(f_v[-1]), N)

    # Interpolar H real e imaginario por separado a la grilla uniforme
    Hr_i = np.interp(f_log, f_v, H_v.real)
    Hi_i = np.interp(f_log, f_v, H_v.imag)

    # Ancho de ventana en número de bins (grilla log-uniforme)
    # 1 octava = log2(f_max/f_min) / N bins
    log_span   = np.log2(f_v[-1] / f_v[0])        # octavas totales
    bins_oct   = N / log_span                      # bins por octava
    half_width = int(np.ceil(bins_oct / fraction / 2))
    if half_width < 1:
        half_width = 1

    win_size = 2 * half_width + 1
    win      = np.hanning(win_size)
    win      /= win.sum()

    # Convolución con modo 'same' y relleno reflejo para no perder bordes
    pad = half_width
    Hr_pad = np.pad(Hr_i, pad, mode='reflect')
    Hi_pad = np.pad(Hi_i, pad, mode='reflect')
    Hr_sm  = np.convolve(Hr_pad, win, mode='valid')[:N]
    Hi_sm  = np.convolve(Hi_pad, win, mode='valid')[:N]

    H_sm_log = Hr_sm + 1j * Hi_sm

    # Interpolar de vuelta a las frecuencias originales
    Hr_out = np.interp(f_v, f_log, H_sm_log.real)
    Hi_out = np.interp(f_v, f_log, H_sm_log.imag)

    result        = H_complex.copy()
    result[valid] = Hr_out + 1j * Hi_out
    return result


def smooth_gamma2(freqs, gamma2, fraction=24):
    """Suaviza γ² (real) con el mismo método log-uniforme."""
    if fraction == 0:
        return gamma2.copy()
    # Reusar smooth_fractional_octave tratando gamma2 como parte real
    g_complex        = gamma2.astype(complex)
    g_sm             = smooth_fractional_octave(freqs, g_complex, fraction)
    return np.clip(g_sm.real, 0, 1)


# ── RunningTF — promedio exponencial estilo SMAART ─────────────────────

class RunningTF:
    """
    Estimador de Transfer Function con promedio exponencial running.

    Por qué es mejor que Welch
    ──────────────────────────
    Welch promedia |X|², |Y|² y |Gxy| → pierde información de fase entre
    bloques. RunningTF promedia el espectro cruzado COMPLEJO:

        Gxy ← α·Gxy + (1-α)·conj(Xi)·Yi

    La fase se conserva en el promedio → H(f) = Gxy/Gxx converge a la
    función de transferencia real mucho más rápido y con menos ruido.
    La coherencia γ² refleja la verdadera correlación lineal entre señales.

    Uso
    ───
    rtf = RunningTF(nperseg=4096, fs=48000, n_averages=8)
    rtf.push(x_block, y_block)      # llamar cada refresh
    freqs, γ², mag, phase, gxx = rtf.get_tf(delay_comp_s=0.003, smooth_fraction=12)
    """

    def __init__(self, nperseg=4096, fs=48000, n_averages=8):
        self.nperseg    = nperseg
        self.fs         = fs
        self.freqs      = np.fft.rfftfreq(nperseg, 1.0 / fs)
        self._win       = np.hanning(nperseg).astype(np.float64)
        self._wpow      = float(np.sum(self._win ** 2))

        nfft        = nperseg // 2 + 1
        self.Gxx    = np.zeros(nfft)
        self.Gyy    = np.zeros(nfft)
        self.Gxy    = np.zeros(nfft, dtype=complex)
        self._ready = False

        self.set_n_averages(n_averages)

    def set_n_averages(self, n):
        """
        Convierte n_averages en alpha.
        n=1 → alpha=0 (sin memoria)
        n=8 → alpha≈0.875
        n=32→ alpha≈0.969
        """
        n = max(1, int(n))
        self.alpha = 1.0 - 1.0 / n

    def push(self, x, y):
        """
        Incorpora el bloque más reciente de audio.
        x, y : arrays de cualquier longitud >= nperseg.
               Se usan las últimas nperseg muestras.
        """
        xw  = x[-self.nperseg:].astype(np.float64) * self._win
        yw  = y[-self.nperseg:].astype(np.float64) * self._win
        Xi  = np.fft.rfft(xw)
        Yi  = np.fft.rfft(yw)

        gxx_new = (Xi.real**2 + Xi.imag**2) / self._wpow
        gyy_new = (Yi.real**2 + Yi.imag**2) / self._wpow
        gxy_new = (Xi.conj() * Yi)           / self._wpow

        if not self._ready:
            self.Gxx    = gxx_new
            self.Gyy    = gyy_new
            self.Gxy    = gxy_new
            self._ready = True
        else:
            a = self.alpha
            self.Gxx = a * self.Gxx + (1.0 - a) * gxx_new
            self.Gyy = a * self.Gyy + (1.0 - a) * gyy_new
            self.Gxy = a * self.Gxy + (1.0 - a) * gxy_new

    def get_tf(self, delay_comp_s=0.0, smooth_fraction=0):
        """
        Calcula y devuelve la TF actualizada.

        Retorna
        ───────
        freqs, gamma2, mag_db, phase_deg, gxx_db
        """
        eps    = 1e-12
        Gxx    = self.Gxx + eps
        Gyy    = self.Gyy + eps
        Gxy    = self.Gxy

        gamma2 = np.clip(np.abs(Gxy)**2 / (Gxx * Gyy), 0.0, 1.0)
        H      = Gxy / Gxx

        # Compensación de retardo — cancela pendiente lineal de fase
        if delay_comp_s != 0.0:
            H = H * np.exp(1j * 2.0 * np.pi * self.freqs * delay_comp_s)

        # Suavizado fraccionado de octava (sobre H complejo → fase coherente)
        if smooth_fraction > 0:
            H      = smooth_fractional_octave(self.freqs, H, fraction=smooth_fraction)
            gamma2 = smooth_gamma2(self.freqs, gamma2,  fraction=smooth_fraction)

        mag_db    = 20.0 * np.log10(np.abs(H)   + eps)
        phase_deg = np.degrees(np.angle(H))
        gxx_db    = 10.0 * np.log10(self.Gxx    + eps)

        return self.freqs, gamma2, mag_db, phase_deg, gxx_db

    def reset(self):
        """Reinicia los acumuladores (p.ej. al cambiar de dispositivo)."""
        nfft        = self.nperseg // 2 + 1
        self.Gxx    = np.zeros(nfft)
        self.Gyy    = np.zeros(nfft)
        self.Gxy    = np.zeros(nfft, dtype=complex)
        self._ready = False

    @property
    def ready(self):
        return self._ready


# ── compute_analysis — compatibilidad con código existente ─────────────

def compute_analysis(x, y, nperseg=4096, overlap=0.5, fs=48000,
                     smooth_fraction=0, delay_comp_s=0.0):
    """
    Wrapper de compatibilidad. Crea un RunningTF temporal, empuja todos
    los bloques disponibles con overlap, y retorna el resultado.

    Para medición en tiempo real usar RunningTF directamente (más eficiente).
    """
    n    = min(len(x), len(y))
    step = max(1, int(nperseg * (1.0 - overlap)))

    rtf = RunningTF(nperseg=nperseg, fs=fs, n_averages=1)
    rtf.alpha = 0.0   # sin decaimiento → promedio lineal clásico

    for start in range(0, n - nperseg + 1, step):
        rtf.alpha = (rtf.alpha * start / max(start + step, 1)
                     ) if False else 0.0
        # Empujar bloque completo
        xi = x[start:start + nperseg]
        yi = y[start:start + nperseg]
        if rtf._ready:
            # Promedio lineal acumulado manualmente
            gxx_new = np.abs(np.fft.rfft(xi * rtf._win))**2 / rtf._wpow
            gyy_new = np.abs(np.fft.rfft(yi * rtf._win))**2 / rtf._wpow
            Xi      = np.fft.rfft(xi * rtf._win)
            Yi      = np.fft.rfft(yi * rtf._win)
            gxy_new = Xi.conj() * Yi / rtf._wpow
            count   = getattr(rtf, '_count', 1)
            rtf.Gxx = (rtf.Gxx * count + gxx_new) / (count + 1)
            rtf.Gyy = (rtf.Gyy * count + gyy_new) / (count + 1)
            rtf.Gxy = (rtf.Gxy * count + gxy_new) / (count + 1)
            rtf._count = count + 1
        else:
            rtf.push(xi, yi)
            rtf._count = 1

    if not rtf._ready:
        nfft  = nperseg // 2 + 1
        freqs = np.fft.rfftfreq(nperseg, 1.0 / fs)
        eps   = 1e-12
        return freqs, np.zeros(nfft), np.full(nfft, -100.0), np.zeros(nfft), np.full(nfft, -100.0), np.full(nfft, -100.0)

    freqs, gamma2, mag_db, phase_deg, gxx_db = rtf.get_tf(
        delay_comp_s=delay_comp_s, smooth_fraction=smooth_fraction)

    eps    = 1e-12
    gyy_db = 10.0 * np.log10(rtf.Gyy + eps)
    return freqs, gamma2, mag_db, phase_deg, gxx_db, gyy_db


# ── welch_spectra — compatibilidad ────────────────────────────────────

def welch_spectra(x, y, nperseg=4096, overlap=0.5, fs=48000):
    """
    Compatibilidad hacia atrás. Usa RunningTF internamente.
    """
    rtf = RunningTF(nperseg=nperseg, fs=fs, n_averages=1)
    n    = min(len(x), len(y))
    step = max(1, int(nperseg * (1.0 - overlap)))
    for start in range(0, n - nperseg + 1, step):
        rtf.push(x[start:start + nperseg], y[start:start + nperseg])
    return rtf.freqs, rtf.Gxx, rtf.Gyy, rtf.Gxy


# ── Retardo por correlación cruzada ───────────────────────────────────

def find_delay(x, y, fs=48000):
    """
    Estima el retardo entre x (referencia) e y (medición)
    usando correlación cruzada normalizada en frecuencia.

    Retorna
    ───────
    delay_ms  : retardo estimado en milisegundos
    delay_smp : retardo en muestras
    """
    n   = min(len(x), len(y))
    win = np.hanning(n)
    X   = np.fft.rfft(x[:n] * win, n=2 * n)
    Y   = np.fft.rfft(y[:n] * win, n=2 * n)
    # GCC-PHAT: normalizar por magnitud → robustez ante coloración espectral
    denom = np.abs(X) * np.abs(Y) + 1e-12
    cc    = np.fft.irfft(np.conj(X) * Y / denom)
    lag   = int(np.argmax(np.abs(cc)))
    if lag > n:
        lag -= 2 * n
    delay_ms = lag / fs * 1000.0
    return delay_ms, lag


# ── Eje logarítmico estilo SMAART ─────────────────────────────────────

def _hz_label(x, _pos):
    """Format frequency: '20 Hz', '1 kHz', '10 kHz', etc."""
    if x >= 1000:
        v = x / 1000
        return f'{int(v)} kHz' if v == int(v) else f'{v:.1f} kHz'
    return f'{int(x)} Hz'

def setup_smaart_axis(ax, fmin=20, fmax=20000, bg='#0d0d0d',
                      show_xlabels=True, show_xlabel=False):
    """
    Configura un eje X logarítmico con ticks en frecuencias estándar,
    estilo SMAART oscuro.

    show_xlabels : muestra los números de frecuencia (20 Hz, 50 Hz, 1 kHz…)
    show_xlabel  : muestra la etiqueta del eje  "Frecuencia (Hz)"
    """
    ax.set_xscale('log')
    ax.set_xlim(fmin, fmax)

    major = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    minor = [25, 31.5, 40, 63, 80, 125, 160, 250, 315, 400,
             630, 800, 1250, 1600, 2500, 3150, 4000, 6300, 8000,
             12500, 16000]

    # Use FuncFormatter — more robust than FixedFormatter on log scale
    ax.set_xticks(major)
    ax.xaxis.set_major_formatter(_mticker.FuncFormatter(_hz_label))
    ax.set_xticks(minor, minor=True)
    ax.xaxis.set_minor_formatter(_mticker.NullFormatter())

    ax.tick_params(axis='y', labelsize=8, colors='#8a9e8a')
    ax.tick_params(axis='x', which='major', labelsize=7, colors='#9e9e9e',
                   labelbottom=show_xlabels)
    ax.tick_params(axis='x', which='minor', colors='#4a5a4a', labelbottom=False)

    if show_xlabel:
        ax.set_xlabel('Frequency (Hz)', fontsize=7, color='#6a7a6a', labelpad=3)

    ax.grid(True, which='major', linestyle='-',  lw=0.55, color='#1e281e')
    ax.grid(True, which='minor', linestyle=':',  lw=0.30, color='#171e17')

    ax.set_facecolor(bg)
    for spine in ax.spines.values():
        spine.set_color('#252e25')
        spine.set_linewidth(0.8)
