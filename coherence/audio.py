"""
coherence.audio
───────────────
Motor de audio: buffer circular, generadores de señal,
y engine de captura/reproducción con sounddevice.

Generadores disponibles: pink, white, tone, sweep
"""

import numpy as np
import threading
import sounddevice as sd


# ── Buffer circular (write-pointer, sin np.roll) ──────────────────────

class CircularBuffer:
    """
    Buffer FIFO de tamaño fijo.

    write() : solo llamado desde el callback de audio (sin lock).
    read()  : llamado desde el hilo de análisis (con lock).

    Usar np.roll en el callback causaba xruns — aquí solo hay
    slice-assignments directas, O(n_write) en lugar de O(buffer_size).
    """

    def __init__(self, size):
        self.buf  = np.zeros(size, dtype=np.float32)
        self.size = size
        self._ptr = 0           # puntero de escritura
        self.lock = threading.Lock()

    def write(self, data):
        """Sin lock — solo el callback escribe; lectura protegida por lock."""
        n   = len(data)
        end = self._ptr + n
        if end <= self.size:
            self.buf[self._ptr:end] = data
        else:
            first = self.size - self._ptr
            self.buf[self._ptr:]  = data[:first]
            self.buf[:n - first]  = data[first:]
        self._ptr = end % self.size   # asignación entera — atómica en CPython

    def read(self, n=None):
        """Retorna las últimas n muestras (o todo el buffer)."""
        with self.lock:
            if n is None or n >= self.size:
                n = self.size
            end   = self._ptr
            start = (end - n) % self.size
            if start < end:
                return self.buf[start:end].copy()
            # Wraps around
            out          = np.empty(n, dtype=np.float32)
            split        = self.size - start
            out[:split]  = self.buf[start:]
            out[split:]  = self.buf[:end]
            return out


# ── Helpers internos ──────────────────────────────────────────────────

def _cyclic(buf, n, pos):
    """Lee n muestras cíclicas desde pos. Retorna (out, new_pos)."""
    size = len(buf)
    end  = pos + n
    if end <= size:
        return buf[pos:end].copy(), end % size
    first       = size - pos
    out         = np.empty(n, dtype=np.float32)
    out[:first] = buf[pos:]
    out[first:] = buf[:n - first]
    return out, n - first


# ── Generadores de señal ──────────────────────────────────────────────

class PinkNoiseGenerator:
    """Ruido rosa: shaping espectral 1/f via FFT, buffer de 8 s cíclico."""

    def __init__(self, fs=48000, buffer_seconds=8):
        n        = fs * buffer_seconds
        white    = np.random.randn(n)
        spectrum = np.fft.rfft(white)
        freqs    = np.fft.rfftfreq(n)
        freqs[0] = 1e-6
        pink     = np.fft.irfft(spectrum / np.sqrt(freqs))[:n].astype(np.float32)
        self.buf = pink / (np.max(np.abs(pink)) + 1e-9)
        self.pos = 0

    def generate(self, n):
        out, self.pos = _cyclic(self.buf, n, self.pos)
        return out


class WhiteNoiseGenerator:
    """Ruido blanco, buffer de 8 s cíclico."""

    def __init__(self, fs=48000, buffer_seconds=8):
        n        = fs * buffer_seconds
        white    = np.random.randn(n).astype(np.float32)
        self.buf = white / (np.max(np.abs(white)) + 1e-9)
        self.pos = 0

    def generate(self, n):
        out, self.pos = _cyclic(self.buf, n, self.pos)
        return out


class ToneGenerator:
    """Tono senoidal continuo — frecuencia configurable en tiempo real."""

    def __init__(self, freq=1000, fs=48000):
        self.freq   = float(freq)
        self.fs     = fs
        self._phase = 0.0

    def generate(self, n):
        t           = np.arange(n, dtype=np.float64)
        out         = np.sin(2.0 * np.pi * self.freq / self.fs * t
                             + self._phase).astype(np.float32)
        self._phase = (self._phase
                       + 2.0 * np.pi * self.freq / self.fs * n) % (2.0 * np.pi)
        return out


class LogSweepGenerator:
    """
    Barrido logarítmico 20 Hz → 20 kHz, cíclico.
    Duración: 5 s → ciclo completo cada 5 s.
    """

    def __init__(self, f1=20, f2=20000, duration=5, fs=48000):
        n      = int(duration * fs)
        t      = np.linspace(0, duration, n, endpoint=False)
        k      = np.log(f2 / f1)
        phase  = 2.0 * np.pi * f1 * duration / k * (np.exp(t / duration * k) - 1.0)
        sweep  = np.sin(phase).astype(np.float32)
        self.buf = sweep / (np.max(np.abs(sweep)) + 1e-9)
        self.pos = 0

    def generate(self, n):
        out, self.pos = _cyclic(self.buf, n, self.pos)
        return out


# ── Motor de audio ────────────────────────────────────────────────────

class AudioEngine:
    """
    Motor principal de captura y reproducción de audio.

    Parámetros
    ----------
    fs          : frecuencia de muestreo (Hz)
    blocksize   : muestras por bloque de callback  (2048 → ~42 ms → tolerante a GIL)
    nperseg     : ventana Welch
    n_averages  : ventanas a promediar (máx 32 sin reiniciar)
    """

    SIGNAL_TYPES = ('pink', 'white', 'tone', 'sweep')

    def __init__(self, fs=48000, blocksize=2048, nperseg=4096, n_averages=8):
        self.fs         = fs
        self.blocksize  = blocksize
        self.nperseg    = nperseg
        self.n_averages = n_averages

        # Buffer pre-dimensionado para hasta 32 averages + 2 de margen
        buf_size = nperseg * 34
        self.buf_ref  = CircularBuffer(buf_size)
        self.buf_meas = CircularBuffer(buf_size)

        # Generadores (inicializados una vez, no en el callback)
        self._generators = {
            'pink':  PinkNoiseGenerator(fs=fs),
            'white': WhiteNoiseGenerator(fs=fs),
            'tone':  ToneGenerator(freq=1000, fs=fs),
            'sweep': LogSweepGenerator(fs=fs),
        }
        self.signal_type = 'pink'

        # Parámetros controlables desde GUI
        self.gain     = 0.3
        self.noise_on = True
        self.ch_meas  = 1       # 1-indexed
        self.ch_ref   = 4       # 1-indexed
        self.dev_in   = 2
        self.dev_out  = 2

        self._stream  = None
        self._running = False

    # ── Frecuencia del tono ───────────────────────────────────────────

    @property
    def tone_freq(self):
        return int(self._generators['tone'].freq)

    @tone_freq.setter
    def tone_freq(self, f):
        self._generators['tone'].freq   = float(f)
        self._generators['tone']._phase = 0.0

    # ── Callback de audio (hilo de alta prioridad) ────────────────────

    def _callback(self, indata, outdata, frames, time, status):
        """
        Llamado por sounddevice cada BLOCKSIZE muestras.
        Solo slice-assignments de numpy — sin allocs, sin locks en write.
        """
        try:
            n_in     = indata.shape[1]
            ref_idx  = min(self.ch_ref  - 1, n_in - 1)
            meas_idx = min(self.ch_meas - 1, n_in - 1)

            self.buf_ref.write( indata[:, ref_idx])
            self.buf_meas.write(indata[:, meas_idx])

            if self.noise_on:
                gen   = self._generators.get(self.signal_type,
                                              self._generators['pink'])
                noise = gen.generate(frames) * self.gain
            else:
                noise = np.zeros(frames, dtype=np.float32)

            for ch in range(outdata.shape[1]):
                outdata[:, ch] = noise

        except Exception:
            outdata.fill(0)   # silencio en caso de error; nunca dejar outdata sin inicializar

    # ── Control del stream ────────────────────────────────────────────

    def start(self):
        if self._running:
            return

        # ── Consultar límites reales del dispositivo ───────────────────
        info_in  = sd.query_devices(self.dev_in)
        info_out = sd.query_devices(self.dev_out)

        max_in  = int(info_in['max_input_channels'])
        max_out = int(info_out['max_output_channels'])

        if max_in < 1:
            raise RuntimeError(
                f"Dispositivo entrada [{self.dev_in}] '{info_in['name']}' "
                f"no tiene canales de entrada.")
        if max_out < 1:
            raise RuntimeError(
                f"Dispositivo salida [{self.dev_out}] '{info_out['name']}' "
                f"no tiene canales de salida.")

        # Clampar a lo que el dispositivo soporta
        n_in  = min(max(self.ch_meas, self.ch_ref), max_in)
        n_out = min(2, max_out)

        # Ajustar fs si el dispositivo no soporta la solicitada
        fs_req = self.fs
        try:
            sd.check_output_settings(device=self.dev_out,
                                     channels=n_out, samplerate=fs_req)
            sd.check_input_settings(device=self.dev_in,
                                    channels=n_in,  samplerate=fs_req)
            fs_use = fs_req
        except sd.PortAudioError:
            # Fallback a la fs nativa del dispositivo de entrada
            fs_use = int(info_in['default_samplerate'])
            self.fs = fs_use   # actualizar engine para que el DSP use la correcta

        self._stream = sd.Stream(
            samplerate = fs_use,
            blocksize  = self.blocksize,
            device     = (self.dev_in, self.dev_out),
            channels   = (n_in, n_out),
            dtype      = 'float32',
            callback   = self._callback,
            latency    = 'low'
        )
        self._stream.start()
        self._running = True

    def stop(self):
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def restart(self):
        """Detiene y reinicia el stream (para cambios de dispositivo/canal)."""
        was_running = self._running
        self.stop()
        if was_running:
            self.start()

    # ── Lectura de buffers ────────────────────────────────────────────

    def get_buffers(self):
        """Retorna (x_ref, y_meas) con n_averages × nperseg muestras."""
        n = self.nperseg * self.n_averages
        return self.buf_ref.read(n), self.buf_meas.read(n)

    @property
    def running(self):
        return self._running

    # ── Utilidades ────────────────────────────────────────────────────

    @staticmethod
    def list_devices():
        devices = sd.query_devices()
        result  = []
        for i, d in enumerate(devices):
            result.append({
                'id'  : i,
                'name': d['name'],
                'in'  : d['max_input_channels'],
                'out' : d['max_output_channels'],
                'fs'  : int(d['default_samplerate']),
            })
        return result
