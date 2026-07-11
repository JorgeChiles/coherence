"""
coherence.audio
───────────────
Motor de audio: buffer circular, generadores de señal,
y engine de captura/reproducción con sounddevice.

Generadores disponibles: pink, white, tone, sweep
"""

import os
import sys
import contextlib
import numpy as np
import threading
import sounddevice as sd

# En Windows usar latencia alta para evitar freezes con WASAPI
_IS_WINDOWS = sys.platform == 'win32'
_LATENCY    = 'high' if _IS_WINDOWS else 'low'


@contextlib.contextmanager
def _suppress_pa_stderr():
    """
    Suprime el ruido de PortAudio/CoreAudio en stderr (macOS).
    En Windows se omite el redirect (os.dup2 es inestable con WASAPI).
    """
    if _IS_WINDOWS:
        yield
        return
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        old_stderr = os.dup(2)
        os.dup2(devnull_fd, 2)
        try:
            yield
        finally:
            os.dup2(old_stderr, 2)
            os.close(old_stderr)
            os.close(devnull_fd)
    except Exception:
        yield   # si falla el redirect, no importa


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
    blocksize   : muestras por bloque de callback  (2048 → ~42 ms)
    nperseg     : ventana Welch
    n_averages  : ventanas a promediar (máx 32 sin reiniciar)

    Multi-canal de medición
    -----------------------
    _ch_meas_list  : lista de canales de medición (1-indexed).
    buf_meas_list  : lista de CircularBuffer correspondiente a cada canal.
    ch_spl         : canal dedicado para el medidor SPL (1-indexed).
    buf_spl        : CircularBuffer del canal SPL.

    Compatibilidad hacia atrás
    --------------------------
    Las propiedades ch_meas y ch_meas2 siguen funcionando para no romper
    el código existente mientras se migra.
    """

    SIGNAL_TYPES = ('pink', 'white', 'tone', 'sweep')

    def __init__(self, fs=48000, blocksize=2048, nperseg=4096, n_averages=8):
        self.fs         = fs
        self.blocksize  = blocksize
        self.nperseg    = nperseg
        self.n_averages = n_averages

        buf_size = nperseg * 34

        # Buffer de referencia (único)
        self.buf_ref  = CircularBuffer(buf_size)

        # Buffers multi-canal de medición (dinámicos)
        self._ch_meas_list : list[int]           = [1, 2]   # canales 1-indexed
        self.buf_meas_list : list[CircularBuffer] = [
            CircularBuffer(buf_size),
            CircularBuffer(buf_size),
        ]

        # Buffer y canal SPL dedicado
        self.ch_spl  : int           = 1
        self.buf_spl : CircularBuffer = CircularBuffer(buf_size)

        # Buffers spectrum multi-canal
        _buf_spec0               = CircularBuffer(buf_size)
        self._ch_spec_list : list[int]           = [1]    # canales 1-indexed
        self.buf_spec_list : list[CircularBuffer] = [_buf_spec0]

        # Generadores
        self._generators = {
            'pink':  PinkNoiseGenerator(fs=fs),
            'white': WhiteNoiseGenerator(fs=fs),
            'tone':  ToneGenerator(freq=1000, fs=fs),
            'sweep': LogSweepGenerator(fs=fs),
        }
        self.signal_type = 'pink'

        # Parámetros controlables desde GUI
        self.gain     = 0.3
        self.noise_on = False   # generador OFF al arrancar — se activa desde Signal Generator
        self.ch_ref   = 2       # canal de referencia (1-indexed)

        # Dispositivos: usar los defaults del sistema, no IDs hardcodeados
        with _suppress_pa_stderr():
            try:
                _def_in, _def_out = sd.default.device
                self.dev_in  = int(_def_in)  if (_def_in  is not None and _def_in  >= 0) else 0
                self.dev_out = int(_def_out) if (_def_out is not None and _def_out >= 0) else 0
            except Exception:
                self.dev_in  = 0
                self.dev_out = 0

        self._stream  = None
        self._running = False

    # ── Compatibilidad hacia atrás — spectrum ─────────────────────────

    @property
    def ch_spec(self) -> int:
        """Canal spectrum primario (backward compat)."""
        return self._ch_spec_list[0] if self._ch_spec_list else 1

    @ch_spec.setter
    def ch_spec(self, v: int):
        if self._ch_spec_list:
            self._ch_spec_list[0] = int(v)
        else:
            self._ch_spec_list = [int(v)]

    @property
    def buf_spec(self) -> 'CircularBuffer':
        """Buffer spectrum primario (backward compat)."""
        return self.buf_spec_list[0]

    # ── Compatibilidad hacia atrás ────────────────────────────────────

    @property
    def ch_meas(self) -> int:
        """Canal de medición primario (TF1)."""
        return self._ch_meas_list[0] if self._ch_meas_list else 1

    @ch_meas.setter
    def ch_meas(self, v: int):
        if self._ch_meas_list:
            self._ch_meas_list[0] = int(v)
        else:
            self._ch_meas_list = [int(v)]

    @property
    def ch_meas2(self) -> int:
        """Canal de medición secundario (TF2)."""
        return self._ch_meas_list[1] if len(self._ch_meas_list) > 1 else 2

    @ch_meas2.setter
    def ch_meas2(self, v: int):
        if len(self._ch_meas_list) > 1:
            self._ch_meas_list[1] = int(v)
        else:
            while len(self._ch_meas_list) < 2:
                self._ch_meas_list.append(int(v))
                buf_size = self.nperseg * 34
                self.buf_meas_list.append(CircularBuffer(buf_size))
            self._ch_meas_list[1] = int(v)

    @property
    def buf_meas(self) -> CircularBuffer:
        """Buffer de medición primario (TF1)."""
        return self.buf_meas_list[0]

    @property
    def buf_meas2(self) -> CircularBuffer:
        """Buffer de medición secundario (TF2)."""
        return self.buf_meas_list[1] if len(self.buf_meas_list) > 1 else self.buf_meas_list[0]

    # ── Gestión dinámica de canales de medición ───────────────────────

    def add_meas_channel(self, ch: int) -> int:
        """
        Agrega un canal de medición.
        Retorna el índice (0-based) del nuevo canal en la lista.
        """
        buf_size = self.nperseg * 34
        self._ch_meas_list.append(int(ch))
        self.buf_meas_list.append(CircularBuffer(buf_size))
        return len(self._ch_meas_list) - 1

    def remove_meas_channel(self, idx: int):
        """
        Elimina el canal de medición en la posición idx.
        Mínimo 1 canal queda siempre.
        """
        if len(self._ch_meas_list) <= 1:
            return
        del self._ch_meas_list[idx]
        del self.buf_meas_list[idx]

    def get_buffer_meas(self, idx: int = 0):
        """
        Retorna nperseg * n_averages muestras del canal de medición idx.
        Si idx está fuera de rango retorna el canal 0.
        """
        n = self.nperseg * self.n_averages
        safe_idx = min(idx, len(self.buf_meas_list) - 1)
        return self.buf_meas_list[safe_idx].read(n)

    # ── Gestión dinámica de canales spectrum ──────────────────────────

    def add_spec_channel(self, ch: int) -> int:
        """Agrega un canal spectrum. Retorna el índice (0-based)."""
        buf_size = self.nperseg * 34
        self._ch_spec_list.append(int(ch))
        self.buf_spec_list.append(CircularBuffer(buf_size))
        return len(self._ch_spec_list) - 1

    def remove_spec_channel(self, idx: int):
        """Elimina el canal spectrum idx. El índice 0 no se puede eliminar."""
        if idx == 0 or len(self._ch_spec_list) <= 1:
            return
        del self._ch_spec_list[idx]
        del self.buf_spec_list[idx]

    def get_buffer_spec(self, idx: int = 0):
        """Retorna nperseg * n_averages muestras del canal spectrum idx."""
        n        = self.nperseg * self.n_averages
        safe_idx = min(idx, len(self.buf_spec_list) - 1)
        return self.buf_spec_list[safe_idx].read(n)

    def get_buffer_spl(self):
        """Retorna nperseg * n_averages muestras del canal SPL."""
        n = self.nperseg * self.n_averages
        return self.buf_spl.read(n)

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
            n_in    = indata.shape[1]
            ref_idx = min(self.ch_ref  - 1, n_in - 1)
            spl_idx = min(self.ch_spl  - 1, n_in - 1)

            self.buf_ref.write(indata[:, ref_idx])
            self.buf_spl.write(indata[:, spl_idx])

            # Todos los canales spectrum
            for sp_buf, sp_ch in zip(self.buf_spec_list, self._ch_spec_list):
                sp_idx = min(sp_ch - 1, n_in - 1)
                sp_buf.write(indata[:, sp_idx])

            # Todos los canales de medición
            for buf, ch in zip(self.buf_meas_list, self._ch_meas_list):
                idx = min(ch - 1, n_in - 1)
                buf.write(indata[:, idx])

            if self.noise_on:
                gen   = self._generators.get(self.signal_type,
                                              self._generators['pink'])
                noise = gen.generate(frames) * self.gain
            else:
                noise = np.zeros(frames, dtype=np.float32)

            for ch in range(outdata.shape[1]):
                outdata[:, ch] = noise

        except Exception:
            outdata.fill(0)

    # ── Control del stream ────────────────────────────────────────────

    # Sample rates to try in order of preference
    _FS_CANDIDATES = [48000, 44100, 96000, 88200, 32000, 22050]

    def _negotiate_fs(self, n_in, n_out):
        """
        Encuentra la primera sample rate que funcione.
        Si n_out == 0, solo verifica la entrada (stream input-only).
        """
        candidates = [self.fs] + [f for f in self._FS_CANDIDATES if f != self.fs]
        info_in  = sd.query_devices(self.dev_in)
        for fs_default in [int(info_in['default_samplerate'])]:
            if fs_default not in candidates:
                candidates.append(fs_default)
        if n_out > 0:
            info_out = sd.query_devices(self.dev_out)
            for fs_default in [int(info_out['default_samplerate'])]:
                if fs_default not in candidates:
                    candidates.append(fs_default)

        last_err = None
        for fs in candidates:
            try:
                sd.check_input_settings(device=self.dev_in, channels=n_in, samplerate=fs)
                if n_out > 0:
                    sd.check_output_settings(device=self.dev_out, channels=n_out,
                                             samplerate=fs)
                return fs
            except sd.PortAudioError as e:
                last_err = e

        raise RuntimeError(
            f"Ninguna sample rate compatible para "
            f"IN '{sd.query_devices(self.dev_in)['name']}'"
            + (f" y OUT '{sd.query_devices(self.dev_out)['name']}'" if n_out > 0 else '')
            + f". Último error: {last_err}"
        )

    def _callback_input_only(self, indata, frames, time, status):
        """Callback para stream sin salida (dispositivo solo-captura)."""
        try:
            n_in    = indata.shape[1]
            ref_idx = min(self.ch_ref  - 1, n_in - 1)
            spl_idx = min(self.ch_spl  - 1, n_in - 1)

            self.buf_ref.write(indata[:, ref_idx])
            self.buf_spl.write(indata[:, spl_idx])

            for sp_buf, sp_ch in zip(self.buf_spec_list, self._ch_spec_list):
                sp_buf.write(indata[:, min(sp_ch - 1, n_in - 1)])

            for buf, ch in zip(self.buf_meas_list, self._ch_meas_list):
                buf.write(indata[:, min(ch - 1, n_in - 1)])

        except Exception:
            pass

    def start(self):
        if self._running:
            return

        # Verificar que los dispositivos guardados siguen existiendo;
        # si no, caer al default del sistema para no crashear.
        n_devs = len(sd.query_devices())
        if self.dev_in >= n_devs or self.dev_in < 0:
            try:
                self.dev_in = int(sd.default.device[0])
            except Exception:
                self.dev_in = 0
        if self.dev_out >= n_devs or self.dev_out < 0:
            try:
                self.dev_out = int(sd.default.device[1])
            except Exception:
                self.dev_out = 0

        with _suppress_pa_stderr():
            info_in  = sd.query_devices(self.dev_in)
            info_out = sd.query_devices(self.dev_out)

        max_in  = int(info_in['max_input_channels'])
        max_out = int(info_out['max_output_channels'])

        # macOS / CoreAudio a veces reporta max_input_channels = 0 en interfaces
        # que sí tienen entrada (bug del driver). Pero hay dispositivos que
        # genuinamente no tienen input (monitores LG, altavoces, etc.).
        # → Probar con 1 canal antes de asumir que el reporte es incorrecto.
        if max_in < 1:
            _probe_ok = False
            with _suppress_pa_stderr():
                try:
                    sd.check_input_settings(
                        device=self.dev_in, channels=1,
                        samplerate=float(info_in['default_samplerate'])
                    )
                    _probe_ok = True
                except Exception:
                    pass

            if _probe_ok:
                # Interfaz que reporta mal — usarla igualmente
                max_in = 2
            else:
                # Dispositivo sin inputs reales (monitor, altavoz USB, etc.)
                # Intentar reutilizar dev_out como entrada si tiene canales
                _out_max_in = int(info_out.get('max_input_channels', 0))
                if _out_max_in > 0:
                    self.dev_in = self.dev_out
                    info_in     = info_out
                    max_in      = _out_max_in
                else:
                    # Último recurso: default del sistema
                    try:
                        self.dev_in = int(sd.default.device[0])
                        with _suppress_pa_stderr():
                            info_in = sd.query_devices(self.dev_in)
                        max_in = max(int(info_in['max_input_channels']), 1)
                    except Exception:
                        self.dev_in = 0
                        max_in = 2

        # Calcular canales de entrada necesarios
        all_ch = self._ch_meas_list + self._ch_spec_list + [self.ch_ref, self.ch_spl]
        n_in   = min(max(all_ch), max_in)

        # Si el dispositivo de salida no tiene canales (ej. micrófono built-in)
        # → abrir stream solo-captura, sin generador de señal
        if max_out < 1:
            fs_use = self._negotiate_fs(n_in, n_out=0)
            self.fs = fs_use
            self._stream = sd.InputStream(
                samplerate = fs_use,
                blocksize  = self.blocksize,
                device     = self.dev_in,
                channels   = n_in,
                dtype      = 'float32',
                callback   = self._callback_input_only,
                latency    = _LATENCY,
            )
        else:
            n_out  = min(2, max_out)
            fs_use = self._negotiate_fs(n_in, n_out)
            self.fs = fs_use
            self._stream = sd.Stream(
                samplerate = fs_use,
                blocksize  = self.blocksize,
                device     = (self.dev_in, self.dev_out),
                channels   = (n_in, n_out),
                dtype      = 'float32',
                callback   = self._callback,
                latency    = _LATENCY,
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

    # ── Lectura de buffers (compatibilidad hacia atrás) ───────────────

    def get_buffers(self):
        """Retorna (x_ref, y_meas) con n_averages × nperseg muestras."""
        n = self.nperseg * self.n_averages
        return self.buf_ref.read(n), self.buf_meas_list[0].read(n)

    def get_buffer_meas2(self):
        """Retorna y_meas2 (2do canal). Compat. hacia atrás."""
        return self.get_buffer_meas(1)

    def get_buffer_spec_legacy(self):
        """Backward compat: retorna el canal spectrum 0."""
        return self.get_buffer_spec(0)

    @property
    def running(self):
        return self._running

    # ── Utilidades ────────────────────────────────────────────────────

    @staticmethod
    def list_devices():
        with _suppress_pa_stderr():
            devices   = sd.query_devices()
            host_apis = sd.query_hostapis()

        if _IS_WINDOWS:
            # En Windows, cada dispositivo físico aparece múltiples veces:
            # una por cada Host API (WASAPI, MME, DirectSound, WDM-KS).
            # Filtramos para mostrar solo la variante WASAPI (la más estable),
            # o si no hay WASAPI para ese dispositivo, la primera que encontremos.
            wasapi_idx = next(
                (i for i, h in enumerate(host_apis) if 'wasapi' in h['name'].lower()),
                None
            )
            # Primero intentamos solo WASAPI; si está vacío, mostramos todo
            result_wasapi = []
            result_all    = []
            for i, d in enumerate(devices):
                entry = {
                    'id'      : i,
                    'name'    : d['name'],
                    'in'      : int(d['max_input_channels']),
                    'out'     : int(d['max_output_channels']),
                    'fs'      : int(d['default_samplerate']),
                    'hostapi' : d.get('hostapi', -1),
                }
                result_all.append(entry)
                if wasapi_idx is not None and d.get('hostapi') == wasapi_idx:
                    result_wasapi.append(entry)

            # Deduplicar por nombre: WASAPI puede exponer el mismo dispositivo
            # físico varias veces (stereo, multi-canal, loopback, etc.).
            # Mantenemos el endpoint con más canales totales.
            seen: dict = {}
            for entry in result_wasapi:
                name = entry['name']
                total = entry['in'] + entry['out']
                if name not in seen or total > seen[name]['in'] + seen[name]['out']:
                    seen[name] = entry
            result_wasapi = list(seen.values())

            return result_wasapi if result_wasapi else result_all

        result = []
        for i, d in enumerate(devices):
            result.append({
                'id'  : i,
                'name': d['name'],
                'in'  : d['max_input_channels'],
                'out' : d['max_output_channels'],
                'fs'  : int(d['default_samplerate']),
            })
        return result
