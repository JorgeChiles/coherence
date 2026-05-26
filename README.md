# Coherence  v0.1

**Real-time audio analyzer for live sound engineers.**  
Free and open-source alternative to SMAART — built for Spanish-speaking audio professionals.

GPL v3  ·  Python 3.11  ·  PyQt6  ·  macOS / Windows

---

## Features

- **Transfer Function** — magnitude (dB) with 1/3, 1/6, 1/12, 1/24-oct smoothing
- **Coherence** overlay aligned to 0 dB reference
- **Phase** — wrapped ±180°, displayed only where γ² ≥ threshold
- **Impulse Response** — fixed ±100 ms axis, delay finder centers the peak
- **Delay Finder** — cross-correlation, shows ms / samples / meters
- **Spectrum RTA** — 1/3-octave CPB, flat pink noise response
- **Signal generators** — pink noise, white noise, sine tone, log sweep
- Multi-average Welch (1 / 4 / 8 / 16 / 32 frames)
- Dark SMAART-style UI — collapsible panels, cursor info bar
- Export to TXT (IR, TF, Phase, Spectrum)

---

## Installation

### Option A — Run from source (recommended for technical users)

Requires Python 3.10+.

```bash
git clone https://github.com/JorgeChiles/coherence.git
cd coherence
pip install -r requirements.txt
python run_coherence.py
```

With conda:
```bash
conda create -n coherence python=3.11
conda activate coherence
pip install -r requirements.txt
python run_coherence.py
```

### Option B — Pre-built executable (no Python required)

Download the latest release for your platform:

**[→ Releases](https://github.com/JorgeChiles/coherence/releases)**

| Platform | File |
|----------|------|
| macOS    | `Coherence-macOS.dmg` |
| Windows  | `Coherence-Windows.exe` |

---

## Routing

Default routing for Focusrite Scarlett 18i8:

| Parameter | Value |
|-----------|-------|
| CH MED    | 1 (microphone) |
| CH REF    | 4 (loopback / DI) |
| DEV IN    | Scarlett 18i8 |
| DEV OUT   | Scarlett 18i8 |

All routing parameters are configurable from the SETTINGS panel.

---

## Usage

1. Connect your audio interface
2. Select DEV IN / DEV OUT in SETTINGS
3. Set CH MED (measurement mic) and CH REF (reference signal)
4. Press **START**
5. Enable **NOISE** to send pink noise to the system under test
6. Press **▶ FIND** in DELAY FINDER to compensate propagation delay
7. Adjust SMOOTH and COH THRESH to taste

---

## Build executable

Requires `pyinstaller`:

```bash
pip install pyinstaller
bash build.sh
```

Output: `dist/Coherence.app` (macOS) or `dist/Coherence.exe` (Windows).

---

## Project structure

```
coherence/
├── __init__.py       version info
├── dsp.py            Welch, coherence, TF, delay finder, CPB
└── app.py            PyQt6 UI — all panels and canvases

run_coherence.py      entry point
requirements.txt
build.sh              PyInstaller build script
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| numpy | DSP math |
| matplotlib | plots embedded in PyQt6 |
| PyQt6 | UI framework |
| sounddevice | real-time audio I/O (PortAudio) |

---

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).

Free to use, modify and distribute. Attribution appreciated.

---

*Built by [@JorgeChiles](https://github.com/JorgeChiles)*  
*Inspired by real touring experience with Dante, Lake, SMAART and large-format PA systems.*
