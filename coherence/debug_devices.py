"""
debug_devices.py — diagnóstico de dispositivos de audio
Correr con:  python3 debug_devices.py
"""
import sounddevice as sd
import sys

print(f"sounddevice version : {sd.__version__}")
print(f"PortAudio version   : {sd.get_portaudio_version()[1]}")
print()

devices = sd.query_devices()
print(f"Total dispositivos encontrados: {len(devices)}")
print("=" * 60)

for i, d in enumerate(devices):
    marker = ""
    name_low = d['name'].lower()
    if 'apogee' in name_low or 'symphony' in name_low or 'duet' in name_low or 'quartet' in name_low or 'ensemble' in name_low:
        marker = "  ◀◀◀ APOGEE"
    print(
        f"[{i:2d}] {d['name'][:40]:<40}  "
        f"IN:{d['max_input_channels']:2d}  OUT:{d['max_output_channels']:2d}  "
        f"FS:{int(d['default_samplerate']):6d}"
        f"{marker}"
    )

print()
print("Dispositivo de entrada por defecto:")
try:
    print(" ", sd.query_devices(kind='input')['name'])
except Exception as e:
    print("  ERROR:", e)

print("Dispositivo de salida por defecto:")
try:
    print(" ", sd.query_devices(kind='output')['name'])
except Exception as e:
    print("  ERROR:", e)

# Test rápido de apertura de stream con cada dispositivo de entrada
print()
print("Test de apertura de stream (cada dispositivo IN):")
print("-" * 60)
for i, d in enumerate(devices):
    if d['max_input_channels'] < 1:
        continue
    try:
        with sd.InputStream(device=i, channels=1,
                            samplerate=int(d['default_samplerate']),
                            blocksize=512):
            print(f"  [{i:2d}] OK  — {d['name'][:40]}")
    except Exception as e:
        print(f"  [{i:2d}] FAIL — {d['name'][:40]}")
        print(f"        {e}")
