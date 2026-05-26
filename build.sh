#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# build.sh  —  Coherence v0.1
# Genera el ejecutable con PyInstaller.
#
# macOS  →  dist/Coherence.app   (doble clic para abrir)
# Windows→  dist/Coherence.exe   (correr desde cmd o doble clic)
#
# Uso:
#   pip install pyinstaller
#   bash build.sh
# ──────────────────────────────────────────────────────────────────────

set -e

echo "==> Coherence — build"
echo "    Python: $(python --version)"
echo ""

# Limpiar builds anteriores
rm -rf build dist Coherence.spec

# Detectar plataforma
PLATFORM=$(uname)

if [ "$PLATFORM" = "Darwin" ]; then
    echo "==> Plataforma: macOS — generando .app"
    pyinstaller \
        --name "Coherence" \
        --onedir \
        --windowed \
        --noconfirm \
        --clean \
        --hidden-import sounddevice \
        --hidden-import PyQt6.sip \
        run_coherence.py

    echo ""
    echo "==> Listo: dist/Coherence.app"
    echo "    Para distribuir: comprimir como Coherence-macOS.zip y subir a GitHub Releases"

elif [ "$PLATFORM" = "Linux" ]; then
    echo "==> Plataforma: Linux — generando binario"
    pyinstaller \
        --name "Coherence" \
        --onefile \
        --noconfirm \
        --clean \
        --hidden-import sounddevice \
        --hidden-import PyQt6.sip \
        run_coherence.py

    echo ""
    echo "==> Listo: dist/Coherence"

else
    echo "==> Plataforma: Windows — generando .exe"
    pyinstaller \
        --name "Coherence" \
        --onefile \
        --windowed \
        --noconfirm \
        --clean \
        --hidden-import sounddevice \
        --hidden-import PyQt6.sip \
        run_coherence.py

    echo ""
    echo "==> Listo: dist/Coherence.exe"
    echo "    Para distribuir: subir a GitHub Releases"
fi
