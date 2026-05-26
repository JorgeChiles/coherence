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
echo "    PyInstaller: $(pyinstaller --version)"
echo ""

# Limpiar builds anteriores
rm -rf build dist Coherence.spec

# Detectar plataforma
PLATFORM=$(uname)

# ── Imports ocultos comunes a todas las plataformas ──────────────────
HIDDEN=(
    --hidden-import sounddevice
    --hidden-import cffi
    --hidden-import _cffi_backend
    --hidden-import PyQt6.sip
    --hidden-import PyQt6.QtCore
    --hidden-import PyQt6.QtGui
    --hidden-import PyQt6.QtWidgets
    --hidden-import matplotlib.backends.backend_qtagg
    --hidden-import matplotlib.backends.backend_agg
    --collect-all sounddevice
)

if [ "$PLATFORM" = "Darwin" ]; then
    echo "==> Plataforma: macOS — generando .app"

    pyinstaller \
        --name "Coherence" \
        --onedir \
        --windowed \
        --noconfirm \
        --clean \
        "${HIDDEN[@]}" \
        run_coherence.py

    # Comprimir para distribución
    cd dist
    zip -r Coherence-macOS.zip Coherence.app
    cd ..

    echo ""
    echo "╔══════════════════════════════════════════════╗"
    echo "║  ✓  dist/Coherence.app       — para usar    ║"
    echo "║  ✓  dist/Coherence-macOS.zip — para subir   ║"
    echo "╚══════════════════════════════════════════════╝"
    echo ""
    echo "Subir a GitHub Releases:"
    echo "  github.com/JorgeChiles/coherence/releases/new"
    echo "  Tag: v0.1.0  |  Adjuntar: dist/Coherence-macOS.zip"

elif [ "$PLATFORM" = "Linux" ]; then
    echo "==> Plataforma: Linux — generando binario"

    pyinstaller \
        --name "Coherence" \
        --onefile \
        --noconfirm \
        --clean \
        "${HIDDEN[@]}" \
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
        "${HIDDEN[@]}" \
        run_coherence.py

    echo ""
    echo "==> Listo: dist/Coherence.exe"
    echo "    Subir a: github.com/JorgeChiles/coherence/releases/new"
fi
