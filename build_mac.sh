#!/bin/bash
# build_mac.sh — Construye Coherence.app para macOS con PyInstaller
# Uso: conda activate audio-ingenieria && bash build_mac.sh

set -e
cd "$(dirname "$0")"

echo "▶ Instalando PyInstaller..."
pip install pyinstaller --quiet

echo "▶ Limpiando builds anteriores..."
rm -rf build/ dist/ __pycache__/

echo "▶ Construyendo Coherence.app..."
pyinstaller Coherence.spec --noconfirm

echo "▶ Comprimiendo con ditto (preserva symlinks, ~62 MB)..."
rm -f Coherence-macOS.zip
ditto -c -k --sequesterRsrc --keepParent dist/Coherence.app Coherence-macOS.zip

echo ""
echo "✅ Listo: Coherence-macOS.zip (~$(du -sh Coherence-macOS.zip | cut -f1))"
echo ""
echo "Siguiente paso:"
echo "  1. Descarga el .zip de Windows desde AppVeyor"
echo "  2. Ve a https://github.com/JorgeChiles/coherence/releases/new"
echo "  3. Crea tag v0.2.0 (o el número que corresponda)"
echo "  4. Sube Coherence-macOS.zip y Coherence-Windows.zip"
echo "  5. Publica el release"
