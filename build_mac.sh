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

echo "▶ Comprimiendo..."
cd dist/
zip -r --symlinks ../Coherence-macOS.zip Coherence.app
cd ..

echo ""
echo "✅ Listo: Coherence-macOS.zip (~$(du -sh Coherence-macOS.zip | cut -f1))"
echo ""
echo "Siguiente paso:"
echo "  1. Ve a https://github.com/JorgeChiles/coherence/releases/new"
echo "  2. Tag: v0.1.0"
echo "  3. Sube Coherence-macOS.zip"
echo "  4. Publica el release"
