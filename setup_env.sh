#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# AUDIO INGENIERÍA — Setup del entorno Conda
# Corre este script UNA SOLA VEZ para configurar todo el proyecto.
#
# Uso:
#   cd "Audio Engineering"
#   bash setup_env.sh
# ──────────────────────────────────────────────────────────────────────

set -e  # detener si hay algún error

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m' # sin color

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║      AUDIO INGENIERÍA — Setup de entorno Conda      ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ── 1. Verificar que conda está instalado ─────────────────────────────
echo -e "${YELLOW}[1/5] Verificando conda...${NC}"
if ! command -v conda &> /dev/null; then
    echo -e "${RED}✗ conda no encontrado.${NC}"
    echo ""
    echo "  Instala Miniconda desde:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    echo ""
    echo "  O con Homebrew:"
    echo "  brew install --cask miniconda"
    exit 1
fi
echo -e "${GREEN}✓ conda $(conda --version)${NC}"

# ── 2. Crear el entorno ───────────────────────────────────────────────
echo ""
echo -e "${YELLOW}[2/5] Creando entorno 'audio-ingenieria'...${NC}"
echo "  (puede tomar 3-5 minutos la primera vez)"
echo ""

# Si ya existe, preguntar si recrear
if conda env list | grep -q "audio-ingenieria"; then
    echo -e "${YELLOW}  El entorno ya existe. ¿Deseas recrearlo? [s/N]${NC}"
    read -r respuesta
    if [[ "$respuesta" =~ ^[Ss]$ ]]; then
        conda env remove -n audio-ingenieria -y
        echo -e "${GREEN}  Entorno anterior eliminado.${NC}"
    else
        echo "  Usando entorno existente."
        SKIP_CREATE=true
    fi
fi

if [ -z "$SKIP_CREATE" ]; then
    conda env create -f environment.yml
    echo -e "${GREEN}✓ Entorno creado exitosamente.${NC}"
fi

# ── 3. Registrar kernel en Jupyter ───────────────────────────────────
echo ""
echo -e "${YELLOW}[3/5] Registrando kernel 'Audio Ingeniería' en Jupyter...${NC}"

conda run -n audio-ingenieria python -m ipykernel install \
    --user \
    --name audio-ingenieria \
    --display-name "🎙️ Audio Ingeniería (Python 3.11)"

echo -e "${GREEN}✓ Kernel registrado.${NC}"

# ── 4. Verificar paquetes clave ───────────────────────────────────────
echo ""
echo -e "${YELLOW}[4/5] Verificando instalación...${NC}"

conda run -n audio-ingenieria python -c "
import numpy as np
import scipy
import matplotlib
import librosa
import soundfile as sf
import jupyter_core

packages = {
    'numpy'      : np.__version__,
    'scipy'      : scipy.__version__,
    'matplotlib' : matplotlib.__version__,
    'librosa'    : librosa.__version__,
    'soundfile'  : sf.__version__,
    'jupyter'    : jupyter_core.__version__,
}

print()
for pkg, ver in packages.items():
    print(f'  ✓  {pkg:<14} {ver}')
print()
print('  Todos los paquetes OK.')
"

# ── 5. Mensaje final ─────────────────────────────────────────────────
echo ""
echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║                    ✓  LISTO                         ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo ""
echo "  Para activar el entorno:"
echo -e "  ${GREEN}conda activate audio-ingenieria${NC}"
echo ""
echo "  Para abrir JupyterLab:"
echo -e "  ${GREEN}jupyter lab${NC}"
echo ""
echo "  En JupyterLab, seleccioná el kernel:"
echo "  🎙️ Audio Ingeniería (Python 3.11)"
echo ""
echo "  Notebook de coherencia:"
echo "  coherencia_audio.ipynb"
echo ""
