"""
run_coherence.py
─────────────────
Lanzador de Coherence — Real-Time Audio Analyzer

Uso:
    conda activate audio-ingenieria
    python run_coherence.py
"""

import sys
import os

# Agrega el directorio padre al path para imports relativos
sys.path.insert(0, os.path.dirname(__file__))

from coherence.app import run

if __name__ == '__main__':
    run()
