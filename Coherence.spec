# -*- mode: python ; coding: utf-8 -*-
# Coherence.spec — PyInstaller build spec for macOS

from PyInstaller.utils.hooks import collect_data_files
import os

# Solo los datos de matplotlib que realmente necesitamos (estilos y fuentes)
mpl_datas = [
    (src, dst) for src, dst in collect_data_files('matplotlib')
    if any(keep in src for keep in ('mpl-data/fonts', 'mpl-data/stylelib', 'mpl-data/matplotlibrc'))
]

# PyQt6: NO usar collect_data_files — PyInstaller lo resuelve solo.
# collect_data_files('PyQt6') arrastra WebEngine, Qt3D, traducciones de 50 idiomas → +400 MB

block_cipher = None

a = Analysis(
    ['run_coherence.py'],
    pathex=['.'],
    binaries=[],
    datas=mpl_datas,
    hiddenimports=[
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        'PyQt6.sip',
        'matplotlib.backends.backend_qtagg',
        'matplotlib.backends.backend_agg',
        'matplotlib.figure',
        'matplotlib.ticker',
        'matplotlib.gridspec',
        'matplotlib.colors',
        'scipy.signal',
        'scipy.signal._upfirdn_apply',
        'scipy._lib.messagestream',
        'sounddevice',
        'numpy',
        'numpy.core._multiarray_umath',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter', 'wx', 'gi',
        'PyQt6.QtWebEngine', 'PyQt6.QtWebEngineCore', 'PyQt6.QtWebEngineWidgets',
        'PyQt6.Qt3DCore', 'PyQt6.Qt3DRender', 'PyQt6.Qt3DInput',
        'PyQt6.QtBluetooth', 'PyQt6.QtNfc', 'PyQt6.QtSerialPort',
        'PyQt6.QtLocation', 'PyQt6.QtPositioning',
        'PyQt6.QtSql', 'PyQt6.QtTest', 'PyQt6.QtXml',
        'PyQt6.QtDesigner', 'PyQt6.QtHelp', 'PyQt6.QtOpenGL',
        'matplotlib.tests', 'matplotlib.testing',
        'scipy.io', 'scipy.optimize', 'scipy.stats', 'scipy.integrate',
        'scipy.interpolate', 'scipy.spatial', 'scipy.sparse',
    ],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Coherence',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=True,
    upx=False,
    upx_exclude=[],
    name='Coherence',
)

app = BUNDLE(
    coll,
    name='Coherence.app',
    icon='Coherence.icns',
    bundle_identifier='mx.coherence.app',
    info_plist={
        'NSPrincipalClass': 'NSApplication',
        'NSHighResolutionCapable': True,
        'CFBundleShortVersionString': '0.1.0',
        'CFBundleVersion': '0.1.0',
        'NSMicrophoneUsageDescription': 'Coherence necesita acceso al audio para medir la función de transferencia.',
        'LSMinimumSystemVersion': '11.0',
    },
)
