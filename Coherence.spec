# -*- mode: python ; coding: utf-8 -*-
# Coherence.spec — PyInstaller build spec for macOS
# Run from the project root:  pyinstaller Coherence.spec

from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

mpl_datas   = collect_data_files('matplotlib')
pyqt6_datas = collect_data_files('PyQt6')

a = Analysis(
    ['run_coherence.py'],
    pathex=['.'],
    binaries=[],
    datas=mpl_datas + pyqt6_datas,
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
    excludes=['tkinter', 'wx', 'gi'],
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
    strip=False,
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
    strip=False,
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
