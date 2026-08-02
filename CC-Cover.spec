# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['gui_launcher.py'],
    pathex=['src'],
    binaries=[],
    datas=[('src', 'src')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'torchaudio', 'torchvision', 'tensorflow', 'funasr', 'modelscope', 'faster_whisper', 'ctranslate2', 'imageio_ffmpeg', 'onnxruntime', 'soundfile', 'transformers', 'librosa'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='CC-Cover',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version='packaging\\version_info.txt',
    icon=['assets\\app.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='CC-Cover',
)
