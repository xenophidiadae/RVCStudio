                                      
import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = []

                                                               
for pkg in ["torch", "onnx", "onnxscript", "ml_dtypes", "faiss"]:
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

datas += [(str(Path("rvc_exporter") / file), "rvc_exporter") for file in
          ["__init__.py", "models.py", "modules.py", "attentions.py", "commons.py", "transforms.py"]]

                                                   
excluded_modules = [
    "tkinter",
    "matplotlib",
    "PIL",
    "cv2",
    "scipy",
    "pandas",
    "IPython",
    "jupyter",
    "torch.distributions",
    "torch.testing",
    "torch.utils.benchmark",
    "torch.cuda",
    "faiss.contrib",
    "pytest",
]

a = Analysis(
    ["convert.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excluded_modules,
    noarchive=False,
)

pyz = PYZ(a.pure)

                               
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="rvc-converter",
    debug=False,
    bootloader_ignore_signals=False,
    strip=(os.name != "nt"),
                                                                                 
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
