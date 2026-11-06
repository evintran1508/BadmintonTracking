import zipfile
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_ZIP = ROOT / "badminton_ai_kaggle.zip"

include_patterns = [
    "ai_core/**/*.py",
    "backend/**/*.py",
    "frontend/public/**/*",
    "frontend/src/**/*",
    "frontend/server.mjs",
    "schema/**/*",
    "config/**/*.yaml",
    "scripts/download_weights.py",
    "fixtures/labels/synthetic_dev_02.json",
    "package.json",
]

with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    for pat in include_patterns:
        for p in ROOT.glob(pat):
            if p.is_file() and not p.name.endswith((".pyc", ".DS_Store")):
                rel = p.relative_to(ROOT)
                zf.write(p, rel)

print(f"Created {OUT_ZIP} ({OUT_ZIP.stat().st_size / 1024:.1f} KB)")
