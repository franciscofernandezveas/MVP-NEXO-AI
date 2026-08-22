# cleanup_chroma.py
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"

if CHROMA_DIR.exists():
    shutil.rmtree(CHROMA_DIR)
    print(f"✅ Carpeta chroma_db eliminada: {CHROMA_DIR}")
else:
    print("ℹ️ La carpeta chroma_db no existe")