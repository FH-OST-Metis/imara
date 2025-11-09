# TODO: This is just a DEMO!

import shutil
from pathlib import Path

src = Path("demo/processed/extracted")
dst = Path("data/processed/extracted")

dst.mkdir(parents=True, exist_ok=True)

for file in src.glob("*"):
    if file.is_file():
        shutil.move(str(file), dst / file.name)

print("✅ Done: all files moved.")
