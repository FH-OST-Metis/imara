import shutil
from pathlib import Path

src = Path("demo/processed/extracted")
dst = Path("data/processed/extracted")

for file in src.rglob("*"):
    if file.is_file():
        target = dst / file.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(file), str(target))

print("✅ Done: all files moved recursively.")
