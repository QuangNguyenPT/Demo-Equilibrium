"""Download ARC-AGI-1 and Sudoku-Extreme (test split) into data/external/ (gitignored)."""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "data" / "external"
ARC_ZIP = "https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip"
SUDOKU_CSV = "https://huggingface.co/datasets/sapientinc/sudoku-extreme/resolve/main/test.csv"


def fetch(url: str) -> bytes:
    print(f"[FETCH] {url}")
    with urllib.request.urlopen(url, timeout=120) as resp:
        return resp.read()


def main() -> int:
    arc_dir = ROOT / "arc"
    if not (arc_dir / "evaluation").exists():
        with zipfile.ZipFile(io.BytesIO(fetch(ARC_ZIP))) as zf:
            for name in zf.namelist():
                parts = name.split("/")
                if len(parts) == 4 and parts[1] == "data" and parts[2] in ("training", "evaluation") and parts[3]:
                    target = arc_dir / parts[2] / parts[3]
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(zf.read(name))
    sudoku = ROOT / "sudoku-extreme" / "test.csv"
    if not sudoku.exists():
        sudoku.parent.mkdir(parents=True, exist_ok=True)
        sudoku.write_bytes(fetch(SUDOKU_CSV))
    for split in ("training", "evaluation"):
        print(f"[FETCH] ARC {split}: {len(list((arc_dir / split).glob('*.json')))} tasks")
    print(f"[FETCH] Sudoku-Extreme test: {sudoku.stat().st_size // 1_000_000} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
