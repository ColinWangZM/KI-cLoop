import os
import subprocess
from pathlib import Path


def build():
    root = Path(__file__).resolve().parent
    source = root.parent / 'kinetics/conservative_solver.cpp'
    library = root / ('conservative_solver.dll' if os.name == 'nt' else 'conservative_solver.so')
    if not library.exists() or library.stat().st_mtime < source.stat().st_mtime:
        subprocess.run([os.environ.get('CXX', 'g++'), '-std=c++17', '-O3', '-shared', '-fPIC', str(source), '-o', str(library)], check=True)
    return library


if __name__ == '__main__':
    print(build().name)
