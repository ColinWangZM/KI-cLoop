import os
from pathlib import Path
import subprocess
import sys
HERE = Path(__file__).resolve().parent

def build():
    if sys.platform != 'linux':
        raise RuntimeError('Use Linux or WSL with a C++17 compiler for the reference solver.')
    source = HERE / 'conservative_solver.cpp'
    output = HERE / 'conservative_solver.so'
    if not output.exists() or output.stat().st_mtime < source.stat().st_mtime:
        temporary = HERE / f'conservative_solver.{os.getpid()}.so'
        subprocess.run([os.environ.get('CXX', 'g++'), '-O3', '-std=c++17', '-shared', '-fPIC', str(source), '-o', str(temporary)], check=True)
        temporary.replace(output)
    return output
if __name__ == '__main__':
    print(build().name)
