from __future__ import annotations

import os
import re
import shutil
import subprocess as sp
import sys
from pathlib import Path

REPO_URL = "https://github.com/X-DataInitiative/tick.git"
REPO_DIR = Path("tick")
DEPS = ["swig", "build-essential", "libopenblas-dev", "pkg-config"]

def replace_double_y(root: Path):
    pat = re.compile(r"\bdouble\s+y\s*;")
    for f in root.rglob("*.[ch]pp"):
        txt = f.read_text()
        new, n = pat.subn("double y = 0;", txt)
        if n:
            f.write_text(new)
            print(f"  • patched {f.relative_to(root)} ({n}×)")


def run(cmd: str | list[str], check=True, **kwargs):
    if isinstance(cmd, str):
        cmd = cmd.split()
    print(f"$ {' '.join(cmd)}")
    return sp.run(cmd, check=check, text=True, **kwargs).returncode

def strict_symlink():
    build_dir = Path("build")
    os.chdir(build_dir)

    # ls lib.linux-x86_64-cpython-310*/tick/array/build/_array*.so
    print("\n── Проверка наличия _array*.so ──")
    result = sp.run("ls lib.linux-x86_64-cpython-310*/tick/array/build/_array*.so",
                    shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print("[ERROR] _array*.so не найден. Сборка array не прошла.")
        print(result.stderr)
        sys.exit(1)
    else:
        print(result.stdout)

    # ln -s lib.linux-x86_64-cpython-310 lib.linux-x86_64-3.10
    print("── Создание символьной ссылки ──")
    link = Path("lib.linux-x86_64-3.10")
    if link.exists() or link.is_symlink():
        link.unlink()
    os.symlink("lib.linux-x86_64-cpython-310", link)
    print(f"✓ ln -s lib.linux-x86_64-cpython-310 → {link}")

    os.chdir("..")


def main():
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    run(["git", "clone", REPO_URL, REPO_DIR.name])
    os.chdir(REPO_DIR)

    run("apt-get update")
    run(["apt-get", "install", "-y", "--no-install-recommends", *DEPS])
    run(["git", "submodule", "update", "--init"])

    replace_double_y(Path("."))

    print("\n── First, quick build to obtain _array.so ──")
    run([sys.executable, "setup.py", "build", "install"], check=False)
    ##
    print("\n── Create linker mirror directory ──")
    strict_symlink()

    env = {**os.environ, "TICK_WERROR": "0"}

    print("\n── Final install (skip build, just copy files) ──")
    run([sys.executable, "setup.py", "install", "--skip-build"], env=env)

    print("\n✓ tick successfully built and installed")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\n[INTERRUPTED]")
