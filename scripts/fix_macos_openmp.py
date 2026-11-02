"""Make faiss-cpu share PyTorch's OpenMP runtime on macOS.

The macOS wheels of torch and faiss-cpu each bundle their own libomp.dylib. Loading
both into one process makes OpenMP abort ("OMP: Error #15") or deadlock/segfault
during CPU inference. This replaces faiss's bundled copy with a symlink to torch's,
so only one runtime is ever loaded. The original is kept as libomp.dylib.orig.

Re-run after reinstalling faiss-cpu. No-op on non-macOS platforms.

Usage (from the project root):
    python -m scripts.fix_macos_openmp           # apply
    python -m scripts.fix_macos_openmp --revert  # restore faiss's own libomp
"""

import argparse
import importlib.util
import sys
from pathlib import Path


def package_dir(name: str) -> Path:
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"Package '{name}' is not installed in this environment.")
    return Path(spec.origin).parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--revert", action="store_true")
    args = parser.parse_args()

    if sys.platform != "darwin":
        print("Not macOS; nothing to do.")
        return

    # find_spec locates packages without importing them, so no libomp is loaded here.
    faiss_omp = package_dir("faiss") / ".dylibs" / "libomp.dylib"
    torch_omp = package_dir("torch") / "lib" / "libomp.dylib"
    backup = faiss_omp.with_name("libomp.dylib.orig")

    if args.revert:
        if not backup.exists():
            print("No backup found; faiss is already using its own libomp.")
            return
        faiss_omp.unlink(missing_ok=True)
        backup.rename(faiss_omp)
        print(f"Restored {faiss_omp}")
        return

    if not torch_omp.exists():
        raise RuntimeError(f"torch libomp not found at {torch_omp}")
    if faiss_omp.is_symlink() and faiss_omp.resolve() == torch_omp.resolve():
        print("Already applied: faiss uses torch's libomp.")
        return
    if not faiss_omp.exists():
        raise RuntimeError(f"faiss libomp not found at {faiss_omp}; faiss layout may have changed.")

    if not backup.exists():
        faiss_omp.rename(backup)
    else:
        faiss_omp.unlink()
    faiss_omp.symlink_to(torch_omp)
    print(f"Linked {faiss_omp} -> {torch_omp}")


if __name__ == "__main__":
    main()
