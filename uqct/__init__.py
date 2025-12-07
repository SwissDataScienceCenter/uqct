import os
import glob
import ctypes
import sys


# Patch to preload Nvidia libraries from site-packages
# This fixes "ImportError: libcusparseLt.so.0" etc. when LD_LIBRARY_PATH is not set
# (e.g. in SLURM jobs or when direnv is not used)
def _preload_nvidia_libs():
    libs_to_preload = [
        "libcusparseLt.so",
        "libcudnn.so",
        "libcudnn_adv.so",
        "libcudnn_cnn.so",
        "libcudnn_engines_precompiled.so",
        "libcudnn_engines_runtime_compiled.so",
        "libcudnn_graph.so",
        "libcudnn_heuristic.so",
        "libcudnn_ops.so",
        "libcublas.so",
        "libcublasLt.so",
        "libnvjitlink.so",
        "libcurand.so",
        "libcusolver.so",
        "libcusparse.so",
        "libnccl.so",
    ]

    site_packages = [p for p in sys.path if "site-packages" in p]
    if not site_packages:
        return

    # Check potential nvidia locations (usually site-packages/nvidia)
    # We look for all .so files in nvidia/ subdirectories
    nvidia_path = os.path.join(site_packages[0], "nvidia")
    if not os.path.exists(nvidia_path):
        return

    # Find all .so files recursively
    # This might be slightly slow but it's executed once on import
    found_libs = glob.glob(os.path.join(nvidia_path, "**", "*.so*"), recursive=True)

    # Map filenames to full paths for faster lookup
    lib_map = {os.path.basename(p): p for p in found_libs}

    for target in libs_to_preload:
        # Find matches. The target might be a prefix or partial match (e.g. libcudnn.so matching libcudnn.so.9)
        # We prefer exact matches or versioned matches
        matches = [p for name, p in lib_map.items() if target in name]
        for match in matches:
            try:
                ctypes.CDLL(match, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


_preload_nvidia_libs()
