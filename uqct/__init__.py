import os
import glob
import ctypes
import sys


# Patch to preload Nvidia libraries from site-packages
# This fixes "ImportError: libcusparseLt.so.0" etc. when LD_LIBRARY_PATH is not set
# (e.g. in SLURM jobs or when direnv is not used)
def _preload_nvidia_libs():
    print("Preloading Nvidia libraries from site-packages")
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

    # Discover .venv relative to this file
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_venv = os.path.join(project_root, ".venv")
    if os.path.exists(local_venv):
        venv_sps = glob.glob(
            os.path.join(local_venv, "lib", "python*", "site-packages")
        )
        site_packages = venv_sps + site_packages

    if not site_packages:
        print("No site-packages found")
        return

    # Check potential nvidia locations (usually site-packages/nvidia)
    # We look for all .so files in nvidia/ subdirectories
    found_libs = []
    for sp in site_packages:
        nvidia_path = os.path.join(sp, "nvidia")
        if os.path.exists(nvidia_path):
            found_libs.extend(
                glob.glob(os.path.join(nvidia_path, "**", "*.so*"), recursive=True)
            )

    if not found_libs:
        print("No Nvidia libraries found in site-packages")
        return

    print(f"Found Nvidia libraries in site-packages: {found_libs}")

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
    print(f"Preloaded Nvidia libraries from site-packages: {found_libs}")


_preload_nvidia_libs()
