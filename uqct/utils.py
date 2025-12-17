import os
import re
from pathlib import Path
from typing import List, Optional

from uqct.logging import get_logger

logger = get_logger(__name__)


def _git_root(start: Path) -> Optional[Path]:
    current = start
    while True:
        if (current / ".git").exists():
            return current
        parent = current.parent
        if parent == current:
            return None
        current = parent


def _candidate_root_dirs() -> List[Path]:
    candidates: List[Path] = []
    env_root = os.environ.get("UQCT_ROOT_DIR")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates += [Path(__file__).parent.parent]
    return candidates


def get_root_dir() -> Path:
    """Return the first existing root directory from the configured candidates."""
    for path in _candidate_root_dirs():
        if path.is_dir():
            return path
    raise FileNotFoundError("None of the configured root directories exist.")


def get_checkpoint_dir() -> Path:
    """Resolve the checkpoint directory with environment variable override."""
    env_ckpt = os.environ.get("UQCT_CKPT_DIR")
    if env_ckpt:
        ckpt_path = Path(env_ckpt).expanduser()
        if ckpt_path.is_dir():
            return ckpt_path
    ckpt_dir = Path("/mydata/chip/shared/checkpoints/uqct/")
    if ckpt_dir.exists():
        return ckpt_dir
    return get_root_dir() / "checkpoints"


def get_results_dir() -> Path:
    """Resolve the results directory with environment variable override."""
    env_results = os.environ.get("UQCT_RESULTS_DIR")
    if env_results:
        results_dir = Path(env_results).expanduser()
    elif Path("/cluster/").exists():
        results_dir = Path("/cluster/scratch/mgaetzner/uqct/results/")
    elif Path("/mydata/").exists():
        results_dir = Path("/mydata/chip/shared/results/uqct/")
    else:
        results_dir = get_root_dir() / "results"
    results_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Results dir: {results_dir}")
    return results_dir


def get_cache_dir() -> Path:
    """Resolve a writable cache directory, creating it when the parent exists."""
    candidates: List[Path] = []
    env_cache = os.environ.get("UQCT_CACHE_DIR")
    if env_cache:
        candidates.append(Path(env_cache).expanduser())
    candidates.extend(
        [
            Path("/mydata/chip/shared/data/caches/"),
            Path("/cluster/scratch/mgaetzner/caches/"),
        ]
    )
    git_root = _git_root(Path(__file__).resolve().parent)
    if git_root:
        candidates.append(git_root / "caches")

    for cache_path in candidates:
        if cache_path.is_dir():
            return cache_path
        parent = cache_path.parent
        parent_exists = parent.exists() if parent != cache_path else cache_path.exists()
        if not parent_exists:
            continue
        try:
            cache_path.mkdir(exist_ok=True)
        except OSError:
            continue
        if cache_path.is_dir():
            return cache_path
    raise FileNotFoundError("Unable to locate or create a cache directory.")


def get_hardware_specific_engine_path(dataset: str) -> Path:
    """
    Returns a unique path inside get_cache_dir() based on the current GPU and TRT version.
    """
    import tensorrt
    import torch

    # 1. Get your project's base cache location
    # Ensure it's a Path object
    base_cache = Path(get_cache_dir())

    if not torch.cuda.is_available():
        # Fallback for CPU/Testing (unlikely to use TRT here, but safe to handle)
        return base_cache / "tensorrt_engines" / "cpu_fallback"

    # 2. Generate Hardware Identity String
    # GPU Name (e.g. "NVIDIA_A100-SXM4-40GB") -> "NVIDIA_A100_SXM4_40GB"
    gpu_name = torch.cuda.get_device_name(0)
    gpu_safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", gpu_name)

    # Compute Capability (e.g. "80" for A100)
    cap = torch.cuda.get_device_capability(0)
    arch_str = f"sm{cap[0]}{cap[1]}"

    # TensorRT Version (e.g. "8.6.1")
    trt_ver = tensorrt.__version__

    # 3. Construct Final Path
    # Structure: /your_cache/tensorrt_engines/NVIDIA_A100_sm80_trt8.6.1
    engine_dir = (
        base_cache
        / "tensorrt_engines"
        / f"{gpu_safe}_{arch_str}_trt{trt_ver}_{dataset}"
    )

    # Create directory if it doesn't exist
    engine_dir.mkdir(parents=True, exist_ok=True)

    return engine_dir
