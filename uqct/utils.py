import os
from pathlib import Path
from typing import List, Optional


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
    candidates += [Path("/cluster/scratch/mgaetzner/uqct/")]
    git_root = _git_root(Path(__file__).resolve().parent)
    if git_root:
        candidates.append(git_root)
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
