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
    candidates.append(Path("/cluster/scratch/mgaetzner/uqct/"))
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
    return get_root_dir() / "checkpoints"
