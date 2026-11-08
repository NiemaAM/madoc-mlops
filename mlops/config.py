"""Settings: params.yaml (what the platform does) + environment variables (secrets and addresses, from .env)."""

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("blla", "htr")
SPLITS = ("train", "val", "test")


@lru_cache(maxsize=1)
def params():
    path = Path(os.environ.get("MLOPS_PARAMS", ROOT / "params.yaml"))
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(name):
    """A folder of params.yaml (paths.data, paths.runs, paths.release); MLOPS_<NAME>_DIR replaces it."""
    folder = Path(os.environ.get(f"MLOPS_{name.upper()}_DIR") or params()["paths"][name])
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def database_url():
    url = os.environ.get("MLOPS_DATABASE_URL")
    if not url:
        raise RuntimeError("MLOPS_DATABASE_URL is not set (see .env)")
    return url


def keyed(value):
    """A keyed hash (MLOPS_SECRET) of a name that must not be stored as is: a group, an act, a writer."""
    import hashlib
    import hmac
    secret = os.environ.get("MLOPS_SECRET", "").encode() or b"change-me"
    return hmac.new(secret, value.encode(), hashlib.sha256).hexdigest()[:16]