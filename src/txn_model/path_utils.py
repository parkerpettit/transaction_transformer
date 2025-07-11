import json
import os
from typing import Optional

try:
    import yaml
except Exception:  # If PyYAML isn't installed
    yaml = None


CONFIG_FILES = [
    os.path.expanduser("~/.txn_data_config.json"),
    os.path.expanduser("~/.txn_data_config.yaml"),
    os.path.expanduser("~/.txn_data_config.yml"),
]


def _load_from_file(path: str) -> Optional[str]:
    """Load the data directory from a JSON/YAML config file."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".json"):
            data = json.load(f)
        else:
            if yaml is None:
                raise RuntimeError("PyYAML required for YAML config files")
            data = yaml.safe_load(f)
    if isinstance(data, dict):
        data_dir = data.get("data_dir")
        if data_dir:
            return os.path.expanduser(str(data_dir))
    return None


def get_data_dir(default: str = "data") -> str:
    """Resolve the directory containing dataset files.

    Order of precedence:
    1. ``TXN_DATA_DIR`` environment variable
    2. Config file ``~/.txn_data_config.[json|yaml|yml]`` with key ``data_dir``
    3. Provided ``default`` relative path
    """
    env_dir = os.getenv("TXN_DATA_DIR")
    if env_dir:
        return os.path.expanduser(env_dir)

    for cfg in CONFIG_FILES:
        data_dir = _load_from_file(cfg)
        if data_dir:
            return data_dir

    return default
