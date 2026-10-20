"""Load the shared config.yaml used by the controller, gesture app and tools."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


def load_config(path=None) -> dict:
    import yaml  # imported here so the rest of robot_core works without PyYAML

    with open(path or DEFAULT_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)
