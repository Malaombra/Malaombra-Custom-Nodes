from __future__ import annotations

import importlib.util
from pathlib import Path

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

_BASE_DIR = Path(__file__).resolve().parent
_NODES_DIR = _BASE_DIR / "nodes"


def _load_node_file(file_path: Path):
    module_name = f"{__name__}.{file_path.parent.name.replace('-', '_')}_nodes"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if _NODES_DIR.exists():
    for node_dir in sorted(_NODES_DIR.iterdir()):
        if not node_dir.is_dir() or node_dir.name.startswith("_"):
            continue

        node_file = node_dir / "nodes.py"
        if not node_file.exists():
            continue

        try:
            module = _load_node_file(node_file)
        except Exception as exc:
            print(f"[Malaombra-Custom-Nodes] Failed loading {node_file}: {exc}")
            continue

        node_mappings = getattr(module, "NODE_CLASS_MAPPINGS", None)
        if isinstance(node_mappings, dict):
            NODE_CLASS_MAPPINGS.update(node_mappings)

        display_mappings = getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", None)
        if isinstance(display_mappings, dict):
            NODE_DISPLAY_NAME_MAPPINGS.update(display_mappings)


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
