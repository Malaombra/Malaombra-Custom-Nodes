from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

_BASE_DIR = Path(__file__).resolve().parent
_NODES_DIR = _BASE_DIR / "nodes"


def _auto_update_repository(repo_dir: Path):
    log_prefix = "[Malaombra-Custom-Nodes][AutoUpdate]"

    # Opt-out with MALAOMBRA_AUTO_UPDATE=0
    if os.environ.get("MALAOMBRA_AUTO_UPDATE", "1").lower() in {"0", "false", "no"}:
        print(f"{log_prefix} check skipped (disabled via MALAOMBRA_AUTO_UPDATE).")
        return

    if not (repo_dir / ".git").exists():
        print(f"{log_prefix} check skipped (.git not found).")
        return

    print(f"{log_prefix} checking for updates...")

    try:
        result = subprocess.run(
            ["git", "-C", str(repo_dir), "pull", "--ff-only"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except FileNotFoundError:
        print(f"{log_prefix} result: failed (git not found).")
        return
    except Exception as exc:
        print(f"{log_prefix} result: failed ({exc}).")
        return

    if result.returncode != 0:
        details = ((result.stderr or "") + "\n" + (result.stdout or "")).strip()
        details_line = details.splitlines()[-1] if details else "unknown error"
        print(f"{log_prefix} result: failed ({details_line}).")
        return

    output = ((result.stdout or "") + "\n" + (result.stderr or "")).strip()
    if "Already up to date." in output:
        print(f"{log_prefix} result: already up to date.")
        return

    print(f"{log_prefix} result: updated successfully.")


def _load_node_file(file_path: Path):
    module_name = f"{__name__}.{file_path.parent.name.replace('-', '_')}_nodes"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_auto_update_repository(_BASE_DIR)

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
