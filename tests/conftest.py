"""
Shared pytest helpers.

The test suite runs without the heavy ML stack (torch, transformers, faiss,
chromadb, langgraph). Modules that import those at the top level are loaded
with lightweight stand-ins so the pure logic inside them can still be tested.
"""

import importlib.util
import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _placeholder_attr(attr: str):
    if attr.startswith("__"):
        raise AttributeError(attr)
    return type(attr, (), {})


def load_module(relative_path: str, name: str | None = None, stubs: tuple[str, ...] = ()):
    """
    Load a single source file as a module without running its package
    __init__ (agents/__init__.py imports the full LangGraph orchestrator).

    `stubs` lists third-party modules to replace with empty placeholders if
    they are not installed.
    """
    path = PROJECT_ROOT / relative_path
    for stub in stubs:
        try:
            importlib.import_module(stub)
        except ImportError:
            placeholder = types.ModuleType(stub)
            # `from stub import Anything` yields an inert placeholder class
            placeholder.__getattr__ = _placeholder_attr
            sys.modules[stub] = placeholder

    # Sibling-module fallbacks such as `from env_loader import ...`
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    module_name = name or f"_test_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
