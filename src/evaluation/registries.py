"""Canonical package registries used to score Package Hallucination Rate (PHR).

Each supported language has a matched pair of name lists under
`data/package_list/`, produced from the same registry snapshot:

  <lang>_package_names_without_standard.txt  -> registry packages only
                                                 (PyPI / npm / crates.io / RubyGems)
  <lang>_package_names_with_standard.txt     -> the above plus the language's
                                                 standard-library module names

The difference between the two (`with` minus `without`) is exactly the set of
standard-library names, which lets us distinguish a "hallucinated" package
(does not exist anywhere) from a package name that is merely a standard-
library module the model imported instead of a third-party dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PACKAGE_LIST_DIR = ROOT / "data" / "package_list"

REGISTRY_FILES: Dict[str, Tuple[str, str]] = {
    "Python": (
        "python_package_names_without_standard.txt",
        "python_package_names_with_standard.txt",
    ),
    "JavaScript": (
        "javascript_package_names_without_standard.txt",
        "javascript_package_names_with_standard.txt",
    ),
    "Rust": (
        "rust_package_names_without_standard.txt",
        "rust_package_names_with_standard.txt",
    ),
    "Ruby": (
        "ruby_package_names_without_standard.txt",
        "ruby_package_names_with_standard.txt",
    ),
}


def _load(path: Path) -> set:
    if not path.is_file():
        raise FileNotFoundError(f"Package list not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip()}


class PackageRegistry:
    """Loads canonical + stdlib package name sets and classifies candidate names."""

    def __init__(self, package_list_dir: Path | str = DEFAULT_PACKAGE_LIST_DIR):
        package_list_dir = Path(package_list_dir)
        self._registry: Dict[str, set] = {}
        self._stdlib: Dict[str, set] = {}
        for language, (without_name, with_name) in REGISTRY_FILES.items():
            without = _load(package_list_dir / without_name)
            with_std = _load(package_list_dir / with_name)
            self._registry[language] = without
            self._stdlib[language] = with_std - without

    @property
    def languages(self) -> List[str]:
        return list(self._registry)

    def classify(self, name: str, language: str) -> str:
        """Classify a candidate package name as 'registry', 'stdlib', or 'hallucinated'."""
        n = name.strip().lower()
        if language not in self._registry:
            raise KeyError(f"No registry loaded for language {language!r}")
        if n in self._registry[language]:
            return "registry"
        if n in self._stdlib[language]:
            return "stdlib"
        return "hallucinated"
