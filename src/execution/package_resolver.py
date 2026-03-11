# -*- coding: utf-8 -*-
"""
Automatic package resolver for Docker code execution.

When generated code fails with ModuleNotFoundError or ImportError,
this resolver:
  1. Detects the missing package from stderr
  2. Prepends a pip install command to the code
  3. Returns the patched code for retry

This solves the recurring issue where LLMs generate code using
packages that are not pre-installed in the sandbox image.
"""

import re
import logging

logger = logging.getLogger(__name__)

# Map of common import names to their pip package names
# (import name != pip name in many cases)
IMPORT_TO_PACKAGE = {
    "sklearn":          "scikit-learn",
    "cv2":              "opencv-python",
    "PIL":              "Pillow",
    "bs4":              "beautifulsoup4",
    "yaml":             "pyyaml",
    "dotenv":           "python-dotenv",
    "Bio":              "biopython",
    "statsmodels":      "statsmodels",
    "scipy":            "scipy",
    "seaborn":          "seaborn",
    "matplotlib":       "matplotlib",
    "pandas":           "pandas",
    "numpy":            "numpy",
    "networkx":         "networkx",
    "gseapy":           "gseapy",
    "lifelines":        "lifelines",
    "umap":             "umap-learn",
    "plotly":           "plotly",
    "bokeh":            "bokeh",
    "xgboost":          "xgboost",
    "lightgbm":         "lightgbm",
    "catboost":         "catboost",
    "shap":             "shap",
    "pydeseq2":         "pydeseq2",
    "pingouin":         "pingouin",
    "mlxtend":          "mlxtend",
}

# Packages that should never be auto-installed (security/stability)
BLOCKED_PACKAGES = {
    "os", "sys", "subprocess", "socket", "requests",
    "urllib", "http", "ftplib", "smtplib",
}


class PackageResolver:
    """
    Detects missing packages from execution errors and patches code.

    Usage:
        resolver = PackageResolver()

        # After a failed execution:
        if "ModuleNotFoundError" in result.stderr:
            patched_code = resolver.patch_code(code, result.stderr)
            if patched_code:
                # retry with patched_code
    """

    def __init__(self):
        self._installed: set = set()  # Track what we've already added

    def detect_missing_packages(self, stderr: str) -> list[str]:
        """
        Extract missing package names from error output.

        Handles patterns like:
          ModuleNotFoundError: No module named 'sklearn'
          ImportError: cannot import name 'linkage' from 'sklearn.cluster'
          ModuleNotFoundError: No module named 'gseapy'

        Args:
            stderr: The captured stderr from Docker execution

        Returns:
            List of pip-installable package names
        """
        packages = []

        # Pattern 1: No module named 'X' or No module named 'X.Y'
        no_module = re.findall(
            r"No module named '([^']+)'",
            stderr
        )
        for match in no_module:
            # Take the top-level module name (before any dot)
            top_level = match.split(".")[0]
            pkg = self._resolve_package_name(top_level)
            if pkg and pkg not in packages:
                packages.append(pkg)

        # Pattern 2: cannot import name 'X' from 'Y'
        bad_import = re.findall(
            r"cannot import name '([^']+)' from '([^']+)'",
            stderr
        )
        for _, module in bad_import:
            top_level = module.split(".")[0]
            pkg = self._resolve_package_name(top_level)
            if pkg and pkg not in packages:
                packages.append(pkg)

        # Filter blocked packages
        packages = [p for p in packages if p not in BLOCKED_PACKAGES]

        if packages:
            logger.info(f"PackageResolver: detected missing packages: {packages}")

        return packages

    def _resolve_package_name(self, import_name: str) -> str:
        """Convert an import name to its pip package name."""
        if not import_name:
            return None

        # Check our mapping first
        if import_name in IMPORT_TO_PACKAGE:
            return IMPORT_TO_PACKAGE[import_name]

        # Default: pip name usually matches import name
        return import_name

    def patch_code(self, code: str, stderr: str) -> str:
        """
        Prepend pip install commands for missing packages.

        Args:
            code:   Original code that failed
            stderr: Error output from failed execution

        Returns:
            Patched code with pip install prepended,
            or original code if no packages detected.
        """
        missing = self.detect_missing_packages(stderr)

        # Only install packages we haven't already tried
        to_install = [p for p in missing if p not in self._installed]

        if not to_install:
            return code

        # Build pip install block
        install_lines = [
            "# -*- coding: utf-8 -*-",
            "# Auto-installing missing packages",
            "import subprocess, sys",
        ]
        for pkg in to_install:
            install_lines.append(
                f"subprocess.run([sys.executable, '-m', 'pip', 'install', "
                f"'{pkg}', '--quiet', '--break-system-packages'], "
                f"capture_output=True)"
            )
            self._installed.add(pkg)
            logger.info(f"PackageResolver: will install '{pkg}'")

        install_lines.append("")  # blank line before user code

        # Remove existing coding declaration from original code to avoid duplicate
        clean_code = re.sub(r"^# -\*- coding.*-\*-\s*\n?", "", code, count=1)

        patched = "\n".join(install_lines) + "\n" + clean_code

        logger.info(
            f"PackageResolver: patched code with {len(to_install)} installs"
        )
        return patched

    def get_stats(self) -> dict:
        return {"packages_installed": sorted(self._installed)}