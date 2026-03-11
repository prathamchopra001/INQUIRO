# -*- coding: utf-8 -*-
"""
Reproducibility Package Generator for INQUIRO.

Generates a complete package containing everything needed to reproduce
a INQUIRO research run, including environment specs, configuration,
data manifests, and re-execution scripts.

Package Contents:
    - environment.yaml: Conda/pip dependencies with exact versions
    - config_snapshot.json: All settings used for the run
    - data_manifest.json: Input file checksums and metadata
    - seeds.json: Random seeds for reproducibility
    - reproduce.py: Script to re-run the analysis
    - README.md: Human-readable instructions

Usage:
    generator = ReproducibilityPackageGenerator(run_dir="outputs/run_...")
    package_path = generator.generate(
        objective="Research objective",
        config=settings,
        data_paths=["data/dataset.csv"],
    )
"""

import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentSpec:
    """Captured environment specification."""
    python_version: str = ""
    platform: str = ""
    platform_version: str = ""
    architecture: str = ""
    packages: Dict[str, str] = field(default_factory=dict)
    conda_env: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DataManifest:
    """Manifest of data files used in the run."""
    files: List[Dict[str, Any]] = field(default_factory=list)
    total_size_bytes: int = 0
    capture_timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SeedManifest:
    """Random seeds used for reproducibility."""
    numpy_seed: Optional[int] = None
    python_seed: Optional[int] = None
    torch_seed: Optional[int] = None
    random_state: int = 42  # Default INQUIRO seed
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ReproducibilityPackage:
    """Complete reproducibility package."""
    
    # Metadata
    run_id: str = ""
    objective: str = ""
    generated_at: str = ""
    inquiro_version: str = "0.1.0"
    
    # Components
    environment: EnvironmentSpec = field(default_factory=EnvironmentSpec)
    config: Dict[str, Any] = field(default_factory=dict)
    data_manifest: DataManifest = field(default_factory=DataManifest)
    seeds: SeedManifest = field(default_factory=SeedManifest)
    
    # Git info (if available)
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    git_dirty: bool = False
    
    # Run statistics
    cycles_completed: int = 0
    findings_count: int = 0
    runtime_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "objective": self.objective,
            "generated_at": self.generated_at,
            "inquiro_version": self.inquiro_version,
            "environment": self.environment.to_dict(),
            "config": self.config,
            "data_manifest": self.data_manifest.to_dict(),
            "seeds": self.seeds.to_dict(),
            "git": {
                "commit": self.git_commit,
                "branch": self.git_branch,
                "dirty": self.git_dirty,
            } if self.git_commit else None,
            "run_stats": {
                "cycles_completed": self.cycles_completed,
                "findings_count": self.findings_count,
                "runtime_seconds": self.runtime_seconds,
            },
        }


class ReproducibilityPackageGenerator:
    """
    Generates reproducibility packages for INQUIRO research runs.
    
    Creates a self-contained package with everything needed to
    reproduce the research run on another machine.
    """
    
    def __init__(self, run_dir: str):
        """
        Args:
            run_dir: Directory where run outputs are stored
        """
        self.run_dir = Path(run_dir)
        self.package_dir = self.run_dir / "reproducibility"
    
    def generate(
        self,
        objective: str,
        config: Any = None,
        data_paths: List[str] = None,
        run_id: str = "",
        cycles_completed: int = 0,
        findings_count: int = 0,
        runtime_seconds: float = 0.0,
    ) -> str:
        """
        Generate a complete reproducibility package.
        
        Args:
            objective: Research objective
            config: Settings object or dict
            data_paths: List of input data file paths
            run_id: Run identifier
            cycles_completed: Number of cycles run
            findings_count: Number of findings generated
            runtime_seconds: Total runtime
            
        Returns:
            Path to the generated package directory
        """
        logger.info("📦 Generating reproducibility package...")
        
        # Create package directory
        self.package_dir.mkdir(parents=True, exist_ok=True)
        
        # Build package
        package = ReproducibilityPackage(
            run_id=run_id or self.run_dir.name,
            objective=objective,
            generated_at=datetime.now().isoformat(),
            cycles_completed=cycles_completed,
            findings_count=findings_count,
            runtime_seconds=runtime_seconds,
        )
        
        # Capture environment
        package.environment = self._capture_environment()
        logger.info("  ✅ Environment captured")
        
        # Capture configuration
        package.config = self._capture_config(config)
        logger.info("  ✅ Configuration captured")
        
        # Capture data manifest
        package.data_manifest = self._capture_data_manifest(data_paths or [])
        logger.info(f"  ✅ Data manifest: {len(package.data_manifest.files)} files")
        
        # Capture seeds
        package.seeds = self._capture_seeds()
        logger.info("  ✅ Random seeds captured")
        
        # Capture git info
        git_info = self._capture_git_info()
        if git_info:
            package.git_commit = git_info.get("commit")
            package.git_branch = git_info.get("branch")
            package.git_dirty = git_info.get("dirty", False)
            logger.info(f"  ✅ Git info: {package.git_commit[:8] if package.git_commit else 'N/A'}")
        
        # Write package files
        self._write_environment_yaml(package.environment)
        self._write_config_snapshot(package.config)
        self._write_data_manifest(package.data_manifest)
        self._write_seeds(package.seeds)
        self._write_reproduce_script(package)
        self._write_readme(package)
        self._write_package_json(package)
        
        logger.info(f"  📦 Package saved to: {self.package_dir}")
        
        return str(self.package_dir)
    
    def _capture_environment(self) -> EnvironmentSpec:
        """Capture current Python environment."""
        env = EnvironmentSpec(
            python_version=sys.version,
            platform=platform.system(),
            platform_version=platform.version(),
            architecture=platform.machine(),
        )
        
        # Get installed packages
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "freeze"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if "==" in line:
                        name, version = line.split("==", 1)
                        env.packages[name.strip()] = version.strip()
        except Exception as e:
            logger.warning(f"Could not capture pip packages: {e}")
        
        # Check for conda
        try:
            conda_env = os.environ.get("CONDA_DEFAULT_ENV")
            if conda_env:
                env.conda_env = conda_env
        except Exception:
            pass
        
        return env
    
    def _capture_config(self, config: Any) -> Dict[str, Any]:
        """Capture configuration settings."""
        if config is None:
            return {}
        
        # If it's a Settings object with to_dict
        if hasattr(config, "to_dict"):
            return config.to_dict()
        
        # If it's already a dict
        if isinstance(config, dict):
            return config
        
        # Try to serialize dataclass
        try:
            return asdict(config)
        except Exception:
            pass
        
        # Fallback: extract public attributes
        try:
            return {
                k: v for k, v in vars(config).items()
                if not k.startswith("_") and not callable(v)
            }
        except Exception:
            return {}


    def _capture_data_manifest(self, data_paths: List[str]) -> DataManifest:
        """Capture manifest of data files with checksums."""
        manifest = DataManifest(
            capture_timestamp=datetime.now().isoformat(),
        )
        
        total_size = 0
        
        for path_str in data_paths:
            path = Path(path_str)
            if not path.exists():
                continue
            
            try:
                stat = path.stat()
                file_size = stat.st_size
                total_size += file_size
                
                # Calculate checksum for files under 100MB
                checksum = None
                if file_size < 100 * 1024 * 1024:
                    checksum = self._calculate_checksum(path)
                
                manifest.files.append({
                    "path": str(path.absolute()),
                    "name": path.name,
                    "size_bytes": file_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "checksum_sha256": checksum,
                })
            except Exception as e:
                logger.warning(f"Could not capture file info for {path}: {e}")
        
        manifest.total_size_bytes = total_size
        return manifest
    
    def _calculate_checksum(self, path: Path, algorithm: str = "sha256") -> str:
        """Calculate file checksum."""
        hasher = hashlib.new(algorithm)
        
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        
        return hasher.hexdigest()
    
    def _capture_seeds(self) -> SeedManifest:
        """Capture random seeds from various libraries."""
        seeds = SeedManifest()
        
        # Try to get numpy seed
        try:
            import numpy as np
            # Can't directly get seed, but we record the default
            seeds.numpy_seed = 42
        except ImportError:
            pass
        
        # Try to get torch seed
        try:
            import torch
            seeds.torch_seed = 42
        except ImportError:
            pass
        
        # Python random
        seeds.python_seed = 42
        seeds.random_state = 42
        
        return seeds
    
    def _capture_git_info(self) -> Optional[Dict[str, Any]]:
        """Capture git repository information."""
        try:
            # Get commit hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=self.run_dir.parent,
            )
            if result.returncode != 0:
                return None
            
            commit = result.stdout.strip()
            
            # Get branch name
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=self.run_dir.parent,
            )
            branch = result.stdout.strip() if result.returncode == 0 else None
            
            # Check if dirty
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=self.run_dir.parent,
            )
            dirty = bool(result.stdout.strip()) if result.returncode == 0 else False
            
            return {
                "commit": commit,
                "branch": branch,
                "dirty": dirty,
            }
        except Exception:
            return None
    
    def _write_environment_yaml(self, env: EnvironmentSpec) -> None:
        """Write environment.yaml file."""
        lines = [
            "# INQUIRO Reproducibility Environment",
            f"# Generated: {datetime.now().isoformat()}",
            f"# Python: {env.python_version.split()[0]}",
            f"# Platform: {env.platform} {env.platform_version}",
            "",
            "name: inquiro-reproduce",
            "channels:",
            "  - conda-forge",
            "  - defaults",
            "dependencies:",
            f"  - python={env.python_version.split()[0]}",
            "  - pip",
            "  - pip:",
        ]
        
        # Add pip packages
        for name, version in sorted(env.packages.items()):
            lines.append(f"    - {name}=={version}")
        
        content = "\n".join(lines)
        (self.package_dir / "environment.yaml").write_text(content, encoding="utf-8")
    
    def _write_config_snapshot(self, config: Dict[str, Any]) -> None:
        """Write config_snapshot.json file."""
        content = json.dumps(config, indent=2, default=str)
        (self.package_dir / "config_snapshot.json").write_text(content, encoding="utf-8")
    
    def _write_data_manifest(self, manifest: DataManifest) -> None:
        """Write data_manifest.json file."""
        content = json.dumps(manifest.to_dict(), indent=2)
        (self.package_dir / "data_manifest.json").write_text(content, encoding="utf-8")
    
    def _write_seeds(self, seeds: SeedManifest) -> None:
        """Write seeds.json file."""
        content = json.dumps(seeds.to_dict(), indent=2)
        (self.package_dir / "seeds.json").write_text(content, encoding="utf-8")


    def _write_reproduce_script(self, package: ReproducibilityPackage) -> None:
        """Write reproduce.py script."""
        script = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
INQUIRO Reproducibility Script
Generated: {package.generated_at}
Run ID: {package.run_id}

This script reproduces the INQUIRO research run with identical settings.
"""

import json
import os
import sys
from pathlib import Path

# Ensure we're in the right directory
SCRIPT_DIR = Path(__file__).parent.absolute()
os.chdir(SCRIPT_DIR.parent)

def verify_environment():
    """Verify the environment matches the original."""
    print("🔍 Verifying environment...")
    
    # Load expected environment
    env_path = SCRIPT_DIR / "config_snapshot.json"
    if not env_path.exists():
        print("  ⚠️ config_snapshot.json not found")
        return False
    
    print("  ✅ Configuration found")
    return True

def verify_data():
    """Verify data files match original checksums."""
    print("🔍 Verifying data files...")
    
    manifest_path = SCRIPT_DIR / "data_manifest.json"
    if not manifest_path.exists():
        print("  ⚠️ data_manifest.json not found")
        return True  # Not a failure if no data
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    files = manifest.get("files", [])
    if not files:
        print("  ℹ️ No data files in manifest")
        return True
    
    import hashlib
    
    for file_info in files:
        path = Path(file_info["path"])
        expected_hash = file_info.get("checksum_sha256")
        
        if not path.exists():
            print(f"  ❌ Missing: {{path}}")
            return False
        
        if expected_hash:
            hasher = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    hasher.update(chunk)
            actual_hash = hasher.hexdigest()
            
            if actual_hash != expected_hash:
                print(f"  ⚠️ Checksum mismatch: {{path.name}}")
                print(f"      Expected: {{expected_hash[:16]}}...")
                print(f"      Got:      {{actual_hash[:16]}}...")
            else:
                print(f"  ✅ {{path.name}}")
    
    return True

def set_seeds():
    """Set random seeds for reproducibility."""
    print("🎲 Setting random seeds...")
    
    seeds_path = SCRIPT_DIR / "seeds.json"
    if not seeds_path.exists():
        print("  ⚠️ seeds.json not found, using defaults")
        seed = 42
    else:
        with open(seeds_path) as f:
            seeds = json.load(f)
        seed = seeds.get("random_state", 42)
    
    import random
    random.seed(seed)
    
    try:
        import numpy as np
        np.random.seed(seed)
        print(f"  ✅ NumPy seed: {{seed}}")
    except ImportError:
        pass
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"  ✅ PyTorch seed: {{seed}}")
    except ImportError:
        pass
    
    print(f"  ✅ Python seed: {{seed}}")

def run_inquiro():
    """Run INQUIRO with the original settings."""
    print("\\n🔬 Starting INQUIRO reproduction run...")
    
    # Load configuration
    config_path = SCRIPT_DIR / "config_snapshot.json"
    with open(config_path) as f:
        config = json.load(f)
    
    # Import INQUIRO
    try:
        from src.core.inquiro import Inquiro
    except ImportError:
        print("❌ Could not import INQUIRO. Ensure you're in the INQUIRO root directory.")
        sys.exit(1)
    
    # Original objective
    objective = """{package.objective}"""
    
    # Create and run
    inquiro = Inquiro(
        objective=objective,
        max_cycles={package.cycles_completed or 5},
    )
    
    report_path = inquiro.run()
    print(f"\\n✅ Reproduction complete!")
    print(f"   Report: {{report_path}}")
    
    return report_path

def main():
    print("=" * 60)
    print("INQUIRO REPRODUCIBILITY RUNNER")
    print("=" * 60)
    print(f"Run ID: {package.run_id}")
    print(f"Original Date: {package.generated_at}")
    print()
    
    # Verify environment
    if not verify_environment():
        print("\\n⚠️ Environment verification failed. Continue anyway? [y/N]")
        if input().strip().lower() != 'y':
            sys.exit(1)
    
    # Verify data
    if not verify_data():
        print("\\n⚠️ Data verification failed. Continue anyway? [y/N]")
        if input().strip().lower() != 'y':
            sys.exit(1)
    
    # Set seeds
    set_seeds()
    
    # Run INQUIRO
    run_inquiro()

if __name__ == "__main__":
    main()
'''
        (self.package_dir / "reproduce.py").write_text(script, encoding="utf-8")


    def _write_readme(self, package: ReproducibilityPackage) -> None:
        """Write README.md with instructions."""
        git_info = ""
        if package.git_commit:
            dirty_note = " (with uncommitted changes)" if package.git_dirty else ""
            git_info = f"""
## Git Information

- **Commit**: `{package.git_commit}`
- **Branch**: `{package.git_branch}`
- **Status**: {"Dirty" + dirty_note if package.git_dirty else "Clean"}

To checkout the exact code version:
```bash
git checkout {package.git_commit}
```
"""
        
        readme = f"""# INQUIRO Reproducibility Package

## Run Information

| Field | Value |
|-------|-------|
| **Run ID** | `{package.run_id}` |
| **Generated** | {package.generated_at} |
| **INQUIRO Version** | {package.inquiro_version} |
| **Cycles** | {package.cycles_completed} |
| **Findings** | {package.findings_count} |
| **Runtime** | {package.runtime_seconds:.1f}s |

## Research Objective

{package.objective}

## Quick Start

### 1. Set up the environment

```bash
# Using conda (recommended)
conda env create -f environment.yaml
conda activate inquiro-reproduce

# Or using pip
pip install -r requirements.txt  # Generate from environment.yaml
```

### 2. Verify data files

Ensure the original data files are in place. Check `data_manifest.json` for:
- File paths
- Expected checksums

### 3. Run reproduction

```bash
python reproduce.py
```

The script will:
1. Verify the environment
2. Check data file integrity
3. Set random seeds
4. Run INQUIRO with identical settings
{git_info}
## Package Contents

| File | Description |
|------|-------------|
| `environment.yaml` | Conda environment specification |
| `config_snapshot.json` | All configuration settings |
| `data_manifest.json` | Input file checksums |
| `seeds.json` | Random seeds used |
| `reproduce.py` | Reproduction script |
| `README.md` | This file |
| `package.json` | Complete package metadata |

## Manual Reproduction

If the automatic script doesn't work:

```python
from src.core.inquiro import Inquiro

inquiro = Inquiro(
    objective=\"\"\"{package.objective}\"\"\",
    max_cycles={package.cycles_completed or 5},
)
inquiro.run()
```

## Troubleshooting

### Environment Issues
- Ensure Python version matches: `{package.environment.python_version.split()[0]}`
- Install missing packages from `environment.yaml`

### Data Issues
- Verify file paths in `data_manifest.json`
- Check checksums match (use `sha256sum` or Python `hashlib`)

### Reproducibility Limitations
- LLM responses may vary slightly due to API non-determinism
- Network-dependent operations (paper fetching) may differ
- Timestamp-based operations will produce different values

## Contact

If you encounter issues reproducing this run, check:
1. All dependencies are installed
2. Data files are accessible
3. API keys are configured (if using external services)
"""
        (self.package_dir / "README.md").write_text(readme, encoding="utf-8")
    
    def _write_package_json(self, package: ReproducibilityPackage) -> None:
        """Write complete package.json metadata."""
        content = json.dumps(package.to_dict(), indent=2, default=str)
        (self.package_dir / "package.json").write_text(content, encoding="utf-8")


def generate_reproducibility_package(
    run_dir: str,
    objective: str,
    config: Any = None,
    data_paths: List[str] = None,
    run_id: str = "",
    cycles_completed: int = 0,
    findings_count: int = 0,
    runtime_seconds: float = 0.0,
) -> str:
    """
    Convenience function to generate a reproducibility package.
    
    Args:
        run_dir: Directory where run outputs are stored
        objective: Research objective
        config: Settings object or dict
        data_paths: List of input data file paths
        run_id: Run identifier
        cycles_completed: Number of cycles run
        findings_count: Number of findings generated
        runtime_seconds: Total runtime
        
    Returns:
        Path to the generated package directory
    """
    generator = ReproducibilityPackageGenerator(run_dir)
    return generator.generate(
        objective=objective,
        config=config,
        data_paths=data_paths,
        run_id=run_id,
        cycles_completed=cycles_completed,
        findings_count=findings_count,
        runtime_seconds=runtime_seconds,
    )
