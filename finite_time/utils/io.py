"""Serialization, hashing, and cache management for simulation runs."""
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np


def digest(path: Path) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, data: Any) -> None:
    """Write dictionary to pretty-printed JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def save_simulation_run(
    folder: Path,
    name: str,
    result: Any,
    config: Optional[Dict[str, Any]] = None,
    metadata_extra: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Save simulation result arrays (.npz) and metadata/metrics (.json)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    npz_path = folder / f"{name}.npz"
    json_path = folder / f"{name}.json"

    arrays = result.to_dict()
    np.savez_compressed(
        npz_path,
        **arrays,
        solver_nodes=result.sol.t,
        solver_states=result.sol.y.T
    )

    meta = {
        'M': (float(result.metrics.get('m', 0.0)) * np.eye(2)).tolist(),
        'd_safe': 1.0,
        'R_c': 6.0,
    }
    if config is None:
        cfg_path = Path(__file__).resolve().parents[2] / 'configs/ecc.json'
        if cfg_path.exists():
            try:
                config = json.loads(cfg_path.read_text())
            except Exception:
                config = {}
        else:
            config = {}
    meta['configuration'] = config
    if metadata_extra:
        meta.update(metadata_extra)

    cert = {
        'certified': result.metrics.get('certified', False),
        'gamma': result.metrics.get('gamma'),
        'gamma_req': result.metrics.get('gamma_req'),
        'm_required': result.metrics.get('m_required'),
        'T_bar': result.metrics.get('T_bar'),
    }

    doc = {
        'metadata': meta,
        'data_sha256': digest(npz_path),
        'solver': result.metrics.get('solver_settings', {}),
        'metrics': result.metrics,
        'certificate': cert,
    }
    write_json(json_path, doc)

    row = dict(name=name, m=float(result.metrics.get('m', 0.0)))
    row.update(result.metrics)
    return row


def load_simulation_run(folder: Path, name: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Load arrays from NPZ and dictionary from JSON."""
    folder = Path(folder)
    with np.load(folder / f"{name}.npz") as data:
        arrays = {key: data[key] for key in data.files}
    meta = json.loads((folder / f"{name}.json").read_text())
    return arrays, meta
