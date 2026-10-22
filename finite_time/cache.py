"""Checked reuse of numerical outputs and completed validation work."""
import hashlib
import json
from pathlib import Path
import numpy as np
import scipy
import numba

SCIENCE_FILES = ('model.py','integrate.py','common.py','theory.py','metrics.py',
                 'formations.py','sampling.py','accelerate.py')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def source_hashes():
    folder = Path(__file__).parent
    return {name:digest(folder/name) for name in SCIENCE_FILES}


def signature(config,extra=None):
    report = Path(__file__).resolve().parents[1]/'ACC_paper/finite-time_report.tex'
    evidence = dict(config=config,source=source_hashes(),theory=digest(report),
                    numpy=np.__version__,scipy=scipy.__version__,numba=numba.__version__,extra=extra)
    return hashlib.sha256(json.dumps(evidence,sort_keys=True).encode()).hexdigest()


def cached_run(folder,name,config,load_arrays=True,hashes=None):
    """Old/unverified or changed outputs are never silently reused."""
    folder = Path(folder)
    path, raw = folder/f'{name}.json', folder/f'{name}.npz'
    try:
        saved = json.loads(path.read_text())
        meta = saved['metadata']
        hashes = source_hashes() if hashes is None else hashes
        checks = saved['metrics']['checks']
        if (meta['configuration'] != config or not checks or not all(checks.values()) or
                any(meta['implementation_sha256'].get(k)!=v for k,v in hashes.items()) or
                meta.get('numpy')!=np.__version__ or meta.get('scipy')!=scipy.__version__ or
                meta.get('numba')!=numba.__version__ or
                meta.get('report_sha256')!=digest(Path(__file__).resolve().parents[1]/'ACC_paper/finite-time_report.tex') or
                saved.get('data_sha256')!=digest(raw)):
            return None
        if load_arrays:
            with np.load(raw) as data:
                arrays = {key:data[key] for key in data.files
                          if key not in ('solver_nodes','solver_states')}
            return arrays,saved
        return saved
    except (OSError,KeyError,ValueError,TypeError):
        return None


def cache_valid(path,token):
    path = Path(path)
    try:
        saved = json.loads(path.read_text())
        return (saved['signature']==token and bool(saved['outputs']) and
                all(digest(path.parent/name)==sha for name,sha in saved['outputs'].items()))
    except (OSError,KeyError,ValueError,TypeError):
        return False


def write_cache(path,token,files):
    from .experiments import write_json
    path = Path(path)
    write_json(path,dict(signature=token,outputs={
        str(Path(file).relative_to(path.parent)):digest(file) for file in files}))


def active_grid_paths(root):
    """Read exactly the current grid, excluding leftover cases from larger grids."""
    folder = Path(root)/'experiment_4'
    if not (folder/'grid.npz').exists():
        return []
    with np.load(folder/'grid.npz') as grid:
        ms,kappas = grid['m'],grid['kappa']
        if grid['classification'].shape != (len(ms),len(kappas)):
            raise ValueError('parameter grid dimensions do not match its coordinates')
    paths = []
    for i,m in enumerate(ms):
        for j,kappa in enumerate(kappas):
            path = folder/'runs'/f'case_{i*len(kappas)+j:04d}.json'
            saved = json.loads(path.read_text())
            if saved.get('grid_m')!=float(m) or saved.get('grid_kappa')!=float(kappa):
                raise ValueError(f'{path}: saved parameters do not match the active grid')
            paths.append(path)
    return paths


def experiment_paths(root,number):
    if number==4:return active_grid_paths(root)
    folder=Path(root)/f'experiment_{number}'
    manifest=folder/'run_manifest.json'
    if manifest.exists():
        return [folder/f'{name}.json' for name in json.loads(manifest.read_text())['runs']]
    patterns={1:'morphing.json',2:'m_*.json',3:'*.json'}
    return [p for p in folder.glob(patterns[number])
            if number!=3 or p.stem.startswith(('two_agent','m_'))]
