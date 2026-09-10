#!/usr/bin/env python3
"""
run_config_dump.py
==================
B1 section 1.5 of EXPERIMENT_PLAN_AUDIT_REMEDIATION.md. A single guarded helper
that writes `run_config.json` into a run's output directory so provenance stops
living only in directory names, plan markdown and shell history.

Follows the `gainjitter` precedent: additive, guarded, never raises into the run,
and RNG-inert with an assertion. Writing a JSON file after argument parsing and
before the fold loop cannot perturb training; the assertion is the house rule.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(_HERE),
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def dump_run_config(out_dir, args, resolved_paths: dict | None = None) -> None:
    """Write out_dir/run_config.json with vars(args), resolved input paths, the
    git commit and library versions. Guarded; RNG state is asserted unchanged."""
    # ---- snapshot RNG (torch if present, numpy always) ----
    _torch = None
    _t_state = None
    try:
        import torch as _torch  # noqa: F811
        _t_state = _torch.get_rng_state().clone()
    except Exception:
        _torch = None
    import numpy as _np
    _np_state = _np.random.get_state()

    try:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg: dict = {
            "argv": list(sys.argv),
            "args": {k: (v if isinstance(v, (int, float, bool, str, type(None), list, dict))
                         else str(v))
                     for k, v in vars(args).items()},
            "resolved_paths": {k: (str(Path(v).resolve()) if v is not None else None)
                               for k, v in (resolved_paths or {}).items()},
            "git_commit": _git_commit(),
            "python": sys.version.split()[0],
        }
        try:
            import torch as _t
            cfg["torch"] = _t.__version__
            cfg["cuda"] = getattr(_t.version, "cuda", None)
            cfg["cudnn"] = (_t.backends.cudnn.version()
                            if _t.backends.cudnn.is_available() else None)
            cfg["gpu"] = (_t.cuda.get_device_name(0) if _t.cuda.is_available() else None)
        except Exception:
            pass
        for _mod in ("numpy", "scipy", "sklearn", "pandas"):
            try:
                cfg[_mod] = __import__(_mod).__version__
            except Exception:
                pass
        (out_dir / "run_config.json").write_text(json.dumps(cfg, indent=2, default=str))
        print(f"[run_config] wrote {out_dir / 'run_config.json'}", flush=True)
    except Exception as e:  # never break a run over provenance
        print(f"[run_config] WARNING could not write run_config.json: {e}", flush=True)

    # ---- RNG-inertness assertions (house rule) ----
    if _torch is not None and _t_state is not None:
        assert _torch.equal(_torch.get_rng_state(), _t_state), \
            "run_config dump perturbed the torch RNG state"
    _np_now = _np.random.get_state()
    assert _np_now[0] == _np_state[0] and _np_now[2] == _np_state[2] and \
        bool((_np_now[1] == _np_state[1]).all()), \
        "run_config dump perturbed the numpy RNG state"
