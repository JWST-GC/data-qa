"""One probe into the pipeline's OWN policy modules, so the trigger's env
defaults cannot disagree with what the reduction actually writes (issue #69).

The pipeline derives cataloging's ``each_suffix`` from
``jwst_gc_pipeline.reduction.destreak_policy.crf_suffix`` (see the pipeline's
``run_pipeline.build_plan``), and stage 1 consults ``destreaks()`` per filter
before writing ``*_destreak_o<obs>_crf.fits`` vs ``*_align_o<obs>_crf.fits``.
The trigger runs in a different (stdlib-only) environment, so it asks the same
policy by subprocessing the pipeline env's python -- one call per (field, obs)
per run, JSON over stdout.

Degrades, never crashes: any probe failure (missing checkout, nonzero exit,
timeout, unparseable output) prints a loud WARNING naming the mismatch risk and
returns None, and the caller falls back to the trigger's historical hardcoded
default -- exactly today's behavior.

Stdlib-only, like the rest of the trigger path.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Dict, Optional, Tuple

#: Same pipe-root convention as pipeline_trigger (the live reduction checkout).
DEFAULT_PIPE_ROOT = "/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline"

#: The pipeline env's python -- the same default the pipeline's own submit
#: scripts use (``PYTHON=${PYTHON:-...}`` in submit_reduction.sbatch).
#: Override with $PIPELINE_PYTHON (or the ``python=`` argument).
DEFAULT_PIPELINE_PYTHON = ("/blue/adamginsburg/adamginsburg/miniconda3/envs/"
                           "python313/bin/python")

#: destreak_policy is a tiny stdlib module; the probe measured ~1 s live.  The
#: generous cap only bounds a wedged filesystem/env, after which we fall back.
PROBE_TIMEOUT_S = 120

# Runs INSIDE the pipeline env: argv = [pipe_root, field, obs, filter, ...].
# pipe_root goes FIRST on sys.path so the requested checkout wins over any
# installed copy of the package.  Mirrors the pipeline's run_pipeline.build_plan
# (each_suffix from destreak_policy.crf_suffix) so trigger and manual runs share
# one source of truth.
PROBE_CODE = """\
import json, sys
sys.path.insert(0, sys.argv[1])
from jwst_gc_pipeline.reduction import destreak_policy
field, obs, filters = sys.argv[2], sys.argv[3], sys.argv[4:]
print(json.dumps({
    "each_suffix": destreak_policy.crf_suffix(field, filters[0], obs),
    "destreaks": {f: bool(destreak_policy.destreaks(field, f))
                  for f in filters},
}))
"""

#: (field, obs, filters, pipe_root, python) -> probe result (None = failed).
#: Failures are cached too, so a broken env is probed once per run, never
#: hammered from a loop over observations.
_CACHE: Dict[Tuple, Optional[dict]] = {}


def pipeline_python() -> str:
    return os.environ.get("PIPELINE_PYTHON", DEFAULT_PIPELINE_PYTHON)


def clear_cache():
    _CACHE.clear()


def _warn_fallback(field, obs, reason):
    print(f"WARNING: pipeline destreak-policy probe FAILED for {field} "
          f"o{obs}: {reason}.\n"
          f"WARNING: falling back to the trigger's hardcoded default "
          f"EACH_SUFFIX=align_o{obs}_crf.  If destreak_policy.destreaks() is "
          f"True for this field (it is for gc-treasury), the reduction writes "
          f"*_destreak_o{obs}_crf.fits and the dependency-chained cataloging "
          f"globs ZERO inputs -- every chain fails at m1 until a human "
          f"resubmits (data-qa#69).", file=sys.stderr)


def probe_policy(field, obs, filters, pipe_root=DEFAULT_PIPE_ROOT,
                 python=None, timeout=PROBE_TIMEOUT_S) -> Optional[dict]:
    """The pipeline's destreak policy for (field, obs, filters), or None.

    Returns ``{"each_suffix": "destreak_o<obs>_crf" | "align_o<obs>_crf",
    "destreaks": {FILTER: bool, ...}}`` -- the same values the pipeline's own
    ``run_pipeline.build_plan`` derives -- by running PROBE_CODE in the
    pipeline env's python.  Cached per (field, obs) within a run.  On ANY
    failure it warns loudly (naming the mismatch risk) and returns None so the
    caller degrades to today's hardcoded default.
    """
    python = python or pipeline_python()
    key = (str(field), str(obs), tuple(filters), pipe_root, python)
    if key in _CACHE:
        return _CACHE[key]
    result = None
    pkg = os.path.join(pipe_root, "jwst_gc_pipeline")
    if not os.path.isdir(pkg):
        # a bogus pipe_root must not silently resolve to an INSTALLED copy of
        # the package (the pipeline env has one on sys.path)
        _warn_fallback(field, obs, f"no jwst_gc_pipeline package under "
                       f"{pipe_root}")
    else:
        argv = [python, "-c", PROBE_CODE, pipe_root, str(field), str(obs),
                *[str(f) for f in filters]]
        try:
            proc = subprocess.run(argv, capture_output=True, text=True,
                                  timeout=timeout)
        except subprocess.TimeoutExpired:
            _warn_fallback(field, obs, f"probe timed out after {timeout}s")
        except OSError as ex:   # FileNotFoundError: missing python binary etc.
            _warn_fallback(field, obs, f"could not run {python}: {ex}")
        else:
            if proc.returncode != 0:
                tail = (proc.stderr or "").strip().splitlines()
                _warn_fallback(field, obs, f"rc={proc.returncode}: "
                               f"{tail[-1] if tail else '?'}")
            else:
                try:
                    parsed = json.loads(proc.stdout)
                except json.JSONDecodeError as ex:
                    parsed = None
                    _warn_fallback(field, obs, f"unparseable probe output: {ex}")
                if parsed is not None:
                    if (isinstance(parsed, dict)
                            and isinstance(parsed.get("each_suffix"), str)
                            and isinstance(parsed.get("destreaks"), dict)):
                        result = parsed
                    else:
                        _warn_fallback(field, obs,
                                       f"malformed probe payload: {parsed!r}")
    _CACHE[key] = result
    return result
