"""Locations data_qa reaches OUTSIDE its own checkout, resolved in one place.

Today that is the jwst-gc-pipeline checkout: the trigger submits its sbatch
scripts, the policy probe imports its modules, and the treasury imaging borrows
``jwst_gc_pipeline.cmz.hips``.  The path was written out in four modules and
only the test copy honoured ``$PIPE_ROOT`` -- the one knob a scrontab entry and
a GitHub runner can both set (issue #85).

``pipe_root()`` mirrors ``pipeline_policy.pipeline_python()``: env first,
HiPerGator default second, read at CALL time so a caller that sets the variable
(or a test that monkeypatches it) is honoured.  Explicit arguments still win --
the ``--pipe-root`` CLI flags pass one through, and an explicit value never
consults the environment.

Stdlib-only, like the rest of the trigger path.
"""
from __future__ import annotations

import os

#: The live reduction checkout on HiPerGator (CLAUDE.md's "active `main`
#: working tree").
DEFAULT_PIPE_ROOT = "/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline"

#: Environment variable naming a different checkout.
PIPE_ROOT_ENV = "PIPE_ROOT"


def pipe_root(default: str = DEFAULT_PIPE_ROOT) -> str:
    """The jwst-gc-pipeline checkout to use: ``$PIPE_ROOT`` when set and
    non-empty, else ``default`` (the HiPerGator path).

    An empty ``$PIPE_ROOT`` falls back rather than resolving every path against
    the current directory.
    """
    return os.environ.get(PIPE_ROOT_ENV) or default
