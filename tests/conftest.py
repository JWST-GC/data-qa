"""Shared test fixtures.

The suite is offline.  ``post_status`` now runs ``_github.check_auth`` (a ``GET
/user`` preflight) before it touches the issue listing, so without a stub every test
that posts would reach api.github.com with a fake token.  The fixture below neutralises
the preflight and clears its per-token memo; the tests that EXERCISE the preflight
(tests/test_github_token.py) re-patch ``check_auth`` or ``_github.request`` themselves,
so the stub hides nothing they assert.
"""
import pytest


@pytest.fixture(autouse=True)
def _offline_auth_preflight(monkeypatch):
    # Import inside the fixture, not at module top: CI runs ``pytest tests data_qa`` (no editable
    # install, no ``python -m``), and pytest imports this conftest during collection BOOTSTRAP --
    # before the repo root reaches sys.path -- so a top-level ``from data_qa import _github`` here
    # dies with ModuleNotFoundError and takes the whole suite down. By fixture-run time the path is
    # set up and the import succeeds.
    from data_qa import _github
    _github._AUTH_CHECKED.clear()
    monkeypatch.setattr(_github, "check_auth", lambda token, force=False:
                        (True, "pytest-stub"))
    yield
    _github._AUTH_CHECKED.clear()
