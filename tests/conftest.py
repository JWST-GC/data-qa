"""Shared test fixtures.

The suite is offline.  ``post_status`` now runs ``_github.check_auth`` (a ``GET
/user`` preflight) before it touches the issue listing, so without a stub every test
that posts would reach api.github.com with a fake token.  The fixture below neutralises
the preflight and clears its per-token memo; the tests that EXERCISE the preflight
(tests/test_github_token.py) re-patch ``check_auth`` or ``_github.request`` themselves,
so the stub hides nothing they assert.
"""
import pytest

from data_qa import _github


@pytest.fixture(autouse=True)
def _offline_auth_preflight(monkeypatch):
    _github._AUTH_CHECKED.clear()
    monkeypatch.setattr(_github, "check_auth", lambda token, force=False:
                        (True, "pytest-stub"))
    yield
    _github._AUTH_CHECKED.clear()
