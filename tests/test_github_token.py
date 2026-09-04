"""get_token()'s ladder: env -> PAT file -> `gh auth token` (issue #148).

The file rung is the headless-safe source: under scron/cron `gh auth token` can hand
back an INVALID token, which is why scripts/refresh_all_issues.sh has read the PAT file
since it was written.  These tests pin the ORDER (the file is tried BEFORE `gh`, never
after) and that `gh` is not consulted at all once the file supplies a token.

No test reads or prints the real ~/.config/data-qa/github_token: every case points
_github.TOKEN_FILE at a tmp_path.
"""
import subprocess

import pytest

from data_qa import _github

# captured at import, before the autouse fixture repoints it at a tmp_path
_DEFAULT_TOKEN_FILE = _github.TOKEN_FILE


@pytest.fixture(autouse=True)
def _no_ambient_token(monkeypatch, tmp_path):
    """Clear the env and point the file rung at an empty tmp dir; ban real `gh`."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setattr(_github, "TOKEN_FILE", str(tmp_path / "absent_token"))

    def _no_subprocess(*a, **kw):
        raise AssertionError(f"`gh` must not be invoked here: {a!r}")

    monkeypatch.setattr(_github.subprocess, "run", _no_subprocess)


def _fake_gh(monkeypatch, stdout, returncode=0):
    def run(cmd, **kw):
        assert cmd[:3] == ["gh", "auth", "token"]
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr="")

    monkeypatch.setattr(_github.subprocess, "run", run)


def test_file_rung_supplies_the_token_and_gh_is_not_consulted(monkeypatch, tmp_path):
    """The regression: no env token, a PAT file on disk -> the file's token."""
    tok_file = tmp_path / "github_token"
    tok_file.write_text("github_pat_fromfile\n")
    monkeypatch.setattr(_github, "TOKEN_FILE", str(tok_file))
    # the autouse fixture makes any subprocess.run call an AssertionError, so this
    # also pins that the `gh` rung is never reached when the file has a token
    assert _github.get_token() == "github_pat_fromfile"


def test_env_still_wins_over_the_file(monkeypatch, tmp_path):
    tok_file = tmp_path / "github_token"
    tok_file.write_text("github_pat_fromfile")
    monkeypatch.setattr(_github, "TOKEN_FILE", str(tok_file))
    monkeypatch.setenv("GITHUB_TOKEN", "env_tok")
    assert _github.get_token() == "env_tok"
    monkeypatch.delenv("GITHUB_TOKEN")
    monkeypatch.setenv("GH_TOKEN", "gh_env_tok")
    assert _github.get_token() == "gh_env_tok"


def test_file_rung_comes_before_the_gh_fallback(monkeypatch, tmp_path):
    """`gh` can hand back an invalid token headless; the file must win over it."""
    tok_file = tmp_path / "github_token"
    tok_file.write_text("github_pat_fromfile")
    monkeypatch.setattr(_github, "TOKEN_FILE", str(tok_file))
    _fake_gh(monkeypatch, "stale_gh_token\n")
    assert _github.get_token() == "github_pat_fromfile"


def test_missing_or_empty_file_falls_through_to_gh(monkeypatch, tmp_path):
    _fake_gh(monkeypatch, "gh_tok\n")
    assert _github.get_token() == "gh_tok"          # TOKEN_FILE does not exist
    empty = tmp_path / "empty_token"
    empty.write_text("   \n")
    monkeypatch.setattr(_github, "TOKEN_FILE", str(empty))
    assert _github.get_token() == "gh_tok"


def test_unreadable_file_falls_through_rather_than_raising(monkeypatch, tmp_path):
    """A directory (or a 000-perm file) at the path must not crash the report path."""
    monkeypatch.setattr(_github, "TOKEN_FILE", str(tmp_path))   # a directory
    _fake_gh(monkeypatch, "gh_tok\n")
    assert _github.get_token() == "gh_tok"


def test_token_file_default_is_the_documented_path():
    """scripts/refresh_all_issues.sh and the scrontab docs name this exact path."""
    assert _DEFAULT_TOKEN_FILE == "~/.config/data-qa/github_token"


def test_token_from_file_strips_and_expands(monkeypatch, tmp_path):
    tok_file = tmp_path / "github_token"
    tok_file.write_text("  github_pat_padded\n\n")
    assert _github.token_from_file(str(tok_file)) == "github_pat_padded"
    monkeypatch.setenv("HOME", str(tmp_path))
    assert _github.token_from_file("~/github_token") == "github_pat_padded"
    assert _github.token_from_file(str(tmp_path / "nope")) is None
